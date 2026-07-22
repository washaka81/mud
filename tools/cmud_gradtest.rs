//! # C-MUD end-to-end gradient check (research §4, #4 — "does the analytic gradient match FD?")
//!
//! Validates `cmud_training_forward`'s analytic gradient against central finite differences on
//! the REAL model forward + exact LM head + collapse-corrected chain rule. This is the integration
//! test that would have caught a sign error in the `|·|` collapse chain (see `wave_collapse`).
//!
//! Usage:
//!   ./mud.sh cmud-gradtest models/smollm2.mud --tokens corpus.txt [--dims 32] [--seqs 2]
//!
//! `corpus.txt`: one token sequence per line, space-separated `u32` ids (target = last id).

use forge_llm::mud::cmud::{
    cmud_backward, CmudLayerParams, CmudLayerParamsGrad, ThinkingState, CMUD_RADIUS_RMS_FACTOR,
    DEFAULT_THINK_ITERS,
};
use forge_llm::mud::inference::forward_last_hidden_and_head;
use forge_llm::mud::MudFile;
use std::process::ExitCode;

struct CachedSeq {
    reg_pre: Vec<f32>,
    target: u32,
}

fn cached_forward_loss_grad(
    cached: &CachedSeq,
    head_w: &[f32],
    hidden: usize,
    vocab: usize,
    params: &CmudLayerParams,
) -> (f32, CmudLayerParamsGrad) {
    let rms = (cached.reg_pre.iter().map(|v| v * v).sum::<f32>() / hidden.max(1) as f32)
        .max(1e-12)
        .sqrt();
    let radius = CMUD_RADIUS_RMS_FACTOR * rms;

    let mut st = ThinkingState::from_real(&cached.reg_pre, radius);
    let mut tapes = Vec::new();
    for _ in 0..DEFAULT_THINK_ITERS {
        st.think_step_trainable_record(params, &mut tapes);
    }
    let mut reg_post = vec![0.0f32; hidden];
    st.collapse_to_real(&mut reg_post);

    let mut logits = vec![0.0f32; vocab];
    for v in 0..vocab {
        let row = &head_w[v * hidden..v * hidden + hidden];
        let mut s = 0.0f32;
        for h in 0..hidden {
            s += reg_post[h] * row[h];
        }
        logits[v] = s;
    }

    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for &l in logits.iter() {
        sum += (l - max).exp();
    }
    let loss = match logits.get(cached.target as usize) {
        Some(&lv) => -(lv - max) + sum.ln(),
        None => 0.0,
    };

    let mut g_logits = vec![0.0f32; vocab];
    for v in 0..vocab {
        g_logits[v] = (logits[v] - max).exp() / sum;
    }
    if (cached.target as usize) < vocab {
        g_logits[cached.target as usize] -= 1.0;
    }

    let mut grad_reg = vec![0.0f32; hidden];
    for v in 0..vocab {
        let gv = g_logits[v];
        if gv == 0.0 {
            continue;
        }
        let row = &head_w[v * hidden..v * hidden + hidden];
        for h in 0..hidden {
            grad_reg[h] += gv * row[h];
        }
    }

    let mut grad = CmudLayerParamsGrad::default();
    let last = tapes.last().unwrap();
    let mut grad_mixed = grad_reg;
    for (gm, (hb, ai)) in grad_mixed
        .iter_mut()
        .zip(last.h_before.iter().zip(last.attn.iter()))
    {
        let mixed = hb + last.alpha * (ai - hb);
        *gm *= mixed.signum();
    }
    cmud_backward(&tapes, &grad_mixed, params, &mut grad);
    (loss, grad)
}

fn main() -> ExitCode {
    let (model_path, tokens_path, dims, seqs) = parse_args();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  C-MUD GRADIENT CHECK  ·  analytic vs finite-difference    ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!("Model: {model_path}  Corpus: {tokens_path}  dims={dims}  seqs={seqs}\n");

    let mud = match MudFile::load(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("  ❌ load failed: {e}");
            return ExitCode::from(1);
        }
    };
    let hidden_meta = mud
        .global_metadata
        .get("hidden_size")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(576);

    let raw = match std::fs::read_to_string(&tokens_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("  ❌ cannot read corpus: {e}");
            return ExitCode::from(1);
        }
    };
    let seqs_data: Vec<(Vec<u32>, u32)> = raw
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| {
            let ids: Vec<u32> = l.split_whitespace().filter_map(|w| w.parse().ok()).collect();
            if ids.len() >= 2 {
                Some((ids[..ids.len() - 1].to_vec(), *ids.last().unwrap()))
            } else {
                None
            }
        })
        .take(seqs)
        .collect();
    if seqs_data.is_empty() {
        eprintln!("  ❌ no usable sequences (need >=2 ids per line)");
        return ExitCode::from(1);
    }

    println!("  caching transformer forwards for {} sequences...", seqs_data.len());
    let mut cached_seqs = Vec::new();
    let mut shared_head_w = Vec::new();
    let mut actual_vocab = 0;
    
    for (i, (c, t)) in seqs_data.iter().enumerate() {
        match forward_last_hidden_and_head(&mud, c, &mut None) {
            Ok((_, reg_pre, head_w, _h, v)) => {
                if i == 0 {
                    shared_head_w = head_w;
                    actual_vocab = v;
                }
                cached_seqs.push(CachedSeq {
                    reg_pre,
                    target: *t,
                });
            }
            Err(e) => {
                eprintln!("  ❌ forward failed: {e}");
                return ExitCode::from(1);
            }
        }
    }

    let hidden = hidden_meta;
    let params = CmudLayerParams::from_defaults(hidden);
    if dims == 0 {
        println!("  checking all {hidden} dims");
    } else {
        println!("  checking first {dims} dims");
    }
    let d = if dims == 0 { hidden } else { dims.min(hidden) };
    let eps = 1e-3f32;

    // Analytic gradient
    let mut g_q = vec![0.0f32; hidden];
    let mut g_k = vec![0.0f32; hidden];
    let mut g_v = vec![0.0f32; hidden];
    let mut g_a = 0.0f32;
    for c in &cached_seqs {
        let (_, grad) = cached_forward_loss_grad(c, &shared_head_w, hidden, actual_vocab, &params);
        for i in 0..hidden {
            g_q[i] += grad.q_phase[i];
            g_k[i] += grad.k_phase[i];
            g_v[i] += grad.v_scale[i];
        }
        g_a += grad.alpha;
    }
    let inv = 1.0 / cached_seqs.len() as f32;
    for i in 0..hidden {
        g_q[i] *= inv;
        g_k[i] *= inv;
        g_v[i] *= inv;
    }
    g_a *= inv;

    // Finite differences: perturb each field, re-run just the thinking pass + LM head
    let loss_of = |mut p: CmudLayerParams, idx: usize, which: u8, delta: f32| -> f32 {
        match which {
            0 => p.q_phase[idx] += delta,
            1 => p.k_phase[idx] += delta,
            2 => p.v_scale[idx] += delta,
            _ => p.alpha += delta,
        }
        let mut total = 0.0f32;
        for c in &cached_seqs {
            total += cached_forward_loss_grad(c, &shared_head_w, hidden, actual_vocab, &p).0;
        }
        total * inv
    };

    let mut max_err = 0.0f32;
    let mut worst = String::new();
    for i in 0..d {
        let fd_q = (loss_of(params.clone(), i, 0, eps) - loss_of(params.clone(), i, 0, -eps))
            / (2.0 * eps);
        let fd_k = (loss_of(params.clone(), i, 1, eps) - loss_of(params.clone(), i, 1, -eps))
            / (2.0 * eps);
        let fd_v = (loss_of(params.clone(), i, 2, eps) - loss_of(params.clone(), i, 2, -eps))
            / (2.0 * eps);

        let err_q = (g_q[i] - fd_q).abs();
        let err_k = (g_k[i] - fd_k).abs();
        let err_v = (g_v[i] - fd_v).abs();

        if err_q > max_err { max_err = err_q; worst = format!("q[{i}] an={:.5} fd={:.5}", g_q[i], fd_q); }
        if err_k > max_err { max_err = err_k; worst = format!("k[{i}] an={:.5} fd={:.5}", g_k[i], fd_k); }
        if err_v > max_err { max_err = err_v; worst = format!("v[{i}] an={:.5} fd={:.5}", g_v[i], fd_v); }
    }
    let fd_a = (loss_of(params.clone(), 0, 3, eps) - loss_of(params.clone(), 0, 3, -eps))
        / (2.0 * eps);
    let err_a = (g_a - fd_a).abs();
    if err_a > max_err {
        max_err = err_a;
        worst = format!("alpha an={g_a:.5} fd={fd_a:.5}");
    }

    println!("  analytic vs FD mismatch (max abs): {max_err:.6}");
    if !worst.is_empty() {
        println!("  worst: {worst}");
    }
    if max_err < 1e-2 {
        println!("\n  ✅ PASS — analytic gradient matches finite differences end-to-end.");
        ExitCode::from(0)
    } else {
        println!("\n  ❌ FAIL — analytic gradient diverges from finite differences.");
        ExitCode::from(1)
    }
}

fn parse_args() -> (String, String, usize, usize) {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model = args
        .iter()
        .find(|a| !a.starts_with('-'))
        .cloned()
        .unwrap_or_else(|| "models/smollm2.mud".to_string());
    let mut tokens = "corpus.txt".to_string();
    let mut dims = 32usize;
    let mut seqs = 2usize;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--tokens" => {
                if let Some(v) = args.get(i + 1) {
                    tokens = v.clone();
                }
            }
            "--dims" => {
                if let Some(v) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                    dims = v;
                }
            }
            "--seqs" => {
                if let Some(v) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                    seqs = v;
                }
            }
            _ => {}
        }
        i += 1;
    }
    (model, tokens, dims, seqs)
}
