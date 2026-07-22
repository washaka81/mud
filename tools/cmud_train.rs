//! # C-MUD Trainer (research §4, #4 entrenable) — experimental
//!
//! Real gradient-descent (Adam) optimizer for the complex-reasoning pass (`CmudLayerParams`),
//! using the EXACT analytic gradient from [`cmud_training_forward`] (backprop through the real
//! model forward + LM head). Minimizes next-token cross-entropy on a token corpus and saves a
//! JSON sidecar consumed at inference via `MUD_CMUD_PARAMS`. Also trains `v_scale` and `alpha`.
//!
//! The production f32 path is untouched; C-MUD stays opt-in.
//!
//! Usage:
//!   ./mud.sh cmud-train models/smollm2.mud --tokens corpus.txt [--dims 0] [--steps 20] [--lr 0.01]
//!
//! `corpus.txt`: one token sequence per line, space-separated `u32` ids (target = last id).
//! `--dims 0` trains all dimensions (default); `--dims N` caps to the first N.

use forge_llm::mud::cmud::{CmudLayerParams, CmudLayerParamsGrad};
use forge_llm::mud::inference::cmud_training_forward;
use forge_llm::mud::MudFile;
use std::path::PathBuf;
use std::process::ExitCode;

fn parse_args() -> (String, String, usize, usize, f32, String, usize) {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model = args
        .iter()
        .find(|a| !a.starts_with('-'))
        .cloned()
        .unwrap_or_else(|| "models/smollm2.mud".to_string());
    let mut tokens = "corpus.txt".to_string();
    let mut dims = 0usize;
    let mut steps = 20usize;
    let mut lr = 0.01f32;
    let mut max_seqs = 128usize;
    let mut out: Option<String> = None;
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
            "--steps" => {
                if let Some(v) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                    steps = v;
                }
            }
            "--lr" => {
                if let Some(v) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                    lr = v;
                }
            }
            "--max-seqs" => {
                if let Some(v) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                    max_seqs = v;
                }
            }
            "--out" => {
                out = args.get(i + 1).cloned();
            }
            _ => {}
        }
        i += 1;
    }
    let out = out.unwrap_or_else(|| {
        CmudLayerParams::sidecar_for(&PathBuf::from(&model))
            .to_string_lossy()
            .to_string()
    });
    (model, tokens, dims, steps, lr, out, max_seqs)
}

fn main() -> ExitCode {
    let (model_path, tokens_path, dims, steps, lr, out_path, max_seqs) = parse_args();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  C-MUD TRAINER  ·  Adam (analytic gradient, real forward)  ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!("Model: {model_path}");
    println!("Corpus: {tokens_path}  dims={dims}  steps={steps}  lr={lr}  max_seqs={max_seqs}");
    println!("Out:    {out_path}\n");

    let mud = match MudFile::load(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("  ❌ load failed: {e}");
            return ExitCode::from(1);
        }
    };
    let hidden = mud
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
    let vocab_size = mud
        .global_metadata
        .get("vocab_size")
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(49152);

    let tokens_str = mud.global_metadata.get("tokenizer.tokens").map(|s| s.as_str()).unwrap_or("");
    let merges_str = mud.global_metadata.get("tokenizer.merges").map(|s| s.as_str()).unwrap_or("");
    let tokenizer = forge_llm::model::tokenizer::Tokenizer::from_mud_metadata(tokens_str, merges_str);

    let mut seqs: Vec<(Vec<u32>, u32)> = raw
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| {
            let ids: Vec<u32> = l
                .split_whitespace()
                .filter_map(|w| w.parse::<u32>().ok())
                .filter(|&id| id < vocab_size)
                .collect();
            if ids.len() >= 2 {
                let target = *ids.last().unwrap();
                Some((ids[..ids.len() - 1].to_vec(), target))
            } else {
                None
            }
        })
        .collect();

    if seqs.is_empty() {
        let ids = tokenizer.encode(&raw);
        let valid_ids: Vec<u32> = ids.into_iter().filter(|&id| id < vocab_size).collect();
        if valid_ids.len() >= 2 {
            for window in valid_ids.windows(16) {
                let target = *window.last().unwrap();
                seqs.push((window[..window.len() - 1].to_vec(), target));
            }
        }
    }
    if seqs.is_empty() {
        eprintln!("  ❌ no usable sequences (need >=2 ids per line)");
        return ExitCode::from(1);
    }
    if max_seqs > 0 && seqs.len() > max_seqs {
        seqs.truncate(max_seqs);
    }
    println!("  corpus sequences: {}", seqs.len());

    let out_pb = PathBuf::from(&out_path);
    std::env::set_var("MUD_CMUD_PARAMS", &out_path);

    // Resume from existing sidecar if present, else identity defaults.
    let mut params = match CmudLayerParams::load_json(&out_pb) {
        Ok(p) if p.q_phase.len() == hidden => p,
        _ => CmudLayerParams::from_defaults(hidden),
    };
    let d = if dims == 0 { hidden } else { dims.min(hidden) };

    let empty_grad = || CmudLayerParamsGrad {
        q_phase: vec![0.0f32; hidden],
        k_phase: vec![0.0f32; hidden],
        v_scale: vec![0.0f32; hidden],
        alpha: 0.0,
    };

    // Adam state.
    let mut mq = vec![0.0f32; hidden];
    let mut vq = vec![0.0f32; hidden];
    let mut mk = vec![0.0f32; hidden];
    let mut vk = vec![0.0f32; hidden];
    let mut ms = vec![0.0f32; hidden];
    let mut vs = vec![0.0f32; hidden];
    let (mut ma, mut va) = (0.0f32, 0.0f32);
    let (b1, b2, ea) = (0.9f32, 0.999f32, 1e-8f32);

    let base = seqs
        .iter()
        .map(|(c, t)| cmud_training_forward(&mud, c, *t).map(|(l, _, _)| l))
        .try_fold(0.0f32, |acc, r| r.map(|v| acc + v))
        .unwrap_or(0.0)
        / seqs.len() as f32;
    println!("  initial CE = {base:.4}");
    use std::io::Write;
    let _ = std::io::stdout().flush();

    let mut best = params.clone();
    let mut best_loss = base;
    for t in 1..=steps {
        // Persist current params so `cmud_training_forward` loads them for the gradient eval.
        let _ = params.save_json(&out_pb);
        let mut g = empty_grad();
        let mut loss = 0.0f32;
        for (c, target) in &seqs {
            let (l, grad, _) = cmud_training_forward(&mud, c, *target)
                .unwrap_or_else(|_| (0.0, empty_grad(), params.clone()));
            loss += l;
            for i in 0..hidden {
                g.q_phase[i] += grad.q_phase[i];
                g.k_phase[i] += grad.k_phase[i];
                g.v_scale[i] += grad.v_scale[i];
            }
            g.alpha += grad.alpha;
        }
        let inv = 1.0 / seqs.len() as f32;
        loss *= inv;
        for i in 0..hidden {
            g.q_phase[i] *= inv;
            g.k_phase[i] *= inv;
            g.v_scale[i] *= inv;
        }
        g.alpha *= inv;

        let bc1 = 1.0 - b1.powi(t as i32);
        let bc2 = 1.0 - b2.powi(t as i32);
        for i in 0..d {
            mq[i] = b1 * mq[i] + (1.0 - b1) * g.q_phase[i];
            vq[i] = b2 * vq[i] + (1.0 - b2) * g.q_phase[i] * g.q_phase[i];
            let mh = mq[i] / bc1;
            let vh = vq[i] / bc2;
            params.q_phase[i] -= lr * mh / (vh.sqrt() + ea);

            mk[i] = b1 * mk[i] + (1.0 - b1) * g.k_phase[i];
            vk[i] = b2 * vk[i] + (1.0 - b2) * g.k_phase[i] * g.k_phase[i];
            let mh = mk[i] / bc1;
            let vh = vk[i] / bc2;
            params.k_phase[i] -= lr * mh / (vh.sqrt() + ea);

            ms[i] = b1 * ms[i] + (1.0 - b1) * g.v_scale[i];
            vs[i] = b2 * vs[i] + (1.0 - b2) * g.v_scale[i] * g.v_scale[i];
            let mh = ms[i] / bc1;
            let vh = vs[i] / bc2;
            params.v_scale[i] -= lr * mh / (vh.sqrt() + ea);
        }
        ma = b1 * ma + (1.0 - b1) * g.alpha;
        va = b2 * va + (1.0 - b2) * g.alpha * g.alpha;
        let mh = ma / bc1;
        let vh = va / bc2;
        params.alpha = (params.alpha - lr * mh / (vh.sqrt() + ea)).clamp(0.0, 1.0);

        if loss < best_loss {
            best_loss = loss;
            best = params.clone();
        }
        println!("  step {t}/{steps}: CE = {loss:.4} (best = {best_loss:.4})");
        use std::io::Write;
        let _ = std::io::stdout().flush();
    }

    if let Err(e) = best.save_json(&out_pb) {
        eprintln!("  ❌ save failed: {e}");
        return ExitCode::from(1);
    }
    println!("\n  ✅ saved trained params → {out_path}");
    println!("  run inference with: MUD_CMUD_THINK=1 MUD_CMUD_PARAMS={out_path} ./mud.sh run {model_path}");
    ExitCode::from(0)
}
