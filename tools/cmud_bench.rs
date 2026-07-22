//! # C-MUD reasoning benchmark (research §5 — "does the complex thinking pass actually help?")
//!
//! On a token corpus, compares next-token cross-entropy and top-1 accuracy WITH vs WITHOUT the
//! complex (`MUD_CMUD_THINK`) pass, and reports how often thinking flips the argmax prediction.
//! Also prints manifold diagnostics (Hermitian norm / phase-lock) for the first sequence so you
//! can see whether thinking converges to a coherent state or collapses.
//!
//! Usage:
//!   ./mud.sh cmud-bench models/smollm2.mud --tokens corpus.txt [--params path.json] [--steps 8]
//!
//! `corpus.txt`: one token sequence per line, space-separated `u32` ids (target = last id).

use forge_llm::mud::cmud::{
    CmudLayerParams, DEFAULT_THINK_ITERS, CMUD_RADIUS_RMS_FACTOR, ThinkingState,
};
use forge_llm::mud::inference::forward_last_hidden_and_head;
use forge_llm::mud::MudFile;
use std::process::ExitCode;

fn ce(logits: &[f32], target: u32) -> f32 {
    let vocab = logits.len();
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum = logits.iter().map(|&l| (l - max).exp()).sum::<f32>().ln();
    let lv = logits.get(target as usize).copied().unwrap_or(0.0);
    -(lv - max) + sum + (vocab as f32).ln() * 0.0 // plain softmax CE
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn main() -> ExitCode {
    let (model_path, tokens_path, params_path, steps) = parse_args();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  C-MUD REASONING BENCH  ·  think ON vs OFF                ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!("Model: {model_path}  Corpus: {tokens_path}  steps={steps}\n");

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

    let mut params = match &params_path {
        Some(p) => CmudLayerParams::load_json(std::path::Path::new(p))
            .unwrap_or_else(|_| CmudLayerParams::from_defaults(hidden)),
        None => CmudLayerParams::from_defaults(hidden),
    };
    let alpha = std::env::var("MUD_CMUD_ALPHA")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(params.alpha);
    params.alpha = alpha.clamp(0.0, 1.0);

    let raw = match std::fs::read_to_string(&tokens_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("  ❌ cannot read corpus: {e}");
            return ExitCode::from(1);
        }
    };
    let seqs: Vec<(Vec<u32>, u32)> = raw
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
        .collect();
    if seqs.is_empty() {
        eprintln!("  ❌ no usable sequences (need >=2 ids per line)");
        return ExitCode::from(1);
    }
    println!("  corpus sequences: {}", seqs.len());

    let mut sum_ce_off = 0.0f32;
    let mut sum_ce_on = 0.0f32;
    let mut acc_off = 0usize;
    let mut acc_on = 0usize;
    let mut flips = 0usize;

    // Diagnostics on the first sequence only.
    let mut diag_done = false;

    for (ci, (ctx, target)) in seqs.iter().enumerate() {
        let (logits_off, reg_pre, head_w, hidden2, vocab) =
            match forward_last_hidden_and_head(&mud, ctx, &mut None) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("  ❌ forward failed: {e}");
                    return ExitCode::from(1);
                }
            };

        // --- thinking ON ---
        let rms = (reg_pre.iter().map(|v| v * v).sum::<f32>() / hidden2.max(1) as f32)
            .max(1e-12)
            .sqrt();
        let radius = CMUD_RADIUS_RMS_FACTOR * rms;
        let mut st = ThinkingState::from_real(&reg_pre, radius);
        let mut max_norm = 0.0f32;
        for _ in 0..steps {
            st.think_step_trainable(&params);
            if !diag_done {
                let n = st.h.iter().map(|h| h.hermite_norm()).fold(0.0f32, f32::max);
                max_norm = max_norm.max(n);
            }
        }
        let mut reg_post = vec![0.0f32; hidden2];
        st.collapse_to_real(&mut reg_post);
        let mut logits_on = vec![0.0f32; vocab];
        for v in 0..vocab {
            let row = &head_w[v * hidden2..v * hidden2 + hidden2];
            let mut s = 0.0f32;
            for h in 0..hidden2 {
                s += reg_post[h] * row[h];
            }
            logits_on[v] = s;
        }

        let ce_off = ce(&logits_off, *target);
        let ce_on = ce(&logits_on, *target);
        sum_ce_off += ce_off;
        sum_ce_on += ce_on;
        let a_off = argmax(&logits_off);
        let a_on = argmax(&logits_on);
        if a_off == *target as usize {
            acc_off += 1;
        }
        if a_on == *target as usize {
            acc_on += 1;
        }
        if a_off != a_on {
            flips += 1;
        }

        if ci == 0 {
            println!(
                "  [diag seq0] radius={radius:.4}  max Hermitian norm={max_norm:.4} (≤{radius:.4})  phase_locked={}",
                st.is_phase_locked()
            );
            diag_done = true;
        }
    }

    let n = seqs.len() as f32;
    let m_ce_off = sum_ce_off / n;
    let m_ce_on = sum_ce_on / n;
    let acc_off_pct = 100.0 * acc_off as f32 / n;
    let acc_on_pct = 100.0 * acc_on as f32 / n;
    let flip_pct = 100.0 * flips as f32 / n;

    println!("\n  ┌─────────────────────┬─────────────┬─────────────┬───────────┐");
    println!("  │  metric             │  think OFF  │  think ON   │  Δ        │");
    println!("  ├─────────────────────┼─────────────┼─────────────┼───────────┤");
    println!(
        "  │  CE (lower=better)  │  {:>9.4}  │  {:>9.4}  │  {:>+7.4}  │",
        m_ce_off, m_ce_on, m_ce_on - m_ce_off
    );
    println!(
        "  │  top-1 acc (%)      │  {:>9.2}  │  {:>9.2}  │  {:>+7.2}  │",
        acc_off_pct, acc_on_pct, acc_on_pct - acc_off_pct
    );
    println!(
        "  │  argmax flips (%)   │  {:>9.1}  │  {:>9.1}  │  {:>7.1}  │",
        flip_pct, flip_pct, 0.0
    );
    println!("  └─────────────────────┴─────────────┴─────────────┴───────────┘");

    if m_ce_on < m_ce_off {
        println!(
            "\n  ✅ thinking LOWERS CE by {:.4} (helps).",
            m_ce_off - m_ce_on
        );
    } else {
        println!(
            "\n  ⚠ thinking RAISES CE by {:.4} (does not help on this corpus / params).",
            m_ce_on - m_ce_off
        );
    }
    ExitCode::from(0)
}

fn parse_args() -> (String, String, Option<String>, usize) {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model = args
        .iter()
        .find(|a| !a.starts_with('-'))
        .cloned()
        .unwrap_or_else(|| "models/smollm2.mud".to_string());
    let mut tokens = "corpus.txt".to_string();
    let mut params: Option<String> = None;
    let mut steps = DEFAULT_THINK_ITERS;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--tokens" => {
                if let Some(v) = args.get(i + 1) {
                    tokens = v.clone();
                }
            }
            "--params" => params = args.get(i + 1).cloned(),
            "--steps" => {
                if let Some(v) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                    steps = v;
                }
            }
            _ => {}
        }
        i += 1;
    }
    (model, tokens, params, steps)
}
