//! `.mud` integrity gates + verifiable-game benchmark.
//!
//! These three functions are the "circuit honors" machinery. The training
//! circuit (`circuit.rs`) calls them after every phase to decide whether the
//! phase's writeback should be kept or rolled back.
//!
//! - [`model_norms_alive`]: fail-fast on a "Dead RMSNorm" model (all-zero
//!   norms in the first layers → forward panics with "Dead RMSNorm").
//! - [`circuit_eval_integrity`]: structural check that the saved `.mud`
//!   has all STE-writable weight tensors present and non-null AND that
//!   the model has not collapsed to token-0 dominance.
//! - [`circuit_benchmark_games`]: runs the model in `games` mode over an
//!   internal channel to compute a verifiable-game win rate.

use crate::mud::corpus_trainer::MudCorpusTrainer;
use crate::mud::{MudFile, MudTensor, MudTensorType};

/// Returns `(alive, detail)` where `alive` is false when the model's
/// RMSNorm weights are all-zero / non-finite in the first layers — the
/// classic "collapsed .mud" that triggers `Dead RMSNorm` on the first
/// forward. Norms are frozen on RO mmap during training, so a collapsed
/// model cannot be repaired in-place and must be replaced with a healthy
/// `.mud` before running the circuit.
pub fn model_norms_alive(path: &str) -> (bool, String) {
    let mud = match MudFile::load(path) {
        Ok(m) => m,
        Err(e) => return (false, format!("load error: {}", e)),
    };
    let n_layers = mud
        .global_metadata
        .get("num_hidden_layers")
        .or_else(|| mud.global_metadata.get("num_layers"))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    if n_layers == 0 {
        return (false, "no layers in metadata".to_string());
    }
    let Some(core) = mud.skills.get("core") else {
        return (false, "no core skill".to_string());
    };
    let mut dead = 0usize;
    let mut checked = 0usize;
    for blk in 0..n_layers.min(4) {
        for name in ["attn_norm.weight", "ffn_norm.weight", "norm.weight"] {
            let key = format!("blk.{}.{}", blk, name);
            let Some(t) = core.tensors.get(&key) else {
                continue;
            };
            if t.t_type != crate::mud::MudTensorType::Float32 {
                continue;
            }
            if t.data_ptr.is_null() {
                dead += 1;
                checked += 1;
                continue;
            }
            let n = t.shape.iter().product::<usize>().clamp(1, 64);
            let slice = unsafe { std::slice::from_raw_parts(t.data_ptr as *const f32, n) };
            let nz = slice.iter().filter(|x| **x != 0.0 && x.is_finite()).count();
            checked += 1;
            if nz == 0 {
                dead += 1;
            }
        }
    }
    if dead > 0 {
        (
            false,
            format!(
                "{} RMSNorm tensors all-zero/non-finite in first layers (model collapsed; needs healthy .mud)",
                dead
            ),
        )
    } else {
        (true, format!("norms alive ({} checked)", checked))
    }
}

/// Honors-mode evaluation of a `.mud` after a training phase.
///
/// Returns `(integrity_ok, detail)`. Two considerations:
///   1. **Norms** — a collapsed model (all-zero RMSNorm) fails fast on the
///      forward and cannot be repaired; flagged here with a clear message.
///   2. **Integrity** — the file loads and its STE-writable weights are
///      present / non-null / non-zero size (no degenerate writeback).
///
/// The circuit uses this to decide whether to KEEP the phase's writeback
/// (honors) or roll it back (keep the previous `.mud`).
pub fn circuit_eval_integrity(path: &str) -> (bool, String) {
    // A collapsed model (all-zero RMSNorm weights) cannot be repaired by the
    // circuit — norms are frozen on RO mmap and the forward fails fast with
    // "Dead RMSNorm". Fail the integrity gate early with a clear message.
    let (norms_ok, norms_detail) = model_norms_alive(path);
    if !norms_ok {
        return (false, norms_detail);
    }
    let mut mud = match MudFile::load(path) {
        Ok(m) => m,
        Err(e) => return (false, format!("load error: {}", e)),
    };
    // Materialize so `data_ptr` is backed by a writable copy (owned_data).
    mud.materialize_writable();
    let n_layers = mud
        .global_metadata
        .get("num_hidden_layers")
        .or_else(|| mud.global_metadata.get("num_layers"))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    if n_layers == 0 {
        return (false, "no layers in metadata".to_string());
    }
    // Structural integrity: every STE-writable weight tensor must be present,
    // have a valid (non-null) data pointer and non-zero size. A degenerate
    // writeback (e.g. a failed STE pack) drops/zeroes tensors — caught here.
    // Per-weight collapse (all-identical bytes) is intentionally NOT flagged,
    // because healthy ternary weights are naturally skewed toward 0x0; the
    // quality benchmark (verifiable-game win rate) is the regression gate.
    let core = match mud.skills.get("core") {
        Some(c) => c,
        None => return (false, "no core skill".to_string()),
    };
    let mut missing = 0usize;
    let mut inspected = 0usize;
    let mut checked: std::collections::HashSet<String> = std::collections::HashSet::new();
    let attn_names = ["attn_q", "attn_k", "attn_v", "attn_output"];
    // Universal FFN-name resolution (2026-08-03): reuse the SAME resolver the
    // forward/inference + checkpoint paths use (`moe_load::dense_ffn_names_for_train`),
    // instead of hardcoding dense-vs-MoE naming here. This keeps `health` ⟷
    // `circuit_eval_integrity` ⟷ training layout-agnostic and in lockstep:
    // if the forward can wire a layer, the gate considers it complete.
    // Dense (SmolLM/forge): ffn_up/gate/down  ·  MoE (DeepSeek): expert.E.w{1,2,3}
    for blk in 0..n_layers {
        for name in attn_names {
            let key = format!("blk.{}.{}.weight", blk, name);
            checked.insert(key);
        }
        let ffn = crate::mud::moe_load::dense_ffn_names_for_train(&core.tensors, blk);
        for n in [&ffn.up, &ffn.gate, &ffn.down] {
            checked.insert(format!("blk.{}.{}.weight", blk, n));
        }
    }
    for key in &checked {
        match core.tensors.get(key) {
            Some(t)
                if t.shape.iter().product::<usize>() > 0
                    && (!t.data_ptr.is_null() || t.owned_data.is_some()) =>
            {
                inspected += 1;
            }
            _ => missing += 1,
        }
    }
    if missing > 0 {
        (
            false,
            format!(
                "{} weight tensors missing/null (inspected {})",
                missing, inspected
            ),
        )
    } else {
        // T0.4 (2026-08-04): mHC radius + saturation gate.
        // Catches models forged/trained with an inflated mHC radius (forge bug:
        // `8.0·√hidden` instead of `1.5·√hidden`) or mHC alpha/beta saturated
        // at clamp bounds (0.0/4.0) from a prior broken-LR training run.
        // These pass the structural gate above but produce exploding activations
        // (cog>500, varh>7, loss above random) with zero weight movement (ΔW=0)
        // because gradients are too small to overcome the STE deadzone.
        let hidden = mud.global_metadata
            .get("hidden_size")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        if hidden > 0 {
            let sqrt_h = (hidden as f32).sqrt();
            let exp_min = sqrt_h;                    // AGENTS.md: ~√hidden
            let exp_max = sqrt_h * 2.0;              // forge(1.5×) + margin
            for blk in 0..n_layers {
                // mHC radius + saturation health (T0.4, 2026-08-04).
                // Use data_ptr (covers both mmap-backed and owned tensors).
                let f32_len = |t: &MudTensor| -> usize {
                    match t.t_type {
                        MudTensorType::Float32 => t.shape.iter().product::<usize>(),
                        _ => 0,
                    }
                };
                for suffix in ["mhc_alpha", "mhc_beta"] {
                    let key = format!("blk.{}.{}", blk, suffix);
                    if let Some(t) = core.tensors.get(&key) {
                        let len = f32_len(t);
                        if len > 0 {
                            let ptr = if let Some(d) = &t.owned_data {
                                d.as_ptr() as *const f32
                            } else if !t.data_ptr.is_null() {
                                t.data_ptr as *const f32
                            } else {
                                continue;
                            };
                            let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                            // P-01: sample every 8th element — zero-alloc, fast
                            let sat: usize = slice.iter().step_by(8)
                                .filter(|&v| *v < 0.01 || *v > 3.99)
                                .count();
                            let total = len.div_ceil(8);
                            if sat as f32 / total as f32 > 0.85 {
                                return (false, format!(
                                    "blk.{}.{} saturated ({}% at clamp bounds 0/4) — re-forge with fixed forge.rs. Override with MUD_SKIP_INTEGRITY=1",
                                    blk, suffix, sat * 100 / total.max(1)
                                ));
                            }
                        }
                    }
                }
                // Check mHC radius inflation/collapse
                let rkey = format!("blk.{}.mhc_radius", blk);
                if let Some(t) = core.tensors.get(&rkey) {
                    let len = f32_len(t);
                    if len > 0 {
                        let ptr = if let Some(d) = &t.owned_data {
                            d.as_ptr() as *const f32
                        } else if !t.data_ptr.is_null() {
                            t.data_ptr as *const f32
                        } else {
                            continue;
                        };
                        let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                        if let Some(&r) = slice.first() {
                            if r < exp_min * 0.5 || r > exp_max * 4.0 {
                                return (false, format!(
                                    "blk.{}.mhc_radius={:.1} outside healthy [{:.1}, {:.1}] — model has degenerate mHC scaling (re-forge). Override with MUD_SKIP_INTEGRITY=1",
                                    blk, r, exp_min * 0.5, exp_max * 4.0
                                ));
                            }
                        }
                    }
                }
            }
        }
        // T0.3: logits-collapse gate
        match MudFile::load(path) {
            Ok(probe) => {
                if crate::mud::inference::model_logits_collapsed(&probe) {
                    (
                        false,
                        "logits collapsed (token-0 dominance across probe prompts)".to_string(),
                    )
                } else {
                    (
                        true,
                        format!("integrity ok ({} weight tensors present)", inspected),
                    )
                }
            }
            Err(e) => (false, format!("probe load error: {}", e)),
        }
    }
}

impl MudCorpusTrainer {
    /// Fast verifiable-game benchmark: the model (player A) plays N matches of
    /// `MathChallenge` / `TicTacToe`; returns `(win_rate, matches_played)`.
    /// Reuses `run_debate_session` in `games` mode over an internal channel so no
    /// RNG / API is needed — pure local inference + `VerifiableJudge`.
    pub fn circuit_benchmark_games(
        &mut self,
        stop_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> anyhow::Result<(f32, usize)> {
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        std::env::set_var("MUD_DEBATE_MODE", "games");
        // Short, bounded benchmark so the circuit never stalls. A MathChallenge
        // match is <=4 attempts; keep tokens tiny so at least one match closes
        // within the time-box even on slow CPU inference (env-overridable).
        if std::env::var("MUD_DEBATE_MAX_TIME").is_err() {
            std::env::set_var("MUD_DEBATE_MAX_TIME", "90");
        }
        std::env::set_var("MUD_DEBATE_MAX_NEW_TOKENS", "3");
        let bench_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = self.run_debate_session(Some(tx), stop_flag);
        }));
        // Parse match results: "...match #N -> A:x.x B:y.y..." ; A>0 means model won.
        let mut wins = 0usize;
        let mut matches = 0usize;
        while let Ok(msg) = rx.try_recv() {
            if let Some(pos) = msg.find("match #") {
                let rest = &msg[pos + "match #".len()..];
                if let Some(ab) = rest.split("->").nth(1) {
                    // Format: "A:+0.00 B:-0.70" — strip the "A:" / "B:" labels.
                    let mut parts = ab.split('B').map(|s| s.trim());
                    if let (Some(a_s), Some(b_s)) = (parts.next(), parts.next()) {
                        let a_num = a_s.trim_start_matches('A').trim_start_matches(':').trim();
                        let b_num = b_s.trim_start_matches(':').trim();
                        if let (Ok(a), Ok(_b)) = (a_num.parse::<f32>(), b_num.parse::<f32>()) {
                            matches += 1;
                            if a > 0.0 {
                                wins += 1;
                            }
                        }
                    }
                }
            }
        }
        if bench_result.is_err() {
            // Benchmark panicked (e.g. model path issue) — treat as 0 quality.
            return Ok((0.0, matches));
        }
        let rate = if matches > 0 {
            wins as f32 / matches as f32
        } else {
            0.0
        };
        Ok((rate, matches))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    fn nonexistent_mud() -> String {
        // No need to actually create the file — `model_norms_alive` returns
        // a clear "load error" on a missing path.
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let p = std::env::temp_dir().join(format!("mud_integrity_test_{n}.mud"));
        let _ = fs::remove_file(&p);
        p.to_string_lossy().to_string()
    }

    #[test]
    fn test_model_norms_alive_missing_file_returns_false() {
        let (ok, detail) = model_norms_alive(&nonexistent_mud());
        assert!(!ok);
        assert!(detail.contains("load error"), "got: {detail}");
    }

    #[test]
    fn test_circuit_eval_integrity_missing_file_returns_false() {
        let (ok, detail) = circuit_eval_integrity(&nonexistent_mud());
        assert!(!ok);
        // Missing file → fails at model_norms_alive with load error.
        assert!(detail.contains("load error"), "got: {detail}");
    }

    /// Gate validation on the REAL divergence-trigger model.
    /// `models/ale_pico_fresh.mud` reports 18 missing tensors (🔴 STRUCTURAL
    /// ISSUES) → must FAIL the gate so the trainer aborts before chunk 0.
    #[test]
    fn test_integrity_gate_rejects_incomplete_model_ale_pico() {
        let path = "models/ale_pico_fresh.mud";
        if !std::path::Path::new(path).exists() {
            eprintln!("skip: {path} not present in this workspace");
            return;
        }
        let (ok, detail) = circuit_eval_integrity(path);
        assert!(!ok, "ale_pico_fresh.mud MUST fail integrity gate — it caused divergence");
        assert!(
            detail.contains("missing") || detail.contains("null") || detail.contains("tensor"),
            "expected missing/null/tensor in detail, got: {detail}"
        );
    }
}
