// ── Macros (must be before `pub mod` for textual visibility in submodules) ───
// Gated println!/print! so the circuit TUI (crossterm) is not contaminated by
// raw stdout when MUD_CIRCUIT_TUI is set.
macro_rules! println {
    () => {
        if std::env::var("MUD_CIRCUIT_TUI").is_err() {
            std::println!();
        }
    };
    ($($arg:tt)*) => {
        if std::env::var("MUD_CIRCUIT_TUI").is_err() {
            std::println!($($arg)*);
        }
    };
}

macro_rules! print {
    ($($arg:tt)*) => {
        if std::env::var("MUD_CIRCUIT_TUI").is_err() {
            std::print!($($arg)*);
        }
    };
}

// ── Submodules (Sprint 4.1 split) ────────────────────────────────────────────
pub mod alignment;
pub mod circuit;
pub mod debate;
pub mod emb;
pub mod integrity;
pub mod metrics;
pub mod session;
pub mod train_step;

use crate::mud::smart_tokenizer::SmartTokenizer;
use crate::mud::MudFile;

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// Live activation metrics sampled **after forward, before backward** (post-BWD kills VarH/VarJ).
pub use metrics::{ActStatsAccum, TrainChunkMetrics};

/// Re-exported embedding helpers (used by the inner alignment step).
#[allow(unused_imports)]
pub(crate) use emb::{load_emb_row_into, slice_stddev};

/// Tamaño de chunk en caracteres para procesamiento del corpus.
pub const CHUNK_SIZE: usize = 50_000;
/// Default hard-checkpoint interval (was 1 = save 100MB every chunk — catastrophic for align).
/// Override: `MUD_TRAIN_CKPT_EVERY=N` (0 = only epoch/end).
fn checkpoint_every_chunks() -> usize {
    std::env::var("MUD_TRAIN_CKPT_EVERY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50)
}
// SPRINT-4 (4.1): `CHECKPOINT_DIR` removed; the path is now derived directly
// from the model file's stem in `corpus_trainer/session.rs`.

pub static SHOULD_TERMINATE: AtomicBool = AtomicBool::new(false);

/// Implements a high-performance local corpus trainer for MUD.
pub struct MudCorpusTrainer {
    pub model_path: String,
    pub corpus_dir: String,
    pub tokenizer: Arc<SmartTokenizer>,
    pub vk: Option<std::sync::Arc<crate::vulkan::ash_backend::AshContext>>,
    /// D4: Pipeline overlap orchestrator for full CPU→GPU→CPU overlap.
    /// Active when MUD_PIPELINE_OVERLAP=1. Uses RefCell for interior mutability
    /// since train_on_sequence takes &self (not &mut self).
    pub pipeline_overlap: std::cell::RefCell<crate::mud::ash_qat_dispatcher::PipelineOverlap>,
    /// FIX (2026-07-31): Global step counter for LR warmup + cosine decay.
    /// Previously LR was fixed at 0.0005 for the entire training run, causing
    /// cog to slowly decay from ~593 to ~250 over 5000 steps before sudden collapse.
    pub step_count: std::sync::atomic::AtomicU64,
    /// Total expected steps for cosine decay (set by run_alignment_session).
    pub total_steps_hint: std::cell::RefCell<Option<usize>>,
    /// Warmup steps (linear ramp from 0 to base_lr). Default: 1% of total.
    pub warmup_steps: std::cell::RefCell<usize>,
    /// FIX (2026-08-03): Best-loss tracking for automatic best-model checkpointing.
    /// Saves `{stem}_best.mud` whenever chunk_loss improves below best_loss,
    /// providing a rollback point if the model collapses (e.g. step-3403 spike).
    pub best_loss: std::cell::RefCell<f32>,
    /// Whether best-model saving is enabled (MUD_BEST_CHECKPOINT, default 1).
    pub best_checkpoint_enabled: bool,
}

impl MudCorpusTrainer {
    pub fn new(model_path: String, corpus_dir: String) -> anyhow::Result<Self> {
        let mud = MudFile::load(&model_path)?;
        Self::validate_metadata(&mud)?;

        let tokens_str = mud
            .global_metadata
            .get("tokenizer.tokens")
            .ok_or_else(|| anyhow::anyhow!("No tokens"))?;
        let merges_str = mud
            .global_metadata
            .get("tokenizer.merges")
            .map(|s| s.as_str())
            .unwrap_or("");
        let tokenizer = SmartTokenizer::from_mud_metadata(tokens_str, merges_str, &mud.global_metadata);

        let vk = crate::vulkan::ash_backend::AshContext::new()
            .map(Arc::new)
            .ok();

        let trainer = Self {
            model_path,
            corpus_dir,
            tokenizer: Arc::new(tokenizer),
            vk,
            pipeline_overlap: std::cell::RefCell::new(
                crate::mud::ash_qat_dispatcher::PipelineOverlap::new(),
            ),
            step_count: std::sync::atomic::AtomicU64::new(0),
            total_steps_hint: std::cell::RefCell::new(None),
            warmup_steps: std::cell::RefCell::new(0),
            best_loss: std::cell::RefCell::new(f32::INFINITY),
            best_checkpoint_enabled: std::env::var("MUD_BEST_CHECKPOINT")
                .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
                .unwrap_or(true),
        };
        trainer.audit_tokenization();
        Ok(trainer)
    }

    /// SPRINT-4 (4.1): thin shim to the extracted module. Behaviour unchanged.
    pub(crate) fn hash_trained_weights(mud: &MudFile) -> u64 {
        crate::mud::checkpoint::hash_trained_weights(mud)
    }

    /// SPRINT-4 (4.1): thin shim to the extracted module. Behaviour unchanged.
    pub(crate) fn sync_shadow_to_mud(
        &self,
        mud: &mut MudFile,
        shadow_emb: &mut [f32],
        shadow_layers: &mut [crate::mud::slime_backward::SlimeLayerShadowF32],
        layers: &[crate::mud::slime_forward::SlimeLayer],
        vk_qat: Option<&mut crate::mud::ash_qat_dispatcher::AshQatDispatcher>,
        final_verify: bool,
    ) {
        crate::mud::checkpoint::sync_shadow_to_mud(
            mud,
            shadow_emb,
            shadow_layers,
            layers,
            vk_qat,
            final_verify,
        );
    }

    pub(crate) fn save_checkpoint(
        &self,
        mud: &mut MudFile,
        shadow_emb: &mut [f32],
        shadow_layers: &mut [crate::mud::slime_backward::SlimeLayerShadowF32],
        layers: &[crate::mud::slime_forward::SlimeLayer],
        _suffix: String,
        mut vk_qat: Option<&mut crate::mud::ash_qat_dispatcher::AshQatDispatcher>,
    ) -> anyhow::Result<()> {
        // SPRINT-4 (4.1): path is now derived from the base model file via the
        // extracted helper. The historical hard-coded
        // `weights/checkpoints/model_latest_checkpoint.mud` is preserved as a
        // fallback when no model_path is known to the caller.
        // FIX (2026-07-31): Always use the SAME checkpoint path (model_latest.mud)
        // and clean up old suffixed checkpoints — no file accumulation.
        let base_path = mud.path
            .as_deref()
            .unwrap_or("models/model.mud");
        let base = std::path::Path::new(base_path);
        let parent = base.parent().unwrap_or_else(|| std::path::Path::new("."));
        let stem = base
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model");
        let ckpt_dir = parent.join("weights").join("checkpoints");
        let _ = std::fs::create_dir_all(&ckpt_dir);

        // FIX (2026-07-31): Clean up ALL old checkpoint files before saving.
        // This ensures only the latest checkpoint is kept — no file accumulation.
        if let Ok(entries) = std::fs::read_dir(&ckpt_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                if let Some(name_str) = name.to_str() {
                    // Match: {stem}_*.mud (chunk_*, epoch_*, etc.) + any .mud files
                    // except the one we're about to write (model_latest.mud)
                    if name_str != format!("{}_latest.mud", stem)
                        && name_str.starts_with(&format!("{}_", stem))
                        && name_str.ends_with(".mud")
                    {
                        let _ = std::fs::remove_file(entry.path());
                    }
                }
            }
        }

        // Always save to the canonical "latest" path — overwrites the previous one.
        let checkpoint_name = ckpt_dir.join(format!("{stem}_latest.mud"));

        if let Some(vk) = vk_qat.as_deref_mut() {
            let _ = vk.sync_all();
        }
        self.sync_shadow_to_mud(mud, shadow_emb, shadow_layers, layers, vk_qat, false);
        let tmp_path = format!("{}.tmp", checkpoint_name.display());
        mud.save(&tmp_path)?;
        std::fs::rename(&tmp_path, &checkpoint_name)?;

        Ok(())
    }

    /// FIX (2026-08-03): Save the best-loss checkpoint to `{stem}_best.mud`.
    /// Called when `chunk_loss < best_loss`. This provides a rollback point
    /// if the model collapses later (e.g. step-3403 gradient spike).
    /// Returns `true` if a new best was saved.
    pub(crate) fn save_best_checkpoint(
        &self,
        chunk_loss: f32,
        mud: &mut MudFile,
        shadow_emb: &mut [f32],
        shadow_layers: &mut [crate::mud::slime_backward::SlimeLayerShadowF32],
        layers: &[crate::mud::slime_forward::SlimeLayer],
        mut vk_qat: Option<&mut crate::mud::ash_qat_dispatcher::AshQatDispatcher>,
    ) -> anyhow::Result<bool> {
        if !self.best_checkpoint_enabled {
            return Ok(false);
        }

        let current_best = *self.best_loss.borrow();
        if !chunk_loss.is_finite() || chunk_loss >= current_best {
            return Ok(false);
        }

        // Update best_loss
        *self.best_loss.borrow_mut() = chunk_loss;

        let base_path = mud.path
            .as_deref()
            .unwrap_or("models/model.mud");
        let base = std::path::Path::new(base_path);
        let parent = base.parent().unwrap_or_else(|| std::path::Path::new("."));
        let stem = base
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model");
        let ckpt_dir = parent.join("weights").join("checkpoints");
        let _ = std::fs::create_dir_all(&ckpt_dir);

        let best_name = ckpt_dir.join(format!("{stem}_best.mud"));

        if let Some(vk) = vk_qat.as_deref_mut() {
            let _ = vk.sync_all();
        }
        self.sync_shadow_to_mud(mud, shadow_emb, shadow_layers, layers, vk_qat, false);
        let tmp_path = format!("{}.tmp", best_name.display());
        mud.save(&tmp_path)?;
        std::fs::rename(&tmp_path, &best_name)?;

        log::info!(
            "[CHECKPOINT] New best loss={:.4} saved to {} (step tracked from chunk metrics)",
            chunk_loss,
            best_name.display()
        );

        Ok(true)
    }
}

// SPRINT-4: `mhc_scale_sgd_step` and `apply_optimizer_cpu_step_and_pack`
// have been moved to `crate::mud::optim` to break the dependency cycle
// between `corpus_trainer` and `slime_backward`. Re-exported here for
// back-compat with the existing call sites (which use the shorter path).
pub use crate::mud::optim::{
    apply_optimizer_cpu_step_and_pack, mhc_scale_sgd_step, OptimizerStrategy,
};

/// Update PRQ scales only: keep existing ELUT codes, fit s per row to shadow.
/// (Implementation moved to `crate::mud::optim` in Sprint 4.)
#[cfg(test)]
mod optimizer_strategy_tests {
    use super::*;
    use crate::mud::slime_backward::{select_optimizer, OptimizerStrategy};
    use std::sync::Mutex;

    // Serialize tests that mutate global optimizer-selection env vars
    // (MUD_TRAIN_MAX_CHUNKS / MUD_OPT / ...). They race under the parallel
    // test runner and produce flaky results otherwise.
    static OPT_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn run_step(
        rows: usize,
        cols: usize,
        strategy: OptimizerStrategy,
        shadow0: f32,
        g0: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<u8>) {
        let mut shadow = vec![shadow0; rows * cols];
        let grad = vec![g0; rows * cols];
        // seed a non-uniform gradient so Muon/GaLore actually transform
        let mut grad = grad;
        for (i, g) in grad.iter_mut().enumerate() {
            *g = g0 * (1.0 + (i % 7) as f32 * 0.1);
        }
        let mut packed = vec![0u8; (rows * cols).div_ceil(2)];
        let mut scales = vec![0.0f32; rows];
        let pool = crate::mud::pcore_pool::PCorePool::new(2);
        let mut adam = crate::mud::adam_state::AdamState::for_strategy(rows * cols, strategy);
        unsafe {
            apply_optimizer_cpu_step_and_pack(
                &mut shadow,
                &grad,
                packed.as_mut_ptr(),
                scales.as_mut_ptr(),
                1e-2,
                0.0,
                1.0,
                cols,
                &pool,
                strategy,
                adam.as_mut(),
            );
        }
        (shadow, scales, packed)
    }

    fn clear_opt_env() {
        std::env::remove_var("MUD_TRAIN_MAX_CHUNKS");
        std::env::remove_var("MUD_OPT");
        std::env::remove_var("MUD_TRAIN_OPT");
        std::env::remove_var("MUD_MUON_NS_ITERS");
    }

    #[test]
    fn test_select_optimizer_square() {
        let _g = OPT_TEST_LOCK.lock().unwrap();
        // Run sequentially to avoid a global env-var race with the quick-mode
        // assertion below (both touch MUD_TRAIN_MAX_CHUNKS via clear_opt_env).
        // Contract: square -> Muon by default; square with MUD_TRAIN_MAX_CHUNKS
        // (quick mode) -> Sgd.
        clear_opt_env();
        assert!(
            matches!(select_optimizer(576, 576), OptimizerStrategy::Muon { .. }),
            "expected Muon for square"
        );
        assert!(
            matches!(select_optimizer(2560, 2560), OptimizerStrategy::Muon { .. }),
            "expected Muon for large square"
        );
        std::env::set_var("MUD_TRAIN_MAX_CHUNKS", "8");
        assert!(
            matches!(select_optimizer(576, 576), OptimizerStrategy::Sgd),
            "expected Sgd in quick mode (MUD_TRAIN_MAX_CHUNKS set)"
        );
        clear_opt_env();
    }

    #[test]
    fn test_select_optimizer_tall_is_galore() {
        let _g = OPT_TEST_LOCK.lock().unwrap();
        clear_opt_env();
        match select_optimizer(1536, 576) {
            OptimizerStrategy::GaLore { rank, .. } => assert!(rank >= 8),
            other => panic!("expected GaLore for tall, got {other:?}"),
        }
    }

    #[test]
    fn test_select_optimizer_wide_is_chunked() {
        let _g = OPT_TEST_LOCK.lock().unwrap();
        clear_opt_env();
        match select_optimizer(576, 1536) {
            OptimizerStrategy::ChunkedAdam { chunk_cols } => assert!(chunk_cols >= 1),
            other => panic!("expected ChunkedAdam for wide, got {other:?}"),
        }
    }

    #[test]
    fn test_apply_optimizer_muon_runs_and_packs() {
        let (shadow, scales, packed) =
            run_step(8, 8, OptimizerStrategy::Muon { ns_iters: 2 }, 0.5, 0.01);
        assert!(scales.iter().all(|s| s.is_finite() && *s > 0.0));
        assert!(shadow.iter().all(|w| w.is_finite()));
        assert!(packed.iter().any(|&b| b != 0));
    }

    #[test]
    fn test_apply_optimizer_galore_tall_finite() {
        let (shadow, scales, packed) = run_step(
            32,
            8,
            OptimizerStrategy::GaLore {
                rank: 4,
                update_freq: 100,
            },
            0.3,
            0.02,
        );
        assert!(shadow.iter().all(|w| w.is_finite()));
        assert!(scales.iter().all(|s| s.is_finite() && *s > 0.0));
        assert!(packed.iter().any(|&b| b != 0));
    }

    #[test]
    fn test_apply_optimizer_chunked_wide_finite() {
        let (shadow, scales, _) = run_step(
            8,
            32,
            OptimizerStrategy::ChunkedAdam { chunk_cols: 8 },
            0.4,
            0.015,
        );
        assert!(shadow.iter().all(|w| w.is_finite()));
        assert!(scales.iter().all(|s| s.is_finite() && *s > 0.0));
    }

    #[test]
    fn test_muon_differs_from_plain_adam_sgd() {
        // Same initial state; Muon preprocess should not match plain Adam path exactly
        let rows = 8usize;
        let cols = 8usize;
        let mut shadow_m = vec![0.5f32; rows * cols];
        let mut shadow_a = vec![0.5f32; rows * cols];
        let mut grad = vec![0.0f32; rows * cols];
        for (i, g) in grad.iter_mut().enumerate() {
            *g = 0.05 * ((i % 5) as f32 - 2.0);
        }
        let mut packed = vec![0u8; (rows * cols).div_ceil(2)];
        let mut scales = vec![0.0f32; rows];
        let pool = crate::mud::pcore_pool::PCorePool::new(2);

        unsafe {
            apply_optimizer_cpu_step_and_pack(
                &mut shadow_m,
                &grad,
                packed.as_mut_ptr(),
                scales.as_mut_ptr(),
                1e-2,
                0.0,
                1.0,
                cols,
                &pool,
                OptimizerStrategy::Muon { ns_iters: 3 },
                None,
            );
        }
        let mut adam = crate::mud::adam_state::AdamState::zeros(rows * cols);
        unsafe {
            apply_optimizer_cpu_step_and_pack(
                &mut shadow_a,
                &grad,
                packed.as_mut_ptr(),
                scales.as_mut_ptr(),
                1e-2,
                0.0,
                1.0,
                cols,
                &pool,
                OptimizerStrategy::Adam,
                Some(&mut adam),
            );
        }

        let max_delta = shadow_m
            .iter()
            .zip(shadow_a.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_delta > 1e-6,
            "Muon path should diverge from plain Adam/SGD path, max_delta={max_delta}"
        );
    }

    #[test]
    fn test_slime_layer_shadow_opts_match_select() {
        let _g = OPT_TEST_LOCK.lock().unwrap();
        // Isolate from quick-train env overrides (MUD_TRAIN_MAX_CHUNKS → SGD).
        clear_opt_env();
        // hidden=64, ffn_mid=192 (ratio 3 > 2.5 → GaLore up; down ratio < 0.4 → Chunked)
        let s = crate::mud::slime_backward::SlimeLayerShadowF32::new(64, 192, 2, 32);
        assert!(matches!(s.q_opt, OptimizerStrategy::Muon { .. }));
        assert!(matches!(s.ffn_up_opt, OptimizerStrategy::GaLore { .. }));
        assert!(matches!(
            s.ffn_down_opt,
            OptimizerStrategy::ChunkedAdam { .. }
        ));
        // GQA k: kv_dim=64, cols=64 → square-ish Muon
        assert!(matches!(s.k_opt, OptimizerStrategy::Muon { .. }));
    }
}
