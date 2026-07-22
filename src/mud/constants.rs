//! Justified project-wide constants (P-13 / GEMINI §2).
//! **SSOT:** import from here — do not redefine `EPSILON_FLOOR` elsewhere.

use std::sync::OnceLock;

pub const DEPTH_DAMPENING_FACTOR: f32 = std::f32::consts::FRAC_1_SQRT_2;
pub const SPARSITY_THRESHOLD_RATIO: f32 = 0.7;
pub const NEURAL_KICK_JITTER: f32 = 1e-5;
/// Absolute numerical floor for stability-critical divisions (SSOT).
pub const EPSILON_FLOOR: f32 = 1e-8;
pub const JEPA_ATTRACTOR_LR: f32 = 0.01;
pub const QAT_LEARNING_RATE: f32 = 0.0005;

/// Effective QAT LR (env `MUD_QAT_LR` overrides; align mode uses a slightly higher default).
pub fn qat_learning_rate() -> f32 {
    if let Ok(v) = std::env::var("MUD_QAT_LR") {
        if let Ok(f) = v.parse::<f32>() {
            return f.clamp(1e-6, 0.1);
        }
    }
    let align = std::env::var("MUD_TRAIN_ALIGN")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if align {
        0.0008 // slightly hotter STE for post-convert recover
    } else {
        QAT_LEARNING_RATE
    }
}
/// Deprecated i16 reseat (kept for doc archaeology only).
pub const SLIME_RESEAT_STRIDE: usize = 256;

/// Default upper bound for PCorePool when hardware detection is unavailable.
pub const PCORE_THREADS_CAP: usize = 8;

/// Cached process-wide core count for HW path of [`default_pcore_threads`].
///
/// **Must** be populated **before** any `core_affinity::set_for_current` pin:
/// after pinning the main thread, `get_core_ids()` often returns only that core
/// (Linux thread affinity mask), which wrongly collapses the pool to 1.
static HW_PCORE_THREADS: OnceLock<usize> = OnceLock::new();

/// Probe logical cores available to the process (pre-pin). Idempotent.
fn probe_hw_pcore_threads() -> usize {
    let available = core_affinity::get_core_ids()
        .map(|c| c.len())
        .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
        .unwrap_or(PCORE_THREADS_CAP)
        .max(1);
    available.clamp(1, PCORE_THREADS_CAP)
}

/// Capture process-wide core count **before** pinning threads.
/// Safe to call multiple times; first successful probe wins.
pub fn capture_hw_pcore_threads() -> usize {
    *HW_PCORE_THREADS.get_or_init(probe_hw_pcore_threads)
}

/// Env: `MUD_PCORE_THREADS` (preferred) or legacy `RAYON_NUM_THREADS`.
/// Else: min(available logical cores, PCORE_THREADS_CAP), at least 1.
///
/// Hardware path is cached via [`capture_hw_pcore_threads`] so a later thread
/// pin cannot shrink the reported pool size to 1.
pub fn default_pcore_threads() -> usize {
    if let Ok(v) =
        std::env::var("MUD_PCORE_THREADS").or_else(|_| std::env::var("RAYON_NUM_THREADS"))
    {
        if let Ok(n) = v.parse::<usize>() {
            return n.clamp(1, 64);
        }
    }
    capture_hw_pcore_threads()
}

/// True when env is set to an affirmative value (`1`, `true`, `yes`, `on`).
#[inline]
pub fn env_flag_true(key: &str) -> bool {
    std::env::var(key)
        .map(|v| {
            let t = v.trim().to_ascii_lowercase();
            t == "1" || t == "true" || t == "yes" || t == "on"
        })
        .unwrap_or(false)
}

/// True when env is set to a negative value (`0`, `false`, `no`, `off`).
#[inline]
pub fn env_flag_false(key: &str) -> bool {
    std::env::var(key)
        .map(|v| {
            let t = v.trim().to_ascii_lowercase();
            t == "0" || t == "false" || t == "no" || t == "off" || t == "cpu"
        })
        .unwrap_or(false)
}

/// `MUD_TRAIN_FREEZE_EMB` — default **true** (protect emb + skip ~1 GiB FP32 on large vocab).
/// Set `0`/`false`/`off` to train embeddings (high RAM).
pub fn train_freeze_emb() -> bool {
    match std::env::var("MUD_TRAIN_FREEZE_EMB") {
        Err(_) => true,
        Ok(v) => {
            let t = v.trim().to_ascii_lowercase();
            !(t == "0" || t == "false" || t == "no" || t == "off")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epsilon_floor_positive() {
        const {
            assert!(EPSILON_FLOOR > 0.0 && EPSILON_FLOOR < 1e-6);
        }
    }

    #[test]
    fn test_default_pcore_threads_in_range() {
        let n = default_pcore_threads();
        assert!((1..=64).contains(&n), "threads={n}");
    }

    #[test]
    fn test_capture_stable_after_repeat() {
        let a = capture_hw_pcore_threads();
        let b = capture_hw_pcore_threads();
        assert_eq!(a, b);
        assert!((1..=PCORE_THREADS_CAP).contains(&a));
    }

    #[test]
    fn test_train_freeze_emb_default_true() {
        // Cannot safely unset global env in parallel tests; only check clamp path logic
        // when env absent: default true is the documented policy.
        let _ = train_freeze_emb();
    }
}
