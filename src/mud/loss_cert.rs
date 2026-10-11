//! # Stream K: Loss trajectory certification
//!
//! Pure policy: given a sequence of training losses, decide whether the
//! optimizer + STE path is learning (loss must fall in a measurable way).
//!
//! Used by:
//! - unit tests (always in CI via `cargo test --lib loss_cert`)
//! - `tools/loss_certification_bench.rs` (optional e2e when a `.mud` is present)
//! - `./mud.sh cert-loss` / optional `MUD_CI_LOSS_CERT=1` in `./mud.sh ci`

use crate::mud::fenwick::FenwickU32;

/// How to score a trajectory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CertMode {
    /// `losses[0] - losses[last] >= min_absolute_drop` **or** relative drop ≥ `min_relative_drop`.
    Endpoints,
    /// Mean(first `window_frac`) − mean(last `window_frac`) must clear relative/absolute floor.
    WindowMeans,
    /// **F5b (Sprint 5+):** percentile gate. Compares the `head_pct`-quantile
    /// of the head window with the `tail_pct`-quantile of the tail window.
    /// Percentiles are recovered via a `FenwickU32` histogram (midpoint
    /// interpolation) — O(n + log B) per query instead of O(n log n) sort.
    /// Median-vs-median (`0.5 / 0.5`) is noise-robust; lower-tail `0.25 /
    /// 0.25` certifies that the *worst* losses improved (alerts on spikes
    /// hidden by mean); upper-tail `0.75 / 0.75` certifies that the *best*
    /// losses improved (catches saturation where the floor won't move).
    Percentile,
}

/// Certification knobs.
#[derive(Clone, Debug)]
pub struct LossCertConfig {
    pub mode: CertMode,
    /// Minimum number of finite loss points.
    pub min_points: usize,
    /// Fraction of the series used for each window (WindowMeans/Percentile). Clamped to (0, 0.5].
    pub window_frac: f32,
    /// Relative drop floor: `(first - last) / max(|first|, eps) >= this`.
    pub min_relative_drop: f32,
    /// Absolute drop floor (nats): `first - last >= this`.
    pub min_absolute_drop: f32,
    /// Reject if any non-finite appears.
    pub reject_non_finite: bool,
    /// F5b head-window quantile (0, 1]. Default 0.5 = median.
    pub head_pct: f32,
    /// F5b tail-window quantile (0, 1]. Default 0.5 = median.
    pub tail_pct: f32,
    /// F5b histogram resolution (more bins → tighter percentile resolution).
    /// Default 256; bench shows this is more than enough for 100k-point
    /// trajectories while keeping Fenwick lookups in L1d.
    pub percentile_bins: usize,
}

impl Default for LossCertConfig {
    fn default() -> Self {
        Self {
            mode: CertMode::WindowMeans,
            min_points: 3,
            window_frac: 0.35,
            min_relative_drop: 0.01, // 1%
            min_absolute_drop: 0.02,
            reject_non_finite: true,
            head_pct: 0.5,
            tail_pct: 0.5,
            percentile_bins: 256,
        }
    }
}

impl LossCertConfig {
    /// Fast / CI-friendly: looser drop floor for short runs.
    pub fn fast() -> Self {
        Self {
            mode: CertMode::WindowMeans,
            min_points: 3,
            window_frac: 0.4,
            min_relative_drop: 0.005,
            min_absolute_drop: 0.01,
            reject_non_finite: true,
            head_pct: 0.5,
            tail_pct: 0.5,
            percentile_bins: 256,
        }
    }

    /// Strict full certification.
    pub fn strict() -> Self {
        Self {
            mode: CertMode::WindowMeans,
            min_points: 8,
            window_frac: 0.3,
            min_relative_drop: 0.02,
            min_absolute_drop: 0.05,
            reject_non_finite: true,
            head_pct: 0.5,
            tail_pct: 0.5,
            percentile_bins: 512,
        }
    }

    /// Noise-robust percentile cert (defaults: median of head vs median of tail).
    pub fn percentile(head_pct: f32, tail_pct: f32) -> Self {
        Self {
            mode: CertMode::Percentile,
            min_points: 4,
            window_frac: 0.35,
            min_relative_drop: 0.01,
            min_absolute_drop: 0.02,
            reject_non_finite: true,
            head_pct: head_pct.clamp(0.01, 0.99),
            tail_pct: tail_pct.clamp(0.01, 0.99),
            percentile_bins: 256,
        }
    }
}

/// Successful certification report.
#[derive(Clone, Debug)]
pub struct LossCertReport {
    pub n: usize,
    pub first: f32,
    pub last: f32,
    /// Representative head value: mean (Endpoints/WindowMeans) or `head_pct`
    /// quantile (Percentile) of the head window.
    pub head_mean: f32,
    /// Representative tail value: mean (Endpoints/WindowMeans) or `tail_pct`
    /// quantile (Percentile) of the tail window.
    pub tail_mean: f32,
    pub absolute_drop: f32,
    pub relative_drop: f32,
    pub mode: CertMode,
}

/// Why certification failed.
#[derive(Clone, Debug, PartialEq)]
pub enum LossCertError {
    TooFewPoints {
        got: usize,
        need: usize,
    },
    NonFinite {
        index: usize,
    },
    InsufficientDrop {
        absolute_drop: f32,
        relative_drop: f32,
        need_abs: f32,
        need_rel: f32,
    },
    Empty,
}

impl std::fmt::Display for LossCertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "empty loss trajectory"),
            Self::TooFewPoints { got, need } => {
                write!(f, "too few points: got {got}, need ≥ {need}")
            }
            Self::NonFinite { index } => write!(f, "non-finite loss at index {index}"),
            Self::InsufficientDrop {
                absolute_drop,
                relative_drop,
                need_abs,
                need_rel,
            } => write!(
                f,
                "loss did not fall enough: Δ={absolute_drop:.4} (need ≥ {need_abs:.4}) \
                 rel={relative_drop:.4} (need ≥ {need_rel:.4})"
            ),
        }
    }
}

impl std::error::Error for LossCertError {}

/// Certify that `losses` show learning.
pub fn certify_trajectory(
    losses: &[f32],
    cfg: &LossCertConfig,
) -> Result<LossCertReport, LossCertError> {
    if losses.is_empty() {
        return Err(LossCertError::Empty);
    }
    if cfg.reject_non_finite {
        for (i, &v) in losses.iter().enumerate() {
            if !v.is_finite() {
                return Err(LossCertError::NonFinite { index: i });
            }
        }
    }
    let finite: Vec<f32> = losses.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.len() < cfg.min_points {
        return Err(LossCertError::TooFewPoints {
            got: finite.len(),
            need: cfg.min_points,
        });
    }

    let n = finite.len();
    let first = finite[0];
    let last = finite[n - 1];

    let (head_mean, tail_mean, absolute_drop) = match cfg.mode {
        CertMode::Endpoints => {
            let d = first - last;
            (first, last, d)
        }
        CertMode::WindowMeans => {
            let frac = cfg.window_frac.clamp(0.05, 0.5);
            let w = ((n as f32) * frac).ceil() as usize;
            let w = w.clamp(1, n / 2).max(1);
            let head: f32 = finite[..w].iter().sum::<f32>() / w as f32;
            let tail: f32 = finite[n - w..].iter().sum::<f32>() / w as f32;
            (head, tail, head - tail)
        }
        CertMode::Percentile => {
            // F5b: head/tail window split (same `window_frac` as WindowMeans),
            // then percentile recovery via `FenwickU32` histogram (midpoint
            // interpolation). Two reuses of a single hist + Fenwick — zero
            // alloc inside the loop other than the two scratch buffers.
            let frac = cfg.window_frac.clamp(0.05, 0.5);
            let w = ((n as f32) * frac).ceil() as usize;
            let w = w.clamp(2, n.saturating_sub(2)).max(2);
            let bins = cfg.percentile_bins.max(8);
            let mut hist = vec![0u32; bins + 1];
            let mut ft = FenwickU32::with_capacity(bins + 1);
            let head_v = percentile_loss(&finite[..w], cfg.head_pct, bins, &mut hist, &mut ft);
            let tail_v = percentile_loss(&finite[n - w..], cfg.tail_pct, bins, &mut hist, &mut ft);
            (head_v, tail_v, head_v - tail_v)
        }
    };

    let denom = head_mean.abs().max(1e-6);
    let relative_drop = absolute_drop / denom;

    let ok = absolute_drop >= cfg.min_absolute_drop || relative_drop >= cfg.min_relative_drop;
    if !ok {
        return Err(LossCertError::InsufficientDrop {
            absolute_drop,
            relative_drop,
            need_abs: cfg.min_absolute_drop,
            need_rel: cfg.min_relative_drop,
        });
    }

    Ok(LossCertReport {
        n,
        first,
        last,
        head_mean,
        tail_mean,
        absolute_drop,
        relative_drop,
        mode: cfg.mode,
    })
}

/// Parse losses from `mud_train_metrics.log` lines.
/// Format (corpus_trainer): `chunk_id 1 loss perplexity ...`
pub fn parse_metrics_log(content: &str) -> Vec<f32> {
    let mut out = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        // chunk_id
        let _ = parts.next();
        // batch marker
        let _ = parts.next();
        if let Some(loss_s) = parts.next() {
            if let Ok(v) = loss_s.parse::<f32>() {
                if v.is_finite() {
                    out.push(v);
                }
            }
        }
    }
    out
}

/// F5b helper: recover the `pct`-quantile of `losses` via a FenwickU32
/// histogram. The caller provides scratch buffers (hist + Fenwick tree so we
/// don't allocate per query when computing head/tail percentiles back-to-back).
///
/// Quantization: linear bins across `[min, max]`. Midpoint interpolation is
/// applied when mapping the recovered bin back to loss-value space — the
/// discrete-bin centroid. For `percentile_bins = 256` the worst-case error
/// is `range/512`, i.e. ~0.2% of the loss spread: well below the drop floor.
///
/// Returns `f32::NAN` for an empty slice (caller must guard `min_points`).
pub fn percentile_loss(
    losses: &[f32],
    pct: f32,
    n_bins: usize,
    hist: &mut [u32],
    ft: &mut FenwickU32,
) -> f32 {
    if losses.is_empty() {
        return f32::NAN;
    }
    let min = losses.iter().copied().fold(f32::INFINITY, f32::min);
    let max = losses.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() || !min.is_finite() {
        return f32::NAN;
    }
    let range = (max - min).max(1e-6);
    // Reset scratch histogram.
    for h in hist.iter_mut().take(n_bins + 1) {
        *h = 0;
    }
    let inv_range_bins = (n_bins as f32) / range;
    for &v in losses {
        // SAFETY: bin is clamped to [0, n_bins] regardless of NaN input.
        let v = if v.is_finite() { v } else { min };
        let bin = (((v - min) * inv_range_bins).floor() as usize).min(n_bins);
        hist[bin] += 1;
    }
    ft.build_from(&hist[..n_bins + 1]);
    // Midpoint-interpolated bin → loss value.
    let total = losses.len() as u32;
    let target = ((pct.clamp(0.0, 1.0) * total as f32).ceil() as u32).max(1).min(total);
    let bin = ft.find_kth(target);
    min + ((bin as f32 + 0.5) / n_bins as f32) * range
}

/// Env-driven config for the cert bench.
pub fn config_from_env() -> LossCertConfig {
    let fast = std::env::var("MUD_LOSS_CERT_FAST")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(true); // product default: fast cert
    let mode = std::env::var("MUD_LOSS_CERT_MODE").ok().unwrap_or_default();
    let mut cfg = if mode.eq_ignore_ascii_case("percentile") {
        let hp = std::env::var("MUD_LOSS_CERT_HEAD_PCT")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.5);
        let tp = std::env::var("MUD_LOSS_CERT_TAIL_PCT")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.5);
        LossCertConfig::percentile(hp, tp)
    } else if fast {
        LossCertConfig::fast()
    } else {
        LossCertConfig::strict()
    };
    if let Ok(v) = std::env::var("MUD_LOSS_CERT_MIN_REL") {
        if let Ok(x) = v.parse() {
            cfg.min_relative_drop = x;
        }
    }
    if let Ok(v) = std::env::var("MUD_LOSS_CERT_MIN_ABS") {
        if let Ok(x) = v.parse() {
            cfg.min_absolute_drop = x;
        }
    }
    if let Ok(v) = std::env::var("MUD_LOSS_CERT_MIN_POINTS") {
        if let Ok(x) = v.parse() {
            cfg.min_points = x;
        }
    }
    if let Ok(v) = std::env::var("MUD_LOSS_CERT_WINDOW_FRAC") {
        if let Ok(x) = v.parse() {
            cfg.window_frac = x;
        }
    }
    if let Ok(v) = std::env::var("MUD_LOSS_CERT_PERCENTILE_BINS") {
        if let Ok(x) = v.parse() {
            cfg.percentile_bins = x;
        }
    }
    cfg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pass_decreasing() {
        let losses = [3.0f32, 2.8, 2.5, 2.2, 2.0, 1.8, 1.5];
        let r = certify_trajectory(&losses, &LossCertConfig::fast()).unwrap();
        assert!(r.absolute_drop > 0.0);
        assert!(r.tail_mean < r.head_mean);
    }

    #[test]
    fn test_fail_flat() {
        let losses = [2.0f32; 10];
        let err = certify_trajectory(&losses, &LossCertConfig::fast()).unwrap_err();
        assert!(matches!(err, LossCertError::InsufficientDrop { .. }));
    }

    #[test]
    fn test_fail_rising() {
        let losses = [1.0f32, 1.2, 1.5, 2.0, 2.5];
        let err = certify_trajectory(&losses, &LossCertConfig::default()).unwrap_err();
        assert!(matches!(err, LossCertError::InsufficientDrop { .. }));
    }

    #[test]
    fn test_reject_nan() {
        let losses = [2.0f32, f32::NAN, 1.0];
        let err = certify_trajectory(&losses, &LossCertConfig::default()).unwrap_err();
        assert_eq!(err, LossCertError::NonFinite { index: 1 });
    }

    #[test]
    fn test_too_few() {
        let losses = [2.0f32, 1.0];
        let err = certify_trajectory(&losses, &LossCertConfig::default()).unwrap_err();
        assert!(matches!(err, LossCertError::TooFewPoints { .. }));
    }

    #[test]
    fn test_endpoints_mode() {
        let mut cfg = LossCertConfig::fast();
        cfg.mode = CertMode::Endpoints;
        cfg.min_points = 2;
        let r = certify_trajectory(&[5.0, 4.0], &cfg).unwrap();
        assert!((r.absolute_drop - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_parse_metrics_log() {
        let sample = "\
# comment
1 1 10.5000 36000.0 0.0 0.1
2 1 9.8000 18000.0 0.0 -0.1
3 1 9.2000 10000.0 0.0 -0.2
";
        let v = parse_metrics_log(sample);
        assert_eq!(v.len(), 3);
        assert!((v[0] - 10.5).abs() < 1e-3);
        assert!((v[2] - 9.2).abs() < 1e-3);
        let r = certify_trajectory(&v, &LossCertConfig::fast()).unwrap();
        assert!(r.absolute_drop > 0.0);
    }

    #[test]
    fn test_noisy_but_learning() {
        // Noisy path with overall decline
        let losses = [3.0, 2.9, 3.1, 2.7, 2.8, 2.4, 2.5, 2.1, 2.0, 1.9];
        let r = certify_trajectory(&losses, &LossCertConfig::fast()).unwrap();
        assert!(r.tail_mean < r.head_mean);
    }

    /// Synthetic “20-step” certification used as CI proxy for nightly e2e.
    #[test]
    fn test_twenty_step_synthetic_must_fall() {
        let mut losses = Vec::with_capacity(20);
        let mut v = 4.0f32;
        for _ in 0..20 {
            losses.push(v);
            v *= 0.97; // monotonic geometric decay
        }
        let r = certify_trajectory(&losses, &LossCertConfig::strict()).unwrap();
        assert!(r.n >= 8);
        assert!(r.relative_drop >= 0.02);
    }

    // ── F5b: Percentile mode tests ─────────────────────────────────────────

    #[test]
    fn test_percentile_mode_pass_decreasing() {
        // Monotone-decreasing trajectory, F5b median-vs-median.
        let losses = [3.0f32, 2.9, 3.05, 2.7, 2.8, 2.4, 2.5, 2.05, 1.9, 1.7];
        let cfg = LossCertConfig::percentile(0.5, 0.5);
        let r = certify_trajectory(&losses, &cfg).unwrap();
        assert_eq!(r.mode, CertMode::Percentile);
        assert!(r.absolute_drop > 0.0);
        assert!(r.tail_mean < r.head_mean);
    }

    #[test]
    fn test_percentile_mode_fail_flat() {
        // Flat trajectory: head/tail percentiles coincide → InsufficientDrop.
        let losses = [2.0f32; 12];
        let cfg = LossCertConfig::percentile(0.5, 0.5);
        let err = certify_trajectory(&losses, &cfg).unwrap_err();
        assert!(matches!(err, LossCertError::InsufficientDrop { .. }));
    }

    #[test]
    fn test_percentile_mode_fail_rising() {
        // Adversarial: bottom-loaded head, top-loaded tail. head_pct < tail_pct.
        let losses = [1.0f32, 1.1, 1.05, 1.05, 2.0, 2.1, 2.2, 1.95, 2.05, 2.15];
        let cfg = LossCertConfig::percentile(0.5, 0.5);
        let err = certify_trajectory(&losses, &cfg).unwrap_err();
        assert!(matches!(err, LossCertError::InsufficientDrop { .. }));
    }

    #[test]
    fn test_percentile_loss_matches_sorted_reference() {
        // Linspace losses give a clean checkerboard histogram (1-2 values per
        // bin when bins ≳ n) so the midpoint-interp recovery is provably within
        // 1 bin width of the true nearest-rank quantile.
        let n = 200usize;
        let losses: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut sorted = losses.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let pcts = [0.1f32, 0.25, 0.5, 0.75, 0.9];
        let n_bins = 256;
        let mut hist = vec![0u32; n_bins + 1];
        let mut ft = FenwickU32::with_capacity(n_bins + 1);
        let min = sorted[0];
        let max = sorted[sorted.len() - 1];
        // 4-bin-widths tolerance (≈ 5.5 step) accommodates midpoint rounding +
        // the ceil-rounded rank target; very loose so we never get a flake on
        // FP rounding.  The function is approximate by design (histogram loss
        // resolution), so the test only sanity-checks the *shape*.
        let tolerance = (max - min).max(1.0) / (n_bins as f32) * 4.0;
        for p in pcts {
            let got = percentile_loss(&losses, p, n_bins, &mut hist, &mut ft);
            // Nearest-rank reference: index = round(p * (n-1))
            let ref_idx = (p * (n as f32 - 1.0)).round() as usize;
            let reference = sorted[ref_idx];
            assert!(
                (got - reference).abs() < tolerance,
                "p={}: got {got}, ref {reference}, tol {tolerance}",
                p
            );
        }
    }

    #[test]
    fn test_percentile_loss_empty_returns_nan() {
        let n_bins = 64;
        let mut hist = vec![0u32; n_bins + 1];
        let mut ft = FenwickU32::with_capacity(n_bins + 1);
        let v = percentile_loss(&[], 0.5, n_bins, &mut hist, &mut ft);
        assert!(v.is_nan());
    }

    #[test]
    fn test_percentile_lower_tail_strictness() {
        // Bottom-tail (0.25) percentile gate catches spike-laden improvement
        // that a WindowMeans cert would miss: tail has a few high-spike outliers
        // but the bulk improved dramatically. Lower-tail pct = 0.25 catches
        // the worst-case improvement.
        let losses = vec![
            // Head window: noisy oscillation around 3.0
            3.0f32, 3.1, 2.9, 3.2, 3.05, 2.85, 3.15, 3.0, 3.1, 2.95,
            // Tail window: mostly ~1.0 but two spikes
            1.0, 1.1, 0.9, 3.4, 0.95, 1.05, 3.5, 1.0, 0.85, 1.05,
        ];
        let cfg = LossCertConfig::percentile(0.25, 0.25);
        let r = certify_trajectory(&losses, &cfg).unwrap();
        // head 25th-percentile ≈ 2.95, tail 25th-percentile ≈ 0.95.
        // Drop is dramatic → cert passes.
        assert!(r.absolute_drop >= 1.0,
            "lower-tail percentile drop too small: {}", r.absolute_drop);
        assert!(r.tail_mean < r.head_mean);
    }
}
