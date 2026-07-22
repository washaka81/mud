//! # Stream K: Loss trajectory certification
//!
//! Pure policy: given a sequence of training losses, decide whether the
//! optimizer + STE path is learning (loss must fall in a measurable way).
//!
//! Used by:
//! - unit tests (always in CI via `cargo test --lib loss_cert`)
//! - `tools/loss_certification_bench.rs` (optional e2e when a `.mud` is present)
//! - `./mud.sh cert-loss` / optional `MUD_CI_LOSS_CERT=1` in `./mud.sh ci`

/// How to score a trajectory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CertMode {
    /// `losses[0] - losses[last] >= min_absolute_drop` **or** relative drop ≥ `min_relative_drop`.
    Endpoints,
    /// Mean(first `window_frac`) − mean(last `window_frac`) must clear relative/absolute floor.
    WindowMeans,
}

/// Certification knobs.
#[derive(Clone, Debug)]
pub struct LossCertConfig {
    pub mode: CertMode,
    /// Minimum number of finite loss points.
    pub min_points: usize,
    /// Fraction of the series used for each window (WindowMeans). Clamped to (0, 0.5].
    pub window_frac: f32,
    /// Relative drop floor: `(first - last) / max(|first|, eps) >= this`.
    pub min_relative_drop: f32,
    /// Absolute drop floor (nats): `first - last >= this`.
    pub min_absolute_drop: f32,
    /// Reject if any non-finite appears.
    pub reject_non_finite: bool,
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
        }
    }
}

/// Successful certification report.
#[derive(Clone, Debug)]
pub struct LossCertReport {
    pub n: usize,
    pub first: f32,
    pub last: f32,
    pub head_mean: f32,
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

/// Env-driven config for the cert bench.
pub fn config_from_env() -> LossCertConfig {
    let fast = std::env::var("MUD_LOSS_CERT_FAST")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(true); // product default: fast cert
    let mut cfg = if fast {
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
}
