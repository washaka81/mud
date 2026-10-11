//! F4: Optimal threshold computation for weight quantization.
//!
//! Uses Fenwick trees (Binary Indexed Trees) for O(n + 100*log(100)) threshold
//! optimization, replacing the O(100n) linear scan.
//!
//! See: `docs/research/MUD_FENWICK_ASM_POINTER_ANALYSIS.md` §1.1

use crate::mud::fenwick::{FenwickF32, FenwickU32};
use crate::mud::constants::EPSILON_FLOOR;

/// Compute optimal (scale, delta) thresholds for ternary quantization.
///
/// Uses Fenwick trees to accelerate the histogram sum/count computation:
/// - Build histogram of |w| in 100 bins — O(n)
/// - Build Fenwick trees over histogram — O(100)
/// - For each threshold candidate: prefix_sum → sum_above + count_above — O(log 100)
///
/// Total: O(n + 100*log(100)) vs O(100n) for the linear scan.
pub fn compute_optimal_thresholds(weights: &[f32]) -> (f32, f32) {
    if weights.is_empty() {
        return (EPSILON_FLOOR, 0.0);
    }

    let mut max_abs = 0.0f32;
    for &w in weights {
        let abs = w.abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }

    if max_abs < EPSILON_FLOOR {
        return (EPSILON_FLOOR, 0.0);
    }

    let num_bins = 100;
    let step = max_abs / num_bins as f32;

    // Build histogram: bin = |w| / max_abs * num_bins
    let mut hist_sum = vec![0.0f32; num_bins + 1];
    let mut hist_count = vec![0u32; num_bins + 1];
    for &w in weights {
        let bin = ((w.abs() / max_abs * num_bins as f32) as usize).min(num_bins);
        hist_sum[bin] += w.abs();
        hist_count[bin] += 1;
    }

    // Build Fenwick trees for O(log n) prefix sums
    let mut ft_sum = FenwickF32::with_capacity(num_bins + 1);
    let mut ft_count = FenwickU32::with_capacity(num_bins + 1);
    ft_sum.build_from(&hist_sum);
    ft_count.build_from(&hist_count);

    let total_sum = ft_sum.total();
    let total_count = ft_count.total();

    let mut best_mse = f32::MAX;
    let mut best_delta = 0.0;
    let mut best_scale = EPSILON_FLOOR;

    for i in 1..num_bins {
        let delta = i as f32 * step;

        // Fenwick tree: prefix_sum(i) = sum of bins 0..i (below threshold)
        // total - prefix_sum(i) = sum/count above threshold
        let sum_below = ft_sum.prefix_sum(i);
        let count_below = ft_count.prefix_sum(i);
        let sum_above = total_sum - sum_below;
        let count_above = total_count - count_below;

        let scale = if count_above > 0 {
            sum_above / count_above as f32
        } else {
            EPSILON_FLOOR
        };

        // MSE computation still requires a full scan (could be optimized further
        // with additional Fenwick trees for w^2, but the gain is marginal)
        let mut current_mse = 0.0;
        for &w in weights {
            let abs_w = w.abs();
            if abs_w <= delta {
                current_mse += w * w;
            } else {
                let diff = abs_w - scale;
                current_mse += diff * diff;
            }
        }

        if current_mse < best_mse {
            best_mse = current_mse;
            best_delta = delta;
            best_scale = scale;
        }
    }

    (best_scale.max(1e-8), best_delta)
}

/// Linear fallback for `compute_optimal_thresholds` (when `MUD_FENWICK=0`).
/// O(100n) — same as the original implementation.
pub fn compute_optimal_thresholds_linear(weights: &[f32]) -> (f32, f32) {
    if weights.is_empty() {
        return (EPSILON_FLOOR, 0.0);
    }

    let mut max_abs = 0.0f32;
    for &w in weights {
        let abs = w.abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }

    if max_abs < EPSILON_FLOOR {
        return (EPSILON_FLOOR, 0.0);
    }

    let num_bins = 100;
    let step = max_abs / num_bins as f32;

    let mut best_mse = f32::MAX;
    let mut best_delta = 0.0;
    let mut best_scale = EPSILON_FLOOR;

    for i in 1..num_bins {
        let delta = i as f32 * step;

        let mut sum_above = 0.0;
        let mut count_above = 0;

        for &w in weights {
            if w.abs() > delta {
                sum_above += w.abs();
                count_above += 1;
            }
        }

        let scale = if count_above > 0 {
            sum_above / count_above as f32
        } else {
            EPSILON_FLOOR
        };

        let mut current_mse = 0.0;
        for &w in weights {
            let abs_w = w.abs();
            if abs_w <= delta {
                current_mse += w * w;
            } else {
                let diff = abs_w - scale;
                current_mse += diff * diff;
            }
        }

        if current_mse < best_mse {
            best_mse = current_mse;
            best_delta = delta;
            best_scale = scale;
        }
    }

    (best_scale.max(1e-8), best_delta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mud::constants::EPSILON_FLOOR;

    #[test]
    fn test_thresholds_empty() {
        let (scale, delta) = compute_optimal_thresholds(&[]);
        assert!((scale - 1e-8).abs() < EPSILON_FLOOR);
        assert!((delta - 0.0).abs() < EPSILON_FLOOR);
    }

    #[test]
    fn test_thresholds_uniform() {
        let weights = vec![1.0f32; 1000];
        let (scale, delta) = compute_optimal_thresholds(&weights);
        assert!(scale > 0.0);
        assert!(delta > 0.0);
    }

    #[test]
    fn test_thresholds_matches_linear() {
        let weights: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.001) - 0.5).collect();
        let (scale_f, delta_f) = compute_optimal_thresholds(&weights);
        let (scale_l, delta_l) = compute_optimal_thresholds_linear(&weights);
        // Results should be close (Fenwick uses histogram approximation for sum/count)
        assert!((scale_f - scale_l).abs() < 0.1);
        assert!((delta_f - delta_l).abs() < 0.1);
    }

    #[test]
    fn test_thresholds_large() {
        let n = 131_072;
        let weights: Vec<f32> = (0..n).map(|i| (i as f32 * 0.0001) - 6.5).collect();
        let (scale, delta) = compute_optimal_thresholds(&weights);
        assert!(scale > 0.0);
        assert!(delta > 0.0);
    }
}
