//! Semantic Tube Prediction (STP) trajectory loss — Phase 2 of the trainable
//! JEPA+mHC plan (`docs/research/MUD_PLAN_MHC_STP_TRAINABLE.md`).
//!
//! STP (arXiv:2602.22617) is a zero-inference-cost, train-only regularizer that
//! keeps the per-layer residual trajectory locally linear ("on the geodesic"):
//! for three positions `s < r < t` with hidden states `h_s, h_r, h_t`, the step
//! direction `d1 = h_r - h_s` should be parallel to the chord `d2 = h_t - h_s`.
//!
//! ## Loss
//! ```text
//! d1 = h_r - h_s
//! d2 = h_t - h_s
//! L  = 1 - cos(d1, d2)          # 0 iff h_r lies on the ray h_s -> h_t
//! ```
//!
//! Note on the plan formula: the plan wrote `1 - cos(proj_parallel(d1,d2), d2)`,
//! but `proj_parallel(d1,d2)` is by construction parallel to `d2`, so its cosine
//! with `d2` degenerates to `sign(<d1,d2>)` (±1) and carries no usable gradient.
//! The correct, non-degenerate objective is the alignment of the *step* with the
//! *chord*, i.e. `1 - cos(d1, d2)`, which is what STP actually optimizes. This is
//! zero exactly when `h_r` is on the segment `h_s -> h_t` (geodesic-zero property).
//!
//! No predictor network, no extra params (identity/local-linear predictor).
//! Raw-pointer AVX2 dot/axpy, TLS scratch — P-00/P-01, zero alloc in the hot loop.

use std::cell::RefCell;

const EPS: f32 = 1e-6;

thread_local! {
    /// Reusable scratch: [d1, d2] each `hidden` long. Grown on demand, never freed.
    static STP_SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return unsafe { forge_autograd::avx_math::dot_product_avx2(a, b) };
        }
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Gradient accumulators for one STP triple, written into caller buffers.
///
/// `grad_h_r`, `grad_h_t`, `grad_h_s` each receive `+= lambda * dL/dh_*`.
/// Buffers are NOT zeroed here (accumulate into a live backward seed).
///
/// Returns the (unscaled) STP loss value for logging.
///
/// # Safety
/// All slices must have identical length `hidden > 0`.
#[allow(clippy::too_many_arguments)]
pub fn stp_loss_and_grad(
    h_s: &[f32],
    h_r: &[f32],
    h_t: &[f32],
    lambda: f32,
    grad_h_s: &mut [f32],
    grad_h_r: &mut [f32],
    grad_h_t: &mut [f32],
) -> f32 {
    let hidden = h_s.len();
    debug_assert_eq!(h_r.len(), hidden);
    debug_assert_eq!(h_t.len(), hidden);
    debug_assert_eq!(grad_h_s.len(), hidden);
    debug_assert_eq!(grad_h_r.len(), hidden);
    debug_assert_eq!(grad_h_t.len(), hidden);

    STP_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        if scratch.len() < 2 * hidden {
            scratch.resize(2 * hidden, 0.0);
        }
        let (d1, d2) = scratch.split_at_mut(hidden);
        let d1 = &mut d1[..hidden];
        let d2 = &mut d2[..hidden];

        for i in 0..hidden {
            d1[i] = h_r[i] - h_s[i];
            d2[i] = h_t[i] - h_s[i];
        }

        let dot12 = dot(d1, d2);
        let n1_sq = dot(d1, d1);
        let n2_sq = dot(d2, d2);
        let n1 = n1_sq.sqrt();
        let n2 = n2_sq.sqrt();
        let denom = n1 * n2;

        // Degenerate (zero-length step or chord): no direction to align, no grad.
        if denom < EPS {
            return 0.0;
        }

        let inv_denom = 1.0 / denom;
        let cos = (dot12 * inv_denom).clamp(-1.0, 1.0);
        let loss = 1.0 - cos;

        // dL/dd1 = -dcos/dd1 = -( d2/(n1 n2) - cos * d1 / n1^2 )
        // dL/dd2 = -dcos/dd2 = -( d1/(n1 n2) - cos * d2 / n2^2 )
        let inv_n1_sq = 1.0 / n1_sq.max(EPS);
        let inv_n2_sq = 1.0 / n2_sq.max(EPS);

        // grad wrt d1 and d2, then chain: h_r += g_d1, h_t += g_d2, h_s += -(g_d1+g_d2)
        for i in 0..hidden {
            let g_d1 = -lambda * (d2[i] * inv_denom - cos * d1[i] * inv_n1_sq);
            let g_d2 = -lambda * (d1[i] * inv_denom - cos * d2[i] * inv_n2_sq);
            grad_h_r[i] += g_d1;
            grad_h_t[i] += g_d2;
            grad_h_s[i] += -(g_d1 + g_d2);
        }

        loss
    })
}

/// Read the `MUD_TRAIN_STP` env flag (default OFF, AWAKE-01 discipline).
pub fn stp_enabled() -> bool {
    std::env::var("MUD_TRAIN_STP")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// STP loss weight `lambda_stp` (default 0.05 per plan §4.1). Override with
/// `MUD_TRAIN_STP_LAMBDA`.
pub fn stp_lambda() -> f32 {
    std::env::var("MUD_TRAIN_STP_LAMBDA")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(0.05)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finite_diff_grad(
        h_s: &[f32],
        h_r: &[f32],
        h_t: &[f32],
        which: usize, // 0=s, 1=r, 2=t
        idx: usize,
    ) -> f32 {
        let h = 1e-3f32;
        let hidden = h_s.len();
        let mut zs = vec![0.0; hidden];
        let mut zr = vec![0.0; hidden];
        let mut zt = vec![0.0; hidden];

        let eval = |s: &[f32], r: &[f32], t: &[f32]| -> f32 {
            let mut gs = vec![0.0; hidden];
            let mut gr = vec![0.0; hidden];
            let mut gt = vec![0.0; hidden];
            stp_loss_and_grad(s, r, t, 1.0, &mut gs, &mut gr, &mut gt)
        };

        let mut sp = h_s.to_vec();
        let mut rp = h_r.to_vec();
        let mut tp = h_t.to_vec();
        let target = match which {
            0 => &mut sp,
            1 => &mut rp,
            _ => &mut tp,
        };
        target[idx] += h;
        let lp = eval(&sp, &rp, &tp);

        let mut sm = h_s.to_vec();
        let mut rm = h_r.to_vec();
        let mut tm = h_t.to_vec();
        let target = match which {
            0 => &mut sm,
            1 => &mut rm,
            _ => &mut tm,
        };
        target[idx] -= h;
        let lm = eval(&sm, &rm, &tm);

        let _ = (&mut zs, &mut zr, &mut zt);
        (lp - lm) / (2.0 * h)
    }

    #[test]
    fn test_stp_zero_on_geodesic() {
        // h_r exactly on the segment h_s -> h_t => L == 0.
        let h_s = vec![0.0f32, 0.0, 0.0, 0.0];
        let h_t = vec![2.0f32, 4.0, 6.0, 8.0];
        let h_r: Vec<f32> = h_s.iter().zip(&h_t).map(|(a, b)| 0.5 * (a + b)).collect();
        let mut gs = vec![0.0; 4];
        let mut gr = vec![0.0; 4];
        let mut gt = vec![0.0; 4];
        let l = stp_loss_and_grad(&h_s, &h_r, &h_t, 1.0, &mut gs, &mut gr, &mut gt);
        assert!(l.abs() < 1e-5, "geodesic loss should be ~0, got {l}");
        for g in gr.iter() {
            assert!(g.abs() < 1e-4, "grad on geodesic should be ~0, got {g}");
        }
    }

    #[test]
    fn test_stp_positive_off_geodesic() {
        let h_s = vec![0.0f32, 0.0, 0.0, 0.0];
        let h_t = vec![2.0f32, 0.0, 0.0, 0.0];
        let h_r = vec![1.0f32, 1.0, 0.0, 0.0]; // bent off the x-axis chord
        let mut gs = vec![0.0; 4];
        let mut gr = vec![0.0; 4];
        let mut gt = vec![0.0; 4];
        let l = stp_loss_and_grad(&h_s, &h_r, &h_t, 1.0, &mut gs, &mut gr, &mut gt);
        assert!(l > 1e-3, "off-geodesic loss should be > 0, got {l}");
    }

    #[test]
    fn test_stp_grad_matches_finite_diff() {
        let h_s = vec![0.3f32, -0.7, 1.2, 0.1, -0.4, 0.9, 0.2, -1.1];
        let h_r = vec![0.9f32, -0.2, 1.6, -0.3, 0.5, 1.4, -0.1, -0.6];
        let h_t = vec![1.7f32, 0.4, 2.1, -0.9, 1.3, 2.0, -0.5, 0.2];
        let hidden = h_s.len();

        let mut gs = vec![0.0; hidden];
        let mut gr = vec![0.0; hidden];
        let mut gt = vec![0.0; hidden];
        stp_loss_and_grad(&h_s, &h_r, &h_t, 1.0, &mut gs, &mut gr, &mut gt);

        for (which, analytic) in [(0usize, &gs), (1, &gr), (2, &gt)] {
            for (idx, &a) in analytic.iter().enumerate().take(hidden) {
                let fd = finite_diff_grad(&h_s, &h_r, &h_t, which, idx);
                assert!(
                    (a - fd).abs() < 1e-3,
                    "which={which} idx={idx} analytic={a} fd={fd}"
                );
            }
        }
    }
}
