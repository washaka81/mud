//! # Adam / SparseAdam state (gap P0 — optimizer honesty)
//!
//! Persistent first/second moments for shadow weights. Used when
//! [`OptimizerStrategy::Adam`] or [`OptimizerStrategy::SparseAdam`] is selected.
//!
//! - **Adam:** full `adam_step_avx2` (or scalar fallback) with bias correction.
//! - **SparseAdam:** same update restricted to rows with non-negligible gradient
//!   (`only_active_rows`), saving bandwidth on huge matrices.

use crate::mud::constants::EPSILON_FLOOR;
use crate::mud::slime_backward::OptimizerStrategy;

/// Default Adam β₁.
pub const ADAM_B1: f32 = 0.9;
/// Default Adam β₂.
pub const ADAM_B2: f32 = 0.999;
/// Default Adam ε.
pub const ADAM_EPS: f32 = 1e-8;
/// Row is “active” if max |g| on the row exceeds this (SparseAdam).
pub const SPARSE_ROW_EPS: f32 = 1e-12;

/// First/second moment buffers + step counter for one matrix.
#[derive(Clone, Debug)]
pub struct AdamState {
    pub m: Vec<f32>,
    pub v: Vec<f32>,
    pub step: u64,
}

impl AdamState {
    pub fn zeros(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            step: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.m.len()
    }

    pub fn is_empty(&self) -> bool {
        self.m.is_empty()
    }

    /// Allocate only if strategy needs moments.
    pub fn for_strategy(n: usize, strategy: OptimizerStrategy) -> Option<Self> {
        match strategy {
            OptimizerStrategy::Adam | OptimizerStrategy::SparseAdam { .. } => Some(Self::zeros(n)),
            _ => None,
        }
    }
}

/// Whether this strategy uses Adam moments (vs Muon/GaLore preprocess + SGD).
#[inline]
pub fn strategy_uses_adam(strategy: OptimizerStrategy) -> bool {
    matches!(
        strategy,
        OptimizerStrategy::Adam | OptimizerStrategy::SparseAdam { .. }
    )
}

/// Bias-corrected Adam scalars for ASM: `lr_bc1 = lr/(1-b1^t)`, `inv_bc2 = 1/(1-b2^t)`.
#[inline]
pub fn adam_bias_scalars(step: u64, lr: f32, b1: f32, b2: f32) -> (f32, f32) {
    let t = step.max(1) as i32;
    let bc1 = 1.0 - b1.powi(t);
    let bc2 = 1.0 - b2.powi(t);
    let lr_bc1 = lr / bc1.max(EPSILON_FLOOR);
    let inv_bc2 = 1.0 / bc2.max(EPSILON_FLOOR);
    (lr_bc1, inv_bc2)
}

/// Full-matrix Adam step (updates `w`, `state.m`, `state.v` in place).
/// Gradients should already be token-normalized if desired.
///
/// # Safety
/// `w`, `g`, and `state` lengths must match; `adam_step_avx2` requires aligned-ish buffers.
pub fn adam_step(
    w: &mut [f32],
    g: &[f32],
    state: &mut AdamState,
    lr: f32,
    weight_decay: f32,
    clip_coef: f32,
) {
    let n = w.len().min(g.len()).min(state.m.len()).min(state.v.len());
    if n == 0 {
        return;
    }
    state.step = state.step.saturating_add(1);
    let (lr_bc1, inv_bc2) = adam_bias_scalars(state.step, lr, ADAM_B1, ADAM_B2);

    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe {
            crate::asm::adam_step_avx2(
                n,
                w.as_mut_ptr(),
                state.m.as_mut_ptr(),
                state.v.as_mut_ptr(),
                g.as_ptr(),
                clip_coef,
                weight_decay,
                ADAM_B1,
                ADAM_B2,
                lr_bc1,
                inv_bc2,
                ADAM_EPS,
            );
        }
    } else {
        adam_step_scalar(
            &mut w[..n],
            &g[..n],
            &mut state.m[..n],
            &mut state.v[..n],
            lr_bc1,
            inv_bc2,
            weight_decay,
            clip_coef,
        );
    }
}

/// Scalar Adam (bias-corrected), same formula as ASM.
#[allow(clippy::too_many_arguments)]
fn adam_step_scalar(
    w: &mut [f32],
    g: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    lr_bc1: f32,
    inv_bc2: f32,
    wd: f32,
    clip: f32,
) {
    for i in 0..w.len() {
        let mut gi = g[i];
        if !gi.is_finite() {
            gi = 0.0;
        }
        gi = gi * clip + wd * w[i];
        m[i] = ADAM_B1 * m[i] + (1.0 - ADAM_B1) * gi;
        v[i] = ADAM_B2 * v[i] + (1.0 - ADAM_B2) * gi * gi;
        let denom = (v[i] * inv_bc2).sqrt() + ADAM_EPS;
        w[i] -= lr_bc1 * m[i] / denom;
    }
}

/// SparseAdam: only rows whose max |g| ≥ [`SPARSE_ROW_EPS`] are updated.
#[allow(clippy::too_many_arguments)]
pub fn sparse_adam_step(
    w: &mut [f32],
    g: &[f32],
    state: &mut AdamState,
    rows: usize,
    cols: usize,
    lr: f32,
    weight_decay: f32,
    clip_coef: f32,
    only_active_rows: bool,
) {
    let cols = cols.max(1);
    let n = rows * cols;
    if w.len() < n || g.len() < n || state.m.len() < n {
        return;
    }
    if !only_active_rows {
        adam_step(w, g, state, lr, weight_decay, clip_coef);
        return;
    }

    state.step = state.step.saturating_add(1);
    let (lr_bc1, inv_bc2) = adam_bias_scalars(state.step, lr, ADAM_B1, ADAM_B2);

    for r in 0..rows {
        let start = r * cols;
        let end = start + cols;
        let row_g = &g[start..end];
        let max_abs = row_g
            .iter()
            .map(|gi| if gi.is_finite() { gi.abs() } else { 0.0 })
            .fold(0.0f32, f32::max);
        if max_abs < SPARSE_ROW_EPS {
            continue;
        }
        // Row-wise scalar Adam (avoids dense ASM when few rows active)
        for c in start..end {
            let mut gi = g[c];
            if !gi.is_finite() {
                gi = 0.0;
            }
            gi = gi * clip_coef + weight_decay * w[c];
            state.m[c] = ADAM_B1 * state.m[c] + (1.0 - ADAM_B1) * gi;
            state.v[c] = ADAM_B2 * state.v[c] + (1.0 - ADAM_B2) * gi * gi;
            let denom = (state.v[c] * inv_bc2).sqrt() + ADAM_EPS;
            w[c] -= lr_bc1 * state.m[c] / denom;
        }
    }
}

/// Normalize grad by num_tokens into `out` (EZOP).
pub fn scale_grad_by_tokens(g: &mut [f32], num_tokens: f32) {
    let ntok = num_tokens.max(1.0);
    if (ntok - 1.0).abs() < 1e-12 {
        return;
    }
    let inv = 1.0 / ntok;
    for x in g.iter_mut() {
        if x.is_finite() {
            *x *= inv;
        } else {
            *x = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adam_reduces_loss_quadratic() {
        // Minimize (w - 1)^2 roughly: g = 2(w-1), start w=0
        let mut w = vec![0.0f32; 64];
        let mut state = AdamState::zeros(64);
        for _ in 0..200 {
            let g: Vec<f32> = w.iter().map(|&wi| 2.0 * (wi - 1.0)).collect();
            adam_step(&mut w, &g, &mut state, 0.05, 0.0, 1.0);
        }
        let mean: f32 = w.iter().sum::<f32>() / w.len() as f32;
        assert!(
            (mean - 1.0).abs() < 0.15,
            "Adam should approach w=1, mean={mean}"
        );
        assert!(state.step >= 200);
    }

    #[test]
    fn test_sparse_skips_zero_rows() {
        let rows = 4usize;
        let cols = 8usize;
        let n = rows * cols;
        let mut w = vec![1.0f32; n];
        let mut g = vec![0.0f32; n];
        // Only row 2 has gradient
        for c in 0..cols {
            g[2 * cols + c] = 0.5;
        }
        let mut state = AdamState::zeros(n);
        let w_before = w.clone();
        sparse_adam_step(&mut w, &g, &mut state, rows, cols, 0.1, 0.0, 1.0, true);
        for r in 0..rows {
            for c in 0..cols {
                let i = r * cols + c;
                if r == 2 {
                    assert!((w[i] - w_before[i]).abs() > 1e-8, "active row should move");
                } else {
                    assert_eq!(w[i], w_before[i], "inactive row must not change");
                    assert_eq!(state.m[i], 0.0);
                }
            }
        }
    }

    #[test]
    fn test_bias_scalars_monotonic() {
        let (lr1, _) = adam_bias_scalars(1, 1e-3, ADAM_B1, ADAM_B2);
        let (lr100, _) = adam_bias_scalars(100, 1e-3, ADAM_B1, ADAM_B2);
        // Early steps have larger effective lr due to bias correction
        assert!(lr1 > lr100);
        assert!(lr100 > 1e-3 * 0.9);
    }

    #[test]
    fn test_for_strategy_alloc() {
        assert!(AdamState::for_strategy(16, OptimizerStrategy::Adam).is_some());
        assert!(AdamState::for_strategy(
            16,
            OptimizerStrategy::SparseAdam {
                only_active_rows: true
            }
        )
        .is_some());
        assert!(AdamState::for_strategy(16, OptimizerStrategy::Muon { ns_iters: 5 }).is_none());
    }
}
