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

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Default Adam β₁.
pub const ADAM_B1: f32 = 0.9;
/// Default Adam β₂.
pub const ADAM_B2: f32 = 0.999;
/// Default Adam ε.
pub const ADAM_EPS: f32 = 1e-8;
/// Row is "active" if max |g| on the row exceeds this (SparseAdam).
pub const SPARSE_ROW_EPS: f32 = 1e-12;

/// D2: Convert f32 → bf16 (truncate mantissa, no rounding).
/// Google bf16 format: 1 sign + 8 exponent + 7 mantissa (same exponent as f32).
#[inline]
pub fn f32_to_bf16(f: f32) -> u16 {
    (f.to_bits() >> 16) as u16
}

/// D2: Convert bf16 → f32 (shift left 16 bits to reconstruct f32).
#[inline]
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// D2: Cached env var check for `MUD_BF16_SHADOW` (P-01 hot-path optimization).
/// `std::env::var` is a syscall — caching via `thread_local` avoids it on every
/// optimizer step (called once per matrix per training chunk).
fn bf16_shadow_enabled() -> bool {
    BF16_SHADOW_CACHE.with(|cache| {
        if let Some(v) = *cache.borrow() {
            return v;
        }
        let v = std::env::var("MUD_BF16_SHADOW").is_ok();
        *cache.borrow_mut() = Some(v);
        v
    })
}

thread_local! {
    #[allow(clippy::missing_const_for_thread_local)]
    static BF16_SHADOW_CACHE: std::cell::RefCell<Option<bool>> = std::cell::RefCell::new(None);
}

/// D2: Reset the bf16 shadow cache (test-only).
#[cfg(test)]
fn reset_bf16_shadow_cache() {
    BF16_SHADOW_CACHE.with(|c| *c.borrow_mut() = None);
}

/// First/second moment buffers + step counter for one matrix.
///
/// D2 (Phase D): `shadow_bf16` optionally stores a bf16 snapshot of the
/// shadow weights after each optimizer step, halving the memory footprint
/// of the shadow (FP32 moments are still kept for numerical stability).
#[derive(Clone, Debug)]
pub struct AdamState {
    pub m: Vec<f32>,
    pub v: Vec<f32>,
    pub step: u64,
    /// Optional bf16 shadow snapshot (MUD_BF16_SHADOW=1). None = FP32 shadow only.
    pub shadow_bf16: Option<Vec<u16>>,
}

impl AdamState {
    pub fn zeros(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            step: 0,
            shadow_bf16: None,
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

    /// D2: Store a bf16 snapshot of the current shadow weights.
    /// Only allocates if `MUD_BF16_SHADOW` is set; reuses the existing
    /// buffer on subsequent calls (P-01: zero-alloc after first call).
    ///
    /// Hot-path optimization: `MUD_BF16_SHADOW` env var is cached via
    /// `OnceLock` (P-01) — avoids a `std::env::var` syscall on every
    /// optimizer step (called once per matrix per chunk).
    pub fn store_shadow_bf16(&mut self, weights: &[f32]) {
        if !bf16_shadow_enabled() {
            return;
        }
        let n = weights.len();
        match &mut self.shadow_bf16 {
            Some(buf) => {
                // Reuse: write in-place, truncate to fit
                let reuse_len = buf.len().min(n);
                // Raw pointer loop — compiler auto-vectorizes f32_to_bf16 (bits >> 16).
                unsafe {
                    let wp = weights.as_ptr();
                    let op = buf.as_mut_ptr();
                    for i in 0..reuse_len {
                        *op.add(i) = f32_to_bf16(*wp.add(i));
                    }
                }
                if buf.len() < n {
                    buf.extend(weights[buf.len()..].iter().map(|&w| f32_to_bf16(w)));
                } else if buf.len() > n {
                    buf.truncate(n);
                }
            }
            None => {
                // P-01: pre-allocate with capacity, raw pointer fill (no iterator map/collect overhead).
                let mut buf: Vec<u16> = Vec::with_capacity(n);
                unsafe {
                    let wp = weights.as_ptr();
                    buf.set_len(n);
                    let op = buf.as_mut_ptr();
                    for i in 0..n {
                        *op.add(i) = f32_to_bf16(*wp.add(i));
                    }
                }
                self.shadow_bf16 = Some(buf);
            }
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
/// R-18 (AdamW desacoplado): el decay NO entra en `gi` (momentos puros sobre
/// el gradiente). Se aplica fuera como `w *= (1 - lr*wd)` ANTES del kernel
/// (exacto: el update no depende de `w` una vez que el kernel recibe `wd=0`).
/// El kernel ASM (`adam_step.s`) conserva su fórmula acoplada pero siempre se
/// invoca con `wd=0`; ver nota en `adam_step.s:71`.
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
    // Decoupled decay first (AdamW). Non-finite weights are left untouched
    // here (el kernel los sanea vía gi=0 y el update los mueve igual).
    if weight_decay.is_finite() && weight_decay != 0.0 && lr.is_finite() && lr != 0.0 {
        let f = (1.0 - lr * weight_decay).clamp(0.0, 1.0);
        if (f - 1.0).abs() > f32::EPSILON {
            for wi in w.iter_mut().take(n) {
                if wi.is_finite() {
                    *wi *= f;
                }
            }
        }
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
                0.0, // R-18: wd=0 — decay ya aplicado fuera (AdamW)
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
            0.0, // R-18: wd=0 — decay ya aplicado fuera (AdamW)
            clip_coef,
        );
    }
}

/// Scalar Adam (bias-corrected), same formula as ASM.
/// P-00: raw-pointer pass — no bounds checks, no iterator overhead.
/// R-18: `wd` es legado — los wrappers (`adam_step`) siempre pasan `wd=0`
/// (decay desacoplado aplicado fuera). Se conserva el parámetro por ABI.
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
    let n = w.len();
    let wp = w.as_mut_ptr();
    let gp = g.as_ptr();
    let mp = m.as_mut_ptr();
    let vp = v.as_mut_ptr();
    for i in 0..n {
        unsafe {
            let mut gi = *gp.add(i);
            if !gi.is_finite() {
                gi = 0.0;
            }
            gi = gi * clip + wd * *wp.add(i);
            *mp.add(i) = ADAM_B1 * *mp.add(i) + (1.0 - ADAM_B1) * gi;
            *vp.add(i) = ADAM_B2 * *vp.add(i) + (1.0 - ADAM_B2) * gi * gi;
            let denom = (*vp.add(i) * inv_bc2).sqrt() + ADAM_EPS;
            *wp.add(i) -= lr_bc1 * *mp.add(i) / denom;
        }
    }
}

/// Row max |g| via AVX2 (skip non-finite). Processes 8 f32/iter.
///
/// # Safety
/// - `g` must point to at least `n` valid f32 values.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn peak_abs_avx2(g: *const f32, n: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut max_abs = 0.0f32;
    let mut i = 0;
    if n >= 8 {
        let zero = _mm256_setzero_ps();
        let mut max_vec = zero;
        while i + 8 <= n {
            let gv = _mm256_loadu_ps(g.add(i));
            // is_finite check: ordered comparison (NaN fails since NaN != NaN)
            let finite = _mm256_cmp_ps(gv, gv, _CMP_ORD_Q);
            let absv = _mm256_andnot_ps(_mm256_set1_ps(-0.0), gv); // |x| via sign-bit clear
            let masked = _mm256_blendv_ps(zero, absv, finite);
            max_vec = _mm256_max_ps(max_vec, masked);
            i += 8;
        }
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), max_vec);
        max_abs = tmp.iter().copied().fold(0.0f32, f32::max);
    }
    // Scalar remainder
    while i < n {
        let gi = *g.add(i);
        if gi.is_finite() {
            let a = gi.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        i += 1;
    }
    max_abs
}

/// Scalar fallback for peak_abs (non-x86_64).
#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub unsafe fn peak_abs_avx2(g: *const f32, n: usize) -> f32 {
    let mut max_abs = 0.0f32;
    for i in 0..n {
        let gi = *g.add(i);
        if gi.is_finite() {
            let a = gi.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
    }
    max_abs
}

/// SparseAdam AVX2 update loop for a single row. Processes 8 f32/iter via FMA.
/// Assumes `max_abs >= SPARSE_ROW_EPS` (row already active).
/// R-18: `weight_decay` es legado — el llamador (`sparse_adam_step`) pasa
/// siempre `0.0` (decay desacoplado por fila fuera del kernel).
///
/// # Safety
/// - `wp`, `gp`, `mp`, `vp` must point to at least `cols` valid f32 values each.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn sparse_adam_row_avx2(
    wp: *mut f32,
    gp: *const f32,
    mp: *mut f32,
    vp: *mut f32,
    cols: usize,
    lr_bc1: f32,
    inv_bc2: f32,
    clip_coef: f32,
    weight_decay: f32,
) {
    let b1 = _mm256_set1_ps(ADAM_B1);
    let b1c = _mm256_set1_ps(1.0 - ADAM_B1);
    let b2 = _mm256_set1_ps(ADAM_B2);
    let b2c = _mm256_set1_ps(1.0 - ADAM_B2);
    let clip = _mm256_set1_ps(clip_coef);
    let wd = _mm256_set1_ps(weight_decay);
    let lr = _mm256_set1_ps(lr_bc1);
    let invbc2 = _mm256_set1_ps(inv_bc2);
    let eps = _mm256_set1_ps(ADAM_EPS);
    let zero = _mm256_setzero_ps();

    let mut i = 0;
    while i + 8 <= cols {
        let g_vec = _mm256_loadu_ps(gp.add(i));
        // is_finite check
        let finite = _mm256_cmp_ps(g_vec, g_vec, _CMP_ORD_Q);
        let gi_vec = _mm256_blendv_ps(zero, g_vec, finite);
        // gi = gi * clip + wd * w
        let wp_vec = _mm256_loadu_ps(wp.add(i));
        let gi_vec = _mm256_fmadd_ps(clip, gi_vec, _mm256_mul_ps(wd, wp_vec));
        // m = b1*m + (1-b1)*gi
        let mp_vec = _mm256_loadu_ps(mp.add(i));
        let m_new = _mm256_fmadd_ps(b1c, gi_vec, _mm256_mul_ps(b1, mp_vec));
        // v = b2*v + (1-b2)*gi*gi
        let vp_vec = _mm256_loadu_ps(vp.add(i));
        let gi_sq = _mm256_mul_ps(gi_vec, gi_vec);
        let v_new = _mm256_fmadd_ps(b2c, gi_sq, _mm256_mul_ps(b2, vp_vec));
        // denom = sqrt(v * inv_bc2) + eps
        let denom = _mm256_add_ps(_mm256_sqrt_ps(_mm256_mul_ps(v_new, invbc2)), eps);
        // w -= lr * m / denom
        let w_delta = _mm256_div_ps(_mm256_mul_ps(lr, m_new), denom);
        let w_new = _mm256_sub_ps(wp_vec, w_delta);
        // Store
        _mm256_storeu_ps(mp.add(i), m_new);
        _mm256_storeu_ps(vp.add(i), v_new);
        _mm256_storeu_ps(wp.add(i), w_new);
        i += 8;
    }
    // Scalar remainder
    while i < cols {
        let mut gi = *gp.add(i);
        if !gi.is_finite() {
            gi = 0.0;
        }
        gi = gi * clip_coef + weight_decay * *wp.add(i);
        *mp.add(i) = ADAM_B1 * *mp.add(i) + (1.0 - ADAM_B1) * gi;
        *vp.add(i) = ADAM_B2 * *vp.add(i) + (1.0 - ADAM_B2) * gi * gi;
        let denom = (*vp.add(i) * inv_bc2).sqrt() + ADAM_EPS;
        *wp.add(i) -= lr_bc1 * *mp.add(i) / denom;
        i += 1;
    }
}

/// SparseAdam: only rows whose max |g| ≥ [`SPARSE_ROW_EPS`] are updated.
/// R-18: AdamW desacoplado por fila activa — `w_fila *= (1-lr*wd)` antes del
/// kernel de fila (que recibe `wd=0`). Las filas inactivas no se tocan
/// (ni update ni decay): es la concesión de esparsidad documentada; el decay
/// total coincide con AdamW denso salvo filas bajo el umbral de actividad.
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

    // P-00: raw-pointer pass — no bounds checks in the inner loop.
    let wp = w.as_mut_ptr();
    let gp = g.as_ptr();
    let mp = state.m.as_mut_ptr();
    let vp = state.v.as_mut_ptr();
    let use_avx = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    for r in 0..rows {
        let start = r * cols;
        let row_ptr = unsafe { gp.add(start) };
        // Row max |g| (skip non-finite)
        let max_abs = if use_avx {
            unsafe { peak_abs_avx2(row_ptr, cols) }
        } else {
            let mut m = 0.0f32;
            for c in start..start + cols {
                unsafe {
                    let gi = *gp.add(c);
                    if gi.is_finite() {
                        let a = gi.abs();
                        if a > m {
                            m = a;
                        }
                    }
                }
            }
            m
        };
        if max_abs < SPARSE_ROW_EPS {
            continue;
        }
        // R-18: decoupled decay on the active row only, then pure-grad kernel.
        if weight_decay.is_finite() && weight_decay != 0.0 && lr.is_finite() && lr != 0.0 {
            let f = (1.0 - lr * weight_decay).clamp(0.0, 1.0);
            if (f - 1.0).abs() > f32::EPSILON {
                unsafe {
                    for c in 0..cols {
                        let p = wp.add(start + c);
                        if (*p).is_finite() {
                            *p *= f;
                        }
                    }
                }
            }
        }
        if use_avx {
            unsafe {
                let row_wp = wp.add(start);
                let row_mp = mp.add(start);
                let row_vp = vp.add(start);
                sparse_adam_row_avx2(
                    row_wp, row_ptr, row_mp, row_vp,
                    cols, lr_bc1, inv_bc2, clip_coef, 0.0, // R-18: wd=0
                );
            }
        } else {
            for c in 0..cols {
                unsafe {
                    let mut gi = *gp.add(start + c);
                    if !gi.is_finite() {
                        gi = 0.0;
                    }
                    gi *= clip_coef; // R-18: sin término wd*w (AdamW)
                    *mp.add(start + c) = ADAM_B1 * *mp.add(start + c) + (1.0 - ADAM_B1) * gi;
                    *vp.add(start + c) = ADAM_B2 * *vp.add(start + c) + (1.0 - ADAM_B2) * gi * gi;
                    let denom = (*vp.add(start + c) * inv_bc2).sqrt() + ADAM_EPS;
                    *wp.add(start + c) -= lr_bc1 * *mp.add(start + c) / denom;
                }
            }
        }
    }
}

/// Normalize grad by num_tokens into `out` (EZOP). Vectorized AVX2 path.
pub fn scale_grad_by_tokens(g: &mut [f32], num_tokens: f32) {
    let ntok = num_tokens.max(1.0);
    if (ntok - 1.0).abs() < 1e-12 {
        return;
    }
    let inv = 1.0 / ntok;
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && !g.is_empty() {
        unsafe {
            use std::arch::x86_64::*;
            let gp = g.as_mut_ptr();
            let n = g.len();
            let zero = _mm256_setzero_ps();
            let inv_vec = _mm256_set1_ps(inv);
            let mut i = 0;
            while i + 8 <= n {
                let gv = _mm256_loadu_ps(gp.add(i));
                // is_finite check: ordered comparison (NaN fails)
                let finite = _mm256_cmp_ps(gv, gv, _CMP_ORD_Q);
                let scaled = _mm256_mul_ps(gv, inv_vec);
                let result = _mm256_blendv_ps(zero, scaled, finite);
                _mm256_storeu_ps(gp.add(i), result);
                i += 8;
            }
            // Scalar remainder
            for j in i..n {
                let mut x = *g.get_unchecked(j);
                if x.is_finite() {
                    x *= inv;
                } else {
                    x = 0.0;
                }
                *g.get_unchecked_mut(j) = x;
            }
            return;
        }
    }
    // Scalar fallback (non-x86_64 or no AVX2)
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

    #[test]
    fn test_bf16_roundtrip() {
        // bf16 has 7 mantissa bits — precision ~1e-3
        let test_vals: [f32; 8] = [
            0.0,
            1.0,
            -1.0,
            0.5,
            -0.5,
            123.456,
            -789.012,
            1e-5,
        ];
        for &v in &test_vals {
            let bf16 = f32_to_bf16(v);
            let back = bf16_to_f32(bf16);
            let rel_err = ((v - back).abs() / v.abs().max(1e-10)).min(1.0);
            assert!(
                rel_err < 0.01,
                "bf16 roundtrip failed for {v}: got {back}, rel_err={rel_err}"
            );
        }
    }

    #[test]
    fn test_bf16_zero_and_sign() {
        assert_eq!(f32_to_bf16(0.0), 0);
        assert_eq!(f32_to_bf16(-0.0), 0x8000);
        assert!(bf16_to_f32(0) == 0.0);
    }

    #[test]
    fn test_shadow_bf16_store() {
        // M-2 PURGE (2026-08-03): formerly `test_shadow_bf16_store_load`.
        // `load_shadow_bf16` was dead (tests-only) — removed, so the load
        // roundtrip assertions are gone. Remaining: store path + env gating.
        let mut state = AdamState::zeros(64);
        assert!(state.shadow_bf16.is_none());

        // Without MUD_BF16_SHADOW, store is a no-op
        let weights: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        std::env::remove_var("MUD_BF16_SHADOW");
        reset_bf16_shadow_cache();
        state.store_shadow_bf16(&weights);
        assert!(state.shadow_bf16.is_none());

        // With MUD_BF16_SHADOW=1, store creates the buffer
        std::env::set_var("MUD_BF16_SHADOW", "1");
        reset_bf16_shadow_cache();
        state.store_shadow_bf16(&weights);
        assert!(state.shadow_bf16.is_some());
        let buf = state.shadow_bf16.as_ref().unwrap();
        assert_eq!(buf.len(), 64);
        // Spot-check: first weight packs to its bf16 truncation.
        assert_eq!(buf[0], f32_to_bf16(weights[0]));

        // Reuse: store again with same size — no reallocation
        state.store_shadow_bf16(&weights);
        assert_eq!(state.shadow_bf16.as_ref().unwrap().len(), 64);

        // Cleanup
        std::env::remove_var("MUD_BF16_SHADOW");
        reset_bf16_shadow_cache();
    }

    #[test]
    fn test_sparse_adam_equivalence_avx2_vs_scalar() {
        // Verify AVX2 and scalar paths produce identical results
        let rows = 8;
        let cols = 16;
        let n = rows * cols;
        let mut w_ref = vec![0.5f32; n];
        let mut w_avx = vec![0.5f32; n];
        let mut g = vec![0.0f32; n];
        // Random-ish gradients (deterministic)
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            g[i] = ((i as f32 * 1.7) % 1.0) - 0.5;
        }
        let mut state_ref = AdamState::zeros(n);
        let mut state_avx = AdamState::zeros(n);
        let lr = 0.01;
        let wd = 0.01;
        let clip = 1.0;

        // Run 5 steps
        for _ in 0..5 {
            sparse_adam_step(&mut w_ref, &g, &mut state_ref, rows, cols, lr, wd, clip, true);
            sparse_adam_step(&mut w_avx, &g, &mut state_avx, rows, cols, lr, wd, clip, true);
        }
        for i in 0..n {
            assert!(
                (w_ref[i] - w_avx[i]).abs() < 1e-5,
                "weight mismatch at {}: ref={} avx={}", i, w_ref[i], w_avx[i]
            );
        }
    }

    #[test]
    fn test_scale_grad_by_tokens_nan_guard() {
        let mut g = vec![1.0f32, 2.0, f32::NAN, -3.0, f32::INFINITY, 4.0];
        scale_grad_by_tokens(&mut g, 2.0);
        // Finite values scaled by 0.5
        assert!((g[0] - 0.5).abs() < 1e-6);
        assert!((g[1] - 1.0).abs() < 1e-6);
        // Non-finite → 0
        assert!(g[2] == 0.0);
        assert!((g[3] - (-1.5)).abs() < 1e-6);
        assert!(g[4] == 0.0);
        assert!((g[5] - 2.0).abs() < 1e-6);
    }

    /// R-18 (AdamW desacoplado): con `g=0` los momentos NO se contaminan
    /// (`m=v=0` tras el paso) y el peso decae exactamente `lr*wd*w`
    /// (sin amplificación por momento). Con el Adam+L2 acoplado anterior,
    /// `gi = wd*w` contaminaba `m,v` y el update era `lr*m/(sqrt(v)+eps)`
    /// sobre un gradiente fantasma.
    #[test]
    fn test_adamw_decoupled_zero_grad_r18() {
        let n = 16;
        let mut w = vec![1.0f32; n];
        let g = vec![0.0f32; n];
        let mut state = AdamState::zeros(n);
        let lr = 0.01f32;
        let wd = 0.01f32;
        adam_step(&mut w, &g, &mut state, lr, wd, 1.0);
        // Momentos limpios: el decay no entra en gi.
        for (i, (mi, vi)) in state.m.iter().zip(state.v.iter()).enumerate() {
            assert_eq!(*mi, 0.0, "R-18: m[{i}] contaminado por WD");
            assert_eq!(*vi, 0.0, "R-18: v[{i}] contaminado por WD");
        }
        // Decoupled decay exacto: w *= (1-lr*wd); update de momento = 0.
        let expected = 1.0 - lr * wd;
        for (i, wi) in w.iter().enumerate() {
            assert!(
                (*wi - expected).abs() < 1e-6,
                "R-18: w[{i}]={wi} debe ser {expected} (decay desacoplado)",
            );
        }
    }
}
