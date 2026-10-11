//! C-MUD thinking kernel — pure complex arithmetic extracted from `cmud.rs`.
//!
//! Contains the mathematical primitives for complex-valued thinking:
//! Gaussian ternary multiplication, wave-function collapse, Hermitian ball
//! projection, and angular phase utilities. All functions are pure (no
//! side-effects on global state) and depend only on `ComplexF32`/`GaussTernary`
//! from `cmud_params`.

use crate::mud::cmud_params::{ComplexF32, GaussTernary};
use crate::mud::constants::EPSILON_FLOOR;

/// Complex multiply by Gaussian ternary (adds/subs only):
/// \(Y_R = X_R W_R - X_I W_I\), \(Y_I = X_R W_I + X_I W_R\).
#[inline]
pub fn gauss_mul(x: ComplexF32, w: GaussTernary) -> ComplexF32 {
    let wr = w.wr as f32;
    let wi = w.wi as f32;
    ComplexF32 {
        re: x.re * wr - x.im * wi,
        im: x.re * wi + x.im * wr,
    }
}

/// Accumulate `out += gauss_mul(x, w) * scale` (PRQ-style).
#[inline]
pub fn gauss_mac(out: &mut ComplexF32, x: ComplexF32, w: GaussTernary, scale: f32) {
    let y = gauss_mul(x, w).scale(scale);
    out.re += y.re;
    out.im += y.im;
}

/// Batch complex multiply: Y = X * W for 8 complex numbers at once (AVX2).
///
/// Processes 8 complex numbers simultaneously using packed f32 (8 values per YMM register).
/// Falls back to scalar `gauss_mul` for remaining elements (< 8).
///
/// # Safety
/// All slices must have at least `n` elements. `n` should be ≥ 8 for AVX2 path.
pub fn gauss_mul_batch_avx2(
    x_re: &[f32],
    x_im: &[f32],
    w_re: &[f32],
    w_im: &[f32],
    y_re: &mut [f32],
    y_im: &mut [f32],
    n: usize,
) {
    let mut i = 0;
    // AVX2 path: process 8 elements at once
    while i + 8 <= n {
        unsafe {
            crate::asm::cmud_gauss_mul_batch_avx2(
                x_re.as_ptr().add(i),
                x_im.as_ptr().add(i),
                w_re.as_ptr().add(i),
                w_im.as_ptr().add(i),
                y_re.as_mut_ptr().add(i),
                y_im.as_mut_ptr().add(i),
            );
        }
        i += 8;
    }
    // Scalar fallback for remaining elements
    while i < n {
        let x = ComplexF32::new(x_re[i], x_im[i]);
        let w = GaussTernary::new(w_re[i] as i8, w_im[i] as i8);
        let y = gauss_mul(x, w);
        y_re[i] = y.re;
        y_im[i] = y.im;
        i += 1;
    }
}

/// Batch wave-collapse: collapse 8 complex states to real values (full AVX2 vectorization).
///
/// Uses the algebraic identity `cos(atan2(im, re)) = re / sqrt(re² + im²)` to eliminate
/// the scalar `atan2` + `cos` calls (the two most expensive transcendentals in the
/// scalar `wave_collapse`), replacing them with vectorized `vsqrtps` + `vdivps`.
/// The `tanh(|im|)` term is computed scalar (platform libm, full precision) since
/// AVX2 has no native `tanh` instruction — the net win is replacing 2 transcendentals
/// (`atan2`, `cos`) with 1 (`sqrt`, `div`) per 8 elements.
///
/// Processes 8 complex numbers simultaneously using packed f32 (8 values per YMM register).
/// Falls back to scalar `wave_collapse` for remaining elements (< 8) and when AVX2 is unavailable.
pub fn wave_collapse_batch_avx2(re: &[f32], im: &[f32], out: &mut [f32], n: usize) {
    use core::arch::x86_64::*;

    let mut i = 0;
    if is_x86_feature_detected!("avx2") {
        // Constants for vectorized computation — initialized in unsafe block (target_feature)
        let (sign_mask, inf, eps, one) = unsafe {
            (
                _mm256_set1_ps(-0.0f32),       // 0x80000000 — clears sign bit via vandnps
                _mm256_set1_ps(f32::INFINITY),
                _mm256_set1_ps(EPSILON_FLOOR),
                _mm256_set1_ps(1.0f32),
            )
        };

        while i + 8 <= n {
            unsafe {
                // Load 8 re and 8 im values
                let v_re = _mm256_loadu_ps(re.as_ptr().add(i));
                let v_im = _mm256_loadu_ps(im.as_ptr().add(i));

                // Finiteness check: |v| < INFINITY catches both NaN and inf
                let abs_re = _mm256_andnot_ps(sign_mask, v_re);
                let abs_im = _mm256_andnot_ps(sign_mask, v_im);
                let re_finite = _mm256_cmp_ps(abs_re, inf, _CMP_LT_OQ);
                let im_finite = _mm256_cmp_ps(abs_im, inf, _CMP_LT_OQ);
                let both_finite = _mm256_and_ps(re_finite, im_finite);

                // tanh(|im|) — scalar (platform libm, full precision)
                // AVX2 has no native tanh; scalar is the same cost as in wave_collapse(h).
                let mut abs_im_arr = [0.0f32; 8];
                _mm256_storeu_ps(abs_im_arr.as_mut_ptr(), abs_im);
                let tanh_arr = [
                    abs_im_arr[0].tanh(),
                    abs_im_arr[1].tanh(),
                    abs_im_arr[2].tanh(),
                    abs_im_arr[3].tanh(),
                    abs_im_arr[4].tanh(),
                    abs_im_arr[5].tanh(),
                    abs_im_arr[6].tanh(),
                    abs_im_arr[7].tanh(),
                ];
                let v_tanh = _mm256_loadu_ps(tanh_arr.as_ptr());

                // 1 + tanh(|im|)
                let one_plus_tanh = _mm256_add_ps(one, v_tanh);

                // re² and im²
                let re2 = _mm256_mul_ps(v_re, v_re);
                let im2 = _mm256_mul_ps(v_im, v_im);

                // r = max(sqrt(re² + im²), EPSILON_FLOOR) — avoids 0/0 → NaN
                let r = _mm256_max_ps(_mm256_sqrt_ps(_mm256_add_ps(re2, im2)), eps);

                // v = re² * (1 + tanh(|im|)) / r
                // (identity: cos(atan2(im, re)) = re / sqrt(re² + im²), so
                //  v = re * (1 + tanh(|im|)) * re / r = re² * (1 + tanh(|im|)) / r)
                let v = _mm256_div_ps(_mm256_mul_ps(re2, one_plus_tanh), r);

                // Zero out non-finite inputs (matches scalar wave_collapse behavior)
                let v = _mm256_and_ps(v, both_finite);

                _mm256_storeu_ps(out.as_mut_ptr().add(i), v);
            }
            i += 8;
        }
    }
    // Scalar fallback for remaining elements
    while i < n {
        let h = ComplexF32::new(re[i], im[i]);
        out[i] = wave_collapse(h);
        i += 1;
    }
}

/// Hermitian ball projection (complex mHC): if \(\|h\|_C > R\), scale onto the sphere.
#[inline]
pub fn project_hermitian(h: ComplexF32, radius: f32) -> ComplexF32 {
    let r = radius.max(EPSILON_FLOOR);
    let n2 = h.hermite_norm_sq();
    if !n2.is_finite() {
        return ComplexF32::default();
    }
    if n2 <= r * r {
        return h;
    }
    let n = n2.sqrt().max(EPSILON_FLOOR);
    h.scale(r / n)
}

/// Angular L1 distance on circle (shortest arc), for phase-lock \(\omega\).
#[inline]
pub fn phase_delta(theta: f32, theta_prev: f32) -> f32 {
    let mut d = theta - theta_prev;
    // wrap to (-π, π]
    const PI: f32 = std::f32::consts::PI;
    const TWO_PI: f32 = 2.0 * PI;
    while d > PI {
        d -= TWO_PI;
    }
    while d <= -PI {
        d += TWO_PI;
    }
    d.abs()
}

/// Wave-function collapse (research §4):
/// \(h_{final} = Re \cdot (1 + \tanh(|Im|)) \cdot \cos(\theta)\).
///
/// Uses the identity `cos(atan2(im, re)) = re / sqrt(re² + im²)` to eliminate
/// the scalar `atan2` + `cos` calls (2 transcendentals), replacing them with
/// `sqrt` + `div` (2 non-transcendentals). Matches the AVX2 path
/// (`wave_collapse_batch_avx2`) which already uses this identity.
#[inline]
pub fn wave_collapse(h: ComplexF32) -> f32 {
    if !h.is_finite() {
        return 0.0;
    }
    let mag_mod = 1.0 + h.im.abs().tanh();
    let r = (h.re * h.re + h.im * h.im).sqrt().max(EPSILON_FLOOR);
    let v = h.re * h.re * mag_mod / r;
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

/// L-14 research register: complex accum + previous phase for \(\omega\).
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default)]
#[repr(C, align(16))]
pub struct ComplexSlimeRegister {
    pub state: ComplexF32,
    pub theta_prev: f32,
    pub _pad: f32,
}

#[cfg(test)]
impl ComplexSlimeRegister {
    #[inline]
    pub fn from_real(re: f32) -> Self {
        let s = ComplexF32::from_real(re);
        Self {
            state: s,
            theta_prev: s.phase(),
            _pad: 0.0,
        }
    }

    #[inline]
    pub fn write_complex(&mut self, h: ComplexF32) {
        self.state = h;
    }

    /// One local phase sample for omega (does not update EMA).
    #[inline]
    pub fn instantaneous_omega(&self) -> f32 {
        phase_delta(self.state.phase(), self.theta_prev)
    }

    #[inline]
    pub fn commit_phase(&mut self) {
        self.theta_prev = self.state.phase();
    }
}
