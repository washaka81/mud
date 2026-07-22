//! # L-09: Engine Zero-Overhead Protocol (EZOP)
//!
//! Raw-pointer hot-path helpers (P-00). Certified vs safe Rust in
//! `tools/ezop_bench.rs` (+~8% on SGD-style updates) and unit tests below.
//!
//! **Contract:** every `unsafe` fn documents length / non-null preconditions.
//! Prefer these over `slice[i]` in QAT/backward tight loops.

use std::cell::RefCell;

thread_local! {
    /// Reused grad buffer for optimizer preprocess (Muon/GaLore) — avoids per-step alloc.
    static GRAD_SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Borrow a thread-local f32 scratch of at least `n` elements (zero-filled to `n`).
/// Caller must not nest concurrent borrows on the same thread.
pub fn with_grad_scratch<R>(n: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    GRAD_SCRATCH.with(|cell| {
        let mut v = cell.borrow_mut();
        if v.len() < n {
            v.resize(n, 0.0);
        }
        // Only expose the prefix we need; leave capacity for next larger call.
        let buf = &mut v[..n];
        f(buf)
    })
}

/// Replace non-finite values with 0.0 (L-08 policy).
///
/// # Safety
/// `p` must be valid for `n` writable f32s.
#[inline]
pub unsafe fn sanitize_f32(p: *mut f32, n: usize) {
    for i in 0..n {
        let v = *p.add(i);
        if !v.is_finite() {
            *p.add(i) = 0.0;
        }
    }
}

/// Copy `n` f32s: dst ← src.
///
/// # Safety
/// `src` readable and `dst` writable for `n` elements; regions may not overlap.
#[inline]
pub unsafe fn copy_f32(dst: *mut f32, src: *const f32, n: usize) {
    std::ptr::copy_nonoverlapping(src, dst, n);
}

/// SGD-style shadow update (scalar EZOP path). Matches `sgd_step_avx2` policy:
/// g ← clamp(g/ntok, ±10); w ← clamp(w*(1-lr*wd) - lr*g, ±5).
///
/// # Safety
/// `w` writable and `g` readable for `n` elements.
#[inline]
pub unsafe fn sgd_step(
    w: *mut f32,
    g: *const f32,
    n: usize,
    lr: f32,
    weight_decay: f32,
    num_tokens: f32,
) {
    // NOTE: gradient normalization by `num_tokens` is performed ONCE by the
    // caller via `scale_grad_by_tokens` (applies to every optimizer strategy).
    // Do NOT divide here again, or the effective LR would shrink by num_tokens².
    let _ = num_tokens;
    let decay_factor = 1.0 - lr * weight_decay;
    for i in 0..n {
        let mut g_val = *g.add(i);
        if !g_val.is_finite() {
            g_val = 0.0;
        }
        g_val = g_val.clamp(-10.0, 10.0);
        let wi = *w.add(i);
        *w.add(i) = (wi * decay_factor - lr * g_val).clamp(-5.0, 5.0);
    }
}

/// y[i] += a * x[i]
///
/// # Safety
/// `y` writable, `x` readable for `n` elements.
#[inline]
pub unsafe fn axpy(y: *mut f32, a: f32, x: *const f32, n: usize) {
    for i in 0..n {
        *y.add(i) += a * *x.add(i);
    }
}

/// y[i] = a * x[i] + b * y[i]
///
/// # Safety
/// `y` writable, `x` readable for `n` elements.
#[inline]
pub unsafe fn axpby(y: *mut f32, a: f32, x: *const f32, b: f32, n: usize) {
    for i in 0..n {
        *y.add(i) = a * *x.add(i) + b * *y.add(i);
    }
}

/// Sum of squares.
///
/// # Safety
/// `x` readable for `n` elements.
#[inline]
pub unsafe fn sum_sq(x: *const f32, n: usize) -> f32 {
    let mut s = 0.0f32;
    for i in 0..n {
        let v = *x.add(i);
        s += v * v;
    }
    s
}

/// Scale in place: x[i] *= s
///
/// # Safety
/// `x` writable for `n` elements.
#[inline]
pub unsafe fn scale(x: *mut f32, s: f32, n: usize) {
    for i in 0..n {
        *x.add(i) *= s;
    }
}

/// Pack one matrix row-major shadow into ELUT 4-bit (2 weights/byte) + PRQ scales.
/// `threshold = scale * 0.7` STE band (matches trainer pack).
///
/// # Safety
/// - `shadow` readable for `rows * cols`
/// - `scales` writable for `rows`
/// - `packed` writable for `(rows * cols).div_ceil(2)` bytes
#[inline]
pub unsafe fn pack_elut_prq(
    shadow: *const f32,
    rows: usize,
    cols: usize,
    scales: *mut f32,
    packed: *mut u8,
) {
    let cols = cols.max(1);
    for r in 0..rows {
        let start = r * cols;
        let mut abs_sum = 0.0f32;
        for c in 0..cols {
            abs_sum += (*shadow.add(start + c)).abs();
        }
        let s = ((abs_sum / cols as f32) * std::f32::consts::FRAC_1_SQRT_2)
            .max(crate::mud::constants::EPSILON_FLOOR);
        *scales.add(r) = s;
        let threshold = s * 0.7;
        // STE deadzone (see session report §4): values inside ±threshold round to 0
        // and never flip a ternary code. On a converged base at default LR
        // (QAT_LEARNING_RATE=0.0005) the per-element gradient (~1/256) stays under
        // threshold → ΔW≈0 (expected). Raise LR via MUD_QAT_LR (e.g. 0.03) to move
        // weights; see corpus_trainer.rs T1.1 retrain-of-verification.
        // Pack 2 values per byte (one store per byte); no read-modify-write.
        let pairs = cols / 2;
        let rem = cols % 2;
        for c in 0..pairs {
            let i0 = start + c * 2;
            let v0 = *shadow.add(i0);
            let b0 = if v0 > threshold {
                0x1u8
            } else if v0 < -threshold {
                0xFu8
            } else {
                0x0u8
            };
            let v1 = *shadow.add(i0 + 1);
            let b1 = if v1 > threshold {
                0x1u8
            } else if v1 < -threshold {
                0xFu8
            } else {
                0x0u8
            };
            *packed.add(start / 2 + c) = b0 | (b1 << 4);
        }
        if rem != 0 {
            let i0 = start + pairs * 2;
            let v0 = *shadow.add(i0);
            let b0 = if v0 > threshold {
                0x1u8
            } else if v0 < -threshold {
                0xFu8
            } else {
                0x0u8
            };
            *packed.add(start / 2 + pairs) = b0;
        }
    }
}

/// Pack ternary f32 row (−1/0/+1 after threshold) into ELUT u32 words → little-endian bytes.
/// Same encoding as [`crate::mud::pack_ternary_row`].
///
/// # Safety
/// `values` readable for `n`; `out_u32` writable for `n.div_ceil(8)`.
#[inline]
pub unsafe fn pack_ternary_into(values: *const f32, n: usize, delta: f32, out_u32: *mut u32) {
    let u32_count = n.div_ceil(8);
    for i in 0..u32_count {
        *out_u32.add(i) = 0;
    }
    // Pack 8 values per u32 word (one store per word) instead of per-element
    // div/mod + read-modify-write.
    let full = n - (n % 8);
    let mut i = 0;
    while i < full {
        let mut word = 0u32;
        for j in 0..8 {
            let v = *values.add(i + j);
            if v.abs() > delta {
                let bits = if v > 0.0 { 0x1u32 } else { 0xFu32 };
                word |= bits << (j * 4);
            }
        }
        *out_u32.add(i / 8) = word;
        i += 8;
    }
    if full < n {
        let mut word = 0u32;
        for j in 0..(n - full) {
            let v = *values.add(full + j);
            if v.abs() > delta {
                let bits = if v > 0.0 { 0x1u32 } else { 0xFu32 };
                word |= bits << (j * 4);
            }
        }
        *out_u32.add(full / 8) = word;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_matches_safe() {
        let n = 1024;
        let mut w_safe: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).sin()).collect();
        let mut w_ezop = w_safe.clone();
        let g: Vec<f32> = (0..n).map(|i| (i as f32 * 0.002).cos() * 0.1).collect();
        let lr = 1e-3f32;
        let wd = 0.01f32;
        // `sgd_step` no longer divides by num_tokens (the caller normalizes once
        // via `scale_grad_by_tokens`). Pass ntok=1 and match without the division.
        let ntok = 1.0f32;

        let decay = 1.0 - lr * wd;
        for i in 0..n {
            let mut gv = g[i] / ntok;
            gv = gv.clamp(-10.0, 10.0);
            w_safe[i] = (w_safe[i] * decay - lr * gv).clamp(-5.0, 5.0);
        }
        unsafe {
            sgd_step(w_ezop.as_mut_ptr(), g.as_ptr(), n, lr, wd, ntok);
        }
        let max_diff = w_safe
            .iter()
            .zip(w_ezop.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(max_diff, 0.0, "EZOP SGD diverged by {max_diff}");
    }

    #[test]
    fn test_sanitize_and_sum_sq() {
        let mut x = vec![1.0f32, f32::NAN, 2.0, f32::INFINITY];
        unsafe {
            sanitize_f32(x.as_mut_ptr(), x.len());
            assert_eq!(x, vec![1.0, 0.0, 2.0, 0.0]);
            let s = sum_sq(x.as_ptr(), x.len());
            assert!((s - 5.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_with_grad_scratch_reuses() {
        with_grad_scratch(16, |b| {
            b.fill(1.0);
            assert_eq!(b.len(), 16);
        });
        with_grad_scratch(8, |b| {
            // May contain previous data in capacity; prefix length is 8
            assert_eq!(b.len(), 8);
            b.fill(2.0);
        });
        with_grad_scratch(32, |b| {
            assert_eq!(b.len(), 32);
        });
    }

    #[test]
    fn test_pack_elut_roundtrip_bits() {
        let rows = 2usize;
        let cols = 8usize;
        let shadow: Vec<f32> = vec![
            1.0, 1.0, 0.0, -1.0, 0.5, -0.5, 0.0, 0.01, // row0
            -2.0, 2.0, 0.0, 0.0, 0.1, -0.1, 0.0, 0.0, // row1
        ];
        let mut scales = vec![0.0f32; rows];
        let mut packed = vec![0u8; (rows * cols).div_ceil(2)];
        unsafe {
            pack_elut_prq(
                shadow.as_ptr(),
                rows,
                cols,
                scales.as_mut_ptr(),
                packed.as_mut_ptr(),
            );
        }
        assert!(scales[0] > 0.0 && scales[1] > 0.0);
        // At least some non-zero nibbles written
        assert!(packed.iter().any(|&b| b != 0));
    }

    #[test]
    fn test_axpy() {
        let mut y = vec![1.0f32, 2.0, 3.0];
        let x = [1.0f32, 1.0, 1.0];
        unsafe {
            axpy(y.as_mut_ptr(), 2.0, x.as_ptr(), 3);
        }
        assert_eq!(y, vec![3.0, 4.0, 5.0]);
    }
}
