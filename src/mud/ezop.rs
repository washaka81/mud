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

/// AVX2 axpy: y[i] += a * x[i] (8 f32/iter via FMA).
///
/// # Safety
/// `y` writable, `x` readable for `n` elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn axpy_avx2(y: *mut f32, a: f32, x: *const f32, n: usize) {
    use std::arch::x86_64::*;
    let av = _mm256_set1_ps(a);
    let mut i = 0;
    while i + 8 <= n {
        let yv = _mm256_loadu_ps(y.add(i));
        let xv = _mm256_loadu_ps(x.add(i));
        let result = _mm256_fmadd_ps(av, xv, yv);
        _mm256_storeu_ps(y.add(i), result);
        i += 8;
    }
    for j in i..n {
        *y.add(j) += a * *x.add(j);
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub unsafe fn axpy_avx2(y: *mut f32, a: f32, x: *const f32, n: usize) {
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

/// Batched AXPY for 8 rows: gw[rr*n + j] += g_y[rr] * x[j] for rr in 0..8.
///
/// Optimisation OPT-4: loads each 8-element chunk of `x` **once** and reuses
/// it across all 8 rows (FMA broadcast), instead of calling `axpy_avx2`
/// 8 times per chunk. This saves 7 redundant loads of `x` per 8-element
/// group and reduces loop overhead.
///
/// # Safety
/// - `g_y` readable for 8 elements.
/// - `x` readable for `n` elements.
/// - `gw` is 8 contiguous rows of `n` elements each (row-major), writable.
#[cfg(target_arch = "x86_64")]
#[inline]
pub unsafe fn axpy_8rows_avx2(g_y: *const f32, x: *const f32, gw: *mut f32, n: usize) {
    use std::arch::x86_64::*;
    // Load all 8 g_y scalars
    let gy_arr = [
        *g_y.add(0), *g_y.add(1), *g_y.add(2), *g_y.add(3),
        *g_y.add(4), *g_y.add(5), *g_y.add(6), *g_y.add(7),
    ];
    let mut j = 0;
    while j + 8 <= n {
        let x_vec = _mm256_loadu_ps(x.add(j));
        for (rr, &gy_val) in gy_arr.iter().enumerate() {
            let gy_vec = _mm256_set1_ps(gy_val);
            let gw_vec = _mm256_loadu_ps(gw.add(rr * n + j));
            let result = _mm256_fmadd_ps(gy_vec, x_vec, gw_vec);
            _mm256_storeu_ps(gw.add(rr * n + j), result);
        }
        j += 8;
    }
    // Scalar remainder
    for (rr, &gy_val) in gy_arr.iter().enumerate() {
        for k in j..n {
            *gw.add(rr * n + k) += gy_val * *x.add(k);
        }
    }
}

/// Scalar fallback for axpy_8rows (non-x86_64 or n < 8).
///
/// # Safety
/// - `g_y` readable for 8 elements.
/// - `x` readable for `n` elements.
/// - `gw` is 8 contiguous rows of `n` elements each (row-major), writable.
#[inline]
pub unsafe fn axpy_8rows(g_y: *const f32, x: *const f32, gw: *mut f32, n: usize) {
    for rr in 0..8 {
        let gy_val = *g_y.add(rr);
        for k in 0..n {
            *gw.add(rr * n + k) += gy_val * *x.add(k);
        }
    }
}

/// Sum of squares.
///
/// # Safety
/// `x` readable for `n` elements.
#[inline]
pub unsafe fn sum_sq(x: *const f32, n: usize) -> f32 {
    // AVX2: 8 f32/iter via vfmadd231ps (acc += v*v, 1 cycle throughput)
    // OPT: 4 independent accumulators (32-element unroll) breaks the serial
    // dependency chain — 4× FP throughput on i7-1260P (5-cycle FMA latency →
    // 6.4 f32/cycle vs 1.6 f32/cycle with single accumulator).
    #[cfg_attr(target_arch = "x86_64", allow(clippy::needless_return))]
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        let mut i = 0;
        let mut s = 0.0f32;
        if n >= 32 {
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();
            while i + 32 <= n {
                let v0 = _mm256_loadu_ps(x.add(i));
                let v1 = _mm256_loadu_ps(x.add(i + 8));
                let v2 = _mm256_loadu_ps(x.add(i + 16));
                let v3 = _mm256_loadu_ps(x.add(i + 24));
                acc0 = _mm256_fmadd_ps(v0, v0, acc0);
                acc1 = _mm256_fmadd_ps(v1, v1, acc1);
                acc2 = _mm256_fmadd_ps(v2, v2, acc2);
                acc3 = _mm256_fmadd_ps(v3, v3, acc3);
                i += 32;
            }
            // Merge 4 accumulators
            let acc01 = _mm256_add_ps(acc0, acc1);
            let acc23 = _mm256_add_ps(acc2, acc3);
            let acc = _mm256_add_ps(acc01, acc23);
            // Horizontal sum
            let mut tmp = [0.0f32; 8];
            _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
            s = tmp.iter().sum();
        }
        // Handle 8-element chunks that didn't fit in 32-unroll
        while i + 8 <= n {
            let v = _mm256_loadu_ps(x.add(i));
            let sq = _mm256_mul_ps(v, v);
            let mut tmp = [0.0f32; 8];
            _mm256_storeu_ps(tmp.as_mut_ptr(), sq);
            s += tmp.iter().sum::<f32>();
            i += 8;
        }
        for j in i..n {
            let v = *x.add(j);
            s += v * v;
        }
        return s;
    }
    // Fallback (non-x86_64)
    #[cfg(not(target_arch = "x86_64"))]
    {
        let mut s = 0.0f32;
        for i in 0..n {
            let v = *x.add(i);
            s += v * v;
        }
        s
    }
}

/// AVX2 sum-of-abs (8 f32/iter). Returns sum(|x[i]|) for i in 0..n.
/// Uses 4 independent accumulators (32-element unroll) for throughput.
///
/// # Safety
/// `x` readable for `n` elements.
#[cfg(target_arch = "x86_64")]
#[inline]
pub unsafe fn sum_abs_avx2(x: *const f32, n: usize) -> f32 {
    use std::arch::x86_64::*;
    let sign_mask = _mm256_set1_ps(-0.0);
    let mut i = 0;
    let mut s = 0.0f32;
    if n >= 32 {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        while i + 32 <= n {
            let v0 = _mm256_loadu_ps(x.add(i));
            let v1 = _mm256_loadu_ps(x.add(i + 8));
            let v2 = _mm256_loadu_ps(x.add(i + 16));
            let v3 = _mm256_loadu_ps(x.add(i + 24));
            // abs(v) = v & ~sign
            let a0 = _mm256_andnot_ps(sign_mask, v0);
            let a1 = _mm256_andnot_ps(sign_mask, v1);
            let a2 = _mm256_andnot_ps(sign_mask, v2);
            let a3 = _mm256_andnot_ps(sign_mask, v3);
            acc0 = _mm256_add_ps(acc0, a0);
            acc1 = _mm256_add_ps(acc1, a1);
            acc2 = _mm256_add_ps(acc2, a2);
            acc3 = _mm256_add_ps(acc3, a3);
            i += 32;
        }
        let acc01 = _mm256_add_ps(acc0, acc1);
        let acc23 = _mm256_add_ps(acc2, acc3);
        let acc = _mm256_add_ps(acc01, acc23);
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
        s = tmp.iter().sum();
    }
    while i + 8 <= n {
        let v = _mm256_loadu_ps(x.add(i));
        let abs_v = _mm256_andnot_ps(sign_mask, v);
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), abs_v);
        s += tmp.iter().sum::<f32>();
        i += 8;
    }
    for j in i..n {
        s += (*x.add(j)).abs();
    }
    s
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub unsafe fn sum_abs_avx2(x: *const f32, n: usize) -> f32 {
    let mut s = 0.0f32;
    for i in 0..n {
        s += (*x.add(i)).abs();
    }
    s
}

/// AVX2 peak-finding: returns max(|x[i]|) for i in 0..n (8 f32/iter).
///
/// # Safety
/// `x` readable for `n` elements.
#[cfg(target_arch = "x86_64")]
#[inline]
pub unsafe fn peak_abs_avx2(x: *const f32, n: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut peak = 0.0f32;
    let mut i = 0;
    if n >= 8 {
        let mut max_vec = _mm256_setzero_ps();
        while i + 8 <= n {
            let v = _mm256_loadu_ps(x.add(i));
            let abs_v = _mm256_andnot_ps(_mm256_set1_ps(-0.0f32), v);
            max_vec = _mm256_max_ps(max_vec, abs_v);
            i += 8;
        }
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), max_vec);
        peak = tmp.iter().copied().fold(0.0f32, f32::max);
    }
    for j in i..n {
        peak = peak.max((*x.add(j)).abs());
    }
    peak
}

/// AVX2 ReLU² activation backward: branchless (g > 0) mask.
/// grad_up[i] = gy[i] * g[i]², grad_gate[i] = gy[i] * 2*g[i]*u[i] (if g > 0, else 0).
///
/// # Safety
/// All pointers valid for `n` elements. `grad_up` and `grad_gate` must not alias.
#[cfg(target_arch = "x86_64")]
#[inline]
pub unsafe fn relu2_backward_avx2(
    grad_y: *const f32,
    gate: *const f32,
    up: *const f32,
    grad_up: *mut f32,
    grad_gate: *mut f32,
    n: usize,
) {
    use std::arch::x86_64::*;
    let two = _mm256_set1_ps(2.0);
    let zero = _mm256_setzero_ps();
    let mut i = 0;
    if n >= 8 {
        while i + 8 <= n {
            let gy = _mm256_loadu_ps(grad_y.add(i));
            let g = _mm256_loadu_ps(gate.add(i));
            let u = _mm256_loadu_ps(up.add(i));
            // mask: 0xFFFFFFFF if g > 0, else 0 (branchless)
            let mask = _mm256_cmp_ps(g, zero, _CMP_GT_OS);
            let g_sq = _mm256_mul_ps(g, g);
            // two_gu = 2.0 * g * u (using vbroadcastss + vmulps)
            let two_g = _mm256_mul_ps(two, g);
            let two_gu = _mm256_mul_ps(two_g, u);
            // grad_up = gy * g² (masked)
            let grad_up_vec = _mm256_and_ps(mask, _mm256_mul_ps(gy, g_sq));
            // grad_gate = gy * 2*g*u (masked)
            let grad_gate_vec = _mm256_and_ps(mask, _mm256_mul_ps(gy, two_gu));
            _mm256_storeu_ps(grad_up.add(i), grad_up_vec);
            _mm256_storeu_ps(grad_gate.add(i), grad_gate_vec);
            i += 8;
        }
    }
    for j in i..n {
        let gy = *grad_y.add(j);
        let g = *gate.add(j);
        let u = *up.add(j);
        if g > 0.0 {
            *grad_up.add(j) = gy * (g * g);
            *grad_gate.add(j) = gy * (2.0 * g * u);
        } else {
            *grad_up.add(j) = 0.0;
            *grad_gate.add(j) = 0.0;
        }
    }
}

/// AVX2 i8→f32 scale reconstruction: out[i] = (i8[i] as f32) * scale (8/iter).
///
/// # Safety
/// `i8_ptr` readable for `n` elements, `out` writable for `n` elements.
#[cfg(target_arch = "x86_64")]
#[inline]
pub unsafe fn scale_i8_to_f32_avx2(i8_ptr: *const i8, out: *mut f32, scale: f32, n: usize) {
    use std::arch::x86_64::*;
    let scale_vec = _mm256_set1_ps(scale);
    let mut i = 0;
    if n >= 8 {
        while i + 8 <= n {
            // vpmovsxbd: load 8 i8 → sign-extend to 8 i32 in one instruction
            let i8_vec = _mm_loadu_si128(i8_ptr.add(i) as *const __m128i);
            let i32_vec = _mm256_cvtepi8_epi32(i8_vec);
            // vcvtpi256ps: i32 → f32
            let f32_vec = _mm256_cvtepi32_ps(i32_vec);
            let result = _mm256_mul_ps(f32_vec, scale_vec);
            _mm256_storeu_ps(out.add(i), result);
            i += 8;
        }
    }
    for j in i..n {
        *out.add(j) = *i8_ptr.add(j) as f32 * scale;
    }
}

/// Scale in place: x[i] *= s (scalar fallback).
///
/// # Safety
/// `x` writable for `n` elements.
#[inline]
pub unsafe fn scale(x: *mut f32, s: f32, n: usize) {
    for i in 0..n {
        *x.add(i) *= s;
    }
}

/// AVX2 element-wise scale: x[i] *= s (8 f32/iter). Broadcasts scalar `s`.
///
/// # Safety
/// `x` writable for `n` elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn scale_vec_avx2(x: *mut f32, s: f32, n: usize) {
    use std::arch::x86_64::*;
    let sv = _mm256_set1_ps(s);
    let mut i = 0;
    while i + 8 <= n {
        let xv = _mm256_loadu_ps(x.add(i));
        let result = _mm256_mul_ps(xv, sv);
        _mm256_storeu_ps(x.add(i), result);
        i += 8;
    }
    // Scalar remainder
    for j in i..n {
        *x.add(j) *= s;
    }
}

/// AVX2 8-row gradient reduction: grad_x[i] += sum_j(scratch[j*n + i]) for j in 0..8.
/// Each 8-element chunk loads from all 8 rows and reduces with 4 adders (ILP).
/// Critical for backward GEMV where 8 output rows' grad_x must be merged into 1.
///
/// # Safety
/// `grad_x` writable for `n` elements. Each of the 8 `scratch` rows readable for `n` elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn grad_x_reduce_8rows_avx2(grad_x: *mut f32, scratch: *const f32, n: usize) {
    use std::arch::x86_64::*;
    let mut i = 0;
    while i + 8 <= n {
        // Load first two rows into independent accumulators (ILP via 2-way unroll)
        let s0 = _mm256_loadu_ps(scratch.add(i));
        let s1 = _mm256_loadu_ps(scratch.add(i).add(n));
        let s2 = _mm256_loadu_ps(scratch.add(i).add(2 * n));
        let s3 = _mm256_loadu_ps(scratch.add(i).add(3 * n));
        let s4 = _mm256_loadu_ps(scratch.add(i).add(4 * n));
        let s5 = _mm256_loadu_ps(scratch.add(i).add(5 * n));
        let s6 = _mm256_loadu_ps(scratch.add(i).add(6 * n));
        let s7 = _mm256_loadu_ps(scratch.add(i).add(7 * n));

        // Tree reduction: 8 → 4 → 2 → 1 (3 independent add chains, ILP)
        let acc0 = _mm256_add_ps(s0, s1);
        let acc1 = _mm256_add_ps(s2, s3);
        let acc2 = _mm256_add_ps(s4, s5);
        let acc3 = _mm256_add_ps(s6, s7);
        let sum0123 = _mm256_add_ps(acc0, acc1);
        let sum4567 = _mm256_add_ps(acc2, acc3);
        let total = _mm256_add_ps(sum0123, sum4567);

        // Read-modify-write grad_x (accumulate)
        let gx = _mm256_loadu_ps(grad_x.add(i));
        let result = _mm256_add_ps(gx, total);
        _mm256_storeu_ps(grad_x.add(i), result);

        i += 8;
    }
    // Scalar remainder
    for j in i..n {
        let mut sum = 0.0f32;
        for row in 0..8 {
            sum += *scratch.add(row * n + j);
        }
        *grad_x.add(j) += sum;
    }
}

/// AVX2 element-wise divide with NaN/Inf guard: out[i] = a[i] / b[i],
/// non-finite results → `default` (P-17). 8 f32/iter via FMA.
///
/// # Safety
/// `a` and `b` readable for `n` elements, `out` writable for `n` elements.
/// `a` and `out` may alias.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn div_avx2(a: *const f32, b: *const f32, out: *mut f32, default: f32, n: usize) {
    use std::arch::x86_64::*;
    let sign_mask = _mm256_set1_ps(-0.0);
    let inf = _mm256_set1_ps(f32::INFINITY);
    let default_v = _mm256_set1_ps(default);

    let mut i = 0;
    while i + 8 <= n {
        let av = _mm256_loadu_ps(a.add(i));
        let bv = _mm256_loadu_ps(b.add(i));
        // Safe divide: where |b| < 1e-8, result = 0 (avoids div-by-zero)
        let abs_b = _mm256_andnot_ps(sign_mask, bv);
        let valid_b = _mm256_cmp_ps(abs_b, _mm256_set1_ps(1e-8), _CMP_GE_OQ);
        let safe_b = _mm256_blendv_ps(_mm256_set1_ps(1.0), bv, valid_b);
        let result = _mm256_div_ps(av, safe_b);
        // NaN/Inf guard: |result| < INF → finite
        let abs_result = _mm256_andnot_ps(sign_mask, result);
        let is_finite = _mm256_cmp_ps(abs_result, inf, _CMP_LT_OQ);
        let guarded = _mm256_blendv_ps(default_v, result, is_finite);
        // Zero where b was too small
        let final_val = _mm256_blendv_ps(default_v, guarded, valid_b);
        _mm256_storeu_ps(out.add(i), final_val);
        i += 8;
    }
    for j in i..n {
        let b = *b.add(j);
        let val = if b.abs() > 1e-8 { *a.add(j) / b } else { default };
        *out.add(j) = if val.is_finite() { val } else { default };
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub unsafe fn div_avx2(a: *const f32, b: *const f32, out: *mut f32, default: f32, n: usize) {
    for i in 0..n {
        let b = *b.add(i);
        let val = if b.abs() > 1e-8 { *a.add(i) / b } else { default };
        *out.add(i) = if val.is_finite() { val } else { default };
    }
}

/// AVX2 scale with NaN/Inf guard: x[i] *= s, non-finite results → 0.
/// Single AVX2 pass (8 f32/iter) with blendv for the guard.
///
/// # Safety
/// `x` writable for `n` elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn scale_finite_avx2(x: *mut f32, s: f32, n: usize) {
    use std::arch::x86_64::*;
    let sv = _mm256_set1_ps(s);
    let zero = _mm256_setzero_ps();
    let sign_mask = _mm256_set1_ps(-0.0);
    let inf = _mm256_set1_ps(f32::INFINITY);
    let mut i = 0;
    while i + 8 <= n {
        let xv = _mm256_loadu_ps(x.add(i));
        let result = _mm256_mul_ps(xv, sv);
        // NaN/Inf guard: |result| < INF → finite
        let abs_result = _mm256_andnot_ps(sign_mask, result);
        let is_finite = _mm256_cmp_ps(abs_result, inf, _CMP_LT_OQ);
        let guarded = _mm256_blendv_ps(zero, result, is_finite);
        _mm256_storeu_ps(x.add(i), guarded);
        i += 8;
    }
    // Scalar remainder
    for j in i..n {
        let val = *x.add(j) * s;
        *x.add(j) = if val.is_finite() { val } else { 0.0 };
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub unsafe fn scale_finite_avx2(x: *mut f32, s: f32, n: usize) {
    for i in 0..n {
        let val = *x.add(i) * s;
        *x.add(i) = if val.is_finite() { val } else { 0.0 };
    }
}

/// AVX2 clamp + NaN guard: x[i] = if x[i].is_finite() { clamp(x[i], lo, hi) } else { default }.
/// Single AVX2 pass (8 f32/iter) with blendv for the NaN guard.
///
/// # Safety
/// `x` writable for `n` elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn clamp_finite_avx2(x: *mut f32, lo: f32, hi: f32, default: f32, n: usize) {
    use std::arch::x86_64::*;
    let lo_v = _mm256_set1_ps(lo);
    let hi_v = _mm256_set1_ps(hi);
    let default_v = _mm256_set1_ps(default);
    let sign_mask = _mm256_set1_ps(-0.0);
    let inf = _mm256_set1_ps(f32::INFINITY);
    let mut i = 0;
    while i + 8 <= n {
        let xv = _mm256_loadu_ps(x.add(i));
        // NaN/Inf guard: |xv| < INF → finite
        let abs_xv = _mm256_andnot_ps(sign_mask, xv);
        let is_finite = _mm256_cmp_ps(abs_xv, inf, _CMP_LT_OQ);
        // Clamp to [lo, hi]
        let clamped = _mm256_min_ps(_mm256_max_ps(xv, lo_v), hi_v);
        // If not finite → default, else → clamped
        let result = _mm256_blendv_ps(default_v, clamped, is_finite);
        _mm256_storeu_ps(x.add(i), result);
        i += 8;
    }
    // Scalar remainder
    for j in i..n {
        let v = *x.add(j);
        *x.add(j) = if v.is_finite() { v.clamp(lo, hi) } else { default };
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub unsafe fn clamp_finite_avx2(x: *mut f32, lo: f32, hi: f32, default: f32, n: usize) {
    for i in 0..n {
        let v = *x.add(i);
        *x.add(i) = if v.is_finite() { v.clamp(lo, hi) } else { default };
    }
}

/// AVX2 element-wise scale with per-element weight: x[i] *= w[i] * s.
/// Loads both x and w as vectors, multiplies x*w*s, stores back to x.
///
/// # Safety
/// `x` writable for `n` elements. `w` readable for `n` elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn scale_with_weight_avx2(x: *mut f32, w: *const f32, s: f32, n: usize) {
    use std::arch::x86_64::*;
    let sv = _mm256_set1_ps(s);
    let mut i = 0;
    while i + 8 <= n {
        let xv = _mm256_loadu_ps(x.add(i));
        let wv = _mm256_loadu_ps(w.add(i));
        let result = _mm256_mul_ps(xv, _mm256_mul_ps(wv, sv));
        _mm256_storeu_ps(x.add(i), result);
        i += 8;
    }
    // Scalar remainder
    for j in i..n {
        *x.add(j) *= *w.add(j) * s;
    }
}

/// AVX2 peak abs for i8 array (32 bytes/iter). Uses _mm256_max_epi8.
///
/// # Safety
/// `x` readable for `n` elements.
#[cfg(target_arch = "x86_64")]
#[inline]
pub unsafe fn peak_abs_i8_avx2(x: *const i8, n: usize) -> i8 {
    use std::arch::x86_64::*;
    let mut max_v: i8 = 0;
    let mut i = 0;
    if n >= 32 {
        let zero = _mm256_setzero_si256();
        let mut max_vec = zero;
        while i + 32 <= n {
            let v = _mm256_loadu_si256(x.add(i) as *const __m256i);
            // Compute abs: (x ^ mask) - mask, where mask = x >> 7 (sign bits)
            let mask = _mm256_cmpgt_epi8(zero, v); // 0xFF where x < 0
            let abs_v = _mm256_sub_epi8(_mm256_xor_si256(v, mask), mask);
            max_vec = _mm256_max_epi8(max_vec, abs_v);
            i += 32;
        }
        // Horizontal reduction: store to array and fold
        let mut tmp = [0i8; 32];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, max_vec);
        for &v in &tmp {
            if v > max_v { max_v = v; }
        }
    }
    // Scalar remainder
    for j in i..n {
        let v = (*x.add(j)).abs();
        if v > max_v { max_v = v; }
    }
    max_v
}

/// AVX2 f32→i8 quantization with NaN/Inf guard (8 lanes/iter).
/// For each i: out[i] = clamp(round(x[i] / scale), -127, 127) as i8.
/// Non-finite results → 0. All NaN/Inf checks before `_mm256_cvtps_epi32` to prevent UB.
///
/// # Safety
/// `x` readable for `n` elements. `out` writable for `n` elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn quantize_i8_avx2(x: *const f32, out: *mut i8, scale: f32, n: usize) {
    use std::arch::x86_64::*;
    let inv_scale = _mm256_set1_ps(1.0 / scale);
    let neg_127 = _mm256_set1_ps(-127.0);
    let pos_127 = _mm256_set1_ps(127.0);
    let inf = _mm256_set1_ps(f32::INFINITY);
    let sign_mask = _mm256_set1_ps(-0.0);
    let mut i = 0;
    while i + 8 <= n {
        let xv = _mm256_loadu_ps(x.add(i));
        // NaN/Inf guard
        let abs_xv = _mm256_andnot_ps(sign_mask, xv);
        let is_finite = _mm256_cmp_ps(abs_xv, inf, _CMP_LT_OQ);
        // Scale → clamp → round → convert to i32 (8 lanes in one pass)
        let scaled = _mm256_mul_ps(xv, inv_scale);
        let clamped = _mm256_min_ps(_mm256_max_ps(scaled, neg_127), pos_127);
        let rounded = _mm256_round_ps::<0>(clamped);
        let ints = _mm256_cvtps_epi32(rounded);
        // Blend: non-finite → 0, finite → int value
        let zero = _mm256_setzero_ps();
        let safe = _mm256_blendv_ps(zero, _mm256_castsi256_ps(ints), is_finite);
        // Extract 8 × i32, store as i8 via temp (packs interleave, temp is simpler)
        let mut tmp = [0i32; 8];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, _mm256_castps_si256(safe));
        for (j, &val) in tmp.iter().enumerate() {
            *out.add(i + j) = val.clamp(-127, 127) as i8;
        }
        i += 8;
    }
    // Scalar remainder
    for j in i..n {
        let v = *x.add(j) / scale;
        *out.add(j) = if v.is_finite() {
            v.round().clamp(-127.0, 127.0) as i8
        } else {
            0
        };
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub unsafe fn quantize_i8_avx2(x: *const f32, out: *mut i8, scale: f32, n: usize) {
    for j in 0..n {
        let v = *x.add(j) / scale;
        *out.add(j) = if v.is_finite() {
            v.round().clamp(-127.0, 127.0) as i8
        } else {
            0
        };
    }
}

/// AVX2 RoPE rotation: complex multiply rotation of interleaved pairs.
/// For i in 0..half_d: x0[i] = x0[i]*cos[i] - x1[i]*sin[i], x1[i] = x1[i]*cos[i] + x0[i]*sin[i].
/// Processes 8 complex pairs per iteration (x0_vec = 8 f32, x1_vec = 8 f32).
///
/// # Safety
/// - `x0` and `x1` writable for `half_d` elements each.
/// - `cos_tbl` and `sin_tbl` readable for `half_d` elements each.
/// - `x0` and `x1` must not overlap (in-place rotation within same buffer requires care).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn rope_rotate_avx2(
    x0: *mut f32, x1: *mut f32,
    cos_tbl: *const f32, sin_tbl: *const f32,
    half_d: usize,
) {
    use std::arch::x86_64::*;
    let mut i = 0;
    while i + 8 <= half_d {
        let x0v = _mm256_loadu_ps(x0.add(i));
        let x1v = _mm256_loadu_ps(x1.add(i));
        let cosv = _mm256_loadu_ps(cos_tbl.add(i));
        let sinv = _mm256_loadu_ps(sin_tbl.add(i));
        // x0' = x0*cos - x1*sin  (FMA: fmsub)
        // x1' = x1*cos + x0*sin  (FMA: fma)
        let x0_new = _mm256_fmsub_ps(x0v, cosv, _mm256_mul_ps(x1v, sinv));
        let x1_new = _mm256_fmadd_ps(x1v, cosv, _mm256_mul_ps(x0v, sinv));
        _mm256_storeu_ps(x0.add(i), x0_new);
        _mm256_storeu_ps(x1.add(i), x1_new);
        i += 8;
    }
    // Scalar remainder
    for j in i..half_d {
        let x0v = *x0.add(j);
        let x1v = *x1.add(j);
        let cosv = *cos_tbl.add(j);
        let sinv = *sin_tbl.add(j);
        let new_x0 = x0v * cosv - x1v * sinv;
        let new_x1 = x1v * cosv + x0v * sinv;
        *x0.add(j) = new_x0;
        *x1.add(j) = new_x1;
    }
}

/// Scalar fallback for rope_rotate (non-x86_64 or n < 8).
#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub unsafe fn rope_rotate_avx2(
    x0: *mut f32, x1: *mut f32,
    cos_tbl: *const f32, sin_tbl: *const f32,
    half_d: usize,
) {
    for j in 0..half_d {
        let x0v = *x0.add(j);
        let x1v = *x1.add(j);
        let cosv = *cos_tbl.add(j);
        let sinv = *sin_tbl.add(j);
        let new_x0 = x0v * cosv - x1v * sinv;
        let new_x1 = x1v * cosv + x0v * sinv;
        *x0.add(j) = new_x0;
        *x1.add(j) = new_x1;
    }
}

/// Element-wise multiply: `out[i] = a[i] * b[i]` (AVX2, 8 f32/iter).
/// Falls back to scalar for remainder.
///
/// # Safety
/// `a` and `b` readable, `out` writable for `n` elements. `out` must not alias `a` or `b`.
#[inline]
pub unsafe fn vmul(a: *const f32, b: *const f32, out: *mut f32, n: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        let mut i = 0;
        while i + 8 <= n {
            let va = _mm256_loadu_ps(a.add(i));
            let vb = _mm256_loadu_ps(b.add(i));
            let vr = _mm256_mul_ps(va, vb);
            _mm256_storeu_ps(out.add(i), vr);
            i += 8;
        }
        while i < n {
            *out.add(i) = *a.add(i) * *b.add(i);
            i += 1;
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        for i in 0..n {
            *out.add(i) = *a.add(i) * *b.add(i);
        }
    }
}

/// AVX2 exp2 approximation: out[i] = 2^x[i] (8 f32/iter via polynomial).
/// Uses the same algorithm as `exp2_approx_local`.
///
/// # Safety
/// `x` readable and `out` writable for `n` elements. Regions may not overlap.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn exp2_avx2(x: *const f32, out: *mut f32, n: usize) {
    use std::arch::x86_64::*;
    let mut i = 0;
    while i + 8 <= n {
        let xv = _mm256_loadu_ps(x.add(i));
        let result = exp2_approx_local(xv);
        _mm256_storeu_ps(out.add(i), result);
        i += 8;
    }
    for j in i..n {
        *out.add(j) = 2.0f32.powf(*x.add(j));
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub unsafe fn exp2_avx2(x: *const f32, out: *mut f32, n: usize) {
    for i in 0..n {
        *out.add(i) = 2.0f32.powf(*x.add(i));
    }
}

/// AVX2 exp: out[i] = e^x[i] = 2^(x * log2(e)) (8 f32/iter).
///
/// # Safety
/// `x` readable and `out` writable for `n` elements. Regions may not overlap.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn exp_avx2(x: *const f32, out: *mut f32, n: usize) {
    use std::arch::x86_64::*;
    let log2e_v = _mm256_set1_ps(std::f32::consts::LOG2_E);
    let zero = _mm256_setzero_ps();
    let sign_mask = _mm256_set1_ps(-0.0);
    let inf = _mm256_set1_ps(f32::INFINITY);
    let mut i = 0;
    while i + 8 <= n {
        let xv = _mm256_loadu_ps(x.add(i));
        let y = _mm256_mul_ps(xv, log2e_v);
        let result = exp2_approx_local(y);
        // P-17: NaN/Inf guard — non-finite exp results → 0 (not propagated through softmax)
        let abs_result = _mm256_andnot_ps(sign_mask, result);
        let is_finite = _mm256_cmp_ps(abs_result, inf, _CMP_LT_OQ);
        let guarded = _mm256_blendv_ps(zero, result, is_finite);
        _mm256_storeu_ps(out.add(i), guarded);
        i += 8;
    }
    for j in i..n {
        let val = (*x.add(j)).exp();
        *out.add(j) = if val.is_finite() { val } else { 0.0 };
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub unsafe fn exp_avx2(x: *const f32, out: *mut f32, n: usize) {
    for i in 0..n {
        *out.add(i) = (*x.add(i)).exp();
    }
}

/// AVX2 horizontal max: returns max(x[i]) for n elements (8 f32/iter).
///
/// # Safety
/// `x` readable for `n` elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn max_avx2(x: *const f32, n: usize) -> f32 {
    use std::arch::x86_64::*;
    if n == 0 { return f32::NEG_INFINITY; }
    let mut max_v = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut i = 0;
    while i + 8 <= n {
        let xv = _mm256_loadu_ps(x.add(i));
        max_v = _mm256_max_ps(max_v, xv);
        i += 8;
    }
    // Horizontal max of 8 lanes → scalar
    let hi = _mm256_extractf128_ps(max_v, 1);
    let lo = _mm256_castps256_ps128(max_v);
    let m128 = _mm_max_ps(lo, hi);
    let mut tmp = [0.0f32; 4];
    _mm_storeu_ps(tmp.as_mut_ptr(), m128);
    let mut result = tmp[0].max(tmp[1]).max(tmp[2]).max(tmp[3]);
    // Scalar remainder
    for j in i..n {
        let v = *x.add(j);
        if v > result { result = v; }
    }
    std::arch::x86_64::_mm256_zeroupper();
    result
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub unsafe fn max_avx2(x: *const f32, n: usize) -> f32 {
    if n == 0 { return f32::NEG_INFINITY; }
    let mut result = f32::NEG_INFINITY;
    for i in 0..n {
        let v = *x.add(i);
        if v > result { result = v; }
    }
    result
}

/// AVX2 in-place subtract broadcast scalar: x[i] -= val (8 f32/iter).
///
/// # Safety
/// `x` writable for `n` elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn broadcast_sub_avx2(x: *mut f32, val: f32, n: usize) {
    use std::arch::x86_64::*;
    let val_v = _mm256_set1_ps(val);
    let mut i = 0;
    while i + 8 <= n {
        let xv = _mm256_loadu_ps(x.add(i));
        let result = _mm256_sub_ps(xv, val_v);
        _mm256_storeu_ps(x.add(i), result);
        i += 8;
    }
    for j in i..n {
        *x.add(j) -= val;
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub unsafe fn broadcast_sub_avx2(x: *mut f32, val: f32, n: usize) {
    for i in 0..n {
        *x.add(i) -= val;
    }
}

/// AVX2 horizontal sum: returns sum(x[i]) for n elements (8 f32/iter).
///
/// # Safety
/// `x` readable for `n` elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn sum_avx2(x: *const f32, n: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let xv = _mm256_loadu_ps(x.add(i));
        acc = _mm256_add_ps(acc, xv);
        i += 8;
    }
    // Horizontal sum of 8 accumulators → scalar
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let s128 = _mm_add_ps(lo, hi);
    let mut tmp = [0.0f32; 4];
    _mm_storeu_ps(tmp.as_mut_ptr(), s128);
    let mut total = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    // Scalar remainder
    for j in i..n {
        total += *x.add(j);
    }
    std::arch::x86_64::_mm256_zeroupper();
    total
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub unsafe fn sum_avx2(x: *const f32, n: usize) -> f32 {
    let mut total = 0.0f32;
    for i in 0..n {
        total += *x.add(i);
    }
    total
}

/// Element-wise multiply in-place: b[i] *= a[i]  (AVX2, 8 f32/iter).
/// `a` and `b` must not alias.
///
/// # Safety
/// - `a` must be readable for `n` elements.
/// - `b` must be writable for `n` elements.
/// - `a` and `b` must not alias (point to overlapping memory).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn vmul_inplace_avx2(a: *const f32, b: *mut f32, n: usize) {
    use std::arch::x86_64::*;
    let mut i = 0;
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.add(i));
        let vb = _mm256_loadu_ps(b.add(i));
        let vr = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(b.add(i), vr);
        i += 8;
    }
    for j in i..n {
        *b.add(j) *= *a.add(j);
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub unsafe fn vmul_inplace_avx2(a: *const f32, b: *mut f32, n: usize) {
    for i in 0..n {
        *b.add(i) *= *a.add(i);
    }
}

/// Local copy of the exp2 approximation (same algorithm as forge_autograd's
/// private `exp2_approx_avx2`). Computes 2^x for 8 lanes.
#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(clippy::excessive_precision)]
unsafe fn exp2_approx_local(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let c0 = _mm256_set1_ps(1.0);
    let c1 = _mm256_set1_ps(std::f32::consts::LN_2);
    let c2 = _mm256_set1_ps(0.24022650);
    let c3 = _mm256_set1_ps(0.05550411);
    let c4 = _mm256_set1_ps(0.00961812);
    let i127 = _mm256_set1_epi32(127);

    // FIX (2026-08-02): Guard against NaN/Inf inputs — `_mm256_cvtps_epi32` on
    // NaN or out-of-i32-range produces UB (indefinite integer 0x80000000).
    // Clamp n to [-127, 127] so n+127 ∈ [0, 254] = valid IEEE-754 exponent range.
    // For NaN: _CMP_ORD_Q detects it, blend to -127 (→ exp2→0). For Inf: clamp to ±127.
    let zero_v = _mm256_setzero_ps();
    let neg127 = _mm256_set1_ps(-127.0);
    let pos127 = _mm256_set1_ps(127.0);
    let is_finite = _mm256_cmp_ps(x, x, _CMP_ORD_Q);  // false for NaN
    let x_clamped = _mm256_min_ps(_mm256_max_ps(x, neg127), pos127);
    let x_safe = _mm256_blendv_ps(zero_v, x_clamped, is_finite);  // NaN → 0 (exp2(0)=1, but NaN→0 is safe)

    let y = _mm256_mul_ps(x_safe, _mm256_set1_ps(1.0)); // identity: x = x * 1
    let n = _mm256_round_ps::<0>(y);
    let f = _mm256_sub_ps(y, n);

    let mut p = c4;
    p = _mm256_fmadd_ps(p, f, c3);
    p = _mm256_fmadd_ps(p, f, c2);
    p = _mm256_fmadd_ps(p, f, c1);
    p = _mm256_fmadd_ps(p, f, c0);

    // Clamp rounded n to [-127, 127] to prevent _mm256_cvtps_epi32 UB
    let n_clamped = _mm256_min_ps(_mm256_max_ps(n, neg127), pos127);
    let n_int = _mm256_cvtps_epi32(n_clamped);
    let n_biased = _mm256_add_epi32(n_int, i127);
    let scale = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(n_biased));

    _mm256_mul_ps(p, scale)
}

/// Vectorized mHC alpha/beta gradient accumulation (OPT-3) — AVX2 core.
///
/// Computes for each i in 0..n:
///   gate[i]   = sigmoid(v_jepa[i])
///   alpha_grad[i] += g[i] * h_in[i]
///   beta_grad[i]  += g[i] * (1 - gate[i]) * f_h[i]
///
/// Uses AVX2+FMA (8 f32/iter) for the sigmoid and elementwise products.
///
/// # Safety
/// - `g`, `v_jepa`, `h_in`, `f_h` readable for `n` elements.
/// - `alpha_grad`, `beta_grad` writable for `n` elements.
/// - No aliasing between input and output pointer sets.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn mhc_alpha_beta_grad_avx2(
    g: *const f32,
    v_jepa: *const f32,
    h_in: *const f32,
    f_h: *const f32,
    alpha_grad: *mut f32,
    beta_grad: *mut f32,
    n: usize,
) {
    use std::arch::x86_64::*;
    let one = _mm256_set1_ps(1.0);
    let mut i = 0;
    if n >= 8 {
        while i + 8 <= n {
            let gv = _mm256_loadu_ps(g.add(i));
            let vj = _mm256_loadu_ps(v_jepa.add(i));
            let hi = _mm256_loadu_ps(h_in.add(i));
            let fh = _mm256_loadu_ps(f_h.add(i));

            // gate = sigmoid(v_jepa)
            let gate = sigmoid_vec_local(vj);

            // alpha_grad += g * h_in
            let alpha_contrib = _mm256_mul_ps(gv, hi);
            let alpha_old = _mm256_loadu_ps(alpha_grad.add(i));
            let alpha_new = _mm256_add_ps(alpha_old, alpha_contrib);
            _mm256_storeu_ps(alpha_grad.add(i), alpha_new);

            // beta_grad += g * (1 - gate) * f_h
            let one_minus_gate = _mm256_sub_ps(one, gate);
            let beta_contrib = _mm256_mul_ps(_mm256_mul_ps(gv, one_minus_gate), fh);
            let beta_old = _mm256_loadu_ps(beta_grad.add(i));
            let beta_new = _mm256_add_ps(beta_old, beta_contrib);
            _mm256_storeu_ps(beta_grad.add(i), beta_new);

            i += 8;
        }
    }
    for j in i..n {
        let gj = *g.add(j);
        let gate = 1.0 / (1.0 + (-*v_jepa.add(j)).exp());
        *alpha_grad.add(j) += gj * *h_in.add(j);
        *beta_grad.add(j) += gj * (1.0 - gate) * *f_h.add(j);
    }
}

/// Vectorized sigmoid using AVX2 (used by mhc_alpha_beta_grad_avx2).
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn sigmoid_vec_local(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let half = _mm256_set1_ps(0.5);
    let neg_one = _mm256_set1_ps(-1.0);
    let one = _mm256_set1_ps(1.0);
    let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
    let neg_x = _mm256_mul_ps(neg_one, x);
    let y = _mm256_mul_ps(neg_x, log2e);
    let exp_val = exp2_approx_local(y);  // now NaN/Inf-safe
    let denom = _mm256_add_ps(one, exp_val);
    let result = _mm256_div_ps(one, denom);
    // P-17: NaN → 0.5 (neutral gate)
    let is_finite = _mm256_cmp_ps(result, result, _CMP_ORD_Q);
    _mm256_blendv_ps(half, result, is_finite)
}

/// Vectorized sigmoid σ(x) = 1 / (1 + exp(-x)), NaN → 0.5 (P-17 neutral).
/// Reuses the same `exp2_approx_local` approximation as `silu_vectorial_avx2`,
/// so a SiLU backward built on this stays consistent with the forward path.
///
/// # Safety
/// `x` readable and `out` writable for `n` elements. Regions may not overlap.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn sigmoid_avx2(x: *const f32, out: *mut f32, n: usize) {
    use std::arch::x86_64::*;
    let mut i = 0;
    while i + 8 <= n {
        let xv = _mm256_loadu_ps(x.add(i));
        _mm256_storeu_ps(out.add(i), sigmoid_vec_local(xv));
        i += 8;
    }
    for j in i..n {
        let v = *x.add(j);
        let val = 1.0 / (1.0 + (-v).exp());
        *out.add(j) = if val.is_finite() { val } else { 0.5 };
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub unsafe fn sigmoid_avx2(x: *const f32, out: *mut f32, n: usize) {
    for i in 0..n {
        let v = *x.add(i);
        let val = 1.0 / (1.0 + (-v).exp());
        *out.add(i) = if val.is_finite() { val } else { 0.5 };
    }
}

/// R-11 FIX: SiLU activation backward for the FFN.
///
/// Forward (slime_forward.rs): `mid[i] = silu(gate[i]) * up[i]`,
///   `silu(x) = x / (1 + exp(-x))`.
/// Backward given upstream grad `gy = dL/d(mid)`:
///   grad_up[i]   = gy[i] * silu(gate[i])
///   grad_gate[i] = gy[i] * up[i] * silu'(gate[i])
///   silu'(x) = σ(x) * (1 + x * (1 - σ(x)))
///
/// This REPLACES the stale ReLU² backward (`d/d g of g²·up`), which computed
/// the gradient of a *different* activation than the forward. Now
/// **backward == forward** (R-11).
///
/// # Safety
/// `gy`, `gate`, `up` readable for `n`; `grad_up`, `grad_gate` writable for `n`.
/// Input regions must not alias the output regions.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn silu_backward_avx2(
    gy: *const f32,
    gate: *const f32,
    up: *const f32,
    grad_up: *mut f32,
    grad_gate: *mut f32,
    n: usize,
) {
    use std::arch::x86_64::*;
    const ONE: f32 = 1.0;
    with_grad_scratch(n, |sigma_buf| {
        sigmoid_avx2(gate, sigma_buf.as_mut_ptr(), n);
        let sigma = sigma_buf.as_ptr();
        let one = _mm256_set1_ps(ONE);
        let mut i = 0;
        while i + 8 <= n {
            let gyv = _mm256_loadu_ps(gy.add(i));
            let gv = _mm256_loadu_ps(gate.add(i));
            let uv = _mm256_loadu_ps(up.add(i));
            let sv = _mm256_loadu_ps(sigma.add(i));

            let silu_g = _mm256_mul_ps(gv, sv); // silu(gate)
            let one_minus_s = _mm256_sub_ps(one, sv);
            let inner = _mm256_fmadd_ps(gv, one_minus_s, one); // 1 + gate*(1-sig)
            let silu_prime = _mm256_mul_ps(sv, inner); // σ*(1 + g*(1-σ))
            let g_up = _mm256_mul_ps(gyv, silu_g); // gy * silu(g)
            let g_gate = _mm256_mul_ps(gyv, _mm256_mul_ps(uv, silu_prime)); // gy*up*silu'(g)

            _mm256_storeu_ps(grad_up.add(i), g_up);
            _mm256_storeu_ps(grad_gate.add(i), g_gate);
            i += 8;
        }
        for j in i..n {
            let gyv = *gy.add(j);
            let gv = *gate.add(j);
            let uv = *up.add(j);
            let sv = *sigma.add(j);
            let silu_g = gv * sv;
            let silu_prime = sv * (ONE + gv * (ONE - sv));
            *grad_up.add(j) = gyv * silu_g;
            *grad_gate.add(j) = gyv * uv * silu_prime;
        }
    });
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub unsafe fn silu_backward_avx2(
    gy: *const f32,
    gate: *const f32,
    up: *const f32,
    grad_up: *mut f32,
    grad_gate: *mut f32,
    n: usize,
) {
    for i in 0..n {
        let gyv = *gy.add(i);
        let gv = *gate.add(i);
        let uv = *up.add(i);
        let sv = 1.0 / (1.0 + (-gv).exp());
        let silu_g = gv * sv;
        let silu_prime = sv * (1.0 + gv * (1.0 - sv));
        *grad_up.add(i) = gyv * silu_g;
        *grad_gate.add(i) = gyv * uv * silu_prime;
    }
}

/// Vectorized mHC alpha/beta gradient accumulation (OPT-3) — public dispatcher.
///
/// # Safety
/// All pointers valid for `n` elements. Input regions must not alias output regions.
#[inline]
pub unsafe fn mhc_backward_vec(
    g: *const f32,
    v_jepa: *const f32,
    h_in: *const f32,
    f_h: *const f32,
    alpha_grad: *mut f32,
    beta_grad: *mut f32,
    n: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        mhc_alpha_beta_grad_avx2(g, v_jepa, h_in, f_h, alpha_grad, beta_grad, n);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        for i in 0..n {
            let gj = *g.add(i);
            let gate = 1.0 / (1.0 + (-*v_jepa.add(i)).exp());
            *alpha_grad.add(i) += gj * *h_in.add(i);
            *beta_grad.add(i) += gj * (1.0 - gate) * *f_h.add(i);
        }
    }
}

/// Vectorized mHC activation backward (R-12 FIX).
///
/// Forward (mhc_residual): `h_out[i] = alpha[i]·h_in[i] + (1-gate[i])·beta[i]·f_h[i]`
/// with `gate[i] = sigmoid(v_jepa[i])`.  Given upstream grad `grad_in` (= dL/d h_out),
/// the activation (residual-stream) gradients are:
///   grad_out[i]  = grad_in[i] * alpha[i]                         (→ dL/d h_in, to residual)
///   grad_branch[i] = (grad_in[i] + kinetic_grad[i]) * (1-gate[i]) * beta[i]  (→ dL/d f_h, i.e. grad_ffn_out)
/// where `kinetic_grad[i] = -2 * kinetic_lambda * v_jepa[i]` (pre-existing JEPA
/// kinetic-energy regularization; preserved in place, only the missing mHC
/// scalars are now applied).  Base models (`alpha_w`/`beta_w` null) use
/// alpha=1, beta=1 (no-op scaling — matches mhc_residual base path).
///
/// `gate` is recomputed from `v_jepa` (same source as `mhc_backward_vec`) so the
/// activation path stays consistent with the alpha/beta param-grad path.
///
/// # Safety
/// - `grad_in`, `v_jepa` readable for `n` elements.
/// - `grad_out`, `grad_branch` writable for `n` elements.
/// - `alpha_w`/`beta_w`: either null (base model) or valid for `n` elements.
/// - Output regions must not alias inputs.
#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn mhc_activation_backward_avx2(
    grad_in: *const f32,
    v_jepa: *const f32,
    alpha_w: *const f32,
    beta_w: *const f32,
    grad_out: *mut f32,
    grad_branch: *mut f32,
    kinetic_lambda: f32,
    n: usize,
) {
    use std::arch::x86_64::*;
    let one = _mm256_set1_ps(1.0);
    let neg2lam = _mm256_set1_ps(-2.0 * kinetic_lambda);
    let has_mhc = !alpha_w.is_null() && !beta_w.is_null();
    let mut i = 0;
    if n >= 8 {
        while i + 8 <= n {
            let gv = _mm256_loadu_ps(grad_in.add(i));
            let vj = _mm256_loadu_ps(v_jepa.add(i));
            let gate = sigmoid_vec_local(vj);
            let one_minus_gate = _mm256_sub_ps(one, gate);
            let kinetic = _mm256_mul_ps(neg2lam, vj); // -2λ·v_jepa

            if has_mhc {
                let av = _mm256_loadu_ps(alpha_w.add(i));
                let bv = _mm256_loadu_ps(beta_w.add(i));
                let g_out = _mm256_mul_ps(gv, av);
                let g_branch = _mm256_mul_ps(
                    _mm256_mul_ps(_mm256_add_ps(gv, kinetic), one_minus_gate),
                    bv,
                );
                _mm256_storeu_ps(grad_out.add(i), g_out);
                _mm256_storeu_ps(grad_branch.add(i), g_branch);
            } else {
                // base model: alpha = beta = 1
                let g_branch = _mm256_mul_ps(_mm256_add_ps(gv, kinetic), one_minus_gate);
                _mm256_storeu_ps(grad_out.add(i), gv);
                _mm256_storeu_ps(grad_branch.add(i), g_branch);
            }
            i += 8;
        }
    }
    for j in i..n {
        let gjv = *grad_in.add(j);
        let vj = *v_jepa.add(j);
        let gate = 1.0 / (1.0 + (-vj).exp());
        let kinetic = -2.0 * kinetic_lambda * vj;
        let one_minus_gate = 1.0 - gate;
        if has_mhc {
            let alpha = *alpha_w.add(j);
            let beta = *beta_w.add(j);
            *grad_out.add(j) = gjv * alpha;
            *grad_branch.add(j) = (gjv + kinetic) * one_minus_gate * beta;
        } else {
            *grad_out.add(j) = gjv;
            *grad_branch.add(j) = (gjv + kinetic) * one_minus_gate;
        }
    }
}

/// Public dispatcher for the mHC activation backward (R-12). AVX2 core on
/// x86_64, scalar fallback otherwise.  See `mhc_activation_backward_avx2`.
///
/// # Safety
/// Same as the AVX2 core.
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn mhc_activation_backward(
    grad_in: *const f32,
    v_jepa: *const f32,
    alpha_w: *const f32,
    beta_w: *const f32,
    grad_out: *mut f32,
    grad_branch: *mut f32,
    kinetic_lambda: f32,
    n: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        mhc_activation_backward_avx2(grad_in, v_jepa, alpha_w, beta_w, grad_out, grad_branch, kinetic_lambda, n);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let has_mhc = !alpha_w.is_null() && !beta_w.is_null();
        for j in 0..n {
            let gjv = *grad_in.add(j);
            let vj = *v_jepa.add(j);
            let gate = 1.0 / (1.0 + (-vj).exp());
            let kinetic = -2.0 * kinetic_lambda * vj;
            let one_minus_gate = 1.0 - gate;
            if has_mhc {
                let alpha = *alpha_w.add(j);
                let beta = *beta_w.add(j);
                *grad_out.add(j) = gjv * alpha;
                *grad_branch.add(j) = (gjv + kinetic) * one_minus_gate * beta;
            } else {
                *grad_out.add(j) = gjv;
                *grad_branch.add(j) = (gjv + kinetic) * one_minus_gate;
            }
        }
    }
}
/// `threshold = scale * 0.7` STE band (matches trainer pack).
///
/// R-15 canonical ternary row-major pack (ELUT 4-bit, 8 weights per u32).
/// Single wire convention shared with the converter (`quantizer.rs`) and
/// `stable_pack_ternary_rows` / `unpack_ternary2bit_to_f32`:
/// per-row padded `rows * cols.div_ceil(8) * 4` bytes, little-endian u32
/// words, nibble `j*4` (`0x1` = +1, `0xF` = -1, else 0), zeroed padding.
/// Deterministic: pure function of `(shadow, scales)`, no RMW on old bytes,
/// no cross-row nibble sharing.
///
/// # Safety
/// - `shadow` readable for `rows * cols`
/// - `scales` writable for `rows`
/// - `packed` writable for `rows * cols.div_ceil(8) * 4` bytes
#[inline]
pub unsafe fn pack_ternary_rows_canonical(
    shadow: *const f32,
    rows: usize,
    cols: usize,
    scales: *mut f32,
    packed: *mut u8,
) {
    let cols = cols.max(1);
    let row_words = cols.div_ceil(8);
    let row_bytes = row_words * 4;
    for r in 0..rows {
        let start = r * cols;
        let mut abs_sum = 0.0f32;
        for c in 0..cols {
            abs_sum += (*shadow.add(start + c)).abs();
        }
        let s = ((abs_sum / cols as f32) * std::f32::consts::FRAC_1_SQRT_2)
            .max(crate::mud::constants::EPSILON_FLOOR);
        *scales.add(r) = s;
        let delta = s * crate::mud::constants::TERNARY_DEADZONE;
        let word_ptr = packed.add(r * row_bytes) as *mut u32;
        pack_ternary_into(shadow.add(start), cols, delta, word_ptr);
    }
}

/// Pack shadow f32 matrix into ELUT 4-bit + per-row PRQ scales.
///
/// R-15: delegates to [`pack_ternary_rows_canonical`] (per-row padded wire
/// format). Kept as a thin wrapper because the optimizer hot path
/// (`optim.rs`) and legacy tests call it by name. Buffer must be
/// `rows * cols.div_ceil(8) * 4` bytes (equals `(rows*cols).div_ceil(2)` iff
/// `cols % 8 == 0`, which holds for all production dims).
///
/// # Safety
/// - `shadow` readable for `rows * cols`
/// - `scales` writable for `rows`
/// - `packed` writable for `rows * cols.div_ceil(8) * 4` bytes
#[inline]
pub unsafe fn pack_elut_prq(
    shadow: *const f32,
    rows: usize,
    cols: usize,
    scales: *mut f32,
    packed: *mut u8,
) {
    pack_ternary_rows_canonical(shadow, rows, cols, scales, packed);
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

/// AVX2 dot3 (fused multiply-add): sum(a[i] * b[i] * c[i]) — 8 f32/iter.
/// Uses FMA: `acc += a*b*c` computed as `(a*b) + c` → FMA.
///
/// # Safety
/// `a`, `b`, `c` readable for `n` elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn dot3_fma_avx2(a: *const f32, b: *const f32, c: *const f32, n: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let av = _mm256_loadu_ps(a.add(i));
        let bv = _mm256_loadu_ps(b.add(i));
        let cv = _mm256_loadu_ps(c.add(i));
        let ab = _mm256_mul_ps(av, bv);
        sum = _mm256_fmadd_ps(ab, cv, sum);
        i += 8;
    }
    // Horizontal sum
    let hi = _mm256_extractf128_ps(sum, 1);
    let lo = _mm256_castps256_ps128(sum);
    let s128 = _mm_add_ps(lo, hi);
    let mut tmp = [0.0f32; 4];
    _mm_storeu_ps(tmp.as_mut_ptr(), s128);
    let mut total = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    // Scalar remainder
    for j in i..n {
        total += *a.add(j) * *b.add(j) * *c.add(j);
    }
    _mm256_zeroupper();
    total
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub unsafe fn dot3_fma_avx2(a: *const f32, b: *const f32, c: *const f32, n: usize) -> f32 {
    let mut total = 0.0f32;
    for i in 0..n {
        total += *a.add(i) * *b.add(i) * *c.add(i);
    }
    total
}

/// AVX2 RMSNorm backward: computes output_norm weight gradient and propagates
/// grad through the normalization. Vectorized version of the 3-loop pattern:
///   sum_dy_xnorm = sum(grad_in * w * (x * rms_inv))
///   w_grad[i] += grad_in[i] * (x[i] * rms_inv)
///   dx = (w * rms_inv) * (grad_in - x_norm * mean_dy_xnorm)
///
/// # Safety
/// - `grad_in` readable+writable for `n` elements.
/// - `x` (pre_norm_x) readable for `n` elements.
/// - `w` (output_norm_w) readable for `n` elements.
/// - `w_grad` writable for `n` elements.
/// - `rms_inv` precomputed as 1/rms(x).
/// - NaN/Inf guard applied per-element (P-17).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn rmsnorm_backward_avx2(
    grad_in: *mut f32,
    x: *const f32,
    w: *const f32,
    w_grad: *mut f32,
    rms_inv: f32,
    mean_dy_xnorm: f32,
    n: usize,
) {
    use std::arch::x86_64::*;
    let rms_inv_v = _mm256_set1_ps(rms_inv);
    let mean_v = _mm256_set1_ps(mean_dy_xnorm);
    let zero = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let gi = _mm256_loadu_ps(grad_in.add(i));
        let xi = _mm256_loadu_ps(x.add(i));
        let wi = _mm256_loadu_ps(w.add(i));

        let x_norm = _mm256_mul_ps(xi, rms_inv_v);
        let wi_rms = _mm256_mul_ps(wi, rms_inv_v);

        // w_grad[i] += grad_in[i] * x_norm[i]
        let wg_old = _mm256_loadu_ps(w_grad.add(i));
        let wg_contrib = _mm256_mul_ps(gi, x_norm);
        let wg_new = _mm256_add_ps(wg_old, wg_contrib);
        _mm256_storeu_ps(w_grad.add(i), wg_new);

        // dx = wi_rms * (gi - x_norm * mean_dy_xnorm)
        // FNMADD(x_norm, mean, gi) = -(x_norm*mean) + gi = gi - x_norm*mean
        // then MUL: wi_rms * inner
        let inner = _mm256_fnmadd_ps(x_norm, mean_v, gi);
        let dx = _mm256_mul_ps(wi_rms, inner);
        // NaN/Inf guard: |dx| < INF → finite (P-17)
        let sign_mask = _mm256_set1_ps(-0.0);
        let inf = _mm256_set1_ps(f32::INFINITY);
        let abs_dx = _mm256_andnot_ps(sign_mask, dx);
        let is_finite = _mm256_cmp_ps(abs_dx, inf, _CMP_LT_OQ);
        let dx_guarded = _mm256_blendv_ps(zero, dx, is_finite);
        _mm256_storeu_ps(grad_in.add(i), dx_guarded);

        i += 8;
    }
    // Scalar remainder with NaN/Inf guard
    for j in i..n {
        let xn = *x.add(j) * rms_inv;
        *w_grad.add(j) += *grad_in.add(j) * xn;
        let dx = (*w.add(j) * rms_inv) * (*grad_in.add(j) - xn * mean_dy_xnorm);
        *grad_in.add(j) = if dx.is_finite() { dx } else { 0.0 };
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub unsafe fn rmsnorm_backward_avx2(
    grad_in: *mut f32,
    x: *const f32,
    w: *const f32,
    w_grad: *mut f32,
    rms_inv: f32,
    mean_dy_xnorm: f32,
    n: usize,
) {
    for i in 0..n {
        let xn = *x.add(i) * rms_inv;
        *w_grad.add(i) += *grad_in.add(i) * xn;
        let dx = (*w.add(i) * rms_inv) * (*grad_in.add(i) - xn * mean_dy_xnorm);
        *grad_in.add(i) = if dx.is_finite() { dx } else { 0.0 };
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

    /// OPT-3: Verify mhc_backward_vec (AVX2) matches a scalar reference.
    #[test]
    fn test_mhc_backward_vec_matches_scalar() {
        let n = 64; // multiple of 8 for full AVX2 coverage
        let g: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1) - 3.0).collect();
        let v_jepa: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2) - 2.0).collect();
        let h_in: Vec<f32> = (0..n).map(|i| (i as f32 * 0.15) - 1.5).collect();
        let f_h: Vec<f32> = (0..n).map(|i| (i as f32 * 0.12) - 0.8).collect();
        let alpha_init: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
        let beta_init: Vec<f32> = (0..n).map(|i| i as f32 * 0.005).collect();

        // Vectorized result
        let mut alpha_vec = alpha_init.clone();
        let mut beta_vec = beta_init.clone();
        unsafe {
            mhc_backward_vec(
                g.as_ptr(),
                v_jepa.as_ptr(),
                h_in.as_ptr(),
                f_h.as_ptr(),
                alpha_vec.as_mut_ptr(),
                beta_vec.as_mut_ptr(),
                n,
            );
        }

        // Scalar reference
        let mut alpha_ref = alpha_init.clone();
        let mut beta_ref = beta_init.clone();
        for i in 0..n {
            let gate = 1.0f32 / (1.0f32 + (-v_jepa[i]).exp());
            alpha_ref[i] += g[i] * h_in[i];
            beta_ref[i] += g[i] * (1.0 - gate) * f_h[i];
        }

        for i in 0..n {
            assert!(
                (alpha_vec[i] - alpha_ref[i]).abs() < 1e-3,
                "alpha_grad mismatch at {i}: vec={} ref={}",
                alpha_vec[i],
                alpha_ref[i]
            );
            assert!(
                (beta_vec[i] - beta_ref[i]).abs() < 1e-3,
                "beta_grad mismatch at {i}: vec={} ref={}",
                beta_vec[i],
                beta_ref[i]
            );
        }
    }

    /// OPT-3: Test with n not a multiple of 8 (scalar remainder path).
    #[test]
    fn test_mhc_backward_vec_remainder() {
        let n = 13; // not a multiple of 8
        let g: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1) - 3.0).collect();
        let v_jepa: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2) - 2.0).collect();
        let h_in: Vec<f32> = (0..n).map(|i| (i as f32 * 0.15) - 1.5).collect();
        let f_h: Vec<f32> = (0..n).map(|i| (i as f32 * 0.12) - 0.8).collect();

        let mut alpha_vec = vec![0.0f32; n];
        let mut beta_vec = vec![0.0f32; n];
        unsafe {
            mhc_backward_vec(
                g.as_ptr(),
                v_jepa.as_ptr(),
                h_in.as_ptr(),
                f_h.as_ptr(),
                alpha_vec.as_mut_ptr(),
                beta_vec.as_mut_ptr(),
                n,
            );
        }

        let mut alpha_ref = vec![0.0f32; n];
        let mut beta_ref = vec![0.0f32; n];
        for i in 0..n {
            let gate = 1.0f32 / (1.0f32 + (-v_jepa[i]).exp());
            alpha_ref[i] += g[i] * h_in[i];
            beta_ref[i] += g[i] * (1.0 - gate) * f_h[i];
        }

        for i in 0..n {
            assert!((alpha_vec[i] - alpha_ref[i]).abs() < 1e-3, "alpha mismatch at {i}");
            assert!((beta_vec[i] - beta_ref[i]).abs() < 1e-3, "beta mismatch at {i}");
        }
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

    /// SPRINT-5 (5.4): branchless pack must produce the same output as
    /// the legacy 3-way if/else. We test the corner cases: positive, negative,
    /// in-deadzone, NaN, ±0, and the rare odd-cols remainder path.
    #[test]
    fn test_pack_elut_prq_branchless_matches_legacy() {
        // Mix of values: positives, negatives, deadzone, NaN.
        let shadow: Vec<f32> = vec![
            1.0, -1.0, 0.5, -0.5, 0.0, -0.0, f32::NAN, 0.1,
            5.0, -5.0, 0.7, -0.7, 0.69, -0.69, 0.0, 0.0,
        ];
        let rows = 1;
        let cols = shadow.len();
        // threshold = s * 0.7; s is set by absmean formula.
        let mut scales_a = vec![0.0f32; rows];
        let mut packed_a = vec![0u8; cols.div_ceil(2)];
        let mut scales_b = vec![0.0f32; rows];
        let mut packed_b = vec![0u8; cols.div_ceil(2)];

        // Reference: 3-way if/else (legacy).
        {
            let start = 0;
            let abs_sum: f32 = shadow[start..start + cols].iter().map(|v| v.abs()).sum();
            let s = (abs_sum / cols as f32 * std::f32::consts::FRAC_1_SQRT_2)
                .max(crate::mud::constants::EPSILON_FLOOR);
            scales_a[0] = s;
            let threshold = s * crate::mud::constants::SPARSITY_THRESHOLD_RATIO;
            for c in 0..cols {
                let v = shadow[start + c];
                let code = if v > threshold {
                    0x1u8
                } else if v < -threshold {
                    0xFu8
                } else {
                    0x0u8
                };
                let byte_idx = (start + c) / 2;
                let nibble_pos = ((start + c) % 2) * 4;
                let cur = packed_a[byte_idx];
                let mask = !(0xFu8 << nibble_pos);
                packed_a[byte_idx] = (cur & mask) | (code << nibble_pos);
            }
        }

        // Branchless: pack_elut_prq.
        unsafe {
            pack_elut_prq(
                shadow.as_ptr(),
                rows,
                cols,
                scales_b.as_mut_ptr(),
                packed_b.as_mut_ptr(),
            );
        }

        // Scales should match exactly (same formula).
        assert_eq!(scales_a, scales_b, "scales diverged");

        // Packed bytes should match exactly.
        assert_eq!(
            packed_a, packed_b,
            "packed bytes diverged:\n  legacy:     {packed_a:?}\n  branchless: {packed_b:?}"
        );
    }

    /// SPRINT-5 (5.4): odd `cols` (rem=1) must still pack the trailing
    /// element correctly. R-15: canonical per-row padded wire format
    /// (`rows * cols.div_ceil(8) * 4` bytes); padding nibbles are zero.
    #[test]
    fn test_pack_elut_prq_odd_cols_remainder() {
        // 5 elements: 2 pairs + 1 remainder.
        let shadow: Vec<f32> = vec![2.0, -2.0, 0.0, 0.5, -3.0];
        let cols: usize = 5;
        let rows = 1;
        let mut scales = vec![0.0f32; rows];
        let mut packed = vec![0u8; rows * cols.div_ceil(8) * 4]; // 4 bytes (1 u32 word)
        unsafe {
            pack_elut_prq(
                shadow.as_ptr(),
                rows,
                cols,
                scales.as_mut_ptr(),
                packed.as_mut_ptr(),
            );
        }
        // Layout (low-nibble first per u32 word, LE bytes):
        //   v=+2.0 → 0x1 ; v=-2.0 → 0xF ; v=0 → 0 ; v=+0.5 → 0 (deadzone) ;
        //   v=-3.0 → 0xF ; padding nibbles → 0
        //   → word = 0x000F_001F? nibbles [1,F,0,0,F,0,0,0] → bytes [F1,00,0F,00]
        assert_eq!(packed, vec![0xF1, 0x00, 0x0F, 0x00]);
    }

    /// R-15: empaquetado determinista + round-trip bit-exacto.
    /// pack → unpack (mismo s) → repack (mismo delta) ⇒ bytes idénticos;
    /// pack dos veces ⇒ bytes idénticos (sin RMW sobre bytes viejos).
    #[test]
    #[allow(clippy::needless_range_loop)]
    fn test_pack_unpack_pack_bit_exact_r15() {
        let rows = 3usize;
        let cols = 12usize; // cols % 8 != 0: ejerce el padding por fila
        let shadow: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 7919) % 17) as f32 * 0.23 - 1.7)
            .collect();
        let row_bytes = cols.div_ceil(8) * 4;
        let mut scales_a = vec![0.0f32; rows];
        let mut packed_a = vec![0u8; rows * row_bytes];
        let mut scales_b = vec![0.0f32; rows];
        let mut packed_b = vec![0xAAu8; rows * row_bytes]; // basura inicial: no debe filtrarse
        unsafe {
            pack_ternary_rows_canonical(
                shadow.as_ptr(),
                rows,
                cols,
                scales_a.as_mut_ptr(),
                packed_a.as_mut_ptr(),
            );
            pack_elut_prq(
                shadow.as_ptr(),
                rows,
                cols,
                scales_b.as_mut_ptr(),
                packed_b.as_mut_ptr(),
            );
        }
        assert_eq!(scales_a, scales_b, "R-15: escalas divergen");
        assert_eq!(packed_a, packed_b, "R-15: pack_elut_prq != canonical");
        // Determinismo: re-pack sobre buffer con basura da lo mismo.
        let mut packed_c = vec![0x55u8; rows * row_bytes];
        let mut scales_c = vec![0.0f32; rows];
        unsafe {
            pack_ternary_rows_canonical(
                shadow.as_ptr(),
                rows,
                cols,
                scales_c.as_mut_ptr(),
                packed_c.as_mut_ptr(),
            );
        }
        assert_eq!(packed_a, packed_c, "R-15: pack no determinista");
        // Round-trip: unpack con las mismas escalas → repack con el mismo
        // delta por fila ⇒ bytes idénticos (los valores ±s están fuera del
        // deadzone 0.7·s y el 0 sigue en 0).
        let mut unpacked = vec![0.0f32; rows * cols];
        crate::mud::slime_backward::unpack_ternary2bit_to_f32(
            &packed_a,
            &scales_a,
            cols,
            &mut unpacked,
        );
        let mut repacked = vec![0u8; rows * row_bytes];
        for r in 0..rows {
            let delta = scales_a[r] * crate::mud::constants::TERNARY_DEADZONE;
            unsafe {
                let word_ptr = repacked.as_mut_ptr().add(r * row_bytes) as *mut u32;
                pack_ternary_into(
                    unpacked.as_ptr().add(r * cols),
                    cols,
                    delta,
                    word_ptr,
                );
            }
        }
        assert_eq!(
            packed_a, repacked,
            "R-15: pack→unpack→pack no es bit-exacto"
        );
    }

    #[test]
    fn test_scale_with_weight_avx2_matches_scalar() {
        let n = 32;
        let mut x = vec![0.0f32; n];
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            x[i] = i as f32;
        }
        let w: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1) + 0.5).collect();
        let s = 0.3;
        let mut ref_x = x.clone();
        for i in 0..n {
            ref_x[i] *= w[i] * s;
        }
        #[cfg(target_arch = "x86_64")]
        {
            unsafe {
                crate::mud::ezop::scale_with_weight_avx2(x.as_mut_ptr(), w.as_ptr(), s, n);
            }
            assert_eq!(x, ref_x, "scale_with_weight_avx2 mismatch");
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            // On non-x86_64, scale_with_weight_avx2 is the scalar fallback
            unsafe {
                crate::mud::ezop::scale_with_weight_avx2(x.as_mut_ptr(), w.as_ptr(), s, n);
            }
            assert_eq!(x, ref_x, "scale_with_weight mismatch");
        }
    }

    #[test]
    fn test_peak_abs_i8_avx2_matches_scalar() {
        let data: Vec<i8> = vec![-100, 50, -127, 0, 127, -1, 1, 42, -50, 100, -127, 1];
        let ref_peak = data.iter().map(|&v| v.abs()).max().unwrap();
        #[cfg(target_arch = "x86_64")]
        {
            unsafe {
                let peak = crate::mud::ezop::peak_abs_i8_avx2(data.as_ptr(), data.len());
                assert_eq!(peak, ref_peak, "peak_abs_i8_avx2 mismatch");
            }
        }
    }

    #[test]
    fn test_rope_rotate_avx2_matches_scalar() {
        let half_d = 16;
        let cos: Vec<f32> = (0..half_d).map(|i| (i as f32 * 0.1).cos()).collect();
        let sin: Vec<f32> = (0..half_d).map(|i| (i as f32 * 0.1).sin()).collect();
        let x0: Vec<f32> = (0..half_d).map(|i| (i as f32 * 0.5) - 4.0).collect();
        let x1: Vec<f32> = (0..half_d).map(|i| (i as f32 * 0.3) - 2.0).collect();

        // Reference (scalar)
        let mut x0_ref = x0.clone();
        let mut x1_ref = x1.clone();
        for i in 0..half_d {
            let new_x0 = x0[i] * cos[i] - x1[i] * sin[i];
            let new_x1 = x1[i] * cos[i] + x0[i] * sin[i];
            x0_ref[i] = new_x0;
            x1_ref[i] = new_x1;
        }

        // AVX2
        let mut x0_avx = x0.clone();
        let mut x1_avx = x1.clone();
        #[cfg(target_arch = "x86_64")]
        unsafe {
            crate::mud::ezop::rope_rotate_avx2(
                x0_avx.as_mut_ptr(), x1_avx.as_mut_ptr(),
                cos.as_ptr(), sin.as_ptr(), half_d,
            );
            for i in 0..half_d {
                assert!((x0_avx[i] - x0_ref[i]).abs() < 1e-5, "x0 mismatch at {i}");
                assert!((x1_avx[i] - x1_ref[i]).abs() < 1e-5, "x1 mismatch at {i}");
            }
        }
    }

    #[test]
    fn test_dot3_fma_avx2_matches_scalar() {
        let n = 64;
        let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1) - 3.0).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2) - 2.0).collect();
        let c: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3) - 1.0).collect();
        let mut ref_total = 0.0f32;
        for i in 0..n {
            ref_total += a[i] * b[i] * c[i];
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let result = dot3_fma_avx2(a.as_ptr(), b.as_ptr(), c.as_ptr(), n);
            assert!((result - ref_total).abs() < 1e-3, "dot3 mismatch: {} vs {}", result, ref_total);
        }
    }

    #[test]
    fn test_rmsnorm_backward_avx2_matches_scalar() {
        let n = 64;
        let grad_in: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1) - 3.0).collect();
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2) - 2.0).collect();
        let w: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05) + 0.9).collect();
        let rms_inv = 0.15;
        let mean_dy_xnorm = 0.3;

        // Reference
        let mut grad_ref = grad_in.clone();
        let mut w_grad_ref = vec![0.0f32; n];
        for i in 0..n {
            let xn = x[i] * rms_inv;
            w_grad_ref[i] += grad_in[i] * xn;
            let dx = (w[i] * rms_inv) * (grad_in[i] - xn * mean_dy_xnorm);
            grad_ref[i] = if dx.is_finite() { dx } else { 0.0 };
        }

        // AVX2
        #[cfg(target_arch = "x86_64")]
        {
            let mut grad_avx = grad_in.clone();
            let mut w_grad_avx = vec![0.0f32; n];
            unsafe {
                rmsnorm_backward_avx2(
                    grad_avx.as_mut_ptr(), x.as_ptr(), w.as_ptr(),
                    w_grad_avx.as_mut_ptr(), rms_inv, mean_dy_xnorm, n,
                );
            }
            for i in 0..n {
                assert!((w_grad_avx[i] - w_grad_ref[i]).abs() < 1e-4,
                    "w_grad mismatch at {i}: {} vs {}", w_grad_avx[i], w_grad_ref[i]);
                assert!((grad_avx[i] - grad_ref[i]).abs() < 1e-4,
                    "grad mismatch at {i}: {} vs {}", grad_avx[i], grad_ref[i]);
            }
        }
    }

    #[test]
    fn test_axpy_avx2_matches_scalar() {
        let n = 32;
        let y = vec![0.0f32; n];
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1) - 1.5).collect();
        let a = 2.5f32;
        // Reference
        let mut y_ref = y.clone();
        for i in 0..n {
            y_ref[i] += a * x[i];
        }
        // AVX2
        let mut y_avx = y.clone();
        #[cfg(target_arch = "x86_64")]
        unsafe {
            axpy_avx2(y_avx.as_mut_ptr(), a, x.as_ptr(), n);
        }
        #[cfg(not(target_arch = "x86_64"))]
        unsafe {
            axpy_avx2(y_avx.as_mut_ptr(), a, x.as_ptr(), n);
        }
        for i in 0..n {
            assert!((y_avx[i] - y_ref[i]).abs() < 1e-5, "axpy mismatch at {i}");
        }
    }

    #[test]
    fn test_scale_finite_avx2_nan_guard() {
        let n = 32;
        let scale = 2.0f32;
        let mut x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.5) - 8.0).collect();
        // Inject NaN and Inf
        x[5] = f32::NAN;
        x[10] = f32::INFINITY;
        x[15] = f32::NEG_INFINITY;
        // Reference
        let mut x_ref = x.clone();
        for v in &mut x_ref {
            let val = *v * scale;
            *v = if val.is_finite() { val } else { 0.0 };
        }
        // AVX2
        let mut x_avx = x.clone();
        #[cfg(target_arch = "x86_64")]
        unsafe {
            scale_finite_avx2(x_avx.as_mut_ptr(), scale, n);
        }
        #[cfg(not(target_arch = "x86_64"))]
        unsafe {
            scale_finite_avx2(x_avx.as_mut_ptr(), scale, n);
        }
        for i in 0..n {
            assert!((x_avx[i] - x_ref[i]).abs() < 1e-5 || (x_avx[i] == 0.0 && x_ref[i] == 0.0),
                "scale_finite mismatch at {i}: {} vs {}", x_avx[i], x_ref[i]);
        }
    }

    #[test]
    fn test_clamp_finite_avx2_matches_scalar() {
        let n = 32;
        let lo = 0.01f32;
        let hi = 10.0f32;
        let default = 1.0f32;
        let mut x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.5) - 5.0).collect();
        x[5] = f32::NAN;
        x[10] = f32::INFINITY;
        // Reference
        let mut x_ref = x.clone();
        for v in &mut x_ref {
            let val = *v;
            *v = if val.is_finite() { val.clamp(lo, hi) } else { default };
        }
        // AVX2
        let mut x_avx = x.clone();
        #[cfg(target_arch = "x86_64")]
        unsafe {
            clamp_finite_avx2(x_avx.as_mut_ptr(), lo, hi, default, n);
        }
        #[cfg(not(target_arch = "x86_64"))]
        unsafe {
            clamp_finite_avx2(x_avx.as_mut_ptr(), lo, hi, default, n);
        }
        for i in 0..n {
            assert!((x_avx[i] - x_ref[i]).abs() < 1e-5 || (x_avx[i] == 0.0 && x_ref[i] == 0.0),
                "clamp_finite mismatch at {i}: {} vs {}", x_avx[i], x_ref[i]);
        }
    }

    #[test]
    fn test_exp_avx2_nan_inf_guard() {
        // FIX (2026-08-02): exp2_approx_local must not crash on NaN/Inf inputs.
        // _mm256_cvtps_epi32 on NaN/Inf is UB — was the SIGSEGV root cause.
        let x: [f32; 8] = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0, -1.0, 0.0, 5.0, -5.0];
        let mut out = vec![0.0f32; 8];
        unsafe {
            exp_avx2(x.as_ptr(), out.as_mut_ptr(), 8);
        }
        // NaN and ±Inf should NOT crash; NaN→0 (non-finite guard), Inf→finite result
        for v in &out {
            assert!(v.is_finite(), "exp_avx2 produced non-finite: {v}");
        }
        assert!((out[3] - 1.0f32.exp()).abs() < 1e-3, "exp(1) mismatch: {}", out[3]);
        assert!((out[5] - 1.0).abs() < 1e-3, "exp(0) should be 1: {}", out[5]);
    }

    #[test]
    fn test_exp_avx2_matches_scalar() {
        let n = 32;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1) - 1.5).collect();
        // Reference
        let mut out_ref = vec![0.0f32; n];
        for i in 0..n {
            out_ref[i] = x[i].exp();
        }
        // AVX2
        let mut out_avx = vec![0.0f32; n];
        unsafe {
            exp_avx2(x.as_ptr(), out_avx.as_mut_ptr(), n);
        }
        for i in 0..n {
            assert!((out_avx[i] - out_ref[i]).abs() < 1e-2,
                "exp mismatch at {i}: {} vs {} (ref={})", out_avx[i], out_ref[i], x[i]);
        }
    }

    #[test]
    fn test_grad_x_reduce_8rows_matches_scalar() {
        let n = 64;
        let scratch: Vec<f32> = (0..(8 * n)).map(|i| (i as f32 * 0.001) - 0.4).collect();
        let mut grad_x_ref: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01) + 0.5).collect();
        let mut grad_x_avx = grad_x_ref.clone();

        // Reference: grad_x[c] += sum_j scratch[j*n + c]
        for j in 0..n {
            let mut sum = 0.0f32;
            for row in 0..8 {
                sum += scratch[row * n + j];
            }
            grad_x_ref[j] += sum;
        }

        // AVX2
        unsafe {
            grad_x_reduce_8rows_avx2(grad_x_avx.as_mut_ptr(), scratch.as_ptr(), n);
        }
        for i in 0..n {
            assert!(
                (grad_x_avx[i] - grad_x_ref[i]).abs() < 1e-5,
                "grad_x_reduce_8rows mismatch at {i}: {} vs {}",
                grad_x_avx[i], grad_x_ref[i]
            );
        }
    }

    #[test]
    fn test_grad_x_reduce_8rows_remainder() {
        let n = 13; // Not a multiple of 8 → tests scalar tail
        let scratch: Vec<f32> = (0..(8 * n)).map(|i| (i as f32 * 0.001) - 0.4).collect();
        let mut grad_x_ref: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01) + 0.5).collect();
        let mut grad_x_avx = grad_x_ref.clone();

        for j in 0..n {
            let mut sum = 0.0f32;
            for row in 0..8 {
                sum += scratch[row * n + j];
            }
            grad_x_ref[j] += sum;
        }

        unsafe {
            grad_x_reduce_8rows_avx2(grad_x_avx.as_mut_ptr(), scratch.as_ptr(), n);
        }
        for i in 0..n {
            assert!(
                (grad_x_avx[i] - grad_x_ref[i]).abs() < 1e-5,
                "remainder mismatch at {i}: {} vs {}",
                grad_x_avx[i], grad_x_ref[i]
            );
        }
    }

    #[test]
    fn test_sum_avx2_matches_scalar() {
        let n = 64;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1) - 3.0).collect();
        let sum_ref: f32 = x.iter().sum();
        let sum_avx;
        unsafe {
            sum_avx = sum_avx2(x.as_ptr(), n);
        }
        assert!((sum_avx - sum_ref).abs() < 1e-3, "sum mismatch: {} vs {}", sum_avx, sum_ref);
    }

    #[test]
    fn test_max_avx2_matches_scalar() {
        let n = 32;
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1) - 1.5).collect();
        let max_ref: f32 = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let max_avx;
        unsafe {
            max_avx = max_avx2(x.as_ptr(), n);
        }
        assert!((max_avx - max_ref).abs() < 1e-5, "max mismatch: {} vs {}", max_avx, max_ref);
    }

    #[test]
    fn test_broadcast_sub_avx2_matches_scalar() {
        let n = 32;
        let mut x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1) - 1.5).collect();
        let val = 0.5f32;
        // Reference
        let mut x_ref = x.clone();
        for v in &mut x_ref {
            *v -= val;
        }
        // AVX2
        unsafe {
            broadcast_sub_avx2(x.as_mut_ptr(), val, n);
        }
        for i in 0..n {
            assert!((x[i] - x_ref[i]).abs() < 1e-5, "broadcast_sub mismatch at {i}: {} vs {}", x[i], x_ref[i]);
        }
    }

    #[test]
    fn test_div_avx2_matches_scalar() {
        let n = 64;
        let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01) - 1.0).collect();
        let b: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.01) - 1.0).abs() + 0.1).collect();
        let mut out_ref = a.clone();
        for i in 0..n {
            let val = a[i] / b[i];
            out_ref[i] = if val.is_finite() { val } else { 0.0 };
        }
        let mut out_avx = vec![0.0f32; n];
        unsafe {
            div_avx2(a.as_ptr(), b.as_ptr(), out_avx.as_mut_ptr(), 0.0, n);
        }
        for i in 0..n {
            assert!((out_avx[i] - out_ref[i]).abs() < 1e-5,
                "div mismatch at {i}: {} vs {}", out_avx[i], out_ref[i]);
        }
    }

    #[test]
    fn test_div_avx2_zero_divisor() {
        let n = 16;
        let a: Vec<f32> = (0..n).map(|i| (i as f32) + 1.0).collect();
        let b: Vec<f32> = vec![0.0f32; n];
        let mut out = vec![999.0f32; n];
        unsafe {
            div_avx2(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 0.0, n);
        }
        for &v in &out {
            assert_eq!(v, 0.0, "zero divisor should yield default 0.0");
        }
    }

    #[test]
    fn test_div_avx2_nan_inf_guard() {
        let n = 32;
        let a: Vec<f32> = vec![f32::INFINITY, f32::NEG_INFINITY, 1.0, 2.0, f32::NAN];
        let a_full: Vec<f32> = (0..n).map(|i| a[i % a.len()]).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01) + 0.5).collect();
        let mut out = vec![0.0f32; n];
        unsafe {
            div_avx2(a_full.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 0.0, n);
        }
        // Non-finite inputs should produce 0.0 (P-17)
        assert!(out[0] == 0.0, "Inf input should yield 0.0");
        assert!(out[1] == 0.0, "-Inf input should yield 0.0");
        assert!(out[4] == 0.0, "NaN input should yield 0.0");
    }

    /// R-11: the FFN activation backward must match the SiLU *forward*
    /// (`silu(gate)*up`), NOT the old ReLU² backward. Verifies the analytical
    /// SiLU derivative against an exact f64 reference; tolerance 1e-2 on AVX2
    /// (exp2-approx propagation, same approx as `silu_vectorial_avx2`) / 1e-5
    /// scalar. Also asserts silu'(1) ≠ 2 (the ReLU² derivative) so a regression
    /// to the old backward cannot silently pass.
    #[test]
    fn test_silu_backward_matches_reference() {
        let n = 32;
        let gy: Vec<f32> = (0..n).map(|i| ((i as f32 + 1.0) * 0.13).sin() * 0.7).collect();
        // gate in ~[-4, 4] keeps silu/exp2-approx well-behaved
        let gate: Vec<f32> = (0..n).map(|i| (i as f32 - 16.0) * 0.25).collect();
        let up: Vec<f32> = (0..n).map(|i| ((i as f32 + 3.0) * 0.17).cos() * 0.5).collect();
        let mut grad_up = vec![0.0f32; n];
        let mut grad_gate = vec![0.0f32; n];
        unsafe {
            silu_backward_avx2(
                gy.as_ptr(),
                gate.as_ptr(),
                up.as_ptr(),
                grad_up.as_mut_ptr(),
                grad_gate.as_mut_ptr(),
                n,
            );
        }
        // Exact f64 reference
        let mut ref_up = vec![0.0f32; n];
        let mut ref_gate = vec![0.0f32; n];
        for i in 0..n {
            let g = gate[i] as f64;
            let s = 1.0 / (1.0 + (-g).exp());
            let silu_g = (g * s) as f32;
            let silu_prime = (s * (1.0 + g * (1.0 - s))) as f32;
            ref_up[i] = gy[i] * silu_g;
            ref_gate[i] = gy[i] * up[i] * silu_prime;
        }
        let tol = if is_x86_feature_detected!("avx2") { 1e-2 } else { 1e-5 };
        for i in 0..n {
            assert!(
                (grad_up[i] - ref_up[i]).abs() < tol,
                "grad_up[{i}] {} vs ref {}",
                grad_up[i],
                ref_up[i]
            );
            assert!(
                (grad_gate[i] - ref_gate[i]).abs() < tol,
                "grad_gate[{i}] {} vs ref {}",
                grad_gate[i],
                ref_gate[i]
            );
        }
        // Regression guard: the OLD ReLU² backward had silu'(g)≡2g for g>0.
        // Real silu'(1) ≈ 0.632, definitely not 2.0.
        let g = 1.0f64;
        let s = 1.0 / (1.0 + (-g).exp());
        let silu_prime = s * (1.0 + g * (1.0 - s));
        assert!(
            (silu_prime as f32 - 2.0).abs() > 1e-3,
            "silu'(1)={silu_prime} should differ from ReLU² derivative 2.0"
        );
    }

    /// R-12: mHC activation backward must scale by alpha / (1-gate)*beta.
    /// Verifies the analytical formula for both the trainable-mHC route
    /// (alpha=0.85, beta=1.2, gate=sigmoid(v_jepa)≠0.5) and the base-model
    /// route (alpha_w/beta_w null → alpha=beta=1), comparing AVX2 vs exact f64.
    #[test]
    fn test_mhc_activation_backward_matches_reference() {
        let n = 32;
        let grad_in: Vec<f32> = (0..n).map(|i| ((i as f32 - 16.0) * 0.07).sin() * 0.3).collect();
        // v_jepa spread so gate is well away from 0.5
        let v_jepa: Vec<f32> = (0..n).map(|i| (i as f32 - 16.0) * 0.25).collect();
        let alpha: Vec<f32> = (0..n).map(|i| 0.85 + (i as f32 - 16.0) * 0.002).collect();
        let beta: Vec<f32> = (0..n).map(|i| 1.2 + (i as f32 - 16.0) * 0.003).collect();
        let kinetic_lambda = 0.005f32;
        let mut grad_out = vec![0.0f32; n];
        let mut grad_branch = vec![0.0f32; n];
        unsafe {
            mhc_activation_backward(
                grad_in.as_ptr(),
                v_jepa.as_ptr(),
                alpha.as_ptr(),
                beta.as_ptr(),
                grad_out.as_mut_ptr(),
                grad_branch.as_mut_ptr(),
                kinetic_lambda,
                n,
            );
        }
        let tol = if is_x86_feature_detected!("avx2") { 1e-3 } else { 1e-5 };
        for i in 0..n {
            let g = grad_in[i] as f64;
            let v = v_jepa[i] as f64;
            let s = 1.0 / (1.0 + (-v).exp());
            let kinetic = -2.0 * kinetic_lambda as f64 * v;
            let one_minus_gate = 1.0 - s;
            // trainable-mHC route
            let ref_out = g * alpha[i] as f64;
            let ref_branch = (g + kinetic) * one_minus_gate * beta[i] as f64;
            assert!(
                (grad_out[i] - ref_out as f32).abs() < tol,
                "mHC grad_out[{i}] {} vs ref {}",
                grad_out[i],
                ref_out
            );
            assert!(
                (grad_branch[i] - ref_branch as f32).abs() < tol,
                "mHC grad_branch[{i}] {} vs ref {}",
                grad_branch[i],
                ref_branch
            );
        }
        // Regression guard: old code returned grad_out = grad_in (identity),
        // i.e. grad_out[i] == grad_in[i] (alpha dropped). With alpha=0.85≠1
        // they must now differ (unless grad_in is 0).
        assert!(
            (grad_out[1] - grad_in[1]).abs() > 1e-4 || grad_in[1].abs() < 1e-9,
            "grad_out should reflect alpha scaling, got identity"
        );

        // Base-model route (null alpha/beta → alpha=beta=1)
        let mut grad_out_b = vec![0.0f32; n];
        let mut grad_branch_b = vec![0.0f32; n];
        unsafe {
            mhc_activation_backward(
                grad_in.as_ptr(),
                v_jepa.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                grad_out_b.as_mut_ptr(),
                grad_branch_b.as_mut_ptr(),
                kinetic_lambda,
                n,
            );
        }
        for i in 0..n {
            let g = grad_in[i] as f64;
            let v = v_jepa[i] as f64;
            let s = 1.0 / (1.0 + (-v).exp());
            let kinetic = -2.0 * kinetic_lambda as f64 * v;
            let ref_out = g; // alpha=1
            let ref_branch = (g + kinetic) * (1.0 - s); // beta=1
            assert!(
                (grad_out_b[i] - ref_out as f32).abs() < tol,
                "base grad_out[{i}] {} vs {}",
                grad_out_b[i],
                ref_out
            );
            assert!(
                (grad_branch_b[i] - ref_branch as f32).abs() < tol,
                "base grad_branch[{i}] {} vs {}",
                grad_branch_b[i],
                ref_branch
            );
        }
    }
}
