use std::arch::x86_64::*;

/// Computes the dot product of two f32 slices using AVX2 and FMA instructions.
/// Assumes both slices have the exact same length.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    let mut sum_vec = _mm256_setzero_ps();
    let len = a.len();
    let mut i = 0;

    while i + 7 < len {
        unsafe {
            let a_chunk = _mm256_loadu_ps(a.as_ptr().add(i));
            let b_chunk = _mm256_loadu_ps(b.as_ptr().add(i));
            sum_vec = _mm256_fmadd_ps(a_chunk, b_chunk, sum_vec);
        }
        i += 8;
    }

    let mut sums = [0.0f32; 8];
    unsafe {
        _mm256_storeu_ps(sums.as_mut_ptr(), sum_vec);
    }
    let mut total = sums.iter().sum();

    while i < len {
        total += a[i] * b[i];
        i += 1;
    }

    total
}

/// Adds vector `b` scaled by `alpha` into vector `a`.
/// `a[i] += alpha * b[i]`
#[target_feature(enable = "avx2,fma")]
pub unsafe fn axpy_avx2(a: &mut [f32], alpha: f32, b: &[f32]) {
    let len = a.len();
    let mut i = 0;
    let alpha_vec = _mm256_set1_ps(alpha);

    while i + 7 < len {
        unsafe {
            let a_chunk = _mm256_loadu_ps(a.as_ptr().add(i));
            let b_chunk = _mm256_loadu_ps(b.as_ptr().add(i));
            let res = _mm256_fmadd_ps(alpha_vec, b_chunk, a_chunk);
            _mm256_storeu_ps(a.as_mut_ptr().add(i), res);
        }
        i += 8;
    }

    while i < len {
        a[i] += alpha * b[i];
        i += 1;
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn sgd_step_avx2(
    shadow_w: &mut [f32],
    grad_w: &[f32],
    lr: f32,
    weight_decay: f32,
    num_tokens: f32,
) {
    // NOTE: gradient normalization by `num_tokens` is performed ONCE by the
    // caller via `scale_grad_by_tokens` (applies to every optimizer strategy).
    // Do NOT divide here again, or the effective LR would shrink by num_tokens².
    let _ = num_tokens;
    let len = shadow_w.len();
    let mut i = 0;

    let g_max_vec = _mm256_set1_ps(10.0);
    let g_min_vec = _mm256_set1_ps(-10.0);

    let neg_lr_vec = _mm256_set1_ps(-lr);
    let decay_factor = 1.0 - lr * weight_decay;
    let decay_vec = _mm256_set1_ps(decay_factor);

    let w_max_vec = _mm256_set1_ps(5.0);
    let w_min_vec = _mm256_set1_ps(-5.0);

    let zero_vec = _mm256_setzero_ps();

    while i + 7 < len {
        let g_chunk = _mm256_loadu_ps(grad_w.as_ptr().add(i));

        let mut g_val = g_chunk;

        // Handle NaN/Infinity: if g_chunk is NaN, cmp_ord is false (all zeros).
        // We mask NaNs to 0.0.
        let is_ord = _mm256_cmp_ps(g_chunk, g_chunk, 0x07); // _CMP_ORD_Q
        g_val = _mm256_blendv_ps(zero_vec, g_val, is_ord);

        // clamp(-10, 10) -> max(min(x, 10), -10)
        g_val = _mm256_max_ps(g_min_vec, _mm256_min_ps(g_max_vec, g_val));

        let w_chunk = _mm256_loadu_ps(shadow_w.as_ptr().add(i));

        // w_val = w_chunk * decay_factor + (-lr) * g_val
        let mut w_val = _mm256_fmadd_ps(decay_vec, w_chunk, _mm256_mul_ps(neg_lr_vec, g_val));

        // clamp(-5, 5)
        w_val = _mm256_max_ps(w_min_vec, _mm256_min_ps(w_max_vec, w_val));

        _mm256_storeu_ps(shadow_w.as_mut_ptr().add(i), w_val);
        i += 8;
    }

    while i < len {
        let g = grad_w[i];
        let mut g_val = if g.is_nan() || g.is_infinite() {
            0.0
        } else {
            g
        };
        g_val = g_val.clamp(-10.0, 10.0);
        let mut w_val = shadow_w[i];
        w_val = w_val * decay_factor - lr * g_val;
        w_val = w_val.clamp(-5.0, 5.0);
        shadow_w[i] = w_val;
        i += 1;
    }
}

/// Fast exp2 approximation using polynomial expansion (same as ASM kernel).
/// Computes 2^x for 8 lanes using: 2^(n+f) = 2^n * poly(f)
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn exp2_approx_avx2(x: __m256) -> __m256 {
    let log2e = _mm256_set1_ps(1.4426950408889634);
    let c0 = _mm256_set1_ps(1.0);
    let c1 = _mm256_set1_ps(0.69314718);
    let c2 = _mm256_set1_ps(0.24022650);
    let c3 = _mm256_set1_ps(0.05550411);
    let c4 = _mm256_set1_ps(0.00961812);
    let i127 = _mm256_set1_epi32(127);

    let y = _mm256_mul_ps(x, log2e);
    let n = _mm256_round_ps::<0>(y);
    let f = _mm256_sub_ps(y, n);

    let mut p = c4;
    p = _mm256_fmadd_ps(p, f, c3);
    p = _mm256_fmadd_ps(p, f, c2);
    p = _mm256_fmadd_ps(p, f, c1);
    p = _mm256_fmadd_ps(p, f, c0);

    let n_int = _mm256_cvtps_epi32(n);
    let n_biased = _mm256_add_epi32(n_int, i127);
    let scale = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(n_biased));

    _mm256_mul_ps(p, scale)
}

/// SiLU activation: f(x) = x * sigmoid(x) = x / (1 + exp(-x))
/// Uses fast AVX2 exp2 approximation. Processes 8 floats per iteration.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn silu_avx2(src: &[f32], dst: &mut [f32]) {
    let len = src.len();
    let mut i = 0;
    let neg_one = _mm256_set1_ps(-1.0);
    let one = _mm256_set1_ps(1.0);

    while i + 7 < len {
        let x = _mm256_loadu_ps(src.as_ptr().add(i));
        let neg_x = _mm256_mul_ps(neg_one, x);
        let exp_neg_x = exp2_approx_avx2(neg_x);
        let denom = _mm256_add_ps(one, exp_neg_x);
        let sigmoid = _mm256_div_ps(one, denom);
        let result = _mm256_mul_ps(x, sigmoid);
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), result);
        i += 8;
    }

    while i < len {
        let x = src[i];
        let sig = 1.0 / (1.0 + (-x).exp());
        dst[i] = x * sig;
        i += 1;
    }
}

/// RMS norm scale factor: 1 / sqrt(mean(x^2) + eps)
/// Uses AVX2 for sum-of-squares reduction.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn rms_norm_scale_avx2(x: &[f32], eps: f32) -> f32 {
    let len = x.len();
    let mut sum_sq = _mm256_setzero_ps();
    let mut i = 0;

    while i + 7 < len {
        let chunk = _mm256_loadu_ps(x.as_ptr().add(i));
        sum_sq = _mm256_fmadd_ps(chunk, chunk, sum_sq);
        i += 8;
    }

    let mut sums = [0.0f32; 8];
    _mm256_storeu_ps(sums.as_mut_ptr(), sum_sq);
    let mut total: f32 = sums.iter().sum();

    while i < len {
        total += x[i] * x[i];
        i += 1;
    }

    1.0 / ((total / len as f32) + eps).sqrt()
}

/// SGEMM: C = A * B^T with AVX2+FMA.
/// A: [m, k], B: [n, k], C: [m, n]
/// Uses 1×8 micro-kernel: each output row processes 8 columns of B at a time.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn sgemm_abt_avx2(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let mut j = 0;

        while j + 7 < n {
            let mut acc = _mm256_setzero_ps();
            for p in 0..k {
                let a_val = _mm256_set1_ps(a_row[p]);
                let b_vals = _mm256_set_ps(
                    b[(j + 7) * k + p],
                    b[(j + 6) * k + p],
                    b[(j + 5) * k + p],
                    b[(j + 4) * k + p],
                    b[(j + 3) * k + p],
                    b[(j + 2) * k + p],
                    b[(j + 1) * k + p],
                    b[j * k + p],
                );
                acc = _mm256_fmadd_ps(a_val, b_vals, acc);
            }
            _mm256_storeu_ps(c.as_mut_ptr().add(i * n + j), acc);
            j += 8;
        }

        while j < n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a_row[p] * b[j * k + p];
            }
            c[i * n + j] = sum;
            j += 1;
        }
    }
}
