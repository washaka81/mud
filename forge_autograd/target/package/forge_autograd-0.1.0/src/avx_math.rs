use std::arch::x86_64::*;

/// Computes the dot product of two f32 slices using AVX2 and FMA instructions.
/// Assumes both slices have the exact same length.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    let mut sum_vec = _mm256_setzero_ps();
    let len = a.len();
    let mut i = 0;

    // Procesa 8 floats a la vez usando Fused Multiply-Add (FMA)
    while i + 7 < len {
        unsafe {
            let a_chunk = _mm256_loadu_ps(a.as_ptr().add(i));
            let b_chunk = _mm256_loadu_ps(b.as_ptr().add(i));
            sum_vec = _mm256_fmadd_ps(a_chunk, b_chunk, sum_vec);
        }
        i += 8;
    }

    // Reduce el vector de 256 bits a un solo f32
    let mut sums = [0.0f32; 8];
    unsafe { _mm256_storeu_ps(sums.as_mut_ptr(), sum_vec); }
    let mut total = sums.iter().sum();

    // Procesa elementos residuales (escalares)
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
            // a_chunk = alpha * b_chunk + a_chunk
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
