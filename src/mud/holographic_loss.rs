use std::arch::x86_64::*;

/// Computes the Cosine Phase Loss gradient between the ideal activation and quantized activation.
/// Uses AVX2 SIMD for maximum throughput across deep layers.
///
/// L_phase = 1 - (x_ideal · x_quant) / (||x_ideal|| * ||x_quant||)
///
/// Returns a scalar phase loss, and mutates `grad_out` by adding the phase gradient,
/// scaled by `lambda`.
///
/// # Safety
/// - The calling CPU MUST support AVX2 (`is_x86_feature_detected!("avx2")`).
///   This function is gated with `#[target_feature(enable = "avx2")]` and will
///   emit AVX2 instructions that trap with `SIGILL` on older CPUs.
/// - All three slices (`x_ideal`, `x_quant`, `grad_out`) MUST have the same
///   length. This is enforced at runtime via `assert_eq!` below.
/// - `grad_out` MUST be uniquely borrowed (no aliasing `&mut` references) for
///   the duration of the call, since it is mutated in place.
#[target_feature(enable = "avx2")]
pub unsafe fn compute_phase_gradients_avx2(
    x_ideal: &[f32],
    x_quant: &[f32],
    grad_out: &mut [f32],
    lambda: f32,
) -> f32 {
    let n = x_ideal.len();
    assert_eq!(n, x_quant.len());
    assert_eq!(n, grad_out.len());

    let mut dot_sum = 0.0;
    let mut norm_ideal_sq = 0.0;
    let mut norm_quant_sq = 0.0;

    let mut dot_v = _mm256_setzero_ps();
    let mut norm_i_v = _mm256_setzero_ps();
    let mut norm_q_v = _mm256_setzero_ps();

    let mut i = 0;
    while i + 8 <= n {
        let id = _mm256_loadu_ps(x_ideal.as_ptr().add(i));
        let qu = _mm256_loadu_ps(x_quant.as_ptr().add(i));

        dot_v = _mm256_fmadd_ps(id, qu, dot_v);
        norm_i_v = _mm256_fmadd_ps(id, id, norm_i_v);
        norm_q_v = _mm256_fmadd_ps(qu, qu, norm_q_v);

        i += 8;
    }

    let mut temp = [0.0; 8];
    _mm256_storeu_ps(temp.as_mut_ptr(), dot_v);
    dot_sum += temp.iter().sum::<f32>();

    _mm256_storeu_ps(temp.as_mut_ptr(), norm_i_v);
    norm_ideal_sq += temp.iter().sum::<f32>();

    _mm256_storeu_ps(temp.as_mut_ptr(), norm_q_v);
    norm_quant_sq += temp.iter().sum::<f32>();

    // Scalar fallback for remainder
    while i < n {
        let id = x_ideal[i];
        let qu = x_quant[i];
        dot_sum += id * qu;
        norm_ideal_sq += id * id;
        norm_quant_sq += qu * qu;
        i += 1;
    }

    let norm_ideal = norm_ideal_sq.sqrt().max(1e-8);
    let norm_quant = norm_quant_sq.sqrt().max(1e-8);
    let denominator = norm_ideal * norm_quant;

    let cos_sim = dot_sum / denominator;
    let phase_loss = 1.0 - cos_sim;

    // Gradient of Cosine Phase Loss with respect to x_ideal
    // dL/dx_ideal_j = - (x_quant_j / denominator) + (cos_sim * x_ideal_j / norm_ideal_sq)
    
    let grad_coeff_quant = -lambda / denominator;
    let grad_coeff_ideal = lambda * cos_sim / norm_ideal_sq;

    let grad_coeff_quant_v = _mm256_set1_ps(grad_coeff_quant);
    let grad_coeff_ideal_v = _mm256_set1_ps(grad_coeff_ideal);

    let mut j = 0;
    while j + 8 <= n {
        let id = _mm256_loadu_ps(x_ideal.as_ptr().add(j));
        let qu = _mm256_loadu_ps(x_quant.as_ptr().add(j));
        let mut g = _mm256_loadu_ps(grad_out.as_ptr().add(j));

        // term1 = grad_coeff_quant * x_quant
        let term1 = _mm256_mul_ps(grad_coeff_quant_v, qu);
        // term2 = grad_coeff_ideal * x_ideal
        let term2 = _mm256_mul_ps(grad_coeff_ideal_v, id);
        
        let grad_phase = _mm256_add_ps(term1, term2);
        g = _mm256_add_ps(g, grad_phase);

        _mm256_storeu_ps(grad_out.as_mut_ptr().add(j), g);
        j += 8;
    }

    while j < n {
        let id = x_ideal[j];
        let qu = x_quant[j];
        let d_phase = (grad_coeff_quant * qu) + (grad_coeff_ideal * id);
        grad_out[j] += d_phase;
        j += 1;
    }

    phase_loss
}

/// Fallback wrapper for safe dynamic dispatch
pub fn compute_phase_gradients(
    x_ideal: &[f32],
    x_quant: &[f32],
    grad_out: &mut [f32],
    lambda: f32,
) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { compute_phase_gradients_avx2(x_ideal, x_quant, grad_out, lambda) };
        }
    }
    
    // Scalar generic fallback
    let mut dot_sum = 0.0;
    let mut norm_ideal_sq = 0.0;
    let mut norm_quant_sq = 0.0;
    
    let n = x_ideal.len();
    for i in 0..n {
        dot_sum += x_ideal[i] * x_quant[i];
        norm_ideal_sq += x_ideal[i] * x_ideal[i];
        norm_quant_sq += x_quant[i] * x_quant[i];
    }
    
    let norm_ideal = norm_ideal_sq.sqrt().max(1e-8);
    let norm_quant = norm_quant_sq.sqrt().max(1e-8);
    let denominator = norm_ideal * norm_quant;
    
    let cos_sim = dot_sum / denominator;
    let grad_coeff_quant = -lambda / denominator;
    let grad_coeff_ideal = lambda * cos_sim / norm_ideal_sq;
    
    for i in 0..n {
        let d_phase = (grad_coeff_quant * x_quant[i]) + (grad_coeff_ideal * x_ideal[i]);
        grad_out[i] += d_phase;
    }
    
    1.0 - cos_sim
}

/// Computes the VicReg Variance Hinge Loss gradient for the JEPA latent space.
///
/// L_V = max(0, gamma - sqrt(Var(V) + eps))
///
///
/// # Safety
/// - The calling CPU MUST support AVX2. This function is gated with `#[target_feature(enable = "avx2")]`
///   and will emit AVX2 instructions that trap with `SIGILL` on older CPUs.
/// - `grad_v_jepa` MUST have exactly the same length as `v_jepa` to avoid memory safety issues.
#[target_feature(enable = "avx2")]
pub unsafe fn compute_vicreg_variance_gradients_avx2(
    v_jepa: &[f32],
    grad_v_jepa: &mut [f32],
    gamma: f32,
    lambda: f32,
) -> f32 {
    let n = v_jepa.len();
    assert_eq!(n, grad_v_jepa.len());
    if n == 0 { return 0.0; }

    let mut sum = 0.0;
    for &v in v_jepa { sum += v; }
    let mu = sum / n as f32;

    let mut var_sum = 0.0;
    for &v in v_jepa {
        let d = v - mu;
        var_sum += d * d;
    }
    let var = var_sum / n as f32;
    let std_dev = (var + 1e-8).sqrt();

    let hinge = gamma - std_dev;
    if hinge <= 0.0 {
        return 0.0; // Variance is high enough, no penalty
    }

    // Gradient of L_V w.r.t v_i:
    // L_V = gamma - std_dev
    // dL/dv_i = -1 * (1 / (2 * std_dev)) * (2 * (v_i - mu) / n)
    //         = -(v_i - mu) / (n * std_dev)
    
    let grad_coeff = -lambda / (n as f32 * std_dev);

    // Add to grad_v_jepa
    for i in 0..n {
        grad_v_jepa[i] += grad_coeff * (v_jepa[i] - mu);
    }

    hinge
}
