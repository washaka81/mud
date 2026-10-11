use crate::mud::slime::SlimeRegister;

// P-13 / L-07: re-export SSOT from constants
pub use crate::mud::constants::EPSILON_FLOOR;

// ── D5: Thread-local scratch for GPU JEPA path (normalize + EMA temporaries).
// Grown on demand, never freed. Holds 2×n f32: [y_norm_tmp, z_next_tmp].
thread_local! {
    static JEPA_GPU_SCRATCH: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

// ── OPT-1: Thread-local scratch for CPU JEPA path. Holds 3×n f32:
// [y_norm_tmp, z_next_tmp, jitter_tmp]. Reuses the same grow-on-demand pattern
// as JEPA_GPU_SCRATCH to eliminate P-01 allocations per jepa_stabilizer call.
thread_local! {
    static JEPA_CPU_SCRATCH: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Fixed EMA alpha for GPU JEPA path (mode 4). Matches AGENTS.md §5:
/// `z_next = 0.9·z + 0.1·y_norm` (pure OU, no adaptive alpha/jitter).
const JEPA_GPU_ALPHA: f32 = 0.1;

/// Per-layer tensor health diagnostics reported after each forward block.
/// Used by the TUI trainer and thermodynamic telemetry.
#[derive(Debug, Default)]
pub struct TensorDiagnostics {
    pub mean_hard: f32,
    pub mean_jepa: f32,
    pub var_hard: f32,
    pub var_jepa: f32,
    pub cov_hard_jepa: f32,
    pub mode_hard: f32,
    pub rho_cross_corr: f32,
    pub jepa_energy: f32,
    pub saturation_ratio: f32,
    pub hard_sparsity: f32,
    pub jepa_sparsity: f32,
    pub regression_slope: f32,
    pub regression_intercept: f32,
    pub softmax_temperature_est: f32,
    pub ternary_jepa_alignment: f32,
    pub z_entropy: f32,
}

/// Compute tensor health statistics from registers.
/// Reads ternary accumulation (native f32) and JEPA integral (native f32).
pub fn check_tensor_health(registers: &[SlimeRegister], gamma: f32) -> TensorDiagnostics {
    let mut jepa_zeros = 0;
    let mut hard_zeros = 0;
    let mut saturated = 0usize;
    let size = registers.len();
    if size == 0 {
        return TensorDiagnostics::default();
    }

    let mut sum_a = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut freqs = std::collections::HashMap::new();

    for reg in registers {
        let hard = reg.read_accum();
        if hard == 0.0 {
            hard_zeros += 1;
        }
        if hard.abs() >= 65500.0 {
            saturated += 1;
        }
        let hard_i32 = hard.round() as i32;
        *freqs.entry(hard_i32).or_insert(0) += 1;

        // JEPA integral zero check
        let integral = reg.read_integral();
        if integral == 0.0 {
            jepa_zeros += 1;
        }

        sum_a += hard as f64;
        sum_b += integral as f64;
    }

    let n = size as f64;
    let mean_a = sum_a / n;
    let mean_b = sum_b / n;

    let mut sum_sq_a = 0.0f64;
    let mut sum_sq_b = 0.0f64;
    let mut sum_cov = 0.0f64;
    let mut jepa_energy_sum = 0.0f64;

    for reg in registers {
        let a = reg.read_accum() as f64;
        let b = reg.read_integral() as f64;

        let diff_a = a - mean_a;
        let diff_b = b - mean_b;

        sum_sq_a += diff_a * diff_a;
        sum_sq_b += diff_b * diff_b;
        sum_cov += diff_a * diff_b;

        let dist = diff_b - gamma as f64;
        jepa_energy_sum += dist * dist;
    }

    let var_a = sum_sq_a / n;
    let var_b = sum_sq_b / n;
    let cov_ab = sum_cov / n;

    let rho = if var_a > 0.0 && var_b > 0.0 {
        cov_ab / (var_a.sqrt() * var_b.sqrt())
    } else {
        0.0
    };

    let slope = if var_a > 0.0 { cov_ab / var_a } else { 0.0 };
    let intercept = mean_b - slope * mean_a;
    let alignment = rho; // same as cross-correlation
    let softmax_temp = if var_a > 0.0 { 1.0 / var_a.sqrt() } else { 1.0 };
    let z_entropy = -(var_b / (var_b + 1.0)).ln();

    let mut mode = 0i32;
    let mut max_freq = 0usize;
    for (&val, &count) in freqs.iter() {
        if count > max_freq {
            max_freq = count;
            mode = val;
        }
    }

    TensorDiagnostics {
        mean_hard: mean_a as f32,
        mean_jepa: mean_b as f32,
        var_hard: var_a as f32,
        var_jepa: var_b as f32,
        cov_hard_jepa: cov_ab as f32,
        mode_hard: mode as f32,
        rho_cross_corr: rho as f32,
        jepa_energy: (jepa_energy_sum / n) as f32,
        saturation_ratio: saturated as f32 / size as f32,
        hard_sparsity: hard_zeros as f32 / size as f32,
        jepa_sparsity: jepa_zeros as f32 / size as f32,
        regression_slope: slope as f32,
        regression_intercept: intercept as f32,
        softmax_temperature_est: softmax_temp as f32,
        ternary_jepa_alignment: alignment as f32,
        z_entropy: z_entropy as f32,
    }
}

/// JEPA Stabilizer v2 — Integral Controller (I-controller)
///
/// Processes a block boundary:
/// 1. Computes z-score normalization of `block_out` (y_norm, RMS=1)
/// 2. Updates per-dimension OU tracker: `z_next = 0.9·z + 0.1·y_norm`
/// 3. Computes `v_jepa = (z - μ_ctx) · inv_sigma_ctx` (centered gate signal)
/// 4. Updates JEPA integral in each register: `I = 0.99·I + 0.01·v_jepa`
///    (I-controller: low-pass, equilibrium I→v_jepa at steady state)
/// 5. Writes v_jepa to tape (for backward pass)
///
/// Returns 0.0 (spring force eliminated — mHC radius handles stabilization).
///
/// GPU path (D5): when `gpu` is `Some`, steps 1-2 (z-score normalize + OU update)
/// are offloaded to `step_elementwise_sync` (elementwise.comp):
///
/// * Step 1 (z-score normalize): mode 5 (normalize), param0=mean, param1=rms
/// * Step 2 (OU update): mode 4 (ema), param0=alpha(0.1), param1=(1-alpha)=0.9
///
/// Uses **fixed** alpha=0.1 (pure OU per AGENTS.md §5). Adaptive alpha + jitter
/// are CPU-only (the generic elementwise shader can't express per-element alpha).
/// If the GPU dispatch fails, falls back to the CPU scalar path.
#[inline]
#[allow(clippy::too_many_arguments, clippy::needless_option_as_deref)]
pub fn jepa_stabilizer(
    block_out: &mut [f32],
    registers: &mut [SlimeRegister],
    mu_ctx: &mut f32,
    inv_sigma_ctx: &mut f32,
    var_ema: &mut f32,
    z_buf: &mut [f32],
    mut tape_out: Option<&mut [f32]>,
    mut gpu: Option<&mut crate::mud::ash_qat_dispatcher::AshQatDispatcher>,
) -> f32 {
    let n = registers.len() as f64;

    // ── Compute y_norm = z-score of block_out (mean 0, std 1) ──────────────
    // P-01: single-pass mean + variance (naive formula safe here since block_out
    // is already z-scored — mean ≈ 0, variance ≈ 1, no catastrophic cancellation).
    let mut sum_y = 0.0f64;
    let mut sum_sq = 0.0f64;
    for b in block_out.iter() {
        let bv = *b as f64;
        sum_y += bv;
        sum_sq += bv * bv;
    }
    let mean_y = sum_y / n;
    let variance = (sum_sq / n - mean_y * mean_y).max(0.0);
    let mean_y_f32 = mean_y as f32;
    let rms = variance.sqrt().max(1e-8) as f32;

    // ── Update z tracker and accumulate statistics ──────────────────────────
    let mut sum_z = 0.0f64;
    let mut sum_z_sq = 0.0f64;

    // ── D5: GPU path (normalize + EMA via elementwise.comp) ───────────────────
    // When AshQatDispatcher is available, offload z-score normalize (mode 5)
    // and OU update (mode 4) to the GPU. Uses fixed alpha=0.1 (pure OU per
    // AGENTS.md §5). Falls back to CPU scalar path on dispatch failure.
    let gpu_ok = if let Some(gpu) = gpu.as_deref_mut() {
        let success = JEPA_GPU_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            if scratch.len() < 2 * registers.len() {
                scratch.resize(2 * registers.len(), 0.0);
            }
            let (y_norm_tmp, z_next_tmp) = scratch.split_at_mut(registers.len());
            let y_norm_tmp = &mut y_norm_tmp[..registers.len()];
            let z_next_tmp = &mut z_next_tmp[..registers.len()];

            // Mode 5 (normalize): y_norm[i] = (block_out[i] - mean) / rms
            let dummy = [0.0f32; 1];
            let norm_ok = unsafe {
                gpu.step_elementwise_sync(
                    block_out, y_norm_tmp, &dummy, registers.len(), 5, mean_y_f32, rms,
                ).is_ok()
            };

            if norm_ok {
                // Mode 4 (ema): z_next[i] = alpha * y_norm[i] + (1-alpha) * z[i]
                let ema_ok = unsafe {
                    gpu.step_elementwise_sync(
                        y_norm_tmp, z_next_tmp, z_buf, registers.len(), 4,
                        JEPA_GPU_ALPHA, 1.0 - JEPA_GPU_ALPHA,
                    ).is_ok()
                };
                if ema_ok {
                    // Copy z_next back to z_buf
                    z_buf[..registers.len()].copy_from_slice(&z_next_tmp[..registers.len()]);
                    true
                } else {
                    false
                }
            } else {
                false
            }
        });

        if success {
            // Accumulate sum_z / sum_z_sq from updated z_buf
            for val in z_buf.iter().take(registers.len()) {
                let val = *val as f64;
                sum_z += val;
                sum_z_sq += val * val;
            }
            true
        } else {
            false
        }
    } else {
        false
    };

    if !gpu_ok {
        // OPT-1/OPT-2: Use thread-local scratch buffer (P-01 zero-alloc)
        // and AVX2 vectorized correction. The flow:
        //   1. Precompute jitter values into thread-local buffer
        //   2. Compute y_norm into thread-local buffer (branchless normalize)
        //   3. jepa_ou_avx2 (ASM) batch EMA with base alpha (0.1)
        //   4. Vectorized adaptive alpha delta + jitter + clamp (AVX2, 8/iter)
        let n = registers.len();
        let alpha = JEPA_GPU_ALPHA;
        let z_buf_ptr = z_buf.as_mut_ptr();

        JEPA_CPU_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            if scratch.len() < 3 * n {
                scratch.resize(3 * n, 0.0);
            }
            // Layout: [y_norm_tmp (n), z_next_tmp (n), jitter_tmp (n)]
            let (y_norm_tmp, rest) = scratch.split_at_mut(n);
            let y_norm_tmp = &mut y_norm_tmp[..n];
            let (z_next_tmp, jitter_tmp) = rest.split_at_mut(n);
            let z_next_tmp = &mut z_next_tmp[..n];
            let jitter_tmp = &mut jitter_tmp[..n];

            // Precompute jitter (deterministic per-index, cached in TLS)
            for (i, jt) in jitter_tmp.iter_mut().enumerate() {
                *jt = (((i * 1_234_567) % 1000) as f32 / 500.0) - 1.0;
            }

            // 1. Compute y_norm (elementwise normalize, branchless)
            for i in 0..n {
                y_norm_tmp[i] = (block_out[i] - mean_y_f32) / rms;
            }

            // 2. ASM batch EMA: z_next = alpha * y_norm + (1-alpha) * z
            unsafe {
                crate::asm::jepa_ou_avx2(
                    n,
                    z_buf_ptr,
                    y_norm_tmp.as_ptr(),
                    z_next_tmp.as_mut_ptr(),
                    alpha.to_bits(),
                );
            }

            // 3. Vectorized adaptive alpha correction + jitter + clamp
            unsafe {
                #[cfg(target_arch = "x86_64")]
                {
                    jepa_adaptive_alpha_correction_avx2(
                        n,
                        z_buf_ptr,
                        y_norm_tmp.as_ptr(),
                        z_next_tmp.as_ptr(),
                        jitter_tmp.as_ptr(),
                        alpha,
                        z_buf_ptr,
                        &mut sum_z,
                        &mut sum_z_sq,
                    );
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    for i in 0..n {
                        let z = *z_buf_ptr.add(i);
                        let diff = *y_norm_tmp.as_ptr().add(i) - z;
                        let delta = diff.abs();
                        let excess = (delta - 2.0).max(0.0);
                        let jepa_alpha = 0.01 + 0.09 * (excess / 5.0).min(1.0);
                        let jitter = *jitter_tmp.as_ptr().add(i);
                        let correction = (jepa_alpha - alpha) * diff + jitter * 1e-4;
                        let z_next =
                            (*z_next_tmp.as_ptr().add(i) + correction).clamp(-50_000.0, 50_000.0);
                        *z_buf_ptr.add(i) = z_next;
                        let zv = z as f64;
                        sum_z += zv;
                        sum_z_sq += zv * zv;
                    }
                }
            }
        });
    } // end if !gpu_ok (CPU path)

    // ── Update EMA statistics ────────────────────────────────────────────────
    let batch_mu_z = (sum_z / n) as f32;
    let z_var = ((sum_z_sq / n) - (batch_mu_z as f64 * batch_mu_z as f64)).max(0.0) as f32;

    if *var_ema == 0.0 {
        *mu_ctx = batch_mu_z;
        *var_ema = z_var.max(EPSILON_FLOOR);
    } else {
        *mu_ctx = 0.9 * (*mu_ctx) + 0.1 * batch_mu_z;
        *var_ema = 0.99 * (*var_ema) + 0.01 * z_var;
    }

    let raw_inv_sigma = 1.0 / ((*var_ema).sqrt() + EPSILON_FLOOR);
    let dynamic_limit = (n as f32).sqrt();
    *inv_sigma_ctx = dynamic_limit * (raw_inv_sigma / dynamic_limit).tanh();

    // ── Compute v_jepa and update integral in each register ──────────────────
    // I-controller: I[t] = 0.99·I[t-1] + 0.01·v_jepa[t]
    // At equilibrium (v_jepa=const), I → v_jepa (DC gain = 1).
    // Transient noise is low-pass filtered.
    // P-01: compute v_jepa once per register, reuse for both integral update
    // and tape write (was previously computed twice in separate loops).
    let inv_sigma = *inv_sigma_ctx;
    let mu = *mu_ctx;
    if let Some(ref mut t) = tape_out {
        for i in 0..registers.len() {
            let v_jepa = (z_buf[i] - mu) * inv_sigma;
            registers[i].update_integral(v_jepa);
            t[i] = v_jepa;
        }
    } else {
        for i in 0..registers.len() {
            let v_jepa = (z_buf[i] - mu) * inv_sigma;
            registers[i].update_integral(v_jepa);
        }
    }

    0.0 // spring force eliminated; mHC radius handles stabilization
}

// ── OPT-2: AVX2 vectorized adaptive alpha correction. ──────────────────────
//
// The scalar CPU path previously did per-element:
//   delta      = |y_norm[i] - z[i]|
//   excess     = max(delta - 2.0, 0.0)
//   jepa_alpha = 0.01 + 0.09 * min(excess / 5.0, 1.0)
//   jitter     = (((i * 1_234_567) % 1000) / 500.0) - 1.0   (deterministic, precomputed)
//   correction = (jepa_alpha - alpha) * (y_norm[i] - z[i]) + jitter * 1e-4
//   z_buf[i]   = clamp(z_next[i] + correction, -50000, 50000)
//
// This kernel processes 8 elements/iteration using FMA intrinsics. The
// jitter term (1e-4 scale) is included from the precomputed buffer.
// Statistics (sum_z, sum_z_sq) use the *old* z values, matching the scalar path.
//
// # Safety
// All pointers must be valid for `n` elements. `out_z` must not overlap
// `z_ptr`, `y_norm_ptr`, `z_next_ptr`, or `jitter_ptr`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn jepa_adaptive_alpha_correction_avx2(
    n: usize,
    z_ptr: *const f32,
    y_norm_ptr: *const f32,
    z_next_ptr: *const f32,
    jitter_ptr: *const f32,
    alpha: f32,
    out_z: *mut f32,
    sum_z: &mut f64,
    sum_z_sq: &mut f64,
) {
    use std::arch::x86_64::*;

    let alpha_vec = _mm256_set1_ps(alpha);
    let two_vec = _mm256_set1_ps(2.0);
    let five_vec = _mm256_set1_ps(5.0);
    let min_alpha = _mm256_set1_ps(0.01);
    let scale_alpha = _mm256_set1_ps(0.09);
    let one_vec = _mm256_set1_ps(1.0);
    let clamp_min = _mm256_set1_ps(-50_000.0);
    let clamp_max = _mm256_set1_ps(50_000.0);
    let jitter_scale = _mm256_set1_ps(1e-4);
    let zero_vec = _mm256_setzero_ps();

    let mut i = 0;
    let mut sum_z_local = 0.0f64;
    let mut sum_z_sq_local = 0.0f64;

    while i + 8 <= n {
        let z_vec = _mm256_loadu_ps(z_ptr.add(i));
        let y_vec = _mm256_loadu_ps(y_norm_ptr.add(i));

        // diff = y_norm - z
        let diff = _mm256_sub_ps(y_vec, z_vec);

        // delta = |diff|  (bitwise AND with fabs mask)
        let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        let delta = _mm256_and_ps(abs_mask, diff);

        // excess = max(delta - 2.0, 0.0)
        let excess = _mm256_max_ps(_mm256_sub_ps(delta, two_vec), zero_vec);

        // scaled = excess / 5.0; clamped = min(scaled, 1.0)
        let scaled = _mm256_div_ps(excess, five_vec);
        let clamped = _mm256_min_ps(scaled, one_vec);

        // jepa_alpha = 0.01 + 0.09 * clamped  (FMA)
        let jepa_alpha = _mm256_fmadd_ps(scale_alpha, clamped, min_alpha);

        // alpha_diff = jepa_alpha - alpha
        let alpha_diff = _mm256_sub_ps(jepa_alpha, alpha_vec);

        // correction = alpha_diff * diff
        let correction = _mm256_mul_ps(alpha_diff, diff);

        // Add jitter * 1e-4
        let jitter_vec = _mm256_loadu_ps(jitter_ptr.add(i));
        let jitter_term = _mm256_mul_ps(jitter_vec, jitter_scale);
        let correction = _mm256_add_ps(correction, jitter_term);

        // z_next_final = clamp(z_next + correction, -50000, 50000)
        let z_next_vec = _mm256_loadu_ps(z_next_ptr.add(i));
        let corrected = _mm256_add_ps(z_next_vec, correction);
        let clamped_val = _mm256_max_ps(clamp_min, _mm256_min_ps(clamp_max, corrected));

        // Write back to z_buf
        _mm256_storeu_ps(out_z.add(i), clamped_val);

        // Accumulate statistics from old z values
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), z_vec);
        for zv in tmp {
            let zv = zv as f64;
            sum_z_local += zv;
            sum_z_sq_local += zv * zv;
        }

        i += 8;
    }

    // Scalar remainder
    while i < n {
        let z = *z_ptr.add(i);
        let delta = (*y_norm_ptr.add(i) - z).abs();
        let excess = (delta - 2.0).max(0.0);
        let jepa_alpha = 0.01 + 0.09 * (excess / 5.0).min(1.0);
        let jitter = *jitter_ptr.add(i);
        let correction = (jepa_alpha - alpha) * (*y_norm_ptr.add(i) - z) + jitter * 1e-4;
        let z_next = (*z_next_ptr.add(i) + correction).clamp(-50_000.0, 50_000.0);
        *out_z.add(i) = z_next;
        let zv = z as f64;
        sum_z_local += zv;
        sum_z_sq_local += zv * zv;
        i += 1;
    }

    *sum_z += sum_z_local;
    *sum_z_sq += sum_z_sq_local;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mud::slime::SlimeRegister;

    #[test]
    fn test_jepa_convergence_equilibrium() {
        let mut registers = vec![SlimeRegister::default(); 128];
        let mut block_out = vec![5.0f32; 128];
        let mut mu: f32 = 5.0;
        let mut inv_sigma = 1.0;
        let mut var_ema = 1.0;
        let mut z_buf = vec![0.0f32; 128];

        // Start z far outside the gate radius to test neural kick
        for i in 0..128 {
            z_buf[i] = 15.0;
            registers[i].write_accum(5.0);
        }

        // Run JEPA for 500 steps
        for _ in 0..500 {
            jepa_stabilizer(
                &mut block_out,
                &mut registers,
                &mut mu,
                &mut inv_sigma,
                &mut var_ema,
                &mut z_buf,
                None,
                None,
            );
            for b in block_out.iter_mut() {
                *b = 5.0; // steady-state input
            }
        }

        let post_jepa_accum = block_out[0];
        let delta_y = (post_jepa_accum - mu).abs();
        println!(
            "post_jepa_accum={}, mu={}, delta_y={}",
            post_jepa_accum, mu, delta_y
        );
        assert!(delta_y < 10.0, "JEPA failed to stabilize output");
    }

    #[test]
    fn test_jepa_integral_update() {
        let mut registers = vec![SlimeRegister::default(); 4];
        let mut block_out = vec![1.0f32; 4];
        let mut mu: f32 = 0.0;
        let mut inv_sigma = 1.0;
        let mut var_ema = 1.0;
        let mut z_buf = vec![0.0f32; 4];

        jepa_stabilizer(
            &mut block_out,
            &mut registers,
            &mut mu,
            &mut inv_sigma,
            &mut var_ema,
            &mut z_buf,
            None,
            None,
        );

        // After one step, integral should be non-zero
        let i0 = registers[0].read_integral();
        assert!(i0.is_finite(), "Integral must be finite");
        // Gate should be near 0.5 for small v_jepa
        let gate = registers[0].gate();
        assert!(
            gate > 0.0 && gate < 1.0,
            "Gate must be in (0,1): got {gate}"
        );
    }

    #[test]
    fn test_jepa_zero_exp_performance() {
        let mut registers = vec![SlimeRegister::default(); 1];
        let mut block_out = vec![0.0f32; 1];
        let mut z_buf = vec![1000.0f32; 1];
        let mut mu: f32 = 0.0;
        let mut inv_sigma = 0.1;
        let mut var_ema = 100.0;

        jepa_stabilizer(
            &mut block_out,
            &mut registers,
            &mut mu,
            &mut inv_sigma,
            &mut var_ema,
            &mut z_buf,
            None,
            None,
        );

        let i = registers[0].read_integral();
        assert!(i.is_finite(), "JEPA integral must be finite");
        assert!(mu.is_finite(), "JEPA mu must be finite");
    }

    #[test]
    fn test_jepa_tape_writes_v_jepa_not_integral() {
        // tape_out should record v_jepa, not the integral (for backward pass)
        let mut registers = vec![SlimeRegister::default(); 8];
        let mut block_out = vec![3.0f32; 8];
        let mut mu = 0.0f32;
        let mut inv_sigma = 1.0;
        let mut var_ema = 1.0;
        let mut z_buf = vec![0.5f32; 8];
        let mut tape = vec![0.0f32; 8];

        jepa_stabilizer(
            &mut block_out,
            &mut registers,
            &mut mu,
            &mut inv_sigma,
            &mut var_ema,
            &mut z_buf,
            Some(&mut tape),
            None,
        );

        // tape should contain v_jepa = (z - mu) * inv_sigma
        // integral = 0.01 * v_jepa (one step from 0)
        let i0 = registers[0].read_integral();
        let t0 = tape[0];
        assert!(
            (i0 - t0).abs() < 0.01,
            "integral = v_jepa at t=1, got i0={i0} t0={t0}"
        );
    }
}
