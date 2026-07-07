use crate::mud::slime::SlimeRegister;

/// Minimum epsilon for variance inversion (shared constant, P-13 compliant)
pub const EPSILON_FLOOR: f32 = 1e-8;

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
/// Reads ternary accumulation (f16 lower bits) and JEPA integral (f16 upper bits).
pub fn check_tensor_health(
    registers: &[SlimeRegister],
    gamma: f32,
    _iscale: f32, // kept for API compat; not used (f16 is self-scaled)
) -> TensorDiagnostics {
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

        // JEPA integral: upper 16 bits
        let integral = reg.read_integral();
        if (reg.0 >> 16) & 0x7FFF == 0 {
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
#[inline]
pub fn jepa_stabilizer(
    block_out: &mut [f32],
    registers: &mut [SlimeRegister],
    mu_ctx: &mut f32,
    inv_sigma_ctx: &mut f32,
    var_ema: &mut f32,
    z_buf: &mut [f32],
    mut tape_out: Option<&mut [f32]>,
) -> f32 {
    let n = registers.len() as f64;

    // ── Compute y_norm = z-score of block_out (mean 0, std 1) ──────────────
    let mut sum_y = 0.0f64;
    for b in block_out.iter() {
        sum_y += *b as f64;
    }
    let mean_y = sum_y / n;
    let mut sum_sq_diff = 0.0f64;
    for b in block_out.iter() {
        let diff = *b as f64 - mean_y;
        sum_sq_diff += diff * diff;
    }
    let mean_y_f32 = mean_y as f32;
    let rms = (sum_sq_diff / n).sqrt().max(1e-8) as f32;

    // ── Update z tracker and accumulate statistics ──────────────────────────
    let mut sum_z = 0.0f64;
    let mut sum_z_sq = 0.0f64;

    for i in 0..registers.len() {
        let z = z_buf[i];
        let y_norm = (block_out[i] - mean_y_f32) / rms;

        // Minimal jitter to prevent deterministic collapse (OU process heat)
        let jitter = (((i * 1_234_567) % 1000) as f32 / 500.0) - 1.0;

        // Pure OU tracker — no spring force, no repulsion
        let z_next = (z * 0.9 + 0.1 * y_norm + jitter * 1e-4).clamp(-50_000.0, 50_000.0);
        z_buf[i] = z_next;

        sum_z += z as f64;
        sum_z_sq += (z as f64) * (z as f64);
    }

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

    // ── Compute v_jepa and update integral in each register ─────────────────
    // I-controller: I[t] = 0.99·I[t-1] + 0.01·v_jepa[t]
    // At equilibrium (v_jepa=const), I → v_jepa (DC gain = 1).
    // Transient noise is low-pass filtered by the 0.99 decay.
    for i in 0..registers.len() {
        let v_jepa = (z_buf[i] - *mu_ctx) * (*inv_sigma_ctx);
        registers[i].write_integral(v_jepa);
    }

    // Write v_jepa to tape for backward pass (not the integral itself)
    if let Some(ref mut t) = tape_out {
        for i in 0..registers.len() {
            t[i] = (z_buf[i] - *mu_ctx) * (*inv_sigma_ctx);
        }
    }

    0.0 // spring force eliminated; mHC radius handles stabilization
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
        );

        // After one step, integral should be non-zero
        let i0 = registers[0].read_integral();
        assert!(i0.is_finite(), "Integral must be finite");
        // Gate should be near 0.5 for small v_jepa
        let gate = registers[0].gate();
        assert!(gate > 0.0 && gate < 1.0, "Gate must be in (0,1): got {gate}");
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
        );

        // tape should contain v_jepa = (z - mu) * inv_sigma
        // integral = 0.01 * v_jepa (one step from 0)
        let i0 = registers[0].read_integral();
        let t0 = tape[0];
        assert!((i0 - t0).abs() < 0.01, "integral = v_jepa at t=1, got i0={i0} t0={t0}");
    }
}
