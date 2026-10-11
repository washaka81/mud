//! Per-chunk activation metrics + Welford-lite running mean.
//!
//! These structs are produced by [`crate::mud::corpus_trainer::train_step::train_on_sequence`]
//! and consumed by the alignment session for telemetry + circuit integrity.

/// Live activation metrics sampled **after forward, before backward** (post-BWD kills VarH/VarJ).
#[derive(Debug, Clone, Copy, Default)]
pub struct TrainChunkMetrics {
    pub loss: f32,
    pub var_h: f32,
    pub var_j: f32,
    pub jepa_integral: f32,
    pub sigma_v_pct: f32,
    pub cognitive: f32,
    pub n_samples: u32,
    pub grad_norm: f32,
    pub entropy: f32,
    pub stp_loss: f32,
}

/// Running mean of per-step activation stats (Welford-lite / online mean).
#[derive(Debug, Default)]
pub struct ActStatsAccum {
    n: u32,
    var_h: f64,
    var_j: f64,
    jepa_i: f64,
    sigma: f64,
    cog: f64,
    grad_norm: f64,
    entropy: f64,
    stp_loss: f64,
}

impl ActStatsAccum {
    pub fn push(&mut self, var_h: f32, var_j: f32, jepa_i: f32, sigma: f32, cog: f32) {
        self.n += 1;
        let n = self.n as f64;
        // online mean: m += (x - m) / n
        self.var_h += (var_h as f64 - self.var_h) / n;
        self.var_j += (var_j as f64 - self.var_j) / n;
        self.jepa_i += (jepa_i as f64 - self.jepa_i) / n;
        self.sigma += (sigma as f64 - self.sigma) / n;
        self.cog += (cog as f64 - self.cog) / n;
    }

    /// Push grad_norm + entropy for the current chunk.
    ///
    /// FIX (2026-08-03): This is called ONCE per chunk with already-averaged
    /// values (`entropy_sum / np`). The previous Welford formula divided by `n`
    /// again (n=64), producing entropy = 8.56/64 = 0.134 — a phantom 64×
    /// under-report that made healthy training look like entropy collapse.
    /// Use simple assignment instead.
    pub fn push_grad(&mut self, grad_norm: f32, entropy: f32, stp_loss: f32) {
        if self.n > 0 {
            self.grad_norm = grad_norm as f64;
            self.entropy = entropy as f64;
            self.stp_loss = stp_loss as f64;
        }
    }

    pub fn finish(self, loss: f32) -> TrainChunkMetrics {
        TrainChunkMetrics {
            loss,
            var_h: self.var_h as f32,
            var_j: self.var_j as f32,
            jepa_integral: self.jepa_i as f32,
            sigma_v_pct: self.sigma as f32,
            cognitive: self.cog as f32,
            n_samples: self.n,
            grad_norm: self.grad_norm as f32,
            entropy: self.entropy as f32,
            stp_loss: self.stp_loss as f32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accum_default_is_zero() {
        let a = ActStatsAccum::default();
        let m = a.finish(0.0);
        assert_eq!(m.n_samples, 0);
        assert_eq!(m.loss, 0.0);
    }

    #[test]
    fn test_accum_running_mean() {
        let mut a = ActStatsAccum::default();
        a.push(1.0, 2.0, 3.0, 4.0, 5.0);
        a.push(3.0, 4.0, 5.0, 6.0, 7.0);
        let m = a.finish(0.5);
        assert_eq!(m.n_samples, 2);
        // mean of (1,3) = 2.0
        assert!((m.var_h - 2.0).abs() < 1e-5);
        // mean of (2,4) = 3.0
        assert!((m.var_j - 3.0).abs() < 1e-5);
        // mean of (3,5) = 4.0
        assert!((m.jepa_integral - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_push_grad_no_double_division() {
        // push_grad receives already-averaged chunk values (not per-step).
        // It must store them directly, NOT apply Welford (which would divide
        // by n again). This is the fix for entropy=0.136 phantom collapse.
        let mut a = ActStatsAccum::default();
        for _ in 0..64 {
            a.push(7.0, 0.3, -0.01, 49.8, 590.0);
        }
        // Simulate train_on_sequence: entropy_sum = 63*8.7 + 0 = 548.1, np=64
        // chunk average = 548.1/64 = 8.563
        let avg_entropy = 548.1 / 64.0;
        a.push_grad(0.001, avg_entropy, 0.0);
        let m = a.finish(6.44);
        // MUST be 8.563, NOT 8.563/64 = 0.134
        assert!((m.entropy - 8.563).abs() < 0.01,
            "entropy was {} (double-division bug: 8.563/64=0.134)", m.entropy);
        assert!((m.grad_norm - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_train_chunk_metrics_default() {
        let m = TrainChunkMetrics::default();
        assert_eq!(m.loss, 0.0);
        assert_eq!(m.n_samples, 0);
    }
}
