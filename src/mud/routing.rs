use std::cmp::Ordering;

/// Implements a Dynamic Router for MUD Mixture of Experts.
pub struct MudRouter {
    /// Number of available experts.
    pub num_experts: usize,
    /// Maximum number of experts to activate per token (Top-K).
    pub max_k: usize,
    /// Minimum probability threshold for an expert to be considered "active".
    pub threshold: f32,
}

impl MudRouter {
    pub fn new(num_experts: usize, max_k: usize) -> Self {
        Self {
            num_experts,
            max_k,
            threshold: 0.1, // Experts with < 10% contribution are deactivated
        }
    }

    pub fn route_in_place(
        &self,
        logits: &[f32],
        indexed: &mut Vec<(usize, f32)>,
        results: &mut Vec<(usize, f32)>,
        dynamic_max_k: Option<usize>,
    ) -> f32 {
        debug_assert_eq!(
            logits.len(),
            self.num_experts,
            "route: logits.len()={} != num_experts={}",
            logits.len(),
            self.num_experts
        );

        results.clear();
        indexed.clear();

        if logits.is_empty() {
            return 0.0;
        }

        // Calculate z-loss = (log Σ exp(logits))^2
        let max_all = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp_all: f32 = logits.iter().map(|&l| (l - max_all).exp()).sum();
        let z_loss = if sum_exp_all > 0.0 {
            (max_all + sum_exp_all.ln()).powi(2)
        } else {
            0.0
        };

        for (i, &l) in logits.iter().enumerate() {
            indexed.push((i, l));
        }

        // Sort by logit value descending
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        // 1. Take Top-K candidates (dynamic override if set)
        let k = dynamic_max_k.unwrap_or(self.max_k).min(indexed.len());
        let candidates = &indexed[..k];

        // 2. Softmax over candidates
        let max_logit = candidates
            .iter()
            .map(|&(_, l)| l)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for &(i, l) in candidates.iter() {
            let exp = (l - max_logit).exp();
            sum_exp += exp;
            results.push((i, exp));
        }

        // Guard against division by zero
        if sum_exp == 0.0 || !sum_exp.is_finite() {
            results.clear();
            results.push((indexed[0].0, 1.0));
            return z_loss;
        }

        for p in results.iter_mut() {
            p.1 /= sum_exp;
        }

        // 3. Filter by threshold
        results.retain(|&(_, p)| p >= self.threshold);

        // 4. Re-normalize
        if !results.is_empty() {
            let new_sum: f32 = results.iter().map(|&(_, p)| p).sum();
            if new_sum > 0.0 && new_sum.is_finite() {
                for p in results.iter_mut() {
                    p.1 /= new_sum;
                }
            }
        } else {
            results.push((indexed[0].0, 1.0));
        }
        z_loss
    }

    /// Q-Head Routing (GRAM) [BIT-02 Roadmap]
    /// Injects stochastic Gumbel noise into the gating mechanism to explore probabilistic paths
    /// and break deterministic "Single Attractor" loops.
    pub fn route_by_q_head(
        &self,
        logits: &[f32],
        temperature: f32,
        seed: u32,
        indexed: &mut Vec<(usize, f32)>,
        results: &mut Vec<(usize, f32)>,
    ) -> f32 {
        debug_assert_eq!(
            logits.len(),
            self.num_experts,
            "route: logits.len()={} != num_experts={}",
            logits.len(),
            self.num_experts
        );

        results.clear();
        indexed.clear();

        if logits.is_empty() {
            return 0.0;
        }

        // Add stochastic Q-Head noise (Pseudo-Gumbel distribution)
        for (i, &l) in logits.iter().enumerate() {
            // Pseudo-random generator without allocations
            let mut state = seed
                .wrapping_add(i as u32)
                .wrapping_mul(747796405)
                .wrapping_add(2891336453);
            state = (state >> ((state >> 28).wrapping_add(4))) ^ state.wrapping_mul(277803737);
            state = (state >> 22) ^ state;

            // Map to (0, 1]
            let u = (state as f32) / (u32::MAX as f32);
            let u = u.clamp(1e-6, 0.999999);

            // Gumbel noise: -ln(-ln(u))
            let gumbel = -(-(u.ln())).ln();

            let noisy_l = l + (gumbel * temperature);
            indexed.push((i, noisy_l));
        }

        // Calculate z-loss using indexed values to avoid allocating a noisy_logits vector
        let max_all = indexed
            .iter()
            .map(|&(_, l)| l)
            .fold(f32::NEG_INFINITY, f32::max);
        let sum_exp_all: f32 = indexed.iter().map(|&(_, l)| (l - max_all).exp()).sum();
        let z_loss = if sum_exp_all > 0.0 {
            (max_all + sum_exp_all.ln()).powi(2)
        } else {
            0.0
        };

        // Sort by noisy logit value descending
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        // 1. Take Top-K candidates
        let k = self.max_k.min(indexed.len());
        let candidates = &indexed[..k];

        // 2. Softmax over candidates
        let max_logit = candidates
            .iter()
            .map(|&(_, l)| l)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for &(i, l) in candidates.iter() {
            let exp = (l - max_logit).exp();
            sum_exp += exp;
            results.push((i, exp));
        }

        // Guard against division by zero
        if sum_exp == 0.0 || !sum_exp.is_finite() {
            results.clear();
            results.push((indexed[0].0, 1.0));
            return z_loss;
        }

        for p in results.iter_mut() {
            p.1 /= sum_exp;
        }

        // 3. Filter by threshold
        results.retain(|&(_, p)| p >= self.threshold);

        // 4. Re-normalize
        if !results.is_empty() {
            let new_sum: f32 = results.iter().map(|&(_, p)| p).sum();
            if new_sum > 0.0 && new_sum.is_finite() {
                for p in results.iter_mut() {
                    p.1 /= new_sum;
                }
            }
        } else {
            results.push((indexed[0].0, 1.0));
        }

        z_loss
    }

    /// Hash Routing MoE [2106.04426]
    /// Deterministic, router-free expert selection based on a hash of the hidden state.
    /// This eliminates the router MLP overhead and the need for z-loss/aux-loss.
    /// O(1) allocation, zero-parameter routing.
    pub fn route_by_hash(&self, hidden_state: &[f32], results: &mut Vec<(usize, f32)>) {
        results.clear();

        if hidden_state.is_empty() || self.num_experts == 0 {
            return;
        }

        // We use a simple structural hash of the continuous hidden state.
        // Hashing the first 32 dimensions provides sufficient deterministic entropy.
        let mut hash_val = 0u64;
        let stride = (hidden_state.len() / 32).max(1);
        for i in 0..32.min(hidden_state.len()) {
            let v = hidden_state[i * stride];
            // Convert float to bits to hash deterministically
            hash_val = hash_val
                .wrapping_mul(1099511628211)
                .wrapping_add(v.to_bits() as u64);
        }

        // Select Top-K experts deterministically based on sequential hashes.
        // Clamp to num_experts so probabilities remain normalized when max_k > num_experts.
        let requested = self.max_k.min(self.num_experts);
        if requested == 0 {
            return;
        }

        let max_retries = requested
            .saturating_mul(self.num_experts)
            .saturating_mul(2)
            .max(1);
        let mut retries = 0;
        while results.len() < requested && retries < max_retries {
            let expert_idx = (hash_val as usize) % self.num_experts;

            if !results.iter().any(|&(idx, _)| idx == expert_idx) {
                let prob = 1.0 / (requested as f32);
                results.push((expert_idx, prob));
            }

            // Perturb hash for the next attempt
            hash_val = hash_val
                .wrapping_mul(1099511628211)
                .wrapping_add(1013904223);
            retries += 1;
        }

        let probs_sum: f32 = results.iter().map(|&(_, p)| p).sum();
        if results.is_empty() {
            results.push((0, 1.0));
        } else if probs_sum > 0.0 && probs_sum.is_finite() {
            for p in results.iter_mut() {
                p.1 /= probs_sum;
            }
        } else {
            results.clear();
            results.push((0, 1.0));
        }
    }

    /// Evaluates the thermodynamic certainty (Entropy) of the router's current decision.
    /// If the certainty is low, the LDT (Lattice-based Deduction) system should
    /// force the model to loop and re-evaluate the hidden states (Slow Thinking).
    /// Returns `true` if the model is 100% certain (or above threshold) and can proceed.
    pub fn evaluate_ldt_certainty(&self, probabilities: &[(usize, f32)]) -> bool {
        if probabilities.len() <= 1 {
            return true; // No ambiguity
        }

        let mut entropy = 0.0f32;
        for &(_, p) in probabilities {
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }

        // If entropy exceeds the thermodynamic limit, the network is "guessing"
        // between multiple experts (high uncertainty). Typical Slow Thinking threshold: 0.35
        let certainty_score = 1.0 - (entropy / (probabilities.len() as f32).ln());

        // Require sufficient certainty (> 80%) to stop recursive thinking.
        // Audit V10: Lowered from 95% to 80% to allow for ternary settling.
        certainty_score > 0.80
    }
}
