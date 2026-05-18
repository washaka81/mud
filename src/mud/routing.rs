use std::cmp::Ordering;

/// Implements a Top-K router for MUD Mixture of Experts.
pub struct MudRouter {
    /// Number of available experts.
    pub num_experts: usize,
    /// Number of experts to activate per token.
    pub top_k: usize,
}

impl MudRouter {
    pub fn new(num_experts: usize, top_k: usize) -> Self {
        Self { num_experts, top_k }
    }

    /// Selects top-K experts based on gate logits.
    /// Returns a list of (expert_id, probability).
    pub fn route(&self, logits: &[f32]) -> Vec<(usize, f32)> {
        let mut indexed: Vec<(usize, f32)> = logits.iter()
            .enumerate()
            .map(|(i, &l)| (i, l))
            .collect();

        // Sort by logit value descending
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        // Take top K
        let top_k = &indexed[..self.top_k.min(self.num_experts)];
        
        // Softmax over top K
        let max_logit = top_k.iter().map(|&(_, l)| l).fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        let mut results: Vec<(usize, f32)> = top_k.iter().map(|&(i, l)| {
            let exp = (l - max_logit).exp();
            sum_exp += exp;
            (i, exp)
        }).collect();

        for (_, p) in results.iter_mut() {
            *p /= sum_exp;
        }

        results
    }
}
