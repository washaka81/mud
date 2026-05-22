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

    /// Selects experts dynamically based on gate logits and threshold.
    /// Returns a list of (expert_id, probability).
    /// INVARIANTE: logits.len() == self.num_experts; expert_id siempre < num_experts.
    pub fn route(&self, logits: &[f32]) -> Vec<(usize, f32)> {
        // Guarda de overflow: si el gate produce un vector de longitud incorrecta, truncar
        // en lugar de indexar fuera de rango.
        debug_assert_eq!(logits.len(), self.num_experts,
            "route: logits.len()={} != num_experts={}", logits.len(), self.num_experts);

        if logits.is_empty() { return vec![]; }

        let mut indexed: Vec<(usize, f32)> = logits.iter()
            .enumerate()
            .map(|(i, &l)| (i, l))
            .collect();

        // Sort by logit value descending
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        // 1. Take Top-K candidates (min evita ir más allá del vector)
        let k = self.max_k.min(indexed.len());
        let candidates = &indexed[..k];
        
        // 2. Softmax over candidates
        let max_logit = candidates.iter().map(|&(_, l)| l).fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        let mut results: Vec<(usize, f32)> = candidates.iter().map(|&(i, l)| {
            let exp = (l - max_logit).exp();
            sum_exp += exp;
            (i, exp)
        }).collect();

        // Guarda de división por cero en la normalización
        if sum_exp == 0.0 || !sum_exp.is_finite() {
            return vec![(indexed[0].0, 1.0)];
        }

        for (_, p) in results.iter_mut() {
            *p /= sum_exp;
        }

        // 3. Filter by threshold (Dynamic Activation)
        results.retain(|&(_, p)| p >= self.threshold);

        // 4. Re-normalize after filtering to maintain energy
        if !results.is_empty() {
            let new_sum: f32 = results.iter().map(|&(_, p)| p).sum();
            if new_sum > 0.0 && new_sum.is_finite() {
                for (_, p) in results.iter_mut() {
                    *p /= new_sum;
                }
            }
        } else {
            // Fallback: tomar el único mejor experto si ninguno superó el umbral
            // indexed[0] siempre existe porque la guarda de empty está al inicio
            results = vec![(indexed[0].0, 1.0)];
        }

        results
    }
}
