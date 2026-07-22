use crate::mud::skills::MudSkill;

/// Implementation of the Logic and Mathematics skill.
/// Biases the router towards experts trained on formal reasoning.
pub struct LogicMathSkill;

impl Default for LogicMathSkill {
    fn default() -> Self {
        Self::new()
    }
}

impl LogicMathSkill {
    pub fn new() -> Self {
        Self
    }
}

impl MudSkill for LogicMathSkill {
    fn name(&self) -> &str {
        "logic_math"
    }

    fn should_activate(&self, _x: &[f32], _context: &str) -> bool {
        // Enrutamiento natural: el modelo MoE debe aprender a activar esta skill
        // basándose únicamente en sus pesos y entrenamiento (Emergencia Cognitiva).
        true
    }

    fn route_bias(&self, _logits: &mut [f32]) {
        // Removed artificial bias. Routing should be learned by the model.
    }

    fn execute_autonomous_action(
        &self,
        _context: &str,
        _engine: &crate::mud::inference::MudInference,
    ) {
        // Delegated to native inference stream. No external Python sandboxing allowed (P-07).
    }

    fn post_process_token(&self, _text: &mut String) {
        // Potential for real-time validation of mathematical syntax
    }
}
