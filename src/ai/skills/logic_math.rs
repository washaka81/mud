use crate::ai::skills::MudSkill;

/// Implementation of the Logic and Mathematics skill.
/// Biases the router towards experts trained on formal reasoning.
pub struct LogicMathSkill;

impl LogicMathSkill {
    pub fn new() -> Self {
        Self
    }
}

impl MudSkill for LogicMathSkill {
    fn name(&self) -> &str {
        "logic_math"
    }

    fn should_activate(&self, _x: &[f32], context: &str) -> bool {
        let math_keywords = ["sum", "sqrt", "solve", "math", "calculate", "+", "-", "*", "/", "="];
        context.chars().any(|c| c.is_digit(10)) || math_keywords.iter().any(|&k| context.contains(k))
    }

    fn route_bias(&self, logits: &mut [f32]) {
        // Favor Experts 2 and 3 for logic/math tasks
        if logits.len() > 3 {
            logits[2] += 5.0;
            logits[3] += 5.0;
        }
    }

    fn pre_process(&self, _x: &mut [f32]) {
        // Logic skill could potentially adjust temperature or precision
    }
}
