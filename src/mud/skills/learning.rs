use crate::mud::skills::MudSkill;

/// Implementation of the Autonomous Learning skill (Legacy - Database system removed).
pub struct LearningSkill;

impl Default for LearningSkill {
    fn default() -> Self {
        Self::new()
    }
}

impl LearningSkill {
    pub fn new() -> Self {
        Self
    }
}

impl MudSkill for LearningSkill {
    fn name(&self) -> &str {
        "autonomous_learning"
    }

    fn should_activate(&self, _x: &[f32], _context: &str) -> bool {
        false
    }

    fn execute_autonomous_action(
        &self,
        _context: &str,
        _engine: &crate::mud::inference::MudInference,
    ) {
        // Learning system disabled with database removal
    }
}
