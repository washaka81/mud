use crate::ai::skills::MudSkill;

/// Implementation of the Personality skill.
/// Shapes the model's tone, style, and identity.
pub struct PersonalitySkill {
    pub identity: String,
}

impl PersonalitySkill {
    pub fn new(identity: &str) -> Self {
        Self { identity: identity.to_string() }
    }
}

impl MudSkill for PersonalitySkill {
    fn name(&self) -> &str {
        "personality"
    }

    fn post_process_token(&self, _text: &mut String) {
        // Example: Subtle adjustments to maintain identity
        // If the identity is "Senior Dev", ensure tone is professional.
    }

    fn route_bias(&self, _logits: &mut [f32]) {
        // Personality can influence routing to "creative" vs "logical" experts
    }
}
