use crate::ai::skills::MudSkill;

/// Implementation of a specialized Language skill (e.g., Spanish or English optimization).
pub struct LanguageSkill {
    pub lang_id: String,
}

impl LanguageSkill {
    pub fn new(lang_id: &str) -> Self {
        Self { lang_id: lang_id.to_string() }
    }
}

impl MudSkill for LanguageSkill {
    fn name(&self) -> &str {
        "language"
    }

    fn post_process_token(&self, _text: &mut String) {
        // Example: Language-specific formatting (e.g., inverted question marks for Spanish)
        if self.lang_id == "es" {
            // Logic to ensure correct Spanish punctuation
        }
    }
}
