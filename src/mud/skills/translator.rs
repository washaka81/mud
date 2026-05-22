use crate::mud::skills::MudSkill;
use std::sync::RwLock;

/// Implementation of the Multilingual Translation skill.
/// Facilitates context-aware translation and cross-lingual reasoning.
pub struct TranslationSkill {
    /// Target language (e.g., "en", "es").
    pub target_lang: RwLock<String>,
}

impl TranslationSkill {
    pub fn new(default_lang: &str) -> Self {
        Self {
            target_lang: RwLock::new(default_lang.to_string()),
        }
    }

    pub fn set_target(&self, lang: &str) {
        let mut target = self.target_lang.write().unwrap();
        *target = lang.to_string();
    }
}

impl MudSkill for TranslationSkill {
    fn name(&self) -> &str {
        "translator"
    }

    fn should_activate(&self, _x: &[f32], context: &str) -> bool {
        let translation_keywords = ["translate", "to english", "en español", "to spanish", "language"];
        translation_keywords.iter().any(|&k| context.to_lowercase().contains(k))
    }

    fn route_bias(&self, _logits: &mut [f32]) {
        // Translation experts should be routed organically.
    }

    fn post_process_token(&self, _text: &mut String) {
        // If the target language is different from detected, 
        // we could apply specific grammatical rules here.
        let target = self.target_lang.read().unwrap();
        if *target == "en" {
            // English-specific post-processing (placeholders)
        } else if *target == "es" {
            // Spanish-specific post-processing
        }
    }

    fn set_param(&self, key: &str, value: &str) {
        if key == "target_lang" {
            self.set_target(value);
        }
    }
}
