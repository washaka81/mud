use crate::mud::skills::MudSkill;
use std::sync::RwLock;

/// Implementation of the Text Autoformatter skill.
/// Cleans up generated text in real-time.
pub struct AutoformatterSkill {
    /// Internal state machine for streaming formatting.
    state: RwLock<FormatterState>,
}

struct FormatterState {
    last_char: Option<char>,
    is_start_of_sentence: bool,
}

impl AutoformatterSkill {
    pub fn new() -> Self {
        Self {
            state: RwLock::new(FormatterState {
                last_char: None,
                is_start_of_sentence: true,
            }),
        }
    }
}

impl MudSkill for AutoformatterSkill {
    fn name(&self) -> &str {
        "autoformatter"
    }

    fn route_bias(&self, logits: &mut [f32]) {
        // Example: Favor Experts 0 and 1 if they are "grammar experts"
        if logits.len() > 1 {
            logits[0] += 2.0;
            logits[1] += 2.0;
        }
    }

    fn post_process_token(&self, text: &mut String) {
        let mut state = self.state.write().unwrap();
        let mut result = String::with_capacity(text.len());

        for c in text.chars() {
            // 1. Avoid multiple spaces
            if c == ' ' && state.last_char == Some(' ') {
                continue;
            }

            // 2. Capitalize start of sentence
            let mut final_char = c;
            if state.is_start_of_sentence && c.is_alphabetic() {
                final_char = c.to_uppercase().next().unwrap_or(c);
                state.is_start_of_sentence = false;
            }

            // 3. Detect end of sentence
            if c == '.' || c == '!' || c == '?' {
                state.is_start_of_sentence = true;
            }

            result.push(final_char);
            state.last_char = Some(final_char);
        }

        // 4. Global replacements (example)
        let processed = result.replace("mud", "MUD").replace("forge llm", "Forge LLM");
        
        *text = processed;
    }
}
