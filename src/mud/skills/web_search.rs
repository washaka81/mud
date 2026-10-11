use crate::mud::inference::MudInference;
use crate::mud::skills::MudSkill;

/// Implementation of the Web Search & Ingestion skill (Legacy - Database system removed).
pub struct WebSearchSkill;

impl Default for WebSearchSkill {
    fn default() -> Self {
        Self::new()
    }
}

impl WebSearchSkill {
    pub fn new() -> Self {
        Self
    }
}

impl MudSkill for WebSearchSkill {
    fn name(&self) -> &str {
        "web_search"
    }

    fn should_activate(&self, _x: &[f32], context: &str) -> bool {
        let keywords = [
            "search",
            "duckduckgo",
            "web",
            "investiga",
            "who is",
            "what is the latest",
            "http",
            "www",
        ];
        keywords.iter().any(|&k| context.to_lowercase().contains(k))
    }

    fn execute_autonomous_action(&self, context: &str, _engine: &MudInference) {
        // Web search action disabled with database removal
        for word in context.split_whitespace() {
            if word.starts_with("http") {
                println!("  [MUD Auto-Action] Researching URL: {}... (Action disabled: No DB)", word);
            }
        }
    }
}
