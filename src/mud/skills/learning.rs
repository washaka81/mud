use crate::mud::skills::MudSkill;
use crate::mud::ingester::MudIngester;
use std::path::Path;

/// Implementation of the Autonomous Learning skill.
/// Automatically detects file paths in conversation and ingests them.
pub struct LearningSkill;

impl LearningSkill {
    pub fn new() -> Self {
        Self
    }
}

impl MudSkill for LearningSkill {
    fn name(&self) -> &str {
        "autonomous_learning"
    }

    fn should_activate(&self, _x: &[f32], context: &str) -> bool {
        // Detect common path patterns or keywords like "read", "load", "file"
        context.contains("/") || context.contains(".txt") || context.contains(".md") || context.contains("read file")
    }

    fn execute_autonomous_action(&self, context: &str, engine: &crate::mud::inference::MudInference) {
        // Simple regex-like path extraction
        for word in context.split_whitespace() {
            if word.contains("/") || word.ends_with(".txt") || word.ends_with(".md") {
                let clean_path = word.trim_matches(|c| c == '(' || c == ')' || c == ',' || c == '\"' || c == '\'');
                if Path::new(clean_path).exists() {
                    println!("  [MUD Auto-Action] Detected path: {}. Learning...", clean_path);
                    let _ = MudIngester::ingest(clean_path, engine);
                }
            }
        }
    }
}
