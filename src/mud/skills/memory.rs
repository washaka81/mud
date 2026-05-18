use crate::mud::skills::MudSkill;
use std::fs::OpenOptions;
use std::io::Write;

/// Implementation of the Long-Term Memory skill.
/// Saves and retrieves conversation history using the Knowledge Index.
pub struct MemorySkill {
    pub history_file: String,
}

impl MemorySkill {
    pub fn new() -> Self {
        Self { history_file: "models/conversation_history.txt".to_string() }
    }

    /// Appends a user/assistant pair to the local history file.
    pub fn save_interaction(&self, user: &str, assistant: &str) {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.history_file)
            .unwrap();
        
        let _ = writeln!(file, "User: {}\nMUD: {}\n---", user, assistant);
    }
}

impl MudSkill for MemorySkill {
    fn name(&self) -> &str {
        "memory"
    }

    fn pre_process(&self, _x: &mut [f32]) {
        // Here we could load past history into the index automatically
    }

    fn route_bias(&self, logits: &mut [f32]) {
        // Bias towards experts that handle context and narrative
        if logits.len() > 7 {
            logits[6] += 1.5;
            logits[7] += 1.5;
        }
    }
}
