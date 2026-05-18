use crate::mud::skills::MudSkill;
use std::sync::RwLock;

/// Implementation of the Retrieval (RAG) skill.
/// Connects the activation stream to the Knowledge Index.
pub struct RetrievalSkill {
    /// Stores the last retrieved fact to inform the engine.
    pub last_fact: RwLock<Option<String>>,
}

impl RetrievalSkill {
    pub fn new() -> Self {
        Self { last_fact: RwLock::new(None) }
    }
}

impl MudSkill for RetrievalSkill {
    fn name(&self) -> &str {
        "retrieval"
    }

    fn should_activate(&self, _x: &[f32], context: &str) -> bool {
        let retrieval_keywords = ["what", "who", "where", "when", "tell me", "info", "fact"];
        context.trim().ends_with('?') || retrieval_keywords.iter().any(|&k| context.to_lowercase().contains(k))
    }

    fn pre_process(&self, x: &mut [f32]) {
        // If we have a retrieved fact, we "inject" its influence
        // into the activation vector. (Simplified: adding a small signature bias)
        let fact_guard = self.last_fact.read().unwrap();
        if fact_guard.is_some() {
            // Signal to the model that context is available
            if x.len() > 0 { x[0] += 1.0; } 
        }
    }

    fn post_process_token(&self, text: &mut String) {
        // If the output text indicates a need for facts, we could trigger a deeper search
        if text.contains("MUD fact:") {
            // Logic to update self.last_fact based on search
        }
    }
}
