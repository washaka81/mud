use serde::{de::DeserializeOwned, Serialize};

/// DECL-01: Rust-Native DSPy Runtime
/// Transition from raw prompting to Declarative Signatures.
/// This allows the engine to enforce output schema and autonomously select experts
/// based on the signature type.
pub trait DeclarativeSignature: Serialize + DeserializeOwned {
    /// The system prompt defining the persona and constraints for this signature.
    fn system_prompt() -> &'static str;
    
    /// Instructions for the reasoning process
    fn instructions() -> &'static str;

    /// The name of the specific skill/expert to activate when evaluating this signature.
    /// By default, it returns None, meaning general reasoning.
    fn required_expert() -> Option<&'static str> {
        None
    }

    /// Serializes the input struct into the MUD Prompt format.
    fn to_prompt(&self) -> String {
        let json_input = serde_json::to_string_pretty(self).unwrap_or_default();
        format!(
            "{} \n\nINSTRUCTIONS: {}\n\nINPUT:\n{}\n\nOUTPUT (Must be valid JSON matching the schema):\n",
            Self::system_prompt(),
            Self::instructions(),
            json_input
        )
    }

    /// Parses the model's textual output back into a strong Rust struct.
    fn parse_response(response: &str) -> Option<Self> {
        // Simple extraction: find the first { and the last }
        let start = response.find('{')?;
        let end = response.rfind('}')?;
        let json_str = &response[start..=end];
        serde_json::from_str(json_str).ok()
    }
}
