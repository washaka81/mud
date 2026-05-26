/// Definition of a MUD Modular Skill.
/// A Skill is an intrinsic capability that can influence the engine's 
/// routing, processing, and output.
pub trait MudSkill: Send + Sync {
    /// Name of the skill (e.g., "autoformatter").
    fn name(&self) -> &str;

    /// Opportunity to modify activations before the forward pass.
    fn pre_process(&self, _x: &mut [f32]) {}

    /// Injects bias into the MoE router's logits.
    /// Used to favor specific experts for this skill.
    fn route_bias(&self, _logits: &mut [f32]) {}

    /// Processes the generated text token before it is displayed.
    /// Useful for real-time formatting.
    fn post_process_token(&self, _text: &mut String) {}

    /// Dynamically sets a parameter for the skill.
    fn set_param(&self, _key: &str, _value: &str) {}

    /// Determines if the skill should be active for the current context.
    fn should_activate(&self, _x: &[f32], _context: &str) -> bool { true }

    /// Executes an autonomous action based on detected intent (e.g., file ingestion).
    fn execute_autonomous_action(&self, _context: &str, _engine: &crate::mud::inference::MudInference) {}
}

pub mod autoformatter;
pub mod logic_math;
pub mod retrieval;
pub mod language;
pub mod personality;
pub mod translator;
pub mod memory;
pub mod learning;
pub mod data_analysis;
pub mod plotting;
pub mod web_search;
pub mod code_formatter;
pub mod logic_marks;
pub mod text_styling;
pub mod coding;
