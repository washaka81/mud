/// Skill para formateo de código profesional y resaltado de sintaxis basado en estándares OpenCode.
pub struct CodeFormatSkill;

impl crate::mud::skills::MudSkill for CodeFormatSkill {
    fn name(&self) -> &str { "code_formatter" }
}
