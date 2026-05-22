/// Skill de análisis lógico para marcar y validar cadenas de pensamiento (CoT).
pub struct LogicMarkSkill;

impl crate::mud::skills::MudSkill for LogicMarkSkill {
    fn name(&self) -> &str { "logic_marks" }
}
