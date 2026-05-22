/// Skill para aplicar estilos ANSI y color al texto en terminales compatibles.
pub struct TextStylingSkill;

impl crate::mud::skills::MudSkill for TextStylingSkill {
    fn name(&self) -> &str { "text_styling" }
}
