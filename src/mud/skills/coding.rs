use crate::mud::skills::MudSkill;
use crate::mud::inference::MudInference;

pub struct CodingExpert;

impl MudSkill for CodingExpert {
    fn name(&self) -> &str {
        "CodingExpert"
    }

    fn should_activate(&self, _x: &[f32], context: &str) -> bool {
        let p = context.to_lowercase();
        p.contains("código") || p.contains("code") || 
        p.contains("rust") || p.contains("python") || 
        p.contains("sql") || p.contains("script") ||
        p.contains("programar") || p.contains("función")
    }

    fn execute_autonomous_action(&self, _context: &str, _engine: &MudInference) {
        println!("  [Skill::CodingExpert] Analizando contexto de programación...");
        // En este hook, la skill puede preparar prompts restrictivos 
        // o inyectar sesgos en el router del MoE para forzar que los "Coding Experts" de MUD
        // (ej. Expertos entrenados en The Stack) se activen.
    }
    
    // Inyecta sesgo para forzar la activación de los expertos de código
    fn route_bias(&self, logits: &mut [f32]) {
        // Asumiendo que el experto 4 y 5 son los entrenados en código
        if logits.len() > 5 {
            logits[4] += 5.0; // Boost masivo al experto de Rust/C++
            logits[5] += 5.0; // Boost masivo al experto de Python/SQL
        }
    }
}
