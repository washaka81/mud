use crate::mud::skills::MudSkill;

/// Implementation of the Logic and Mathematics skill.
/// Biases the router towards experts trained on formal reasoning.
pub struct LogicMathSkill;

impl Default for LogicMathSkill {
    fn default() -> Self {
        Self::new()
    }
}

impl LogicMathSkill {
    pub fn new() -> Self {
        Self
    }
}

impl MudSkill for LogicMathSkill {
    fn name(&self) -> &str {
        "logic_math"
    }

    fn should_activate(&self, _x: &[f32], _context: &str) -> bool {
        // Enrutamiento natural: el modelo MoE debe aprender a activar esta skill
        // basándose únicamente en sus pesos y entrenamiento (Emergencia Cognitiva).
        true
    }

    fn route_bias(&self, _logits: &mut [f32]) {
        // Removed artificial bias. Routing should be learned by the model.
    }

    fn execute_autonomous_action(&self, context: &str, _engine: &crate::mud::inference::MudInference) {
        // Router de Delegación Matemática
        // Detects if a formula is present and attempts to evaluate it
        if context.contains('+') || context.contains('-') || context.contains('*') || context.contains('/') || context.contains('^') {
            println!("  [LogicMath] 🧠 Mathematical intent detected. Delegating to external sandbox...");
            
            // Extract potential expression (simplistic extraction for demonstration)
            let words: Vec<&str> = context.split_whitespace().collect();
            let mut expr = String::new();
            for word in words {
                if word.chars().all(|c| c.is_ascii_digit() || "+-*/^().".contains(c)) {
                    expr.push_str(word);
                }
            }

            if !expr.is_empty() {
                // Replace standard math symbols with Python ones
                let python_expr = expr.replace('^', "**");
                
                let output = std::process::Command::new("python3")
                    .arg("tools/math_sandbox.py")
                    .arg(&python_expr)
                    .output();

                match output {
                    Ok(out) => {
                        let result_str = String::from_utf8_lossy(&out.stdout);
                        if result_str.starts_with("SUCCESS:") {
                            let answer = result_str.replace("SUCCESS:", "").trim().to_string();
                            println!("  [Sandbox] Exact calculation result: {} = {}", python_expr, answer);
                            // TODO: Inject this exact answer into the inference stream context
                        } else {
                            println!("  [Sandbox] Parsing failed or expression invalid: {}", String::from_utf8_lossy(&out.stderr));
                        }
                    },
                    Err(e) => println!("  [Sandbox Error] Failed to invoke Python runtime: {}", e),
                }
            }
        }
    }

    fn post_process_token(&self, _text: &mut String) {
        // Potential for real-time validation of mathematical syntax
    }
}
