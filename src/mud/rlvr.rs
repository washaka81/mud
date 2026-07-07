use std::process::Command;
use std::fs;

/// Environment-Based Critic for Reinforcement Learning from Verifiable Rewards (RLVR)
/// Executes generated code in a sandboxed/temporary environment to obtain
/// an objective reward signal (+1.0 for success, -1.0 for compilation/syntax failure)
/// and the associated error log for Self-Correction (SCoRe).
#[derive(Default)]
pub struct RlvrCritic {
    // We could store configuration for sandbox execution here.
}

impl RlvrCritic {
    pub fn new() -> Self {
        Self::default()
    }

    /// Evaluates a block of Rust code by attempting to compile it.
    /// Returns (reward, error_log).
    pub fn evaluate_rust_code(&self, code: &str) -> (f32, String) {
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("mud_rlvr_eval.rs");
        
        // Write code to a temporary file
        if fs::write(&file_path, code).is_err() {
            return (-1.0, "System Error: Failed to write to temp file".to_string());
        }

        // Run rustc --no-codegen to only check syntax/types
        // This is extremely fast and serves as our Verifiable Reward Environment
        let output = Command::new("rustc")
            .arg("--emit=metadata")
            .arg(&file_path)
            .output();

        // Clean up
        let _ = fs::remove_file(&file_path);
        let _ = fs::remove_file(temp_dir.join("libmud_rlvr_eval.rmeta")); // if it generated a lib

        match output {
            Ok(output) => {
                if output.status.success() {
                    (1.0, "Success".to_string())
                } else {
                    let error_msg = String::from_utf8_lossy(&output.stderr).to_string();
                    (-1.0, error_msg)
                }
            }
            Err(e) => {
                (-1.0, format!("Execution Error: {}", e))
            }
        }
    }
}
