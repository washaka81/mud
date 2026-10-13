//! SAFETY contract (R-36): this module contains no `unsafe` code. The token the
//! gate flags is the cargo feature name `unsafe-rce` inside `#[cfg]` attributes
//! and inside the warning/error strings of the safe stub — never an unsafe
//! operation. The feature gates `evaluate_rust_code`, which intentionally runs
//! `rustc` on attacker-controlled input (RCE by design, research only);
//! production builds compile the stub that always returns `-1.0`.

#[cfg(feature = "unsafe-rce")]
use std::fs;
#[cfg(feature = "unsafe-rce")]
use std::process::Command;

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
    ///
    /// # SECURITY (V25 audit SEC-1)
    ///
    /// This function executes the system `rustc` binary on attacker-controlled
    /// input. `rustc` runs procedural macros, `build.rs` scripts, and processes
    /// `#[...]` attributes during compilation — equivalent to arbitrary code
    /// execution with the privileges of the calling process.
    ///
    /// **Gated behind the `unsafe-rce` cargo feature** so that production builds
    /// (`cargo build --release` with no extra features) do not include this
    /// function. To opt in for research/eval: `cargo build --features unsafe-rce`.
    #[cfg(feature = "unsafe-rce")]
    pub fn evaluate_rust_code(&self, code: &str) -> (f32, String) {
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("mud_rlvr_eval.rs");

        // Write code to a temporary file
        if fs::write(&file_path, code).is_err() {
            return (
                -1.0,
                "System Error: Failed to write to temp file".to_string(),
            );
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
            Err(e) => (-1.0, format!("Execution Error: {}", e)),
        }
    }

    /// Safe stub: always returns the negative reward.
    /// When the `unsafe-rce` feature is enabled, `evaluate_rust_code` is the
    /// real implementation (and shadows this one).
    // SAFETY: `unsafe-rce` is a cargo feature name only (no unsafe operation);
    // the code below is the production stub compiled when the feature is off.
    #[cfg(not(feature = "unsafe-rce"))]
    pub fn evaluate_rust_code(&self, _code: &str) -> (f32, String) {
        log::warn!(
            "[SECURITY] RlvrCritic::evaluate_rust_code is a no-op in this build. \
             Recompile with `--features unsafe-rce` to enable (research only)."
        );
        (-1.0, "RLVR critic disabled in production build (compile with --features unsafe-rce to enable)".to_string())
    }
}

// ============================================================
// R-37: build-gate test — the default (production/CI) build MUST
// NOT compile/execute rustc. Only compiled when the feature is OFF.
// ============================================================
#[cfg(all(test, not(feature = "unsafe-rce")))]
mod tests {
    use super::*;

    #[test]
    fn test_rlvr_critic_disabled_by_default() {
        let critic = RlvrCritic::new();
        let (reward, msg) = critic.evaluate_rust_code("fn main() {}");
        assert_eq!(reward, -1.0, "default build must not reward code execution");
        assert!(
            msg.contains("disabled"),
            "stub must explain the feature gate, got: {msg}"
        );
    }
}
