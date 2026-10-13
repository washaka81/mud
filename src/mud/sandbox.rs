//! SAFETY contract (R-36): this module contains no `unsafe` code. The lines the
//! R-36 gate flags are the cargo feature name `unsafe-cli` inside `#[cfg]`
//! attributes and inside user-facing error strings — never unsafe operations.
//! The feature gates `execute_command`, which by design runs
//! attacker-controlled `bash -c` (see its SECURITY note); production builds
//! compile the safe stub that always returns `PermissionDenied`.

#[cfg(feature = "unsafe-cli")]
use std::io::Read;
#[cfg(feature = "unsafe-cli")]
use std::process::{Command, Stdio};
#[cfg(feature = "unsafe-cli")]
use std::time::Duration;

/// Priority 11: Sandboxed Terminal Execution
/// Allows the MUD Engine to execute secure sandboxed terminal commands (e.g., `cargo check`, `bash script.sh`)
/// and receive the stdout/stderr as context for RLVR (Self-Correction).
pub struct TerminalSandbox {
    pub allow_network: bool,
    pub timeout_seconds: u64,
}

impl TerminalSandbox {
    pub fn new(timeout_seconds: u64) -> Self {
        Self {
            allow_network: false,
            timeout_seconds,
        }
    }

    /// Executes a shell command and captures its output, with strict timeouts.
    ///
    /// # SECURITY (V25 audit SEC-2)
    ///
    /// This function runs the system `bash -c <command>` with attacker-controlled
    /// input. There is no whitelist, no parsing, and no real sandboxing — a
    /// malicious caller can do anything the process can do (read ~/.ssh/, pivot
    /// to other systems, etc.).
    ///
    /// **Gated behind the `unsafe-cli` cargo feature** so that production
    /// builds do not include this function. To opt in: `cargo build
    /// --features unsafe-cli`. Default builds get a safe stub.
#[cfg(feature = "unsafe-cli")]
pub fn execute_command(&self, command: &str, cwd: &std::path::Path) -> std::io::Result<String> {
    use wait_timeout::ChildExt;
    let mut child = Command::new("bash")
        .arg("-c")
        .arg(command)
        .current_dir(cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let timeout = Duration::from_secs(self.timeout_seconds);
    match child.wait_timeout(timeout).unwrap() {
            Some(status) => {
                let mut out_str = String::new();
                if let Some(mut stdout) = child.stdout.take() {
                    stdout.read_to_string(&mut out_str)?;
                }
                let mut err_str = String::new();
                if let Some(mut stderr) = child.stderr.take() {
                    stderr.read_to_string(&mut err_str)?;
                }

                let final_output = if status.success() {
                    format!("Command Output:\n{}", out_str)
                } else {
                    format!(
                        "Command Failed (Exit Code: {}):\nSTDOUT:\n{}\nSTDERR:\n{}",
                        status.code().unwrap_or(-1),
                        out_str,
                        err_str
                    )
                };
                Ok(final_output)
            }
            None => {
                let _ = child.kill();
                let _ = child.wait();
                Ok(format!(
                    "Command Timeout Reached ({}s). Process terminated.",
                    self.timeout_seconds
                ))
            }
        }
    }

    /// Safe stub for the default build. Returns an error so the caller cannot
    /// mistakenly rely on shell execution in production.
    // SAFETY: `unsafe-cli` is only a cargo feature name (no unsafe operation
    // here); this stub is compiled precisely when that feature is disabled.
    #[cfg(not(feature = "unsafe-cli"))]
    pub fn execute_command(&self, _command: &str, _cwd: &std::path::Path) -> std::io::Result<String> {
        Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "TerminalSandbox::execute_command is disabled in production builds. \
             Recompile with `--features unsafe-cli` to enable (research only).",
        ))
    }
}

// ============================================================
// R-37: build-gate test — the default (production/CI) build MUST
// refuse shell execution. Only compiled when the feature is OFF.
// ============================================================
#[cfg(all(test, not(feature = "unsafe-cli")))]
mod tests {
    use super::*;

    #[test]
    fn test_terminal_sandbox_disabled_by_default() {
        let sb = TerminalSandbox::new(5);
        let err = sb
            .execute_command("echo pwn", std::path::Path::new("."))
            .expect_err("default build must refuse bash -c");
        assert_eq!(err.kind(), std::io::ErrorKind::PermissionDenied);
        assert!(err.to_string().contains("disabled"));
    }
}
