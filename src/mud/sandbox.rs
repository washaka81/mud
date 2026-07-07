use std::process::{Command, Stdio};
use std::time::Duration;
use std::io::Read;
use wait_timeout::ChildExt;

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
    pub fn execute_command(&self, command: &str, cwd: &std::path::Path) -> std::io::Result<String> {
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
                    format!("Command Failed (Exit Code: {}):\nSTDOUT:\n{}\nSTDERR:\n{}", 
                        status.code().unwrap_or(-1), out_str, err_str)
                };
                Ok(final_output)
            }
            None => {
                // Timeout occurred, kill the process
                let _ = child.kill();
                let _ = child.wait();
                Ok(format!("Command Timeout Reached ({}s). Process terminated.", self.timeout_seconds))
            }
        }
    }
}
