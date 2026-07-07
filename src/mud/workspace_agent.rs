use std::path::{Path, PathBuf};
use std::fs;

/// Priority 9: Autonomous Workspace Integration
/// This module provides the MUD Engine with direct read/write access to the host
/// filesystem, enabling autonomous project traversal outside the standard CLI loop.
pub struct AgentWorkspace {
    root_dir: PathBuf,
}

impl AgentWorkspace {
    /// Mounts an autonomous workspace at the specified directory.
    pub fn mount<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let root_dir = path.as_ref().to_path_buf();
        if !root_dir.exists() {
            fs::create_dir_all(&root_dir)?;
        }
        Ok(Self { root_dir })
    }

    /// Recursively lists files to build an internal mental map of the project.
    pub fn scan_project_map(&self) -> Vec<PathBuf> {
        let mut map = Vec::new();
        let mut stack = vec![self.root_dir.clone()];

        while let Some(current_dir) = stack.pop() {
            if let Ok(entries) = fs::read_dir(&current_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();

                    // Skip common hidden/binary directories to prevent memory bloat
                    let file_name = path.file_name().unwrap_or_default().to_string_lossy();
                    if file_name.starts_with('.') || file_name == "target" || file_name == "node_modules" {
                        continue;
                    }

                    if path.is_dir() {
                        stack.push(path);
                    } else if path.is_file() {
                        if let Ok(rel_path) = path.strip_prefix(&self.root_dir) {
                            map.push(rel_path.to_path_buf());
                        } else {
                            map.push(path);
                        }
                    }
                }
            }
        }
        
        map.sort();
        map
    }
    
    /// Reads a file into the engine's context buffer.
    pub fn read_file(&self, relative_path: &str) -> std::io::Result<String> {
        let full_path = self.root_dir.join(relative_path);
        fs::read_to_string(full_path)
    }

    /// Writes generated code autonomously back to the filesystem.
    pub fn write_file(&self, relative_path: &str, content: &str) -> std::io::Result<()> {
        let full_path = self.root_dir.join(relative_path);
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(full_path, content)
    }
}
