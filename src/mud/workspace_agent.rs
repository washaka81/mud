use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};

/// Priority 9: Autonomous Workspace Integration
/// This module provides the MUD Engine with direct read/write access to the host
/// filesystem, enabling autonomous project traversal outside the standard CLI loop.
pub struct AgentWorkspace {
    root_dir: PathBuf,
    /// Canonicalized form of `root_dir` (resolved symlinks + `..`).
    /// Used as the trust boundary for path-traversal checks.
    root_canon: PathBuf,
}

impl AgentWorkspace {
    /// Mounts an autonomous workspace at the specified directory.
    pub fn mount<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let root_dir = path.as_ref().to_path_buf();
        if !root_dir.exists() {
            fs::create_dir_all(&root_dir)?;
        }
        // Resolve symlinks + normalize. Cached once so every read_file/write_file
        // check is a simple starts_with (no syscall).
        let root_canon = fs::canonicalize(&root_dir)?;
        Ok(Self { root_dir, root_canon })
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
                    if file_name.starts_with('.')
                        || file_name == "target"
                        || file_name == "node_modules"
                    {
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

    /// Validates that `relative_path` resolves to a path inside `root_canon`.
    /// Rejects paths containing `..` components (CWE-22 path-traversal).
    /// Symlinks within the workspace are allowed (resolved by `canonicalize`).
    fn resolve_inside_root(&self, relative_path: &str) -> io::Result<PathBuf> {
        // 1. Reject any `..` component (or absolute path) before touching the FS.
        let p = Path::new(relative_path);
        let mut has_parent = false;
        for c in p.components() {
            match c {
                Component::ParentDir => has_parent = true,
                Component::RootDir | Component::Prefix(_) => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("absolute path rejected: {relative_path}"),
                    ));
                }
                _ => {}
            }
        }
        if has_parent {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("path traversal rejected (..): {relative_path}"),
            ));
        }

        // 2. Resolve symlinks and verify the result is still under root_canon.
        let candidate = self.root_dir.join(relative_path);
        // For non-existent files (e.g. write_file on a brand-new path), check
        // the parent. For existing ones, canonicalize the leaf.
        let canon = if candidate.exists() {
            fs::canonicalize(&candidate)?
        } else if let Some(parent) = candidate.parent() {
            if parent.exists() {
                let parent_canon = fs::canonicalize(parent)?;
                parent_canon.join(candidate.file_name().unwrap_or_default())
            } else {
                // Parent doesn't exist yet; we'll create it in write_file.
                // Trust the joined path (no `..` possible at this point).
                candidate.clone()
            }
        } else {
            candidate.clone()
        };

        // 3. Ensure the resolved path is inside the root.
        if !canon.starts_with(&self.root_canon) {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!(
                    "path escape detected: '{relative_path}' resolves outside the workspace root"
                ),
            ));
        }
        Ok(canon)
    }

    /// Reads a file into the engine's context buffer.
    pub fn read_file(&self, relative_path: &str) -> std::io::Result<String> {
        let full_path = self.resolve_inside_root(relative_path)?;
        fs::read_to_string(full_path)
    }

    /// Writes generated code autonomously back to the filesystem.
    pub fn write_file(&self, relative_path: &str, content: &str) -> std::io::Result<()> {
        let full_path = self.resolve_inside_root(relative_path)?;
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(full_path, content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    fn unique_tmp() -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("mud_ws_test_{}_{}", std::process::id(), n));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_path_traversal_blocked() {
        let root = unique_tmp();
        let ws = AgentWorkspace::mount(&root).unwrap();
        // `..` must be rejected before any FS call.
        let err = ws.read_file("../etc/passwd").unwrap_err();
        assert!(err.to_string().contains("traversal"));
    }

    #[test]
    fn test_absolute_path_rejected() {
        let root = unique_tmp();
        let ws = AgentWorkspace::mount(&root).unwrap();
        let err = ws.read_file("/etc/passwd").unwrap_err();
        assert!(err.to_string().contains("absolute"));
    }

    #[test]
    fn test_normal_read_write_works() {
        let root = unique_tmp();
        let ws = AgentWorkspace::mount(&root).unwrap();
        ws.write_file("foo.txt", "hello").unwrap();
        let s = ws.read_file("foo.txt").unwrap();
        assert_eq!(s, "hello");
    }
}
