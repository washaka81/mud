use std::fs;
use std::io::{self, Read, Write};
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
    ///
    /// SEC-WS-01 (R-38): the previous implementation used `Path::exists()`,
    /// which follows the final symlink — a *dangling* link whose target sits
    /// outside the root was therefore treated as a brand-new file and
    /// `write_file`/`read_file` then followed it outside the workspace. The
    /// resolver below canonicalizes the **deepest existing ancestor** and
    /// rejects dangling links explicitly (fail-closed), so every path handed
    /// to the IO helpers is either a real path under `root_canon` or an
    /// error.
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

        // 2. Walk up to the deepest prefix that exists on disk. Use
        //    `symlink_metadata` (no follow): a dangling symlink leaf reports
        //    as existing here, and `canonicalize` below then fails on it —
        //    which is exactly the fail-closed behaviour we want.
        let candidate = self.root_dir.join(relative_path);
        let mut probe = candidate.clone();
        loop {
            if fs::symlink_metadata(&probe).is_ok() {
                break;
            }
            match probe.parent() {
                Some(parent) => probe = parent.to_path_buf(),
                None => break,
            }
        }

        // 3. Canonicalize the existing prefix (resolves every symlink in it)
        //    and require it to stay under the workspace root. Any dangling
        //    link in the prefix (including the leaf) makes this fail => Err.
        let resolved_prefix = fs::canonicalize(&probe).map_err(|e| {
            io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!(
                    "symlink escape (dangling link outside workspace) rejected: \
                     '{relative_path}': {e}"
                ),
            )
        })?;
        if !resolved_prefix.starts_with(&self.root_canon) {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!(
                    "path escape detected: '{relative_path}' resolves outside the workspace root"
                ),
            ));
        }

        // 4. Rebuild the final path as `resolved_prefix + verbatim remainder`.
        //    The remainder does not exist yet (probe was the deepest existing
        //    prefix) and contains no `..` (rejected above), so it holds no
        //    symlink: `create_dir_all`/writes cannot leave the root.
        //    NOTE: when the whole path exists, `probe == candidate` and the
        //    prefix IS the answer — `join("")` would append a trailing slash
        //    and make opening a regular file fail with ENOTDIR.
        if probe == candidate {
            return Ok(resolved_prefix);
        }
        let suffix = candidate.strip_prefix(&probe).unwrap_or(Path::new(""));
        Ok(resolved_prefix.join(suffix))
    }

    /// Opens a resolved leaf without following a final symlink (SEC-WS-01
    /// defence-in-depth against a TOCTOU swap after resolution): on Linux and
    /// macOS a symlinked leaf yields `ELOOP` instead of a file outside the
    /// root. `O_NOFOLLOW` never affects `O_CREAT` for a genuinely new file.
    fn open_leaf(path: &Path, write: bool) -> io::Result<fs::File> {
        #[allow(unused_mut)]
        let mut opts = fs::OpenOptions::new();
        if write {
            opts.write(true).create(true).truncate(true);
        } else {
            opts.read(true);
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            // O_NOFOLLOW values from the platform headers (not in std).
            #[cfg(target_os = "linux")]
            opts.custom_flags(0o400000);
            #[cfg(target_os = "macos")]
            opts.custom_flags(0x0100);
        }
        opts.open(path)
    }

    /// Reads a file into the engine's context buffer.
    pub fn read_file(&self, relative_path: &str) -> std::io::Result<String> {
        let full_path = self.resolve_inside_root(relative_path)?;
        let mut s = String::new();
        Self::open_leaf(&full_path, false)?.read_to_string(&mut s)?;
        Ok(s)
    }

    /// Writes generated code autonomously back to the filesystem.
    pub fn write_file(&self, relative_path: &str, content: &str) -> std::io::Result<()> {
        let full_path = self.resolve_inside_root(relative_path)?;
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent)?;
        }
        Self::open_leaf(&full_path, true)?.write_all(content.as_bytes())
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

    // ============================================================
    // SEC-WS-01 (R-38): symlink escape / dangling symlink
    // ============================================================
    #[cfg(unix)]
    #[test]
    fn test_dangling_symlink_rejected() {
        use std::os::unix::fs::symlink;
        let root = unique_tmp();
        let ws = AgentWorkspace::mount(&root).unwrap();
        // Dangling symlink pointing OUTSIDE the root (target does not exist).
        let outside = root
            .parent()
            .unwrap()
            .join(format!("mud_r38_dangling_{}", std::process::id()));
        let _ = fs::remove_file(&outside);
        symlink(&outside, root.join("link")).unwrap();

        let err = ws.read_file("link").unwrap_err();
        assert!(
            err.to_string().contains("dangling") || err.to_string().contains("escape"),
            "read: got {err}"
        );
        let err = ws.write_file("link", "pwn").unwrap_err();
        assert!(
            err.to_string().contains("dangling") || err.to_string().contains("escape"),
            "write: got {err}"
        );
        assert!(
            !outside.exists(),
            "SEC-WS-01: write through dangling symlink escaped the root"
        );
        let _ = fs::remove_file(&outside);
    }

    #[cfg(unix)]
    #[test]
    fn test_symlink_to_outside_rejected() {
        use std::os::unix::fs::symlink;
        let root = unique_tmp();
        let ws = AgentWorkspace::mount(&root).unwrap();
        let outside = unique_tmp();
        fs::write(outside.join("secret.txt"), "secret").unwrap();
        symlink(outside.join("secret.txt"), root.join("leak.txt")).unwrap();

        let err = ws.read_file("leak.txt").unwrap_err();
        assert!(err.to_string().contains("escape"), "got {err}");
    }

    #[cfg(unix)]
    #[test]
    fn test_symlink_dir_ancestor_escape_rejected() {
        use std::os::unix::fs::symlink;
        let root = unique_tmp();
        let ws = AgentWorkspace::mount(&root).unwrap();
        let outside = unique_tmp();
        // Symlinked directory inside root pointing outside; a write "into" it
        // must be rejected (the leaf does not exist, the ancestor escapes).
        symlink(&outside, root.join("outdir")).unwrap();

        let err = ws.write_file("outdir/new.txt", "pwn").unwrap_err();
        assert!(err.to_string().contains("escape"), "got {err}");
        assert!(
            !outside.join("new.txt").exists(),
            "SEC-WS-01: write escaped through a symlinked ancestor"
        );
    }

    #[cfg(unix)]
    #[test]
    fn test_symlink_inside_root_ok() {
        use std::os::unix::fs::symlink;
        let root = unique_tmp();
        let ws = AgentWorkspace::mount(&root).unwrap();
        fs::write(root.join("real.txt"), "ok").unwrap();
        symlink(root.join("real.txt"), root.join("alias.txt")).unwrap();
        // Symlinks that stay inside the workspace remain allowed.
        assert_eq!(ws.read_file("alias.txt").unwrap(), "ok");
    }
}
