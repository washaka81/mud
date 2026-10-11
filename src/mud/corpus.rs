//! SPRINT-4 (4.1): corpus file collection utilities.
//!
//! Extracted from `corpus_trainer.rs` to break the file into smaller, focused
//! modules. These functions are pure filesystem scanning — no model state,
//! no GPU, no I/O beyond `std::fs::read_dir`.
//!
//! Filter rules (frozen in code, not configurable from CLI):
//!   - skip dotfiles
//!   - skip `_stash*`, `dumps`, `dumps_archive`, `downloads` dirs
//!   - skip `project_corpus.txt` (its own pipeline)
//!   - root only scans `src/`, `forge_autograd/`, `training/`
//!   - `training/corpus/`: only `*.txt` and `*.md`
//!   - elsewhere: `*.rs` (unless `MUD_TRAIN_TEXT_ONLY=1`)
//!
//! Returned vector is sorted by file size (smallest first) so quick AOT
//! fills the budget from short alignment files.

use std::path::{Path, PathBuf};

/// SEC-H7 (2026-07-26): hard cap on directory-walk depth. The project tree
/// is shallow (≤8 levels in practice); a symlink loop or a malicious
/// `tree-of-doom` payload next to `training/corpus/` used to be able to
/// drive unbounded stack recursion. 32 covers any natural repo while still
/// rejecting pathological cases in microseconds (no extra syscalls).
pub(crate) const COLLECT_FILES_MAX_DEPTH: usize = 32;

/// True if any path component is a known non-train dir.
pub(crate) fn path_is_excluded(path: &Path) -> bool {
    path.components().any(|c| {
        let s = c.as_os_str().to_string_lossy();
        s.starts_with("_stash") || s == "dumps" || s == "dumps_archive" || s == "downloads"
    })
}

/// Recursively collect training-eligible files into `files`.
///
/// `dir` is the directory to scan. `is_root` distinguishes the project root
/// (where we whitelist `src/`, `forge_autograd/`, `training/`) from nested
/// subdirs (which are scanned unconditionally, modulo the exclusion rules).
pub(crate) fn collect_files(dir: &Path, files: &mut Vec<PathBuf>, is_root: bool) {
    collect_files_inner(dir, files, is_root, 0, COLLECT_FILES_MAX_DEPTH);
}

fn collect_files_inner(
    dir: &Path,
    files: &mut Vec<PathBuf>,
    is_root: bool,
    depth: usize,
    max_depth: usize,
) {
    // SEC-H7: depth-bounded recursion. A symlink loop or a tree-of-doom
    // payload no longer blows the stack; we stop descending past the cap.
    if depth > max_depth {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name().unwrap_or_default().to_string_lossy();

        if name.starts_with('.') {
            continue;
        }
        // Skip stash/backup/download dirs (quick-train isolation)
        if name.starts_with("_stash")
            || name == "dumps"
            || name == "dumps_archive"
            || name == "downloads"
        {
            continue;
        }
        if path_is_excluded(&path) {
            continue;
        }
        if name == "project_corpus.txt" {
            continue;
        }

        if path.is_dir() {
            if !is_root || name == "src" || name == "forge_autograd" || name == "training" {
                collect_files_inner(&path, files, false, depth + 1, max_depth);
            }
        } else if let Some(ext) = path.extension() {
            let ext_str = ext.to_string_lossy();
            if ext_str == "txt" || ext_str == "rs" || ext_str == "md" {
                let path_str = path.to_string_lossy();
                // Normalize: accept training/corpus even with ./ prefix
                let in_corpus = path_str.contains("training/corpus/")
                    || path_str.contains("training\\corpus\\");
                let text_only = std::env::var("MUD_TRAIN_TEXT_ONLY")
                    .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    .unwrap_or(false);
                if in_corpus {
                    if ext_str == "txt" || ext_str == "md" {
                        files.push(path);
                    }
                } else if ext_str == "rs" && !text_only {
                    files.push(path);
                }
            }
        }
    }
}

/// Scan the project root with the corpus filter rules, returning a
/// `Vec<PathBuf>` sorted by file size ascending.
pub(crate) fn scan_training_files() -> anyhow::Result<Vec<PathBuf>> {
    let mut text_files = Vec::new();
    collect_files(Path::new("."), &mut text_files, true);
    text_files.sort_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(u64::MAX));
    if text_files.is_empty() {
        anyhow::bail!("No training files found in the project root!");
    }
    Ok(text_files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU32, Ordering};

    static COUNTER: AtomicU32 = AtomicU32::new(0);

    fn fresh_dir(name: &str) -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let p = std::env::temp_dir().join(format!("mud_corpus_test_{name}_{n}"));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn test_path_is_excluded_detects_stash() {
        let p = Path::new("/foo/_stash_2026/x.txt");
        assert!(path_is_excluded(p));
    }

    #[test]
    fn test_path_is_excluded_detects_dumps() {
        let p = Path::new("/foo/dumps/x.txt");
        assert!(path_is_excluded(p));
    }

    #[test]
    fn test_path_is_excluded_passes_normal() {
        let p = Path::new("/foo/src/x.txt");
        assert!(!path_is_excluded(p));
    }

    #[test]
    fn test_collect_files_respects_exclusion() {
        // Save/restore MUD_TRAIN_TEXT_ONLY — mud.sh exports it as 1 by default,
        // which would skip .rs files and break this test.
        let prev = std::env::var("MUD_TRAIN_TEXT_ONLY").ok();
        std::env::set_var("MUD_TRAIN_TEXT_ONLY", "0");
        let dir = fresh_dir("excl");
        // Outside training/corpus/, only *.rs is eligible.
        fs::write(dir.join("keep.rs"), "fn x(){}").unwrap();
        fs::create_dir(dir.join("_stash")).unwrap();
        fs::write(dir.join("_stash/secret.rs"), "no").unwrap();
        let mut files = Vec::new();
        collect_files(&dir, &mut files, false);
        assert_eq!(files.len(), 1);
        assert!(files[0].ends_with("keep.rs"));
        let _ = fs::remove_dir_all(&dir);
        // Restore
        match prev {
            Some(v) => std::env::set_var("MUD_TRAIN_TEXT_ONLY", v),
            None => std::env::remove_var("MUD_TRAIN_TEXT_ONLY"),
        }
    }

    #[test]
    fn test_collect_files_skips_project_corpus() {
        // Save/restore MUD_TRAIN_TEXT_ONLY — mud.sh exports it as 1 by default.
        let prev = std::env::var("MUD_TRAIN_TEXT_ONLY").ok();
        std::env::set_var("MUD_TRAIN_TEXT_ONLY", "0");
        let dir = fresh_dir("pcc");
        // Outside training/corpus/, only *.rs is eligible. project_corpus.txt
        // would be matched by extension but is excluded by name.
        fs::write(dir.join("ok.rs"), "fn a(){}").unwrap();
        fs::write(dir.join("project_corpus.txt"), "b").unwrap();
        let mut files = Vec::new();
        collect_files(&dir, &mut files, false);
        assert_eq!(files.len(), 1);
        assert!(files[0].ends_with("ok.rs"));
        let _ = fs::remove_dir_all(&dir);
        // Restore
        match prev {
            Some(v) => std::env::set_var("MUD_TRAIN_TEXT_ONLY", v),
            None => std::env::remove_var("MUD_TRAIN_TEXT_ONLY"),
        }
    }
}
