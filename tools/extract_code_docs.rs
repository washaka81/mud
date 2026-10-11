//! Extract documentation and explanatory comments from Rust source files
//! and Markdown docs into a clean text corpus for training.
//!
//! Usage: `cargo run --release --bin extract_code_docs`
//!
//! Output: `training/corpus/mud_docs.txt` — one clean text line per doc
//! comment paragraph, suitable for tokenization via SmartTokenizer.
//!
//! Sources:
//!   - `src/mud/**/*.rs` — `///`, `//!`, and explanatory `//` comments
//!   - `docs/architecture/**/*.md` — arch design docs
//!   - `docs/sessions/*.md` — session reports (trim boilerplate)
//!   - `docs/research/*.md` — research notes (trim boilerplate)
//!
//! The tool strips code syntax, normalizes whitespace, and ensures each
//! line is a coherent natural-language sentence (or short paragraph).
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

/// Maximum depth for directory walks (SEC-H7 style guard).
const MAX_DEPTH: usize = 24;

fn main() -> anyhow::Result<()> {
    let out_path = PathBuf::from("training/corpus/mud_docs.txt");
    fs::create_dir_all(out_path.parent().unwrap())?;

    let mut out = BufWriter::new(fs::File::create(&out_path)?);
    let mut count = 0usize;

    // 1. Rust doc comments from src/mud/
    count += extract_rust_docs("src/mud", &mut out)?;

    // 2. Architecture docs
    count += extract_markdown("docs/architecture", &mut out)?;

    // 3. Session reports
    count += extract_markdown("docs/sessions", &mut out)?;

    out.flush()?;
    println!("extract_code_docs: wrote {} lines to {}", count, out_path.display());
    Ok(())
}

/// Recursively find files with a given extension, skipping excluded dirs.
fn find_files(dir: &Path, exts: &[&str], depth: usize, out: &mut Vec<PathBuf>) {
    if depth > MAX_DEPTH {
        return;
    }
    let Ok(entries) = fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
        if name.starts_with('.') || name.starts_with("_stash") {
            continue;
        }
        if path.is_dir() {
            find_files(&path, exts, depth + 1, out);
        } else if let Some(ext) = path.extension() {
            let ext_str = ext.to_string_lossy();
            if exts.iter().any(|e| e == &ext_str) {
                out.push(path);
            }
        }
    }
}

/// Extract doc/explanatory comments from Rust files.
fn extract_rust_docs(root: &str, out: &mut BufWriter<fs::File>) -> anyhow::Result<usize> {
    let root_path = Path::new(root);
    if !root_path.exists() {
        return Ok(0);
    }

    let mut files = Vec::new();
    find_files(root_path, &["rs"], 0, &mut files);
    files.sort();

    let mut count = 0usize;
    for fpath in &files {
        let file = match fs::File::open(fpath) {
            Ok(f) => f,
            Err(_) => continue,
        };
        let reader = BufReader::new(file);
        let mut current_doc: Vec<String> = Vec::new();

        for line_result in reader.lines() {
            let line = match line_result {
                Ok(l) => l,
                Err(_) => continue,
            };

            let trimmed = line.trim_start();

            if let Some(rest) = trimmed.strip_prefix("///") {
                let text = rest.trim_start();
                if !text.is_empty() {
                    current_doc.push(text.to_string());
                }
            } else if let Some(rest) = trimmed.strip_prefix("//!") {
                let text = rest.trim_start();
                if !text.is_empty() {
                    current_doc.push(text.to_string());
                }
            } else if trimmed.starts_with("//") && !trimmed.starts_with("///") {
                // Explanatory comment — flush doc block first, then emit if substantive
                if !current_doc.is_empty() {
                    flush_doc(out, &current_doc, &mut count);
                    current_doc.clear();
                }
                // Skip TODO/FIXME/copyright/short noise
                let text = trimmed.trim_start_matches("//").trim_start();
                if text.len() > 12 && !is_noise_comment(text) {
                    let cleaned = clean_text(text);
                    if cleaned.len() > 20 {
                        writeln!(out, "{}", cleaned)?;
                        count += 1;
                    }
                }
            } else {
                // Non-comment line — flush pending doc block
                if !current_doc.is_empty() {
                    flush_doc(out, &current_doc, &mut count);
                    current_doc.clear();
                }
            }
        }
        // Flush at EOF
        if !current_doc.is_empty() {
            flush_doc(out, &current_doc, &mut count);
            current_doc.clear();
        }
    }

    Ok(count)
}

/// Flush a doc comment block as paragraphs.
fn flush_doc(out: &mut BufWriter<fs::File>, doc: &[String], count: &mut usize) {
    if doc.is_empty() {
        return;
    }
    // Join doc lines into paragraphs (blank gap = paragraph break)
    let mut para: Vec<String> = Vec::new();
    for line in doc {
        if line.is_empty() {
            if !para.is_empty() {
                let combined = para.join(" ");
                let cleaned = clean_text(&combined);
                if cleaned.len() > 20 {
                    writeln!(out, "{}", cleaned).ok();
                    *count += 1;
                }
                para.clear();
            }
        } else {
            para.push(line.clone());
        }
    }
    if !para.is_empty() {
        let combined = para.join(" ");
        let cleaned = clean_text(&combined);
        if cleaned.len() > 20 {
            writeln!(out, "{}", cleaned).ok();
            *count += 1;
        }
    }
}

/// Filter out boilerplate/noise comments.
fn is_noise_comment(text: &str) -> bool {
    let lower = text.to_lowercase();
    lower.starts_with("todo:")
        || lower.starts_with("fixme:")
        || lower.starts_with("hack:")
        || lower.starts_with("note:")
        || lower.starts_with("safety:")
        || lower.starts_with("test")
        || lower.starts_with("#[")
        || lower.starts_with("derive(")
        || lower.starts_with("allow(")
        || lower.starts_with("deny(")
        || lower.starts_with("warn(")
        || lower.starts_with("cfg(")
        || lower.contains("copyright")
}

/// Clean a text line: strip markdown code spans, normalize whitespace.
fn clean_text(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '`' {
            // Skip inline code: consume until matching backtick (or end of word)
            let mut code = String::new();
            while let Some(&nc) = chars.peek() {
                if nc == '`' {
                    chars.next();
                    break;
                }
                code.push(nc);
                chars.next();
            }
            // If it's a code reference (like a function name), keep the last component
            if code.contains('(') || code.contains(']') || code.chars().filter(|&c| c == ':').count() >= 2 {
                // Keep as-is for identifier-like code
                result.push_str(&code);
            } else {
                result.push_str(&code);
            }
        } else {
            result.push(c);
        }
    }
    // Normalize whitespace
    let parts: Vec<&str> = result.split_whitespace().collect();
    parts.join(" ")
}

/// Extract text content from Markdown files.
fn extract_markdown(root: &str, out: &mut BufWriter<fs::File>) -> anyhow::Result<usize> {
    let root_path = Path::new(root);
    if !root_path.exists() {
        return Ok(0);
    }

    let mut files = Vec::new();
    find_files(root_path, &["md"], 0, &mut files);
    files.sort();

    let mut count = 0usize;
    for fpath in &files {
        let content = match fs::read_to_string(fpath) {
            Ok(c) => c,
            Err(_) => continue,
        };

        for raw_line in content.lines() {
            let line = raw_line.trim();
            // Skip markdown structure, code blocks, headers, links
            if line.is_empty()
                || line.starts_with('#')
                || line.starts_with("```")
                || line.starts_with('|')
                || line.starts_with("- ")
                || line.starts_with("* ")
                || line.starts_with(">")
                || line.starts_with('[')
                || line.starts_with("---")
                || line.starts_with("===")
            {
                continue;
            }
            // Skip lines that are mostly markdown boilerplate
            if line.starts_with("**") && line.ends_with("**") {
                continue;
            }
            // Clean and emit
            let cleaned = clean_text(line);
            if cleaned.len() > 25 {
                writeln!(out, "{}", cleaned)?;
                count += 1;
            }
        }
    }

    Ok(count)
}
