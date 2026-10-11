//! SPRINT-2: SlimeX file fusion (merge multiple .mud into one).
//!
//! Thin CLI wrapper around `MudFileMerger::merge_files`. Scans `models/*.mud`,
//! loads them all, and writes `models/merged.mud`.
//!
//! Usage:
//!   cargo run --release --bin slimex_fuse -- [INPUT_DIR] [OUTPUT_PATH]
//!   cargo run --release --bin slimex_fuse -- models/ models/merged.mud
//!
//! Default: input=`models/`, output=`models/merged.mud`.

use forge_llm::mud::file_fusion::MudFileMerger;
use forge_llm::mud::MudFile;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let input_dir = args.get(1).cloned().unwrap_or_else(|| "models".to_string());
    let output_path = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "models/merged.mud".to_string());

    println!("[slimex-fuse] scanning {input_dir} for .mud files...");
    let path = Path::new(&input_dir);
    if !path.is_dir() {
        anyhow::bail!("{input_dir} is not a directory");
    }

    let entries: Vec<_> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|s| s.to_str())
                .map(|s| s == "mud")
                .unwrap_or(false)
        })
        .collect();

    if entries.is_empty() {
        anyhow::bail!("no .mud files found in {input_dir}");
    }

    println!("[slimex-fuse] found {} .mud file(s):", entries.len());
    for e in &entries {
        if let Some(name) = e.path().file_name().and_then(|s| s.to_str()) {
            println!("  • {name}");
        }
    }

    // Load all .mud files.
    let mut muds: Vec<MudFile> = Vec::new();
    for e in &entries {
        let p = e.path();
        match MudFile::load(p.to_str().unwrap_or_default()) {
            Ok(m) => muds.push(m),
            Err(err) => {
                eprintln!("[slimex-fuse] WARN: failed to load {}: {err}", p.display());
            }
        }
    }

    if muds.len() < 2 {
        println!(
            "[slimex-fuse] only {} valid .mud file(s); nothing to fuse.",
            muds.len()
        );
        return Ok(());
    }

    // Borrow slice for the merger.
    let refs: Vec<&MudFile> = muds.iter().collect();
    println!(
        "[slimex-fuse] fusing {} files → {output_path}",
        refs.len()
    );
    MudFileMerger::merge_files(&refs, &output_path)?;
    println!("[slimex-fuse] OK  wrote {output_path}");

    // Verify with the loader.
    println!("[slimex-fuse] verifying merged file...");
    let merged = MudFile::load(&output_path)?;
    let n_tensors: usize = merged
        .skills
        .values()
        .map(|s| s.tensors.len())
        .sum();
    println!(
        "[slimex-fuse] verification passed  ({} skills, {} tensors, {:.1} MiB)",
        merged.skills.len(),
        n_tensors,
        std::fs::metadata(&output_path)?.len() as f64 / 1_048_576.0
    );

    Ok(())
}
