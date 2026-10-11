//! # Pointer-address calculation audit for a real `.mud` file
//!
//! Validates the raw-pointer ELUT/ternary layout of every Ternary2Bit tensor
//! against `dequantize_ternary_row`, using the exact address math
//! (`u32_idx = k/8; shift = (k%8)*4`) from `tools/training_healthcheck.rs`.
//!
//! Usage: `./mud.sh pointer-audit [model.mud]`

use forge_llm::mud::pointer_audit::audit_model_pointers;
use forge_llm::mud::MudFile;
use std::process::ExitCode;

fn main() -> ExitCode {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/smollm2.mud".to_string());

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  POINTER-ADDRESS AUDIT  ·  ELUT/ternary layout (P-00)    ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!("Model: {model_path}\n");

    let mud = match MudFile::load(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("  ❌ load failed: {e}");
            return ExitCode::from(1);
        }
    };

    let rep = audit_model_pointers(&mud);
    println!(
        "  tensors_checked : {}",
        rep.tensors_checked
    );
    println!("  elements_checked: {}", rep.elements_checked);
    println!("  mismatches      : {}", rep.mismatches);
    println!("  max_abs_err     : {:.2e}", rep.max_abs_err);

    println!("\n--- Verdict ---");
    if rep.mismatches == 0 && rep.elements_checked > 0 {
        println!(
            "  🟢 POINTER LAYOUT OK — {} ternary elements decode identically via raw mmap pointers",
            rep.elements_checked
        );
        ExitCode::SUCCESS
    } else {
        println!(
            "  🔴 POINTER LAYOUT MISMATCH — {} bad elements (max_err {:.2e})",
            rep.mismatches, rep.max_abs_err
        );
        ExitCode::from(2)
    }
}
