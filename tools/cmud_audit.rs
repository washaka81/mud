//! # C-MUD Reasoning Audit (research §3 — new work)
//!
//! Standalone audit of the complex-valued reasoning path (`MUD_CMUD_THINK=1`).
//! Runs a real forward over probe prompts and reports readable, validated metrics:
//! finite logits, dynamic range, token-0 dominance, phase-lock and Hermitian ball.
//!
//! Usage: `./mud.sh cmud-audit [model.mud]`

use forge_llm::mud::inference::cmud_audit;
use forge_llm::mud::MudFile;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmp = args.iter().any(|a| a == "--cmp");
    let model_path = args
        .into_iter()
        .find(|a| !a.starts_with('-'))
        .unwrap_or_else(|| "models/smollm2.mud".to_string());

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  C-MUD REASONING AUDIT  ·  complex thinking manifold     ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!("Model: {model_path}\n");

    let mud = match MudFile::load(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("  ❌ load failed: {e}");
            return ExitCode::from(1);
        }
    };

    if cmp {
        let c = forge_llm::mud::inference::cmud_compare(&mud);
        println!("--- Baseline vs C-MUD (quality probe) ---");
        println!("  baseline_argmax : {}", c.baseline_argmax);
        println!("  cmud_argmax     : {}", c.cmud_argmax);
        println!("  argmax_changed  : {}", c.argmax_changed);
        println!("  logit_l2 (Δ)    : {:.4}", c.logit_l2);
        println!("  baseline_entropy: {:.4}", c.baseline_entropy);
        println!("  cmud_entropy    : {:.4}", c.cmud_entropy);
        println!(
            "  🟢 probe complete (Δ is expected; validates the path shifts the distribution)"
        );
        return ExitCode::SUCCESS;
    }

    let cka = cmud_audit(&mud);
    println!("--- Metrics ---");
    for line in cka.summary_lines() {
        println!("{line}");
    }

    println!("\n--- Verdict ---");
    if cka.healthy() {
        println!("  🟢 C-MUD reasoning HEALTHY");
        println!("    (phase-coherent attention + CUE phase-repulsion + wave collapse)");
        ExitCode::SUCCESS
    } else {
        let mut reasons = Vec::new();
        if !cka.forward_ok {
            reasons.push("forward error");
        }
        if !cka.logits_finite {
            reasons.push("non-finite logits");
        }
        if cka.logit_range_min <= 0.0 {
            reasons.push("zero dynamic range");
        }
        if cka.token0_dominant {
            reasons.push("token-0 dominance");
        }
        if cka.think.max_herm_norm > cka.think.radius * 1.01 {
            reasons.push("Hermitian ball violated");
        }
        println!("  🔴 C-MUD reasoning UNHEALTHY: {}", reasons.join(", "));
        ExitCode::from(2)
    }
}
