//! # GEMV auto-threshold bench (stream C)
//!
//! Profiles CPU (AVX2 PCorePool) vs GPU (ash tiled) ternary GEMV on this device
//! and prints the break-even work threshold used by `MUD_GPU_GEMV=auto`.
//!
//! ```bash
//! cargo run --release --bin gemv_auto_bench
//! MUD_GPU_GEMV_LOG=1 cargo run --release --bin gemv_auto_bench
//! ```

use forge_llm::vulkan::ash_backend::AshContext;
use forge_llm::vulkan::gemv_policy::{self, GemvGpuMode, GEMV_NEVER};

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  GEMV auto-bench  ·  CPU vs GPU break-even (stream C)    ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!("Policy: {}\n", gemv_policy::policy_summary());

    if !gemv_policy::vulkan_not_disabled() {
        eprintln!("MUD_USE_VULKAN=0 — cannot profile GPU.");
        std::process::exit(2);
    }

    let mut ctx = match AshContext::new() {
        Ok(c) if c.is_available() => c,
        Ok(_) => {
            eprintln!("AshContext created but not available.");
            std::process::exit(2);
        }
        Err(e) => {
            eprintln!("AshContext::new failed: {e}");
            std::process::exit(2);
        }
    };

    // Force auto calibration path for reporting
    unsafe {
        std::env::set_var("MUD_GPU_GEMV_LOG", "1");
    }
    let report = unsafe { gemv_policy::calibrate(&mut ctx) };
    gemv_policy::publish_calibration(report.clone());

    println!("\n── Summary ──────────────────────────────────────────────");
    println!("  mode           : {:?}", GemvGpuMode::Auto);
    println!("  device         : {}", report.device_available);
    if report.min_work >= GEMV_NEVER {
        println!("  min_work       : NEVER (keep AVX2 for all shapes)");
    } else {
        println!(
            "  min_work       : {}  (≈{:.0}²)",
            report.min_work,
            (report.min_work as f64).sqrt()
        );
    }
    println!("  note           : {}", report.note);
    println!("\nEnv knobs:");
    println!("  MUD_GPU_GEMV=auto|1|0     policy (default auto)");
    println!("  MUD_GPU_GEMV_MIN=<work>   force threshold");
    println!("  MUD_GPU_GEMV_LOG=1        print sample table");
    println!("  MUD_USE_VULKAN=0          disable ash entirely");
}
