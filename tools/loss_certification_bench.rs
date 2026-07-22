//! # Stream K — Loss certification bench
//!
//! Runs a short (or full) STE QAT alignment session and asserts the loss
//! trajectory falls ([`forge_llm::mud::loss_cert`]).
//!
//! ```bash
//! # Fast gate (default): few epochs, exit 0/1
//! cargo run --release --bin loss_certification_bench -- --fast models/smollm2.mud
//!
//! # Full historical long run
//! MUD_LOSS_CERT_FAST=0 cargo run --release --bin loss_certification_bench -- models/smollm2.mud
//!
//! ./mud.sh cert-loss [model]
//! ```
//!
//! Exit codes:
//! - `0` — certified
//! - `1` — train failed or loss did not fall
//! - `2` — model missing (CI soft-skip)

use forge_llm::mud::corpus_trainer::MudCorpusTrainer;
use forge_llm::mud::loss_cert::{
    certify_trajectory, config_from_env, parse_metrics_log, LossCertConfig,
};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut fast = env::var("MUD_LOSS_CERT_FAST")
        .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(true);
    let mut model = None::<String>;
    for a in &args {
        if a == "--fast" {
            fast = true;
        } else if a == "--full" {
            fast = false;
        } else if a == "--help" || a == "-h" {
            eprintln!(
                "usage: loss_certification_bench [--fast|--full] [model.mud]\n\
                 env: MUD_LOSS_CERT_FAST=0|1  MUD_LOSS_CERT_EPOCHS  MUD_LOSS_CERT_BATCH\n\
                      MUD_LOSS_CERT_MIN_REL  MUD_LOSS_CERT_MIN_ABS  MUD_LOSS_CERT_MIN_POINTS"
            );
            return ExitCode::SUCCESS;
        } else if !a.starts_with('-') {
            model = Some(a.clone());
        }
    }

    let model_path = model.unwrap_or_else(|| {
        if PathBuf::from("models/smollm2.mud").is_file() {
            "models/smollm2.mud".into()
        } else if PathBuf::from("models/test_model.mud").is_file() {
            "models/test_model.mud".into()
        } else {
            String::new()
        }
    });

    if model_path.is_empty() || !PathBuf::from(&model_path).is_file() {
        eprintln!("[loss-cert] no model.mud found — soft-skip (exit 2)");
        eprintln!("  place models/smollm2.mud or pass path as argv");
        return ExitCode::from(2);
    }

    // Prefer CPU for deterministic short cert (optional override).
    if env::var("MUD_GPU_GEMV").is_err() {
        unsafe { env::set_var("MUD_GPU_GEMV", "0") };
    }
    if fast {
        unsafe {
            env::set_var("MUD_LOSS_CERT_FAST", "1");
            if env::var("MUD_TRAIN_SEQ_LEN").is_err() {
                env::set_var("MUD_TRAIN_SEQ_LEN", "16");
            }
        }
    }

    let epochs: usize = env::var("MUD_LOSS_CERT_EPOCHS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(if fast { 2 } else { 45 });
    let batch: usize = env::var("MUD_LOSS_CERT_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(if fast { 8 } else { 4 });

    let corpus_dir = "training/corpus_cert".to_string();
    let _ = fs::remove_dir_all(&corpus_dir);
    fs::create_dir_all(&corpus_dir).expect("corpus dir");
    // Deterministic multi-sentence corpus (enough tokens for several steps)
    let dummy = "\
Alice was beginning to get very tired of sitting by her sister on the bank.
The quick brown fox jumps over the lazy dog near the river bank at dawn.
MUD ternary training certifies that gradients flow and loss falls over steps.
Uno dos tres cuatro cinco seis siete ocho nueve diez once doce trece.
";
    fs::write(format!("{corpus_dir}/cert.txt"), dummy).expect("write corpus");

    // Fresh metrics log for this run only
    let metrics_path = "mud_train_metrics.log";
    let _ = fs::write(metrics_path, "");

    // Avoid resuming mid-cert
    let checkpoint_path = "weights/checkpoints/model_latest_checkpoint.mud";
    if PathBuf::from(checkpoint_path).is_file() {
        let _ = fs::remove_file(checkpoint_path);
        println!("[loss-cert] cleared checkpoint for clean cert");
    }

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  MUD LOSS CERTIFICATION  ·  stream K                     ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!("  model   : {model_path}");
    println!("  mode    : {}", if fast { "FAST" } else { "FULL" });
    println!("  epochs  : {epochs}   batch: {batch}");

    let trainer = match MudCorpusTrainer::new(model_path.clone(), corpus_dir.clone()) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[loss-cert] trainer init failed: {e}");
            cleanup(&corpus_dir);
            return ExitCode::from(1);
        }
    };

    let t0 = Instant::now();
    if let Err(e) = trainer.run_alignment_session(batch, epochs) {
        eprintln!("[loss-cert] training failed: {e}");
        cleanup(&corpus_dir);
        return ExitCode::from(1);
    }
    println!("  train_s : {:.1}", t0.elapsed().as_secs_f32());

    let content = fs::read_to_string(metrics_path).unwrap_or_default();
    let losses = parse_metrics_log(&content);
    println!("  points  : {}", losses.len());
    if !losses.is_empty() {
        println!(
            "  loss    : {:.4} → {:.4}",
            losses[0],
            losses[losses.len() - 1]
        );
    }

    let cfg = if fast {
        let mut c = LossCertConfig::fast();
        // Allow env overrides on top of fast defaults
        let env_c = config_from_env();
        c.min_relative_drop = env_c.min_relative_drop;
        c.min_absolute_drop = env_c.min_absolute_drop;
        c.min_points = env_c.min_points.min(c.min_points.max(3));
        c
    } else {
        config_from_env()
    };

    match certify_trajectory(&losses, &cfg) {
        Ok(r) => {
            println!("\n  ✅ CERTIFIED");
            println!(
                "     n={}  head_mean={:.4}  tail_mean={:.4}  Δ={:.4}  rel={:.2}%",
                r.n,
                r.head_mean,
                r.tail_mean,
                r.absolute_drop,
                r.relative_drop * 100.0
            );
            cleanup(&corpus_dir);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("\n  ❌ NOT CERTIFIED: {e}");
            if losses.len() < 3 {
                eprintln!(
                    "     tip: increase MUD_LOSS_CERT_EPOCHS or ensure telemetry writes \
                     mud_train_metrics.log"
                );
            }
            cleanup(&corpus_dir);
            ExitCode::from(1)
        }
    }
}

fn cleanup(corpus_dir: &str) {
    let _ = fs::remove_dir_all(corpus_dir);
}
