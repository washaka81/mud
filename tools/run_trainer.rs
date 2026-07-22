//! MUD corpus trainer entrypoint.
//!
//! Boot order (critical for Iris Xe / 15 GiB hosts):
//! 1. Parse CLI flags
//! 2. Apply host-safe train env defaults (no clobber of user exports)
//! 3. Capture HW cores + init PCorePool **before** any thread pin
//! 4. Pin main thread, print header, run session

use forge_llm::mud::corpus_trainer::{MudCorpusTrainer, SHOULD_TERMINATE};
use forge_llm::mud::trainer_ui::{box_bottom, box_kv, box_section, box_title, box_top};
use std::sync::atomic::Ordering;

/// Set env only if the user has not already exported it.
fn set_if_absent(k: &str, v: &str) {
    if std::env::var(k).is_err() {
        // SAFETY: single-threaded CLI setup before any trainer worker threads
        unsafe { std::env::set_var(k, v) };
    }
}

/// Host-safe defaults for **every** train mode (Iris Xe UMA / ~15 GiB design target).
/// User exports always win.
fn apply_host_train_defaults() {
    // Capture cores before any pin; also seeds OnceLock for default_pcore_threads.
    let n = forge_llm::mud::constants::capture_hw_pcore_threads();
    set_if_absent("MUD_PCORE_THREADS", &n.to_string());

    // iGPU GEMV thrash: upload+readback loses to AVX2 on UMA (see ASM_VULKAN_BOTTLENECK).
    set_if_absent("MUD_GPU_GEMV", "0");
    // Optional ash ctx for heartbeat / future NS; GEMV stays CPU via above.
    set_if_absent("MUD_USE_VULKAN", "1");
    // Skip ash QAT HOST_VISIBLE alloc (OOM → drop is fine, but avoid the attempt).
    set_if_absent("MUD_TRAIN_EZOP", "0");
    // Protect emb + skip ~1 GiB FP32 materialize on large vocab.
    set_if_absent("MUD_TRAIN_FREEZE_EMB", "1");
    // More signal per chunk → stable loss descent (finding: 16-32 steps/chunk
    // produced noisy 3.4↔8.9 oscillation on low-resource hosts). 64 validated.
    set_if_absent("MUD_TRAIN_STEPS_PER_CHUNK", "64");
    // Corpus txt/md only — do not scrape entire src/**/*.rs into AOT by default.
    set_if_absent("MUD_TRAIN_TEXT_ONLY", "1");
}

/// Post-convert recovery recipe (`--align` / `--post-convert`).
fn apply_align_defaults(batch_size: &mut usize, epochs: &mut usize) {
    set_if_absent("MUD_TRAIN_ALIGN", "1");
    set_if_absent("MUD_TRAIN_TEXT_ONLY", "1");
    set_if_absent("MUD_TRAIN_MAX_CHUNKS", "32");
    set_if_absent("MUD_TRAIN_SEQ_LEN", "32");
    set_if_absent("MUD_TRAIN_STEPS_PER_CHUNK", "64");
    set_if_absent("MUD_TRAIN_NUM_NEG", "63");
    set_if_absent("MUD_TRAIN_LAST_N_LAYERS", "8");
    set_if_absent("MUD_TRAIN_FREEZE_EMB", "1");
    set_if_absent("MUD_TRAIN_CKPT_EVERY", "0");
    set_if_absent("MUD_OPT", "sgd");
    set_if_absent("MUD_QAT_LR", "0.0008");
    set_if_absent("MUD_PCORE_THREADS", "8");
    set_if_absent("MUD_GPU_GEMV", "0");
    set_if_absent("MUD_TRAIN_EZOP", "0");

    let align_opts = format!(
        "TEXT_ONLY MAX_CHUNKS={} SEQ={} STEPS={} NUM_NEG={} OPT={} LR={} LAST_N={} GEMV=0 EZOP=0",
        std::env::var("MUD_TRAIN_MAX_CHUNKS").unwrap_or_default(),
        std::env::var("MUD_TRAIN_SEQ_LEN").unwrap_or_default(),
        std::env::var("MUD_TRAIN_STEPS_PER_CHUNK").unwrap_or_default(),
        std::env::var("MUD_TRAIN_NUM_NEG").unwrap_or_default(),
        std::env::var("MUD_OPT").unwrap_or_default(),
        std::env::var("MUD_QAT_LR").unwrap_or_default(),
        std::env::var("MUD_TRAIN_LAST_N_LAYERS").unwrap_or_default(),
    );
    println!(
        "{}",
        forge_llm::mud::trainer_ui::note("warn", &format!("--align mode: {}", align_opts))
    );
    println!(
        "{}",
        forge_llm::mud::trainer_ui::note(
            "ram",
            "telemetry -> mud_train_metrics.log + stderr [TELEM]  ·  TUI: cargo run --release --bin train_telemetry"
        )
    );
    *batch_size = 32;
    *epochs = 1;
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut batch_size = 16usize;
    let mut epochs = 3usize;

    let align_mode = args.iter().any(|x| x == "--align" || x == "--post-convert");

    // ── 1) Env defaults (before pool / pin) ──────────────────────────────────
    apply_host_train_defaults();
    if align_mode {
        apply_align_defaults(&mut batch_size, &mut epochs);
    }

    if let Some(pos) = args.iter().position(|x| x == "--epochs") {
        if let Some(val) = args.get(pos + 1) {
            epochs = val.parse()?;
        }
    }
    if let Some(pos) = args.iter().position(|x| x == "--batch") {
        if let Some(val) = args.get(pos + 1) {
            batch_size = val.parse()?;
        }
    }

    // ── 2) PCorePool + core count before pin (L-07) ──────────────────────────
    // Must read affinity **before** set_for_current — post-pin get_core_ids → 1 core.
    let num_p_cores = core_affinity::get_core_ids()
        .map(|ids| ids.len().clamp(1, 8))
        .unwrap_or_else(forge_llm::mud::constants::capture_hw_pcore_threads);
    let pool_n = forge_llm::mud::pcore_pool::global_pool_threads();

    // ── 3) Pin main thread (after core count is cached) ──────────────────────
    if let Some(core_ids) = core_affinity::get_core_ids() {
        if let Some(p_core) = core_ids.iter().min_by_key(|id| id.id) {
            core_affinity::set_for_current(*p_core);
        }
    }

    // Real device probe (ash) — not just the env flag, so the banner can't lie.
    let (vulkan_active, gpu_desc) = forge_llm::vulkan::ash_backend::probe_gpu();
    let mkl_type = std::env::var("MKL_DEBUG_CPU_TYPE").unwrap_or_else(|_| "not set".into());
    // GEMV dispatch: default policy is `auto` (one-shot micro-bench), NOT 0.
    let gpu_gemv = std::env::var("MUD_GPU_GEMV").unwrap_or_else(|_| "auto".into());

    // ── 4) Resolve model path ────────────────────────────────────────────────
    let passed_model_path = args.iter().rfind(|a| a.ends_with(".mud")).cloned();

    let checkpoint_path = "weights/checkpoints/model_latest_checkpoint.mud";
    let cwd_mud = std::fs::read_dir(".").ok().and_then(|mut d| {
        d.find_map(|e| {
            e.ok()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|ext| ext == "mud"))
        })
    });

    // Explicit CLI .mud always wins (don't silently resume a stale checkpoint).
    // Checkpoint auto-resume only when no model path was passed.
    let (model_path, resumed) = if let Some(path) = passed_model_path {
        (path, false)
    } else if std::path::Path::new(checkpoint_path).exists() {
        (checkpoint_path.to_string(), true)
    } else if let Some(path) = cwd_mud {
        (path.to_string_lossy().to_string(), false)
    } else {
        let p = std::fs::read_dir("models")
            .ok()
            .and_then(|mut d| {
                d.find_map(|e| {
                    e.ok()
                        .map(|e| e.path())
                        .filter(|p| p.extension().is_some_and(|ext| ext == "mud"))
                })
            })
            .map(|p| p.to_string_lossy().to_string())
            .expect("No .mud model found in arguments, checkpoint, cwd, or models/ directory!");
        (p, false)
    };

    let model_size_bytes = std::fs::metadata(&model_path).map(|m| m.len()).unwrap_or(0);
    let model_size_mb = model_size_bytes as f64 / 1_048_576.0;

    let corpus_dir = "training/corpus";
    std::fs::create_dir_all(corpus_dir)?;
    std::fs::create_dir_all("weights/checkpoints")?;

    let mut corpus_files = 0usize;
    let mut corpus_bytes = 0u64;
    if let Ok(entries) = std::fs::read_dir(corpus_dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.extension()
                .is_some_and(|x| x == "txt" || x == "rs" || x == "md")
            {
                corpus_files += 1;
                corpus_bytes += std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
            }
        }
    }
    let _ = corpus_bytes; // corpus size surfaced by trainer core box, not the header

    let status_str = if resumed {
        "Resumed from checkpoint (checkpoint auto-detected)"
    } else {
        "Fresh run (no checkpoint found)"
    };

    // ── 5) HEADER (slim; the full Session/Training/Architecture box is printed
    //        by the trainer core once the model is loaded — single authority). ──
    println!("{}", box_top());
    println!("{}", box_title("MUD NATIVE CORPUS ALIGNER  ·  STE QAT"));
    println!("{}", box_section("Hardware"));
    println!(
        "{}",
        box_kv(
            "PCorePool",
            &format!("{pool_n} threads  (MUD_PCORE_THREADS / L-07)  ·  Rayon OFF (P-27)")
        )
    );
    println!(
        "{}",
        box_kv(
            "Cores",
            &format!("{num_p_cores} detected (affinity scan)  ·  MKL {mkl_type}")
        )
    );
    println!(
        "{}",
        box_kv(
            "Vulkan",
            &if vulkan_active {
                format!("ash ON · {gpu_desc}  ·  GEMV={gpu_gemv}")
            } else {
                format!("OFF ({gpu_desc})  ·  GEMV={gpu_gemv}")
            }
        )
    );
    println!("{}", box_section("Session"));
    println!("{}", box_kv("Model", &model_path));
    println!(
        "{}",
        box_kv(
            "Size",
            &format!("{model_size_mb:.2} MB  ({model_size_bytes} bytes)")
        )
    );
    println!("{}", box_kv("Status", status_str));
    println!(
        "{}",
        box_kv(
            "Corpus",
            &format!(
                "{} files, {:.2} MB (training/corpus)",
                corpus_files,
                corpus_bytes as f64 / 1_048_576.0
            )
        )
    );
    println!("{}", box_bottom());
    println!();

    // ── SIGINT Handler ────────────────────────────────────────────────────────
    ctrlc::set_handler(move || {
        println!(
            "\n{}",
            forge_llm::mud::trainer_ui::note("warn", "termination received — saving weights...")
        );
        SHOULD_TERMINATE.store(true, Ordering::SeqCst);
    })?;

    let mut trainer = MudCorpusTrainer::new(model_path.to_string(), corpus_dir.to_string())?;

    if let Some(pos) = args.iter().position(|x| x == "--distill") {
        let file = args.get(pos + 1).map(|s| s.as_str()).unwrap_or("");
        eprintln!(
            "  \x1b[1;31m[distill]\x1b[0m Agentic distillation is not yet wired to the STE QAT core (training stub removed)."
        );
        eprintln!(
            "  \x1b[1;36m[distill]\x1b[0m Falling back to corpus alignment on trace file '{}'.",
            file
        );
        trainer.run_alignment_session(batch_size, epochs)?;
    } else if args.iter().any(|x| x == "--debate") {
        trainer.run_debate_session(
            None,
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        )?;
    } else if args.iter().any(|x| x == "--circuit") {
        trainer.run_training_circuit(
            None,
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        )?;
    } else {
        trainer.run_alignment_session(batch_size, epochs)?;
    }

    Ok(())
}
