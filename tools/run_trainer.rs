use forge_llm::mud::corpus_trainer::{MudCorpusTrainer, SHOULD_TERMINATE};
use std::sync::atomic::Ordering;

fn main() -> anyhow::Result<()> {
    // Hardware-specific optimization: Initialize P-Core Pool before pinning the main thread.
    // This ensures the pool workers can correctly query and bind to all available P-cores 
    // before the main thread restricts its own CPU affinity mask.
    let _ = forge_llm::mud::pcore_pool::get_pool();

    // Hardware-specific optimization: Pin main thread to a P-Core
    if let Some(core_ids) = core_affinity::get_core_ids() {
        if let Some(p_core) = core_ids.iter().find(|id| id.id < 8) {
            core_affinity::set_for_current(*p_core);
            let num_p = core_ids.iter().filter(|id| id.id < 8).count();
            println!("  🧠 Pinned trainer thread to P-core {}, {} P-cores available", p_core.id, num_p);
        }
    }

    let args: Vec<String> = std::env::args().collect();
    let mut batch_size = 16;
    let mut epochs = 3;

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
    let passed_model_path = args
        .iter().rfind(|a| a.ends_with(".mud"))
        .cloned();
        
    // Priority 13 & Checkpoint Auto-Resume:
    let checkpoint_path = "weights/checkpoints/model_latest_checkpoint.mud";
    let cwd_mud = std::fs::read_dir(".").ok().and_then(|mut d| d.find_map(|e| e.ok().map(|e| e.path()).filter(|p| p.extension().map_or(false, |ext| ext == "mud"))));
    
    let model_path = if std::path::Path::new(checkpoint_path).exists() {
        println!("  ♻️  Found latest checkpoint. Resuming from: {}", checkpoint_path);
        checkpoint_path.to_string()
    } else if let Some(path) = passed_model_path {
        path
    } else if let Some(path) = cwd_mud {
        path.to_string_lossy().to_string()
    } else {
        // Fallback scan inside models/
        std::fs::read_dir("models").ok().and_then(|mut d| d.find_map(|e| e.ok().map(|e| e.path()).filter(|p| p.extension().map_or(false, |ext| ext == "mud"))))
            .map(|p| p.to_string_lossy().to_string())
            .expect("No .mud model found in arguments, checkpoint, cwd, or models/ directory!")
    };
    let corpus_dir = "training/corpus";
    println!("  🚀 Starting Deep Epoch Linguistic Alignment Session...");

    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!(
        " 🌀 MUD NATIVE CORPUS ALIGNER v1.1 (Batch: {}, Epochs: {})",
        batch_size, epochs
    );
    println!(" ══════════════════════════════════════════════════════════════════════");

    // Create corpus and checkpoint directories
    std::fs::create_dir_all(corpus_dir)?;
    std::fs::create_dir_all("weights/checkpoints")?;

    let mut trainer = MudCorpusTrainer::new(model_path.to_string(), corpus_dir.to_string())?;

    let mut distill_file = None;
    if let Some(pos) = args.iter().position(|x| x == "--distill") {
        if let Some(val) = args.get(pos + 1) {
            distill_file = Some(val.clone());
        }
    }

    // SIGINT Handler
    ctrlc::set_handler(move || {
        println!("\n🛑 Termination signal received. Saving weights and shutting down...");
        SHOULD_TERMINATE.store(true, Ordering::SeqCst);
    })?;

    if let Some(file) = distill_file {
        println!("🧠 Modality: Agentic Distillation (QAT) using {}", file);
        let mut t = trainer;
        t.distill_workflow(&file)?;
    } else if args.iter().any(|x| x == "--debate") {
        println!("⚔️  Modality: RLVR Debate Arena");
        trainer.run_debate_session(None)?;
    } else {
        println!("📝 Modality: Standard Corpus Alignment (Self-Supervised)");
        trainer.run_alignment_session(batch_size, epochs)?;
    }

    Ok(())
}
