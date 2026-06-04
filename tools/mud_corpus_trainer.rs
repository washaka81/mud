use forge_llm::mud::corpus_trainer::{MudCorpusTrainer, SHOULD_TERMINATE};
use std::sync::atomic::Ordering;

fn main() -> anyhow::Result<()> {
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
    let model_path = args
        .iter()
        .find(|a| a.ends_with(".mud"))
        .cloned()
        .unwrap_or_else(|| "models/core_skills.mud".to_string());
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

    let trainer = MudCorpusTrainer::new(model_path.to_string(), corpus_dir.to_string())?;

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
    } else {
        println!("📝 Modality: Standard Corpus Alignment (Self-Supervised)");
        trainer.run_alignment_session(batch_size, epochs)?;
    }

    Ok(())
}
