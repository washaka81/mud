use forge_llm::mud::corpus_trainer::{MudCorpusTrainer, SHOULD_TERMINATE};
use std::sync::atomic::Ordering;
use std::thread;
use std::time::Duration;

fn main() -> anyhow::Result<()> {
    let model_path = "models/core_skills.mud";
    let corpus_dir = "training/corpus";
    
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!(" 🌀 MUD NATIVE CORPUS ALIGNER v1.0");
    println!(" ══════════════════════════════════════════════════════════════════════");

    // Create corpus directory if not exists
    std::fs::create_dir_all(corpus_dir)?;

    let trainer = MudCorpusTrainer::new(model_path.to_string(), corpus_dir.to_string())?;

    // SIGINT Handler
    ctrlc::set_handler(move || {
        println!("\n🛑 Termination signal received. Saving weights and shutting down...");
        SHOULD_TERMINATE.store(true, Ordering::SeqCst);
    })?;

    trainer.run_alignment_session(16, 3)?; // Default: Batch 16, 3 Epochs

    Ok(())
}
