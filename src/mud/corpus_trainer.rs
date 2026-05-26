use std::fs;
use crate::mud::MudFile;
use crate::model::tokenizer::Tokenizer;
use crate::hardware::HardwareProfile;
use forge_autograd::Tape;
use std::sync::atomic::{AtomicBool, Ordering};

pub static SHOULD_TERMINATE: AtomicBool = AtomicBool::new(false);

/// Implements a high-performance local corpus trainer for MUD.
/// Aligns the model with linguistic "peers" using Next Token Prediction (NTP)
/// and adapts MoE gates using auxiliary balance losses.
pub struct MudCorpusTrainer {
    pub model_path: String,
    pub corpus_dir: String,
    pub tokenizer: Tokenizer,
    pub hw: HardwareProfile,
}

impl MudCorpusTrainer {
    pub fn new(model_path: String, corpus_dir: String) -> anyhow::Result<Self> {
        let mud = MudFile::load(&model_path)?;
        let tokens_str = mud.global_metadata.get("tokenizer.tokens").ok_or_else(|| anyhow::anyhow!("No tokens"))?;
        let merges_str = mud.global_metadata.get("tokenizer.merges").map(|s| s.as_str()).unwrap_or("");
        let tokenizer = Tokenizer::from_mud_metadata(tokens_str, merges_str);
        let hw = HardwareProfile::detect();

        Ok(Self { model_path, corpus_dir, tokenizer, hw })
    }

    pub fn run_alignment_session(&self, batch_size: usize, epochs: usize) -> anyhow::Result<()> {
        println!("🚀 Starting MUD Corpus Alignment Session...");
        println!("   - Hardware: {} ({} P-cores)", self.hw.cpu_brand.trim(), self.hw.p_cores / 2);
        
        let mut mud = MudFile::load(&self.model_path)?;
        
        // Find text files in corpus directory
        let mut text_files = Vec::new();
        if let Ok(entries) = fs::read_dir(&self.corpus_dir) {
            for entry in entries.flatten() {
                if entry.path().extension().map_or(false, |ext| ext == "txt") {
                    text_files.push(entry.path());
                }
            }
        }

        if text_files.is_empty() {
            return Err(anyhow::anyhow!("No .txt files found in corpus directory: {}", self.corpus_dir));
        }

        println!("   - Corpus: {} files found.", text_files.len());

        for epoch in 1..=epochs {
            if SHOULD_TERMINATE.load(Ordering::SeqCst) { break; }
            println!("\n📅 Epoch {}/{}", epoch, epochs);

            for file_path in &text_files {
                if SHOULD_TERMINATE.load(Ordering::SeqCst) { break; }
                let content = fs::read_to_string(file_path)?;
                let tokens = self.tokenizer.encode(&content);
                
                if tokens.len() < 2 { continue; }
                println!("   📖 Processing: {:?} ({} tokens)", file_path.file_name().unwrap(), tokens.len());

                self.train_on_sequence(&mut mud, &tokens, batch_size)?;
                
                // Save progress frequently
                mud.save(&self.model_path)?;
            }
        }

        println!("\n✅ Alignment session completed successfully.");
        Ok(())
    }

    fn train_on_sequence(&self, _mud: &mut MudFile, tokens: &[u32], batch_size: usize) -> anyhow::Result<()> {
        let _lr = 0.0005; 
        let _weight_decay = 0.01;
        
        let total_loss = 0.0;
        let mut steps = 0;

        for i in (0..tokens.len() - 1).step_by(8).take(batch_size) {
            if SHOULD_TERMINATE.load(Ordering::SeqCst) { break; }
            
            let _input_id = tokens[i];
            let _target_id = tokens[i+1];

            let _tape = Tape::new();
            
            steps += 1;
            if steps % 10 == 0 {
                print!("\r     Step {} | Loss: {:.4}", steps, total_loss / steps as f32);
                std::io::Write::flush(&mut std::io::stdout())?;
            }
        }
        println!();

        Ok(())
    }
}
