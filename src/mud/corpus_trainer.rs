use std::fs;
use std::time::{Instant, Duration};
use crate::mud::{MudFile, MudTensorType};
use crate::model::tokenizer::Tokenizer;
use crate::hardware::HardwareProfile;
use forge_autograd::Tape;
use std::sync::atomic::{AtomicBool, Ordering};

pub static SHOULD_TERMINATE: AtomicBool = AtomicBool::new(false);

/// Implements a high-performance local corpus trainer for MUD.
pub struct MudCorpusTrainer {
    pub model_path: String,
    pub corpus_dir: String,
    pub tokenizer: Tokenizer,
    pub hw: HardwareProfile,
}

impl MudCorpusTrainer {
    pub fn new(model_path: String, corpus_dir: String) -> anyhow::Result<Self> {
        let mud = MudFile::load(&model_path)?;
        Self::validate_metadata(&mud)?;

        let tokens_str = mud.global_metadata.get("tokenizer.tokens").ok_or_else(|| anyhow::anyhow!("No tokens"))?;
        let merges_str = mud.global_metadata.get("tokenizer.merges").map(|s| s.as_str()).unwrap_or("");
        let tokenizer = Tokenizer::from_mud_metadata(tokens_str, merges_str);
        let hw = HardwareProfile::detect();

        let trainer = Self { model_path, corpus_dir, tokenizer, hw };
        trainer.audit_tokenization();
        Ok(trainer)
    }

    fn validate_metadata(mud: &MudFile) -> anyhow::Result<()> {
        println!("📊 Phase 0: Metadata Integrity Validation...");
        let required_keys = ["hidden_size", "num_layers", "num_experts", "tokenizer.tokens"];
        for key in required_keys {
            if !mud.global_metadata.contains_key(key) {
                anyhow::bail!("CRITICAL: Missing essential metadata key: '{}'", key);
            }
        }
        if let Some(core) = mud.skills.get("core") {
            let mut ternary_count = 0;
            let mut scale_count = 0;
            for (name, tensor) in &core.tensors {
                if tensor.t_type == MudTensorType::Ternary2Bit { ternary_count += 1; }
                if name.ends_with(".scale") { scale_count += 1; }
            }
            println!("   - Tensors: Found {} ternary weights and {} scales.", ternary_count, scale_count);
        }
        println!("   ✅ Metadata validated successfully.");
        Ok(())
    }

    fn audit_tokenization(&self) {
        println!("📊 Phase 1: Tokenization Sync Audit...");
        let test_phrases = ["MUD engine optimized.", "Inteligencia artificial.", "BPE Hello World!"];
        for phrase in test_phrases {
            let ids = self.tokenizer.encode(phrase);
            let decoded = self.tokenizer.decode(&ids);
            println!("   - Original: \"{}\" | Decoded: \"{}\"", phrase, decoded.trim());
        }
        println!("   ✅ Tokenization audit complete.");
    }

    pub fn run_alignment_session(&self, batch_size: usize, epochs: usize) -> anyhow::Result<()> {
        println!("🚀 Starting MUD Corpus Alignment Session...");
        let mut mud = MudFile::load(&self.model_path)?;
        
        let mut shadow_emb = {
            let core = mud.skills.get("core").ok_or_else(|| anyhow::anyhow!("No core skill"))?;
            let emb_tensor = core.tensors.get("token_embd.weight").ok_or_else(|| anyhow::anyhow!("No embedding"))?;
            let elements = emb_tensor.shape[0] * emb_tensor.shape[1];
            let mut data = vec![0.0f32; elements];
            unsafe {
                if emb_tensor.t_type == MudTensorType::Ternary2Bit {
                    crate::mud::dequantize_ternary_row(emb_tensor.data_ptr as *const u32, &mut data, elements);
                } else {
                    std::ptr::copy_nonoverlapping(emb_tensor.data_ptr as *const f32, data.as_mut_ptr(), elements);
                }
            }
            data
        };
        
        let mut text_files = Vec::new();
        if let Ok(entries) = fs::read_dir(&self.corpus_dir) {
            for entry in entries.flatten() {
                if entry.path().extension().map_or(false, |ext| ext == "txt") {
                    text_files.push(entry.path());
                }
            }
        }
        if text_files.is_empty() { anyhow::bail!("No .txt files in {}", self.corpus_dir); }

        let resume_epoch = mud.global_metadata.get("trainer.current_epoch").and_then(|v| v.parse::<usize>().ok()).unwrap_or(1);
        let resume_file_idx = mud.global_metadata.get("trainer.current_file_idx").and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
        let resume_chunk_idx = mud.global_metadata.get("trainer.current_chunk_idx").and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);

        let mut _total_chars = 0u64;
        for path in &text_files { _total_chars += fs::metadata(path)?.len(); }
        let chunk_size = 50000;
        let chunks_per_file: Vec<usize> = text_files.iter().map(|p| (fs::metadata(p).unwrap().len() as usize).div_ceil(chunk_size)).collect();
        let total_chunks_per_epoch: usize = chunks_per_file.iter().sum();
        let total_chunks_all_epochs = total_chunks_per_epoch * epochs;

        let start_time = Instant::now();
        let mut global_chunks_processed = 0usize;

        for epoch in 1..=epochs {
            if SHOULD_TERMINATE.load(Ordering::SeqCst) { break; }
            if epoch < resume_epoch { global_chunks_processed += total_chunks_per_epoch; continue; }
            
            for (f_idx, file_path) in text_files.iter().enumerate() {
                if SHOULD_TERMINATE.load(Ordering::SeqCst) { break; }
                if epoch == resume_epoch && f_idx < resume_file_idx { global_chunks_processed += chunks_per_file[f_idx]; continue; }
                
                let content = fs::read_to_string(file_path)?;
                let chars: Vec<char> = content.chars().collect();
                let file_chunks = chunks_per_file[f_idx];

                for (c_idx, chunk) in chars.chunks(chunk_size).enumerate() {
                    if SHOULD_TERMINATE.load(Ordering::SeqCst) { break; }
                    if epoch == resume_epoch && f_idx == resume_file_idx && c_idx < resume_chunk_idx { global_chunks_processed += 1; continue; }
                    
                    let chunk_str: String = chunk.iter().collect();
                    let tokens = self.tokenizer.encode(&chunk_str);
                    if tokens.len() < 2 { continue; }
                    global_chunks_processed += 1;
                    
                    let elapsed = start_time.elapsed();
                    let progress = (global_chunks_processed as f32 / total_chunks_all_epochs as f32) * 100.0;
                    let chunks_per_sec = global_chunks_processed as f32 / elapsed.as_secs_f32();
                    let remaining_chunks = total_chunks_all_epochs.saturating_sub(global_chunks_processed);
                    let eta = if chunks_per_sec > 0.0 { Duration::from_secs_f32(remaining_chunks as f32 / chunks_per_sec) } else { Duration::ZERO };

                    print!("\x1B[2J\x1B[H");
                    println!("╔══════════════════════════════════════════════════════════════════════╗");
                    println!("║  🌀 MUD LINGUISTIC RECALIBRATION MONITOR                             ║");
                    println!("╠══════════════════════════════════════════════════════════════════════╣");
                    println!("║  PROGRESS: [{:<20}] {:>5.1}% | EPOCH: {}/{}             ║", "█".repeat((progress / 5.0) as usize) + &"░".repeat(20usize.saturating_sub((progress / 5.0) as usize)), progress, epoch, epochs);
                    println!("║  CURRENT : {:<57} ║", file_path.file_name().unwrap().to_string_lossy());
                    println!("║  CHUNK   : {:>5}/{} in file | GLOBAL: {:>6}/{}            ║", c_idx + 1, file_chunks, global_chunks_processed, total_chunks_all_epochs);
                    println!("╠══════════════════════════════════════════════════════════════════════╣");
                    println!("║  VELOCITY: {:>10.1} chunks/hr | ELAPSED: {:>10.1?}      ║", chunks_per_sec * 3600.0, elapsed);
                    println!("║  TOTAL ETA: \x1B[1;32m{:>21.1?}\x1B[0m | BATCH: {:>12}      ║", eta, batch_size);
                    println!("╚══════════════════════════════════════════════════════════════════════╝");
                    let _ = std::io::Write::flush(&mut std::io::stdout());

                    self.train_on_sequence(&mut mud, &mut shadow_emb, &tokens, batch_size)?;

                    if global_chunks_processed > 0 && global_chunks_processed % 5000 == 0 {
                        self.save_checkpoint(&mut mud, &shadow_emb, format!("chunk_{}", global_chunks_processed))?;
                    }
                }
                mud.global_metadata.insert("trainer.current_epoch".to_string(), epoch.to_string());
                mud.global_metadata.insert("trainer.current_file_idx".to_string(), f_idx.to_string());
                mud.global_metadata.insert("trainer.current_chunk_idx".to_string(), file_chunks.to_string());
                self.sync_shadow_to_mud(&mut mud, &shadow_emb);
                mud.save(&self.model_path)?;
            }
            if !SHOULD_TERMINATE.load(Ordering::SeqCst) {
                self.save_checkpoint(&mut mud, &shadow_emb, format!("epoch_{}", epoch))?;
            }
        }
        println!("\n✅ Alignment session completed.");
        Ok(())
    }

    fn sync_shadow_to_mud(&self, mud: &mut MudFile, shadow_emb: &[f32]) {
        let core = mud.skills.get_mut("core").unwrap();
        let emb_tensor = core.tensors.get_mut("token_embd.weight").unwrap();
        emb_tensor.t_type = MudTensorType::Float32;
        let bytes = unsafe { std::slice::from_raw_parts(shadow_emb.as_ptr() as *const u8, shadow_emb.len() * 4) }.to_vec();
        emb_tensor.owned_data = Some(bytes);
    }

    fn save_checkpoint(&self, mud: &mut MudFile, shadow_emb: &[f32], suffix: String) -> anyhow::Result<()> {
        let checkpoint_name = format!("weights/checkpoints/core_skills_{}.mud", suffix);
        self.sync_shadow_to_mud(mud, shadow_emb);
        mud.save(&checkpoint_name)?;
        Ok(())
    }

    fn train_on_sequence(&self, mud: &mut MudFile, shadow_emb: &mut [f32], tokens: &[u32], batch_size: usize) -> anyhow::Result<()> {
        let lr = 0.001; 
        let hidden_size = mud.global_metadata.get("hidden_size").and_then(|v| v.parse::<usize>().ok()).unwrap_or(512);
        let vocab_size = shadow_emb.len() / hidden_size;
        for i in (0..tokens.len() - 1).step_by(8).take(batch_size) {
            if SHOULD_TERMINATE.load(Ordering::SeqCst) { break; }
            let input_id = tokens[i] as usize;
            if input_id >= vocab_size { continue; }
            let mut tape = Tape::new();
            let x_data = shadow_emb[input_id * hidden_size .. (input_id + 1) * hidden_size].to_vec();
            let x_node = tape.push_leaf(x_data, vec![1, hidden_size]);
            let loss_node = tape.cross_entropy(x_node, 0); 
            tape.backward(loss_node);
            let x_grad = &tape.nodes[x_node.0].grad;
            for j in 0..hidden_size { shadow_emb[input_id * hidden_size + j] -= lr * x_grad[j]; }
        }
        Ok(())
    }
}
