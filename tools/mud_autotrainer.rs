use forge_llm::mud::auto_trainer::MudAutoTrainer;
use std::thread;
use std::time::Duration;
use rusqlite::Connection;
use std::path::Path;

// ANSI Colors for premium dashboard
const CYAN: &str = "\x1b[36m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const RED: &str = "\x1b[31m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

fn main() -> anyhow::Result<()> {
    println!("{}=================================================={}", BOLD, RESET);
    println!("{}🌀  MUD ENGINE NATIVE AUTONOMOUS LEARNING DAEMON 🌀{}", BOLD, RESET);
    println!("{}=================================================={}", BOLD, RESET);

    let model_path = std::env::args().nth(1).unwrap_or_else(|| "models/core_skills.mud".to_string());
    let db_path = std::env::args().nth(2).unwrap_or_else(|| "models/knowledge.db".to_string());
    
    if !Path::new(&db_path).exists() {
        println!("{}⚠️  Knowledge base database not found at {}. Creating new database...{}", YELLOW, db_path, RESET);
        let conn = Connection::open(&db_path)?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                embedding BLOB,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status INTEGER DEFAULT 0
            )",
            [],
        )?;
        println!("{}✅ Database created successfully.{}", GREEN, RESET);
    }

    if !Path::new(&model_path).exists() {
        println!("{}❌ Critical Error: Model file '{}' not found. Cannot train.{}", RED, model_path, RESET);
        std::process::exit(1);
    }

    let trainer = MudAutoTrainer::new(db_path.to_string(), 1, model_path.to_string());
    println!("{}🤖 Auto-Trainer instantiated:{}", CYAN, RESET);
    println!("   - Knowledge Base: {}{}{}", BOLD, db_path, RESET);
    println!("   - Model Weights:  {}{}{}", BOLD, model_path, RESET);
    println!("   - Threshold:      {}1 fact(s){}", BOLD, RESET);
    println!("\n{}🚀 Monitoring knowledge base for new chunks in the background...{}", GREEN, RESET);
    println!("   (Press Ctrl+C to terminate the daemon gracefully)\n");

    // Register Ctrl+C handler for graceful shutdown
    ctrlc::set_handler(|| {
        println!("\n{}🛑 Ctrl+C detected! Requesting graceful shutdown... Saving weights.{}", RED, RESET);
        forge_llm::mud::auto_trainer::SHOULD_TERMINATE.store(true, std::sync::atomic::Ordering::SeqCst);
    }).expect("Error setting Ctrl+C handler");

    loop {
        if forge_llm::mud::auto_trainer::SHOULD_TERMINATE.load(std::sync::atomic::Ordering::SeqCst) {
            println!("{}👋 Graceful shutdown complete. Exiting autotrainer safely.{}", GREEN, RESET);
            break;
        }

        let conn = Connection::open(&db_path)?;
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM facts WHERE status = 0",
            [],
            |row| row.get(0)
        ).unwrap_or(0);

        if count > 0 {
            println!("{}   ⏳ Found {} unassimilated fact(s). Triggering Hot Ternary SGD training cycle...{}", YELLOW, count, RESET);
            match trainer.run_training_cycle() {
                Ok(n) => {
                    println!("{}   ✅ Successfully assimilated {} facts into the ternary expert weights!{}", GREEN, n, RESET);
                    // Print updated model metrics by calling the system audit tool's weight_audit logic
                    println!("{}   📊 Running quick weight health audit...{}", CYAN, RESET);
                    if let Ok(model) = forge_llm::mud::MudFile::load(&model_path) {
                        if let Some(core) = model.skills.get("core") {
                            if let Some(tensor) = core.tensors.get("blk.0.expert.0.w1.weight") {
                                let n_elements = tensor.shape.iter().copied().product::<usize>();
                                let n_u32 = (n_elements + 15) / 16;
                                let data_ptr = tensor.data_ptr as *const u32;
                                let packed_data = unsafe { std::slice::from_raw_parts(data_ptr, n_u32) };
                                
                                let mut counts = [0usize; 3];
                                for &val in packed_data {
                                    for i in 0..16 {
                                        let bits = (val >> (i * 2)) & 3;
                                        if bits == 1 { counts[1] += 1; }
                                        else if bits == 2 { counts[2] += 1; }
                                        else { counts[0] += 1; }
                                    }
                                }
                                let total = counts[0] + counts[1] + counts[2];
                                let variance = (counts[1] as f32 * 1.0 + counts[2] as f32 * 1.0) / total as f32;
                                let sigma = variance.sqrt();
                                println!("      Expert 0.w1 | Sigma: {:.4} | Pos/Neg Ratio: {:.1}% / {:.1}%", 
                                         sigma, (counts[1] as f32 / total as f32) * 100.0, (counts[2] as f32 / total as f32) * 100.0);
                            }
                        }
                    }
                    println!();
                }
                Err(e) => {
                    println!("{}   ❌ Training Cycle failed: {}{}", RED, e, RESET);
                }
            }
        }

        // Fast-response sleep loop (5 seconds total in 100ms intervals)
        for _ in 0..50 {
            if forge_llm::mud::auto_trainer::SHOULD_TERMINATE.load(std::sync::atomic::Ordering::SeqCst) {
                break;
            }
            thread::sleep(Duration::from_millis(100));
        }
    }

    Ok(())
}
