use std::thread;
use std::time::Duration;
use std::sync::Arc;
use crate::ai::store::MudStore;

/// Background Daemon for MUD Autonomous Learning.
pub struct MudAutoTrainer {
    store: Arc<MudStore>,
    threshold: usize,
}

impl MudAutoTrainer {
    pub fn new(store: Arc<MudStore>, threshold: usize) -> Self {
        Self { store, threshold }
    }

    /// Starts a background thread that monitors the store for new knowledge.
    pub fn start_background_monitor(&self) {
        let store = self.store.clone();
        let threshold = self.threshold;

        thread::spawn(move || {
            println!("  [Auto-Trainer] Background monitor active.");
            loop {
                match store.get_unassimilated() {
                    Ok(unassimilated) => {
                        if unassimilated.len() >= threshold {
                            println!("  [Auto-Trainer] Knowledge threshold reached ({} facts). Triggering digestion...", unassimilated.len());
                            self::trigger_training();
                        }
                    }
                    Err(e) => eprintln!("  [Auto-Trainer] Monitor error: {}", e),
                }
                thread::sleep(Duration::from_secs(60)); // Check every minute
            }
        });
    }
}

fn trigger_training() {
    println!("  [Auto-Trainer] ETA for Digestion: ~4.5 minutes (based on GPU P100 throughput).");
    // Trigger the shell script for Kaggle push
    let _ = std::process::Command::new("./training/push_to_kaggle.sh").output();
}
