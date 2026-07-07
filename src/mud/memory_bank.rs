use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Priority 12: Continuous Context Persistence (Memory Banks)
/// Replaces ephemeral KV-caching with persistent vector-mapped memory banks (RAG-lite),
/// enabling the agent to maintain context across multi-day coding sessions and reboots.
pub struct MemoryBank {
    db_path: PathBuf,
    memories: HashMap<String, MemoryRecord>,
}

#[derive(Clone, Debug)]
pub struct MemoryRecord {
    pub id: String,
    pub content: String,
    pub timestamp: u64,
    // In a full implementation, this would contain a compressed KV-cache state or embedding vector
    // pub vector: Vec<f32>, 
}

impl MemoryBank {
    /// Initializes a new Memory Bank mapped to a persistent directory.
    pub fn new<P: AsRef<Path>>(storage_dir: P) -> std::io::Result<Self> {
        let db_path = storage_dir.as_ref().join("memory_bank.json");
        let mut bank = Self {
            db_path: db_path.clone(),
            memories: HashMap::new(),
        };

        if db_path.exists() {
            bank.load_from_disk()?;
        } else {
            if let Some(parent) = db_path.parent() {
                fs::create_dir_all(parent)?;
            }
        }

        Ok(bank)
    }

    /// Stores a new memory record and flushes to disk.
    pub fn store(&mut self, key: &str, content: &str) -> std::io::Result<()> {
        let record = MemoryRecord {
            id: key.to_string(),
            content: content.to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        self.memories.insert(key.to_string(), record);
        self.flush_to_disk()
    }

    /// Retrieves a memory record by its semantic key.
    pub fn retrieve(&self, key: &str) -> Option<String> {
        self.memories.get(key).map(|r| r.content.clone())
    }

    /// Scans memories for a keyword (Basic RAG-lite fallback).
    pub fn search(&self, keyword: &str) -> Vec<String> {
        self.memories
            .values()
            .filter(|r| r.content.contains(keyword) || r.id.contains(keyword))
            .map(|r| format!("[{}] {}", r.id, r.content))
            .collect()
    }

    /// Loads serialized memories from disk.
    fn load_from_disk(&mut self) -> std::io::Result<()> {
        let mut file = File::open(&self.db_path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        // Very basic manual parsing to avoid heavy external dependencies like serde if not present.
        for line in contents.lines() {
            let parts: Vec<&str> = line.splitn(3, '|').collect();
            if parts.len() == 3 {
                let id = parts[0].to_string();
                let timestamp = parts[1].parse::<u64>().unwrap_or(0);
                let content = parts[2].replace("\\n", "\n");

                self.memories.insert(id.clone(), MemoryRecord {
                    id,
                    content,
                    timestamp,
                });
            }
        }
        Ok(())
    }

    /// Persists all memories to the filesystem.
    fn flush_to_disk(&self) -> std::io::Result<()> {
        let mut file = File::create(&self.db_path)?;
        for record in self.memories.values() {
            let safe_content = record.content.replace('\n', "\\n");
            writeln!(file, "{}|{}|{}", record.id, record.timestamp, safe_content)?;
        }
        Ok(())
    }
}
