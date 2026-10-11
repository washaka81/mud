//! AOT Engine Boot Indexer (`BootMemoryIndexer`)
//! Scans, m-maps, and seats all `.mud` memory blocks into the 256-slot Continuity Table at boot time (t_0).

use std::sync::Arc;
use crate::mud::memory_continuity::MemoryContinuityTable;
use crate::mud::MudFile;

/// AOT Boot Indexer for zero cold-start neural memory bank seating.
pub struct BootMemoryIndexer {
    pub models_dir: String,
    pub continuity_table: MemoryContinuityTable,
    pub loaded_count: u8,
}

impl BootMemoryIndexer {
    pub fn boot_index(models_dir: &str) -> anyhow::Result<Self> {
        let mut continuity_table = MemoryContinuityTable::new();
        let mut loaded_count = 0u8;

        let path = std::path::Path::new(models_dir);
        if path.exists() && path.is_dir() {
            let entries = std::fs::read_dir(path)?;
            for entry in entries {
                let entry = entry?;
                let file_path = entry.path();
                if file_path.extension().and_then(|s| s.to_str()) == Some("mud") {
                    if loaded_count == 255 {
                        log::warn!("[BOOT_INDEXER] 256 slot limit reached.");
                        break;
                    }
                    if let Ok(mud_file) = MudFile::load(file_path.to_str().unwrap_or_default()) {
                        let slot_id = loaded_count;
                        continuity_table.seat_module(slot_id, Arc::new(mud_file));
                        loaded_count += 1;
                    }
                }
            }
        }

        Ok(Self {
            models_dir: models_dir.to_string(),
            continuity_table,
            loaded_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boot_indexer_non_existent_dir() {
        let indexer = BootMemoryIndexer::boot_index("/tmp/non_existent_mud_dir_12345").unwrap();
        assert_eq!(indexer.loaded_count, 0);
    }
}
