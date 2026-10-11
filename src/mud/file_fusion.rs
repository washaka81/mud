//! Dynamic Background File Fusion (`MudFileMerger`)
//! Consolidates multiple active .mud modules into a single unified .mud binary on disk.

use std::collections::HashMap;
use crate::mud::{MudFile, StreamingMudWriter};

pub struct MudFileMerger;

impl MudFileMerger {
    /// Consolidates a slice of MudFiles into a single unified MudFile binary container on disk.
    pub fn merge_files(files: &[&MudFile], output_path: &str) -> anyhow::Result<()> {
        let mut global_metadata = HashMap::new();
        global_metadata.insert("merged_files_count".to_string(), files.len().to_string());
        global_metadata.insert("engine".to_string(), "SlimeX LLM / MUD".to_string());

        let mut tensors_meta = Vec::new();
        for (f_idx, file) in files.iter().enumerate() {
            for skill in file.skills.values() {
                for (key, tensor) in &skill.tensors {
                    let merged_key = if files.len() > 1 {
                        format!("mod_{}.{}", f_idx, key)
                    } else {
                        key.clone()
                    };
                    tensors_meta.push((merged_key, tensor.t_type, tensor.shape.clone()));
                }
            }
        }

        let mut writer = StreamingMudWriter::create(output_path, &global_metadata, &tensors_meta)?;

        for file in files {
            for skill in file.skills.values() {
                for tensor in skill.tensors.values() {
                    let data_len = tensor.data_size();
                    if !tensor.data_ptr.is_null() && data_len > 0 {
                        // SAFETY: tensor.data_ptr is guaranteed non-null and valid for data_len bytes
                        let slice = unsafe { std::slice::from_raw_parts(tensor.data_ptr, data_len) };
                        writer.write_tensor_data(slice)?;
                    } else if let Some(owned) = &tensor.owned_data {
                        writer.write_tensor_data(owned)?;
                    }
                }
            }
        }

        writer.close(output_path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mud::{MudSkill, MudTensor};

    #[test]
    fn test_file_merger_single_file() {
        let owned_bytes = vec![0u8; 16];
        let dummy_tensor = MudTensor {
            name: "token_embd.weight".to_string(),
            t_type: crate::mud::MudTensorType::Float32,
            shape: vec![4],
            data_ptr: owned_bytes.as_ptr(),
            offset: 0,
            data_base: 0,
            mmap: None,
            owned_data: Some(owned_bytes),
        };

        let mut tensors = HashMap::new();
        tensors.insert("token_embd.weight".to_string(), dummy_tensor);

        let skill = MudSkill {
            name: "default".to_string(),
            tensors,
            metadata: HashMap::new(),
        };

        let mut skills = HashMap::new();
        skills.insert("default".to_string(), skill);

        let mud_file = MudFile {
            mmap: None,
            skills,
            global_metadata: HashMap::new(),
            path: None,
        };

        let temp_out = "/tmp/test_merged_single.mud";
        let res = MudFileMerger::merge_files(&[&mud_file], temp_out);
        assert!(res.is_ok());

        let _ = std::fs::remove_file(temp_out);
    }
}
