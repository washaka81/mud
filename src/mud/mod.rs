use memmap2::Mmap;
use std::collections::HashMap;
use std::sync::Arc;

pub mod corpus_trainer;
pub mod debate_trainer;
pub mod arena_games;
pub mod slime_backward;
pub mod dspy;
pub mod slime;
pub mod slime_jepa;
pub mod slime_forward;
pub mod ecc;
pub mod ldt_micro;
pub mod rlvr;
pub mod routing;
pub mod workspace;
pub mod constants;
pub mod workspace_agent;
pub mod holographic_loss;
pub mod galore;
pub mod sandbox;
pub mod subagents;
pub mod memory_bank;
pub mod memory_profiler;
pub mod qat_dispatcher;
pub mod self_play;
pub mod muon;
pub mod speculative;#[cfg(test)]
mod tests;

/// MUD: Modular Understanding Dynamics
/// File version 1.0
pub const MUD_MAGIC: &[u8; 4] = b"MUD\x01";

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MudTensorType {
    Ternary2Bit = 0,
    Float32 = 1,
    Float16 = 2,
    Int4 = 3,
    Uint8 = 4,
}

pub struct MudTensor {
    pub name: String,
    pub t_type: MudTensorType,
    pub shape: Vec<usize>,
    pub data_ptr: *const u8,
    pub offset: usize,
    /// Keep the mmap alive if this tensor was loaded from a file
    pub mmap: Option<Arc<Mmap>>,
    /// Optional owned data for newly created tensors
    pub owned_data: Option<Vec<u8>>,
}

impl Clone for MudTensor {
    fn clone(&self) -> Self {
        let mut cloned = Self {
            name: self.name.clone(),
            t_type: self.t_type,
            shape: self.shape.clone(),
            data_ptr: std::ptr::null(),
            offset: self.offset,
            mmap: self.mmap.clone(),
            owned_data: self.owned_data.clone(),
        };
        if let Some(owned) = &cloned.owned_data {
            cloned.data_ptr = owned.as_ptr();
        } else if let Some(mmap) = &cloned.mmap {
            let data_start = (cloned.offset + 31) & !31;
            cloned.data_ptr = unsafe { mmap.as_ptr().add(data_start) };
        }
        cloned
    }
}

impl MudTensor {
    pub fn data_size(&self) -> usize {
        let elements: usize = self.shape.iter().product();
        match self.t_type {
            MudTensorType::Ternary2Bit => elements.div_ceil(8) * 4,
            MudTensorType::Float32 => elements * 4,
            MudTensorType::Float16 => elements * 2,
            MudTensorType::Int4 => elements.div_ceil(2),
            MudTensorType::Uint8 => elements,
        }
    }
}

// 🛡️ THREAD SAFETY: MudTensor is safe to send between threads because data_ptr
// is always derived from either Arc<Mmap> (kept alive by mmap field) or
// owned_data (kept alive by owned_data field). Clone maintains this invariant.
unsafe impl Send for MudTensor {}
unsafe impl Sync for MudTensor {}

#[derive(Clone)]
pub struct MudSkill {
    pub name: String,
    pub tensors: HashMap<String, MudTensor>,
    pub metadata: HashMap<String, String>,
}

// SAFETY: MudSkill owns only String and HashMap, both Send+Sync
unsafe impl Send for MudSkill {}
unsafe impl Sync for MudSkill {}

pub struct MudFile {
    pub mmap: Option<Arc<Mmap>>,
    pub skills: HashMap<String, MudSkill>,
    pub global_metadata: HashMap<String, String>,
}

// SAFETY: MudFile owns Arc<Mmap> and HashMap<String, MudSkill>; all fields are Send+Sync
unsafe impl Send for MudFile {}
unsafe impl Sync for MudFile {}

pub struct StreamingMudWriter {
    file: std::fs::File,
    current_data_offset: usize,
}

impl StreamingMudWriter {
    pub fn create(
        path: &str,
        global_metadata: &HashMap<String, String>,
        tensors_meta: &[(String, MudTensorType, Vec<usize>)],
    ) -> anyhow::Result<Self> {
        use byteorder::{LittleEndian, WriteBytesExt};
        use std::io::Write;

        let temp_path = format!("{}.tmp", path);
        let mut file = std::fs::File::create(&temp_path)?;
        file.write_all(MUD_MAGIC)?;

        file.write_u32::<LittleEndian>(global_metadata.len() as u32)?;
        for (k, v) in global_metadata {
            let kb = k.as_bytes();
            file.write_u32::<LittleEndian>(kb.len() as u32)?;
            file.write_all(kb)?;
            let vb = v.as_bytes();
            file.write_u32::<LittleEndian>(vb.len() as u32)?;
            file.write_all(vb)?;
        }

        file.write_u32::<LittleEndian>(tensors_meta.len() as u32)?;

        let mut header_data = Vec::new();
        let mut curr_offset = 0;
        for (name, t_type, shape) in tensors_meta {
            let name_b = name.as_bytes();
            header_data.write_u32::<LittleEndian>(name_b.len() as u32)?;
            header_data.write_all(name_b)?;
            header_data.write_u32::<LittleEndian>(*t_type as u32)?;
            header_data.write_u32::<LittleEndian>(shape.len() as u32)?;
            for &d in shape {
                header_data.write_u64::<LittleEndian>(d as u64)?;
            }

            let elements: usize = shape.iter().product();
            let s = match t_type {
                MudTensorType::Ternary2Bit => elements.div_ceil(8) * 4,
                MudTensorType::Float32 => elements * 4,
                MudTensorType::Float16 => elements * 2,
                MudTensorType::Int4 => elements.div_ceil(2),
                MudTensorType::Uint8 => elements,
            };

            header_data.write_u64::<LittleEndian>(curr_offset as u64)?;
            let padding = (32 - (s % 32)) % 32;
            curr_offset += s + padding;
        }

        file.write_all(&header_data)?;

        let current_pos = file.metadata()?.len() as usize;
        let padding = (32 - (current_pos % 32)) % 32;
        file.write_all(&[0u8; 32][..padding])?;

        Ok(Self {
            file,
            current_data_offset: 0,
        })
    }

    pub fn write_tensor_data(&mut self, data: &[u8]) -> anyhow::Result<()> {
        use std::io::Write;
        let s = data.len();
        self.file.write_all(data)?;
        let padding = (32 - (s % 32)) % 32;
        if padding > 0 {
            self.file.write_all(&[0u8; 32][..padding])?;
        }
        self.current_data_offset += s + padding;
        Ok(())
    }

    pub fn close(self, final_path: &str) -> anyhow::Result<()> {
        let f = self.file;
        f.sync_all()?;
        let temp_path = format!("{}.tmp", final_path);
        drop(f);

        let metadata = std::fs::metadata(&temp_path)?;
        let file_size = metadata.len();
        if file_size == 0 {
            anyhow::bail!("StreamingMudWriter::close: temp file is empty (0 bytes)");
        }

        let expected_min_size = 4 + 4 + 4;
        if file_size < expected_min_size {
            anyhow::bail!(
                "StreamingMudWriter::close: temp file too small ({} bytes, expected at least {})",
                file_size, expected_min_size
            );
        }

        std::fs::rename(&temp_path, final_path)?;

        let final_metadata = std::fs::metadata(final_path)?;
        if final_metadata.len() != file_size {
            anyhow::bail!(
                "StreamingMudWriter::close: file size mismatch after rename (temp: {}, final: {})",
                file_size, final_metadata.len()
            );
        }

        Ok(())
    }
}

impl MudFile {
    /// Deep Configuration Incrustation: Process and parse the dynamically embedded configuration at load time.
    pub fn raw_config(&self) -> Option<serde_json::Value> {
        self.global_metadata.get("raw_config_json").and_then(|s| serde_json::from_str(s).ok())
    }

    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        use byteorder::{LittleEndian, WriteBytesExt};
        use std::fs::File;
        use std::io::Write;

        let temp_path = format!("{}.tmp", path);
        let mut file = File::create(&temp_path)?;
        file.write_all(MUD_MAGIC)?;

        file.write_u32::<LittleEndian>(self.global_metadata.len() as u32)?;
        for (k, v) in &self.global_metadata {
            let kb = k.as_bytes();
            file.write_u32::<LittleEndian>(kb.len() as u32)?;
            file.write_all(kb)?;
            let vb = v.as_bytes();
            file.write_u32::<LittleEndian>(vb.len() as u32)?;
            file.write_all(vb)?;
        }

        let mut all_tensors = Vec::new();
        for skill in self.skills.values() {
            for tensor in skill.tensors.values() {
                all_tensors.push(tensor);
            }
        }

        file.write_u32::<LittleEndian>(all_tensors.len() as u32)?;

        let mut header_data = Vec::new();
        let mut curr_offset = 0;

        for tensor in &all_tensors {
            let name_b = tensor.name.as_bytes();
            header_data.write_u32::<LittleEndian>(name_b.len() as u32)?;
            header_data.write_all(name_b)?;
            header_data.write_u32::<LittleEndian>(tensor.t_type as u32)?;
            header_data.write_u32::<LittleEndian>(tensor.shape.len() as u32)?;
            for &d in &tensor.shape {
                header_data.write_u64::<LittleEndian>(d as u64)?;
            }

            let s = if let Some(owned) = &tensor.owned_data {
                owned.len()
            } else {
                let elements: usize = tensor.shape.iter().product();
                match tensor.t_type {
                    MudTensorType::Ternary2Bit => elements.div_ceil(8) * 4,
                    MudTensorType::Float32 => elements * 4,
                    MudTensorType::Float16 => elements * 2,
                    MudTensorType::Int4 => elements.div_ceil(2),
                    MudTensorType::Uint8 => elements,
                }
            };

            header_data.write_u64::<LittleEndian>(curr_offset as u64)?;
            let padding = (32 - (s % 32)) % 32;
            curr_offset += s + padding;
        }

        file.write_all(&header_data)?;
        let current_pos = file.metadata()?.len() as usize;
        let padding = (32 - (current_pos % 32)) % 32;
        file.write_all(&[0u8; 32][..padding])?;

        // Second pass: Write data directly to disk without loading everything into memory
        for tensor in &all_tensors {
            let s = if let Some(owned) = &tensor.owned_data {
                file.write_all(owned)?;
                owned.len()
            } else {
                let elements: usize = tensor.shape.iter().product();
                let s_expected = match tensor.t_type {
                    MudTensorType::Ternary2Bit => elements.div_ceil(8) * 4,
                    MudTensorType::Float32 => elements * 4,
                    MudTensorType::Float16 => elements * 2,
                    MudTensorType::Int4 => elements.div_ceil(2),
                    MudTensorType::Uint8 => elements,
                };
                
                let slice = if tensor.mmap.is_some() {
                    unsafe { std::slice::from_raw_parts(tensor.data_ptr, s_expected) }
                } else {
                    &[]
                };
                file.write_all(slice)?;
                s_expected
            };

            let padding = (32 - (s % 32)) % 32;
            file.write_all(&[0u8; 32][..padding])?;
        }

        file.sync_all()?;
        let written_size = file.metadata()?.len();
        drop(file);
        std::fs::rename(&temp_path, path)?;

        let final_metadata = std::fs::metadata(path)?;
        let final_size = final_metadata.len();
        if final_size != written_size {
            anyhow::bail!(
                "MudFile::save: file size mismatch after rename (written: {}, final: {})",
                written_size, final_size
            );
        }

        let verify_file = std::fs::File::open(path)?;
        let verify_mmap = unsafe { Mmap::map(&verify_file)? };
        if verify_mmap.len() != final_size as usize {
            anyhow::bail!(
                "MudFile::save: mmap size mismatch (mmap: {}, file: {})",
                verify_mmap.len(), final_size
            );
        }

        if verify_mmap.len() < 4 || &verify_mmap[0..4] != MUD_MAGIC {
            anyhow::bail!("MudFile::save: output file has invalid magic number");
        }

        Ok(())
    }

    pub fn load(path: &str) -> anyhow::Result<Self> {
        let file = std::fs::File::open(path)?;
        // SAFETY: file is a valid open File handle; Mmap guarantees the mapping is readable
        let mmap = Arc::new(unsafe { Mmap::map(&file)? });
        if mmap.len() < 16 {
            anyhow::bail!("File too small");
        }

        let mut start_offset = 0;
        if &mmap[0..4] != MUD_MAGIC {
            // Check for MUD-Executable trailer (8 bytes size + 8 bytes "MUDEXEC\0")
            let trailer_pos = mmap.len().saturating_sub(8);
            if &mmap[trailer_pos..mmap.len()] == b"MUDEXEC\0" {
                let offset_pos = trailer_pos.saturating_sub(8);
                let mud_offset = u64::from_le_bytes(mmap[offset_pos..offset_pos + 8].try_into()?) as usize;
                
                if mud_offset + 4 <= mmap.len() && &mmap[mud_offset..mud_offset + 4] == MUD_MAGIC {
                    start_offset = mud_offset;
                } else {
                    anyhow::bail!("Invalid MUD magic number inside MUD-Executable");
                }
            } else {
                anyhow::bail!("Invalid MUD magic number and no MUDEXEC trailer found");
            }
        }

        let mut pos = start_offset + 4;

        let meta_count = u32::from_le_bytes(mmap[pos..pos + 4].try_into()?) as usize;
        pos += 4;
        let mut global_metadata = HashMap::new();
        for _ in 0..meta_count {
            let k_len = u32::from_le_bytes(mmap[pos..pos + 4].try_into()?) as usize;
            pos += 4;
            let key = String::from_utf8_lossy(&mmap[pos..pos + k_len]).into_owned();
            pos += k_len;

            let v_len = u32::from_le_bytes(mmap[pos..pos + 4].try_into()?) as usize;
            pos += 4;
            let val = String::from_utf8_lossy(&mmap[pos..pos + v_len]).into_owned();
            pos += v_len;
            global_metadata.insert(key, val);
        }

        let tensor_count = u32::from_le_bytes(mmap[pos..pos + 4].try_into()?) as usize;
        pos += 4;
        let mut tensors = HashMap::new();
        for _ in 0..tensor_count {
            let n_len = u32::from_le_bytes(mmap[pos..pos + 4].try_into()?) as usize;
            pos += 4;
            let name = String::from_utf8_lossy(&mmap[pos..pos + n_len]).into_owned();
            pos += n_len;

            let t_type_val = u32::from_le_bytes(mmap[pos..pos + 4].try_into()?);
            pos += 4;
            let t_type = match t_type_val {
                0 => MudTensorType::Ternary2Bit,
                1 => MudTensorType::Float32,
                2 => MudTensorType::Float16,
                3 => MudTensorType::Int4,
                4 => MudTensorType::Uint8,
                _ => anyhow::bail!("Unknown tensor type: {}", t_type_val),
            };

            let s_len = u32::from_le_bytes(mmap[pos..pos + 4].try_into()?) as usize;
            pos += 4;
            let mut shape = Vec::new();
            for _ in 0..s_len {
                shape.push(u64::from_le_bytes(mmap[pos..pos + 8].try_into()?) as usize);
                pos += 8;
            }

            let offset = u64::from_le_bytes(mmap[pos..pos + 8].try_into()?) as usize;
            pos += 8;

            tensors.insert(
                name.clone(),
                MudTensor {
                    name,
                    t_type,
                    shape,
                    data_ptr: std::ptr::null(),
                    offset,
                    mmap: Some(mmap.clone()),
                    owned_data: None,
                },
            );
        }

        let data_start = (pos + 31) & !31; // Align start to 32-byte boundary (matching writer)
        let mmap_len = mmap.len();

        for tensor in tensors.values_mut() {
            let ptr_offset = data_start
                .checked_add(tensor.offset)
                .expect("load: data_start + tensor.offset overflow");

            let n_elements = tensor.shape.iter().product::<usize>();
            let expected_bytes = match tensor.t_type {
                MudTensorType::Ternary2Bit => n_elements.div_ceil(8) * 4,
                MudTensorType::Float32 => n_elements * 4,
                MudTensorType::Float16 => n_elements * 2,
                MudTensorType::Int4 => n_elements.div_ceil(2),
                MudTensorType::Uint8 => n_elements,
            };

            if ptr_offset + expected_bytes > mmap_len {
                anyhow::bail!(
                    "CORRUPTION DETECTED: Tensor '{}' (type {:?}) requires {} bytes at offset 0x{:x}, but mmap only has {} bytes remaining.",
                    tensor.name, tensor.t_type, expected_bytes, ptr_offset, mmap_len.saturating_sub(ptr_offset)
                );
            }

            // SAFETY: ptr_offset + expected_bytes was validated against mmap_len on lines 335-339
            tensor.data_ptr = unsafe { mmap.as_ptr().add(ptr_offset) };
        }

        let mut skills = HashMap::new();
        skills.insert(
            "core".to_string(),
            MudSkill {
                name: "core".to_string(),
                tensors,
                metadata: HashMap::new(),
            },
        );
        Ok(Self {
            mmap: Some(mmap),
            skills,
            global_metadata,
        })
    }

    pub fn get_tensor_ternary(&self, skill: &str, name: &str) -> Option<*const u32> {
        self.skills
            .get(skill)?
            .tensors
            .get(name)
            .filter(|t| t.t_type == MudTensorType::Ternary2Bit)
            .map(|t| t.data_ptr as *const u32)
    }

    /// Load a .mud file and verify ECC parity for all ternary tensors.
    /// Single-bit errors are corrected; parity-bit errors are logged.
    pub fn load_verified(path: &str) -> anyhow::Result<Self> {
        let mut mud = Self::load(path)?;
        let (corrected, parity_err) = mud.ecc_verify_all();
        if corrected > 0 || parity_err > 0 {
            eprintln!(
                "[ECC] {} single-bit corrected, {} parity errors (data intact)",
                corrected, parity_err
            );
        }
        Ok(mud)
    }

    /// Generate ECC parity for all Ternary2Bit tensors.
    /// Creates .ecc sibling tensors containing 1 parity byte per u32 of packed weights.
    /// Safe to call multiple times — replaces existing .ecc tensors.
    pub fn ecc_generate_all(&mut self) -> usize {
        use crate::mud::ecc::as_u32_slice_le;
        use crate::mud::ecc::{ecc_compute_buf, ecc_tensor_name, is_ecc_tensor};

        let mut count = 0;

        let work: Vec<(String, String, Vec<u8>)> = {
            let mut list = Vec::new();
            for (skill_name, skill) in &self.skills {
                for (t_name, tensor) in &skill.tensors {
                    if tensor.t_type != MudTensorType::Ternary2Bit || is_ecc_tensor(t_name) {
                        continue;
                    }
                    let n: usize = tensor.shape.iter().product();
                    let u32_count = n.div_ceil(8);
                    let parity = if let Some(owned) = &tensor.owned_data {
                        ecc_compute_buf(as_u32_slice_le(owned))
                    } else if tensor.mmap.is_some() {
                        let slice = unsafe { std::slice::from_raw_parts(tensor.data_ptr, u32_count * 4) };
                        ecc_compute_buf(bytemuck::try_cast_slice(slice).unwrap_or(&[]))
                    } else {
                        continue;
                    };
                    list.push((skill_name.clone(), t_name.clone(), parity));
                }
            }
            list
        };

        for (skill_name, t_name, parity) in work {
            let ecc_name = ecc_tensor_name(&t_name);
            if let Some(skill) = self.skills.get_mut(&skill_name) {
                skill.tensors.insert(
                    ecc_name.clone(),
                    MudTensor {
                        name: ecc_name,
                        t_type: MudTensorType::Uint8,
                        shape: vec![parity.len()],
                        data_ptr: std::ptr::null(),
                        offset: 0,
                        mmap: None,
                        owned_data: Some(parity),
                    },
                );
                count += 1;
            }
        }

        count
    }

    /// Verify and correct all Ternary2Bit tensors using stored .ecc parity.
    /// Single-bit flips are corrected in owned_data.
    /// Returns (total_single_bit_corrected, total_parity_bit_errors).
    pub fn ecc_verify_all(&mut self) -> (u32, u32) {
        use crate::mud::ecc::{
            as_u32_slice_le_mut, base_tensor_name, ecc_verify_buf, is_ecc_tensor,
        };

        let mut total_corrected = 0u32;
        let mut total_parity_err = 0u32;

        let work: Vec<(String, String, Vec<u8>)> = {
            let mut list = Vec::new();
            for (skill_name, skill) in &self.skills {
                let keys: Vec<String> = skill.tensors.keys().cloned().collect();
                for ecc_name in keys {
                    if !is_ecc_tensor(&ecc_name) {
                        continue;
                    }
                    let Some(base_name) = base_tensor_name(&ecc_name) else {
                        continue;
                    };
                    let Some(base_tensor) = skill.tensors.get(base_name) else {
                        continue;
                    };
                    if base_tensor.t_type != MudTensorType::Ternary2Bit {
                        continue;
                    }
                    let Some(ecc_tensor) = skill.tensors.get(&ecc_name) else {
                        continue;
                    };
                    let n: usize = base_tensor.shape.iter().product();
                    let u32_count = n.div_ceil(8);
                    let ecc_data: Vec<u8> = if let Some(owned) = &ecc_tensor.owned_data {
                        owned.clone()
                    } else if let Some(mmap) = &ecc_tensor.mmap {
                        mmap[ecc_tensor.offset..ecc_tensor.offset + u32_count].to_vec()
                    } else {
                        continue;
                    };
                    // Truncate to actual u32 count
                    let ecc_data = ecc_data.into_iter().take(u32_count).collect();
                    list.push((skill_name.clone(), base_name.to_string(), ecc_data));
                }
            }
            list
        };

        for (skill_name, base_name, ecc_data) in &work {
            let Some(skill) = self.skills.get_mut(skill_name) else {
                continue;
            };
            let Some(tensor) = skill.tensors.get_mut(base_name) else {
                continue;
            };

            let n: usize = tensor.shape.iter().product();
            let u32_count = n.div_ceil(8);

            if let Some(owned) = &mut tensor.owned_data {
                if owned.len() >= u32_count * 4 {
                    let slice = as_u32_slice_le_mut(owned);
                    let (c, p) = ecc_verify_buf(slice, ecc_data);
                    total_corrected += c;
                    total_parity_err += p;
                }
            } else if tensor.mmap.is_some() {
                let src = unsafe { std::slice::from_raw_parts(tensor.data_ptr, u32_count * 4) };
                let mut owned = crate::mud::ecc::aligned_copy(src);
                {
                    let slice = as_u32_slice_le_mut(&mut owned);
                    let (c, p) = ecc_verify_buf(slice, ecc_data);
                    total_corrected += c;
                    total_parity_err += p;
                }
                tensor.owned_data = Some(owned);
                tensor.data_ptr = std::ptr::null();
                tensor.mmap = None;
            }
        }

        (total_corrected, total_parity_err)
    }
}

/// Packs a row of ternary values (-1, 0, 1) into a little-endian byte vector.
pub fn pack_ternary_row(values: &[f32], delta: f32) -> Vec<u8> {
    let n = values.len();
    let u32_count = n.div_ceil(8);
    let mut packed = vec![0u32; u32_count];
    
    for i in 0..n {
        let v = values[i];
        if v.abs() > delta {
            let bits = if v > 0.0 { 0x1u32 } else { 0xFu32 };
            packed[i / 8] |= bits << ((i % 8) * 4);
        }
    }
    
    let mut bytes = vec![0u8; u32_count * 4];
    unsafe {
        std::ptr::copy_nonoverlapping(
            packed.as_ptr() as *const u8,
            bytes.as_mut_ptr(),
            bytes.len(),
        );
    }
    bytes
}

/// Dequantizes a row of packed ternary weights (1.58-bit) to f32.
///
/// # Safety
/// * `packed` must point to at least `n.div_ceil(8)` valid `u32` elements.
/// * `out` must have at least `n` elements.
pub unsafe fn dequantize_ternary_row(packed: *const u32, out: &mut [f32], n: usize) {
    assert!(
        out.len() >= n,
        "dequantize_ternary_row: output buffer too small ({} < {})",
        out.len(),
        n
    );
    if packed.is_null() {
        return;
    }

    let u32_count = n / 8;
    let remainder = n % 8;
    for i in 0..u32_count {
        let val = *packed.add(i);
        for j in 0..8 {
            let bits = (val >> (j * 4)) & 0xF;
            out[i * 8 + j] = match bits {
                0x1 => 1.0,
                0xF => -1.0,
                _ => 0.0,
            };
        }
    }
    if remainder > 0 {
        let val = *packed.add(u32_count);
        for j in 0..remainder {
            let bits = (val >> (j * 4)) & 0xF;
            out[u32_count * 8 + j] = match bits {
                0x1 => 1.0,
                0xF => -1.0,
                _ => 0.0,
            };
        }
    }
}
pub mod integral_threshold;
pub mod pcore_pool;
