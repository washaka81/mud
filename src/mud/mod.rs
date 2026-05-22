use std::collections::HashMap;
use std::sync::Arc;
use memmap2::Mmap;

pub mod routing;
pub mod inference;
pub mod skills;
pub mod ingester;
pub mod store;
pub mod graph;
pub mod auto_trainer;

/// MUD: Modular Understanding Dynamics
/// File version 1.0
pub const MUD_MAGIC: &[u8; 4] = b"MUD\x01";

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MudTensorType {
    Ternary2Bit = 0,
    Float32 = 1,
    Float16 = 2,
}

#[derive(Clone)]
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

#[derive(Clone)]
pub struct MudSkill {
    pub name: String,
    pub tensors: HashMap<String, MudTensor>,
    pub metadata: HashMap<String, String>,
}

pub struct MudFile {
    pub mmap: Option<Arc<Mmap>>,
    pub skills: HashMap<String, MudSkill>,
    pub global_metadata: HashMap<String, String>,
}

impl MudFile {
    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        use std::io::Write;
        use std::fs::File;
        use byteorder::{WriteBytesExt, LittleEndian};

        let mut file = File::create(path)?;
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
        
        let mut tensor_bytes = Vec::new();
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
            
            let size = if let Some(owned) = &tensor.owned_data {
                let s = owned.len();
                tensor_bytes.push(owned.clone());
                s
            } else {
                let elements: usize = tensor.shape.iter().product();
                let s = match tensor.t_type {
                    MudTensorType::Ternary2Bit => elements.div_ceil(16) * 4,
                    MudTensorType::Float32 => elements * 4,
                    MudTensorType::Float16 => elements * 2,
                };
                let slice = unsafe { std::slice::from_raw_parts(tensor.data_ptr, s) };
                tensor_bytes.push(slice.to_vec());
                s
            };

            header_data.write_u64::<LittleEndian>(curr_offset as u64)?;
            curr_offset += size;
        }

        file.write_all(&header_data)?;
        let current_pos = file.metadata()?.len() as usize;
        let padding = (32 - (current_pos % 32)) % 32;
        file.write_all(&vec![0u8; padding])?;

        for data in tensor_bytes {
            file.write_all(&data)?;
        }

        Ok(())
    }

    pub fn load(path: &str) -> anyhow::Result<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = Arc::new(unsafe { Mmap::map(&file)? });
        let mut pos = 0;

        if &mmap[pos..pos + 4] != MUD_MAGIC { anyhow::bail!("Invalid MUD magic number"); }
        pos += 4;

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
            let name_len = u32::from_le_bytes(mmap[pos..pos + 4].try_into()?) as usize;
            pos += 4;
            let name = String::from_utf8_lossy(&mmap[pos..pos + name_len]).into_owned();
            pos += name_len;
            let t_type_raw = u32::from_le_bytes(mmap[pos..pos + 4].try_into()?);
            pos += 4;
            let t_type = match t_type_raw {
                0 => MudTensorType::Ternary2Bit, 1 => MudTensorType::Float32, 2 => MudTensorType::Float16,
                _ => anyhow::bail!("Unsupported MUD tensor type: {}", t_type_raw),
            };
            let n_dims = u32::from_le_bytes(mmap[pos..pos + 4].try_into()?) as usize;
            pos += 4;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(u64::from_le_bytes(mmap[pos..pos + 8].try_into()?) as usize);
                pos += 8;
            }
            let offset = u64::from_le_bytes(mmap[pos..pos + 8].try_into()?) as usize;
            pos += 8;
            tensors.insert(name.clone(), MudTensor { 
                name, t_type, shape, data_ptr: std::ptr::null(), offset, mmap: Some(mmap.clone()), owned_data: None,
            });
        }

        let data_start = (pos + 31) & !31;
        let mmap_len = mmap.len();
        for tensor in tensors.values_mut() {
            let ptr_offset = data_start.checked_add(tensor.offset)
                .expect("load: data_start + tensor.offset overflow");
            assert!(ptr_offset < mmap_len, "load: offset 0x{:x} fuera del mmap (len=0x{:x})", ptr_offset, mmap_len);
            tensor.data_ptr = unsafe { mmap.as_ptr().add(ptr_offset) };
        }

        let mut skills = HashMap::new();
        skills.insert("core".to_string(), MudSkill { name: "core".to_string(), tensors, metadata: HashMap::new() });
        Ok(Self { mmap: Some(mmap), skills, global_metadata })
    }

    pub fn get_tensor_ternary(&self, skill: &str, name: &str) -> Option<*const u32> {
        self.skills.get(skill)?.tensors.get(name).filter(|t| t.t_type == MudTensorType::Ternary2Bit).map(|t| t.data_ptr as *const u32)
    }
}

pub fn dequantize_ternary_row(packed: *const u32, out: &mut [f32], n: usize) {
    // Guarda: out debe tener al menos n elementos
    debug_assert!(out.len() >= n, "dequantize_ternary_row: out.len()={} < n={}", out.len(), n);
    let u32_count = n / 16;        // bloques completos
    let remainder = n % 16;        // elementos residuales (sin bloque completo)
    unsafe {
        for i in 0..u32_count {
            let val = *packed.add(i);
            for j in 0..16 {
                let bits = (val >> (j * 2)) & 3;
                out[i * 16 + j] = match bits { 1 => 1.0, 2 => -1.0, _ => 0.0 };
            }
        }
        // Desempaqueta los bits residuales del bloque parcial final
        if remainder > 0 {
            let val = *packed.add(u32_count);
            for j in 0..remainder {
                let bits = (val >> (j * 2)) & 3;
                out[u32_count * 16 + j] = match bits { 1 => 1.0, 2 => -1.0, _ => 0.0 };
            }
        }
    }
}
