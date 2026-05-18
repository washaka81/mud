use memmap2::Mmap;
use std::fs::File;
use std::collections::HashMap;
use std::sync::Arc;
use crate::asm::BlockQ4_0;

/// Represents the various tensor types supported by GGUF.
/// Only a subset is currently implemented for computation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TensorType {
    /// 32-bit Floating Point
    F32 = 0, 
    /// 16-bit Floating Point
    F16 = 1, 
    /// 4-bit Quantized (Type 0)
    Q4_0 = 2, 
    /// 4-bit Quantized (Type 1)
    Q4_1 = 3, 
    /// 5-bit Quantized (Type 0)
    Q5_0 = 6, 
    /// 5-bit Quantized (Type 1)
    Q5_1 = 7, 
    /// 8-bit Quantized (Type 0)
    Q8_0 = 8, 
    /// 8-bit Quantized (Type 1)
    Q8_1 = 9,
    /// K-Quants (various levels)
    Q2K = 10, Q3K = 11, Q4K = 12, Q5K = 13, Q6K = 14, Q8K = 15,
}

/// Metadata about a single tensor in the GGUF file.
pub struct Tensor {
    /// Unique name of the tensor (e.g., 'blk.0.attn_q.weight')
    pub name: String,
    /// Semantic type of the tensor data
    pub t_type: TensorType,
    /// Raw type ID as stored in the GGUF file
    pub raw_type: u32,
    /// Dimensions of the tensor [rows, cols, ...]
    pub shape: Vec<usize>,
    /// Pointer to the mapped data in memory
    pub data_ptr: *const u8,
}

/// Variant for holding different types of metadata values.
#[derive(Debug, Clone)]
pub enum MetadataValue {
    Uint8(u8), Int8(i8), Uint16(u16), Int16(i16), Uint32(u32), Int32(i32),
    Float32(f32), Bool(bool), String(String), Array(Vec<MetadataValue>),
    Uint64(u64), Int64(i64), Float64(f64),
}

/// A loaded GGUF model container.
/// Uses memory mapping for zero-copy access to weights.
pub struct GGUFModel {
    /// The underlying memory map of the model file
    pub mmap: Arc<Mmap>,
    /// Map of tensor names to their descriptors
    pub tensors: HashMap<String, Tensor>,
    /// Map of global metadata keys to their values
    pub metadata: HashMap<String, MetadataValue>,
    /// Byte alignment for tensor data (usually 32)
    pub alignment: usize,
}

impl GGUFModel {
    /// Loads a GGUF model from the specified file path.
    /// Parses the header, metadata, and tensor descriptors, and sets up memory mapping.
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let mmap = Arc::new(unsafe { Mmap::map(&file)? });
        let mut pos = 0;

        // Verify GGUF Magic Number
        if &mmap[pos..pos+4] != b"GGUF" { anyhow::bail!("No GGUF magic"); }
        pos += 4;
        
        // Parse Version and Counts
        let version = u32::from_le_bytes(mmap[pos..pos+4].try_into()?);
        pos += 4;
        let tensor_count = u64::from_le_bytes(mmap[pos..pos+8].try_into()?) as usize;
        pos += 8;
        let kv_count = u64::from_le_bytes(mmap[pos..pos+8].try_into()?) as usize;
        pos += 8;

        println!("GGUF V{}, Tensors={}, KVs={}", version, tensor_count, kv_count);

        // Parse Metadata Key-Value pairs
        let mut metadata = HashMap::new();
        let mut alignment = 32;
        for _ in 0..kv_count {
            let key_len = u64::from_le_bytes(mmap[pos..pos+8].try_into()?) as usize;
            pos += 8;
            let key = String::from_utf8_lossy(&mmap[pos..pos+key_len]).into_owned();
            pos += key_len;
            let val_type = u32::from_le_bytes(mmap[pos..pos+4].try_into()?);
            pos += 4;
            
            let (val, next_pos) = read_metadata_value(&mmap, pos, val_type)?;
            pos = next_pos;

            if key == "general.alignment" {
                if let MetadataValue::Uint32(align) = val { alignment = align as usize; }
            }
            metadata.insert(key, val);
        }

        // Parse Tensor Descriptors
        let mut tensors = HashMap::new();
        for _ in 0..tensor_count {
            let name_len = u64::from_le_bytes(mmap[pos..pos+8].try_into()?) as usize;
            pos += 8;
            let name = String::from_utf8_lossy(&mmap[pos..pos+name_len]).into_owned();
            pos += name_len;
            let n_dims = u32::from_le_bytes(mmap[pos..pos+4].try_into()?) as usize;
            pos += 4;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(u64::from_le_bytes(mmap[pos..pos+8].try_into()?) as usize);
                pos += 8;
            }
            let t_type_raw = u32::from_le_bytes(mmap[pos..pos+4].try_into()?);
            pos += 4;
            let offset = u64::from_le_bytes(mmap[pos..pos+8].try_into()?) as usize;
            pos += 8;
            
            let t_type = match t_type_raw {
                0 => TensorType::F32, 
                1 => TensorType::F16, 
                2 => TensorType::Q4_0,
                12 => TensorType::Q4K, 
                _ => TensorType::Q4_0, // Default fallback
            };
            tensors.insert(name.clone(), Tensor { name, t_type, raw_type: t_type_raw, shape, data_ptr: offset as *const u8 });
        }

        // Calculate start of data section and finalize tensor pointers
        let data_start = (pos + alignment - 1) & !(alignment - 1);
        for tensor in tensors.values_mut() {
            tensor.data_ptr = unsafe { mmap.as_ptr().add(data_start + (tensor.data_ptr as usize)) };
        }

        Ok(Self { mmap, tensors, metadata, alignment })
    }

    /// Helper to retrieve an array of metadata values for a given key.
    pub fn get_metadata_array(&self, key: &str) -> Option<&Vec<MetadataValue>> {
        if let Some(MetadataValue::Array(arr)) = self.metadata.get(key) {
            Some(arr)
        } else {
            None
        }
    }

    /// Convenience helper to get a pointer to Q4_0 tensor data.
    pub fn get_tensor_q4_0(&self, name: &str) -> Option<*const BlockQ4_0> {
        self.tensors.get(name).filter(|t| t.t_type == TensorType::Q4_0).map(|t| t.data_ptr as *const BlockQ4_0)
    }
}

fn read_metadata_value(mmap: &[u8], mut pos: usize, val_type: u32) -> anyhow::Result<(MetadataValue, usize)> {
    let val = match val_type {
        0 => { let v = mmap[pos]; pos += 1; MetadataValue::Uint8(v) },
        1 => { let v = mmap[pos] as i8; pos += 1; MetadataValue::Int8(v) },
        2 => { let v = u16::from_le_bytes(mmap[pos..pos+2].try_into()?); pos += 2; MetadataValue::Uint16(v) },
        3 => { let v = i16::from_le_bytes(mmap[pos..pos+2].try_into()?); pos += 2; MetadataValue::Int16(v) },
        4 => { let v = u32::from_le_bytes(mmap[pos..pos+4].try_into()?); pos += 4; MetadataValue::Uint32(v) },
        5 => { let v = i32::from_le_bytes(mmap[pos..pos+4].try_into()?); pos += 4; MetadataValue::Int32(v) },
        6 => { let v = f32::from_le_bytes(mmap[pos..pos+4].try_into()?); pos += 4; MetadataValue::Float32(v) },
        7 => { let v = mmap[pos] != 0; pos += 1; MetadataValue::Bool(v) },
        8 => {
            let len = u64::from_le_bytes(mmap[pos..pos+8].try_into()?) as usize;
            pos += 8;
            let s = String::from_utf8_lossy(&mmap[pos..pos+len]).into_owned();
            pos += len;
            MetadataValue::String(s)
        },
        9 => {
            let itype = u32::from_le_bytes(mmap[pos..pos+4].try_into()?);
            let n = u64::from_le_bytes(mmap[pos+4..pos+12].try_into()?) as usize;
            pos += 12;
            let mut arr = Vec::with_capacity(n);
            for _ in 0..n {
                let (v, next_pos) = read_metadata_value(mmap, pos, itype)?;
                pos = next_pos;
                arr.push(v);
            }
            MetadataValue::Array(arr)
        },
        10 => { let v = u64::from_le_bytes(mmap[pos..pos+8].try_into()?); pos += 8; MetadataValue::Uint64(v) },
        11 => { let v = i64::from_le_bytes(mmap[pos..pos+8].try_into()?); pos += 8; MetadataValue::Int64(v) },
        12 => { let v = f64::from_le_bytes(mmap[pos..pos+8].try_into()?); pos += 8; MetadataValue::Float64(v) },
        _ => anyhow::bail!("Unsupported GGUF type: {} at pos {}", val_type, pos),
    };
    Ok((val, pos))
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tokenizer_test;
#[cfg(test)]
mod dump_metadata;
#[cfg(test)]
mod dump_tensors;
