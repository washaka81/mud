use crate::asm::BlockQ4_0;
// R-32: reuse the SEC-H1 caps (same untrusted-input threat model as .mud:
// GGUF files are user-supplied downloads/conversions). Real GGUFs carry
// ~10-50 KV pairs + hundreds of tensors; the caps leave ~50x headroom.
use crate::mud::{
    MAX_META_ENTRIES, MAX_SHAPE_DIMS, MAX_STRING_LEN, MAX_TENSOR_DIM, MAX_TENSOR_ENTRIES,
};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;

/// R-32: max nesting depth for GGUF metadata arrays (type 9 recurses).
/// Legit files nest at most 1 level (array of scalars/strings); anything
/// deeper is hostile (stack exhaustion via recursion).
const GGUF_MAX_ARRAY_DEPTH: u8 = 8;
/// R-32: max element count of a single metadata array. The tokenizer token
/// list is the largest legit array (~256k strings); 4M leaves 16x headroom
/// while bounding the `Vec::push` loop on hostile `n = u64::MAX`.
const GGUF_MAX_ARRAY_LEN: usize = 4 * 1024 * 1024;
/// R-32: sanity cap for the `general.alignment` KV (spec default 32).
/// A hostile alignment (0, non-power-of-two, gigantic) previously corrupted
/// the `data_start` rounding (`(pos + align - 1) & !(align - 1)` underflows
/// on 0, mis-rounds on non-powers of two).
const GGUF_MAX_ALIGNMENT: usize = 1 << 20;

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
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
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
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
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
    ///
    /// R-32: every read is bounds-checked against `mmap.len()` (truncated
    /// files return `Err`, never panic on slice indexing), counts/lengths
    /// are capped (SEC-H1 values), array nesting is depth-limited, and
    /// tensor data offsets are validated with `checked_add` before the
    /// `unsafe` pointer is formed.
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        // SAFETY: `file` is a valid read-only handle; memmap2 maps exactly
        // `file`'s length. The mapping outlives every `data_ptr` derived
        // from it because this `Arc<Mmap>` is stored in the returned model.
        let mmap = Arc::new(unsafe { Mmap::map(&file)? });
        let buf: &[u8] = &mmap;
        let mut cur = GgufCursor { buf, pos: 0 };

        // Verify GGUF Magic Number (bounds-checked: files < 4 bytes => Err).
        if cur.read_bytes(4)? != b"GGUF" {
            anyhow::bail!("No GGUF magic");
        }

        // Parse Version and Counts.
        let version = cur.read_u32()?;
        let tensor_count = cur.read_u64()? as usize;
        let kv_count = cur.read_u64()? as usize;
        // R-32: cap counts before looping (a hostile u64::MAX count
        // previously meant a near-infinite loop / HashMap blowup).
        if kv_count > MAX_META_ENTRIES {
            anyhow::bail!(
                "R-32: kv_count {kv_count} exceeds cap {MAX_META_ENTRIES}"
            );
        }
        if tensor_count > MAX_TENSOR_ENTRIES {
            anyhow::bail!(
                "R-32: tensor_count {tensor_count} exceeds cap {MAX_TENSOR_ENTRIES}"
            );
        }

        println!(
            "GGUF V{}, Tensors={}, KVs={}",
            version, tensor_count, kv_count
        );

        // Parse Metadata Key-Value pairs
        let mut metadata = HashMap::new();
        let mut alignment = 32usize;
        for _ in 0..kv_count {
            let key_len = cur.read_u64()? as usize;
            // R-32: cap key length before slicing (hostile u64::MAX
            // previously panicked on `mmap[pos..pos+key_len]`).
            if key_len > MAX_STRING_LEN {
                anyhow::bail!("R-32: metadata key len {key_len} exceeds cap {MAX_STRING_LEN}");
            }
            let key = String::from_utf8_lossy(cur.read_bytes(key_len)?).into_owned();
            let val_type = cur.read_u32()?;

            let (val, next_pos) = read_metadata_value(buf, cur.pos, val_type, 0)?;
            cur.pos = next_pos;

            if key == "general.alignment" {
                if let MetadataValue::Uint32(align) = val {
                    // R-32: validate before use (0 => underflow, non-pow2 =>
                    // mis-round, gigantic => data_start past EOF).
                    let a = align as usize;
                    if a == 0 || a > GGUF_MAX_ALIGNMENT || !a.is_power_of_two() {
                        anyhow::bail!("R-32: hostile general.alignment={a}");
                    }
                    alignment = a;
                }
            }
            metadata.insert(key, val);
        }

        // Parse Tensor Descriptors
        let mut tensors = HashMap::new();
        for _ in 0..tensor_count {
            let name_len = cur.read_u64()? as usize;
            if name_len > MAX_STRING_LEN {
                anyhow::bail!("R-32: tensor name len {name_len} exceeds cap {MAX_STRING_LEN}");
            }
            let name = String::from_utf8_lossy(cur.read_bytes(name_len)?).into_owned();
            let n_dims = cur.read_u32()? as usize;
            // R-32: cap rank before `with_capacity` + loop.
            if n_dims > MAX_SHAPE_DIMS {
                anyhow::bail!("R-32: tensor '{name}' rank {n_dims} exceeds cap {MAX_SHAPE_DIMS}");
            }
            let mut shape = Vec::with_capacity(n_dims);
            let mut elements = 1usize;
            for _ in 0..n_dims {
                let d = cur.read_u64()? as usize;
                // R-32: per-dim magnitude cap (R-31 analogue): an arbitrary
                // u64 dim previously flowed into unchecked size math.
                if d > MAX_TENSOR_DIM {
                    anyhow::bail!(
                        "R-32: tensor '{name}' dim {d} exceeds cap {MAX_TENSOR_DIM}"
                    );
                }
                elements = elements.checked_mul(d).ok_or_else(|| {
                    anyhow::anyhow!("R-32: tensor '{name}' shape product overflows")
                })?;
                shape.push(d);
            }
            let t_type_raw = cur.read_u32()?;
            let offset = cur.read_u64()? as usize;

            let t_type = match t_type_raw {
                0 => TensorType::F32,
                1 => TensorType::F16,
                2 => TensorType::Q4_0,
                12 => TensorType::Q4K,
                _ => {
                    // SEC-H10 (fixed 2026-07-27): Do not silently fall back.
                    // Warn via the structured `log` macro so the operator knows
                    // the tensor type was unrecognized and data may be
                    // misinterpreted. Retaining Q4_0 as a best-effort fallback
                    // for backward compatibility, but the warning makes the
                    // silent corruption visible. (SEC-H6: migrated from
                    // eprintln! to log::warn! for structured logging.)
                    log::warn!(
                        "SEC-H10: Unknown GGUF tensor type {} for tensor '{}', \
                         falling back to Q4_0 (data may be misinterpreted)",
                        t_type_raw, name
                    );
                    TensorType::Q4_0
                }
            };
            tensors.insert(
                name.clone(),
                Tensor {
                    name,
                    t_type,
                    raw_type: t_type_raw,
                    shape,
                    data_ptr: offset as *const u8,
                },
            );
        }

        // Calculate start of data section and finalize tensor pointers.
        // R-32: checked rounding (alignment already validated pow2/<=cap)
        // + checked per-tensor offset and end (fixed-ratio types verify the
        // full byte range; variable types at least the offset itself).
        let data_start = cur
            .pos
            .checked_add(alignment - 1)
            .ok_or_else(|| anyhow::anyhow!("R-32: data_start offset overflows"))?
            & !(alignment - 1);
        // NOTE: data_start may legally sit past EOF (empty model, no data
        // section); the per-tensor offset/end checks below are the real gate.
        for tensor in tensors.values_mut() {
            let rel = tensor.data_ptr as usize;
            let abs = data_start
                .checked_add(rel)
                .ok_or_else(|| anyhow::anyhow!("R-32: tensor '{}' offset overflows", tensor.name))?;
            if abs > buf.len() {
                anyhow::bail!(
                    "R-32: tensor '{}' offset {abs} past EOF {}",
                    tensor.name,
                    buf.len()
                );
            }
            // Exact byte-size check for fixed-ratio types (ggml blocks).
            if let Some(size) = gguf_tensor_byte_size(tensor.t_type, &tensor.shape) {
                let end = abs.checked_add(size).ok_or_else(|| {
                    anyhow::anyhow!("R-32: tensor '{}' end overflows", tensor.name)
                })?;
                if end > buf.len() {
                    anyhow::bail!(
                        "R-32: tensor '{}' end {end} past EOF {}",
                        tensor.name,
                        buf.len()
                    );
                }
            }
            // SAFETY: `abs <= buf.len()` checked above; `mmap` outlives the
            // model via the `Arc` held in `GGUFModel.mmap`. The pointer is
            // only dereferenced by callers after re-validating length
            // against the tensor's own byte size.
            tensor.data_ptr = unsafe { mmap.as_ptr().add(abs) };
        }

        Ok(Self {
            mmap,
            tensors,
            metadata,
            alignment,
        })
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
        self.tensors
            .get(name)
            .filter(|t| t.t_type == TensorType::Q4_0)
            .map(|t| t.data_ptr as *const BlockQ4_0)
    }
}

/// R-32: bounds-checked byte cursor. Every primitive read validates
/// `pos + size <= buf.len()` with `checked_add` and returns `Err` on
/// truncation — a truncated/hostile file can never panic the parser via
/// slice indexing.
struct GgufCursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl GgufCursor<'_> {
    fn need(&self, len: usize) -> anyhow::Result<usize> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| anyhow::anyhow!("R-32: cursor offset overflows"))?;
        if end > self.buf.len() {
            anyhow::bail!("R-32: truncated GGUF (need {end} bytes, have {})", self.buf.len());
        }
        Ok(end)
    }
    fn read_bytes(&mut self, len: usize) -> anyhow::Result<&[u8]> {
        let end = self.need(len)?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn read_u8(&mut self) -> anyhow::Result<u8> {
        Ok(self.read_bytes(1)?[0])
    }
    fn read_u16(&mut self) -> anyhow::Result<u16> {
        Ok(u16::from_le_bytes(self.read_bytes(2)?.try_into().unwrap()))
    }
    fn read_u32(&mut self) -> anyhow::Result<u32> {
        Ok(u32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }
    fn read_u64(&mut self) -> anyhow::Result<u64> {
        Ok(u64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap()))
    }
}

/// R-32: exact in-file byte size for fixed-ratio GGUF types, `None` for
/// variable/complex layouts (K-quants) where only the offset is checked.
/// Block layouts follow ggml (`Q4_0`: 2B scale + 16B qs per 32 elems, etc.).
fn gguf_tensor_byte_size(t_type: TensorType, shape: &[usize]) -> Option<usize> {
    let mut elements = 1usize;
    for &d in shape {
        elements = elements.checked_mul(d)?;
    }
    match t_type {
        TensorType::F32 => elements.checked_mul(4),
        TensorType::F16 => elements.checked_mul(2),
        TensorType::Q4_0 => elements.div_ceil(32).checked_mul(18),
        TensorType::Q4_1 => elements.div_ceil(32).checked_mul(20),
        TensorType::Q5_0 => elements.div_ceil(32).checked_mul(22),
        TensorType::Q5_1 => elements.div_ceil(32).checked_mul(24),
        TensorType::Q8_0 => elements.div_ceil(32).checked_mul(34),
        TensorType::Q8_1 => elements.div_ceil(32).checked_mul(36),
        _ => None,
    }
}

fn read_metadata_value(
    buf: &[u8],
    pos: usize,
    val_type: u32,
    depth: u8,
) -> anyhow::Result<(MetadataValue, usize)> {
    // R-32: depth cap — nested arrays (type 9) recurse; hostile nesting
    // previously exhausted the stack.
    if depth > GGUF_MAX_ARRAY_DEPTH {
        anyhow::bail!("R-32: metadata array nesting exceeds {GGUF_MAX_ARRAY_DEPTH}");
    }
    let mut cur = GgufCursor { buf, pos };
    let val = match val_type {
        0 => MetadataValue::Uint8(cur.read_u8()?),
        1 => MetadataValue::Int8(cur.read_u8()? as i8),
        2 => MetadataValue::Uint16(cur.read_u16()?),
        3 => MetadataValue::Int16(cur.read_u16()? as i16),
        4 => MetadataValue::Uint32(cur.read_u32()?),
        5 => MetadataValue::Int32(cur.read_u32()? as i32),
        6 => MetadataValue::Float32(f32::from_le_bytes(cur.read_bytes(4)?.try_into().unwrap())),
        7 => MetadataValue::Bool(cur.read_u8()? != 0),
        8 => {
            let len = cur.read_u64()? as usize;
            // R-32: cap string length before slicing (hostile u64::MAX
            // previously panicked on `buf[pos..pos+len]`).
            if len > MAX_STRING_LEN {
                anyhow::bail!("R-32: metadata string len {len} exceeds cap {MAX_STRING_LEN}");
            }
            let s = String::from_utf8_lossy(cur.read_bytes(len)?).into_owned();
            MetadataValue::String(s)
        }
        9 => {
            let itype = cur.read_u32()?;
            let n = cur.read_u64()? as usize;
            // R-32: cap element count (hostile n previously meant a
            // `Vec::with_capacity(n)` multi-GB allocation attempt / abort).
            if n > GGUF_MAX_ARRAY_LEN {
                anyhow::bail!("R-32: metadata array len {n} exceeds cap {GGUF_MAX_ARRAY_LEN}");
            }
            // Reserve incrementally, not `with_capacity(n)`: a hostile-but-
            // under-cap n must still not pre-balloon the Vec before the
            // per-element bounds checks below terminate the parse.
            let mut arr = Vec::new();
            arr.try_reserve(n.min(1024)).map_err(|_| {
                anyhow::anyhow!("R-32: metadata array reserve failed for len {n}")
            })?;
            let mut p = cur.pos;
            for _ in 0..n {
                let (v, next_pos) = read_metadata_value(buf, p, itype, depth + 1)?;
                p = next_pos;
                arr.push(v);
            }
            cur.pos = p;
            MetadataValue::Array(arr)
        }
        10 => MetadataValue::Uint64(cur.read_u64()?),
        11 => MetadataValue::Int64(cur.read_u64()? as i64),
        12 => MetadataValue::Float64(f64::from_le_bytes(cur.read_bytes(8)?.try_into().unwrap())),
        _ => anyhow::bail!("Unsupported GGUF type: {} at pos {}", val_type, pos),
    };
    Ok((val, cur.pos))
}

#[cfg(test)]
mod dump_metadata;
#[cfg(test)]
mod dump_tensors;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tokenizer_test;

// ============================================================
// R-32 (2026-10-13): GGUF parser bounds — truncated/hostile files
// must return Err, never panic. All fixtures are synthetic and
// written to temp_dir (no real model needed).
// ============================================================
#[cfg(test)]
mod r32_bounds_tests {
    use super::*;

    fn write_tmp(name: &str, bytes: &[u8]) -> String {
        let path = std::env::temp_dir().join(format!(
            "gguf_r32_{name}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, bytes).unwrap();
        path.to_string_lossy().into_owned()
    }

    /// Minimal valid header: magic + version + tensor_count + kv_count.
    fn header(tensor_count: u64, kv_count: u64) -> Vec<u8> {
        let mut b = b"GGUF".to_vec();
        b.extend_from_slice(&3u32.to_le_bytes()); // version
        b.extend_from_slice(&tensor_count.to_le_bytes());
        b.extend_from_slice(&kv_count.to_le_bytes());
        b
    }

    #[test]
    fn test_r32_truncated_file_is_err_not_panic() {
        for len in [0usize, 1, 3, 10, 23] {
            let full = header(0, 0);
            let path = write_tmp("trunc", &full[..len.min(full.len())]);
            let r = GGUFModel::load(&path);
            assert!(r.is_err(), "len {len} must be Err, got Ok");
            let _ = std::fs::remove_file(&path);
        }
    }

    #[test]
    fn test_r32_empty_header_valid() {
        // tensor_count=0, kv_count=0 parses fine (data_start == EOF is OK).
        let path = write_tmp("empty", &header(0, 0));
        let m = GGUFModel::load(&path).expect("empty GGUF must load");
        assert!(m.tensors.is_empty() && m.metadata.is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_r32_huge_kv_count_rejected() {
        let path = write_tmp("kvcount", &header(0, u64::MAX));
        let err = GGUFModel::load(&path).err().expect("kv_count=u64::MAX must be Err");
        assert!(err.to_string().contains("R-32"), "got: {err}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_r32_huge_tensor_count_rejected() {
        let path = write_tmp("tcount", &header(u64::MAX, 0));
        let err = GGUFModel::load(&path).err().expect("tensor_count=u64::MAX must be Err");
        assert!(err.to_string().contains("R-32"), "got: {err}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_r32_huge_key_len_rejected() {
        let mut b = header(0, 1);
        b.extend_from_slice(&u64::MAX.to_le_bytes()); // key_len
        let path = write_tmp("keylen", &b);
        let err = GGUFModel::load(&path).err().expect("key_len=u64::MAX must be Err");
        assert!(err.to_string().contains("R-32"), "got: {err}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_r32_huge_string_value_rejected() {
        // kv: key "k" + type 8 (string) + len u64::MAX.
        let mut b = header(0, 1);
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(b"k");
        b.extend_from_slice(&8u32.to_le_bytes());
        b.extend_from_slice(&u64::MAX.to_le_bytes());
        let path = write_tmp("strlen", &b);
        let err = GGUFModel::load(&path).err().expect("string len=u64::MAX must be Err");
        assert!(err.to_string().contains("R-32"), "got: {err}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_r32_huge_array_len_rejected() {
        // kv: key "k" + type 9 (array of u8) + n u64::MAX. Previously this
        // hit `Vec::with_capacity(n)` (abort/OOM); now capped before alloc.
        let mut b = header(0, 1);
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(b"k");
        b.extend_from_slice(&9u32.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes()); // itype u8
        b.extend_from_slice(&u64::MAX.to_le_bytes()); // n
        let path = write_tmp("arrlen", &b);
        let err = GGUFModel::load(&path).err().expect("array len=u64::MAX must be Err");
        assert!(err.to_string().contains("R-32"), "got: {err}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_r32_nested_array_depth_rejected() {
        // 16 nested arrays (type 9, n=1) + innermost u8. Depth cap is 8.
        let mut inner = vec![0u8]; // u8 value
        for _ in 0..16 {
            let mut outer = Vec::new();
            outer.extend_from_slice(&9u32.to_le_bytes()); // itype = array
            outer.extend_from_slice(&1u64.to_le_bytes()); // n = 1
            outer.extend_from_slice(&inner);
            inner = outer;
        }
        let mut b = header(0, 1);
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(b"k");
        b.extend_from_slice(&9u32.to_le_bytes());
        b.extend_from_slice(&9u32.to_le_bytes()); // outer itype = array
        b.extend_from_slice(&1u64.to_le_bytes()); // n = 1
        b.extend_from_slice(&inner);
        let path = write_tmp("nest", &b);
        let err = GGUFModel::load(&path).err().expect("16-deep nesting must be Err");
        assert!(err.to_string().contains("R-32"), "got: {err}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_r32_hostile_alignment_rejected() {
        // kv general.alignment = 0 => previously underflowed data_start math.
        let mut b = header(0, 1);
        let key = b"general.alignment";
        b.extend_from_slice(&(key.len() as u64).to_le_bytes());
        b.extend_from_slice(key);
        b.extend_from_slice(&4u32.to_le_bytes()); // Uint32
        b.extend_from_slice(&0u32.to_le_bytes()); // align = 0
        let path = write_tmp("align0", &b);
        let err = GGUFModel::load(&path).err().expect("alignment=0 must be Err");
        assert!(err.to_string().contains("R-32"), "got: {err}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_r32_tensor_rank_and_dim_rejected() {
        // tensor "w", n_dims = u32::MAX => rank cap.
        let mut b = header(1, 0);
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(b"w");
        b.extend_from_slice(&u32::MAX.to_le_bytes()); // n_dims
        let path = write_tmp("rank", &b);
        let err = GGUFModel::load(&path).err().expect("rank=u32::MAX must be Err");
        assert!(err.to_string().contains("R-32"), "got: {err}");
        let _ = std::fs::remove_file(&path);

        // tensor "w", dims [8, u64::MAX] => dim magnitude cap.
        let mut b = header(1, 0);
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(b"w");
        b.extend_from_slice(&2u32.to_le_bytes()); // n_dims = 2
        b.extend_from_slice(&8u64.to_le_bytes());
        b.extend_from_slice(&u64::MAX.to_le_bytes());
        let path = write_tmp("dim", &b);
        let err = GGUFModel::load(&path).err().expect("dim=u64::MAX must be Err");
        assert!(err.to_string().contains("R-32"), "got: {err}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_r32_tensor_offset_oob_rejected() {
        // tensor "w" F32 [4] with data offset far past EOF.
        let mut b = header(1, 0);
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(b"w");
        b.extend_from_slice(&1u32.to_le_bytes()); // n_dims = 1
        b.extend_from_slice(&4u64.to_le_bytes()); // dim = 4
        b.extend_from_slice(&0u32.to_le_bytes()); // F32
        b.extend_from_slice(&u64::MAX.to_le_bytes()); // offset
        let path = write_tmp("offset", &b);
        let err = GGUFModel::load(&path).err().expect("offset=u64::MAX must be Err");
        assert!(err.to_string().contains("R-32"), "got: {err}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_r32_tensor_end_oob_rejected() {
        // tensor "w" F32 [1024] with offset 0 but no data section: the
        // 4 KiB body check must fail even though the offset itself is small.
        let mut b = header(1, 0);
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(b"w");
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&1024u64.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes()); // F32
        b.extend_from_slice(&0u64.to_le_bytes()); // offset = 0
        let path = write_tmp("end", &b);
        let err = GGUFModel::load(&path).err().expect("body past EOF must be Err");
        assert!(err.to_string().contains("R-32"), "got: {err}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_r32_valid_minimal_model_loads() {
        // tensor "w" F32 [4] with 16 bytes of data, alignment 32.
        let mut b = header(1, 0);
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(b"w");
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&4u64.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes()); // F32
        b.extend_from_slice(&0u64.to_le_bytes()); // offset = 0
        let pos = b.len();
        let data_start = (pos + 31) & !31;
        b.resize(data_start, 0);
        b.extend_from_slice(&[1.0f32, 2.0, 3.0, 4.0].iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<_>>());
        let path = write_tmp("valid", &b);
        let m = GGUFModel::load(&path).expect("minimal valid GGUF must load");
        let t = m.tensors.get("w").expect("tensor w present");
        assert_eq!(t.shape, vec![4]);
        assert_eq!(t.t_type, TensorType::F32);
        let _ = std::fs::remove_file(&path);
    }
}

// ============================================================
// R-32 proptest fuzz (SEC-H2 analogue): arbitrary byte sequences
// must never panic GGUFModel::load. Run with:
//   cargo test --lib --features proptest -- --test-threads=2
// ============================================================
#[cfg(all(test, feature = "proptest"))]
mod r32_fuzz {
    use proptest::prelude::*;

    fn load_bytes_no_panic(bytes: &[u8]) {
        let path = format!(
            "/tmp/gguf_fuzz_{}.bin",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let _ = std::fs::write(&path, bytes);
        let _result = super::GGUFModel::load(&path); // MUST NOT panic
        let _ = std::fs::remove_file(&path);
    }

    proptest! {
        /// Arbitrary bytes (0..4096) must never panic the GGUF loader.
        #[test]
        fn fuzz_gguf_load_arbitrary_bytes(bytes in prop::collection::vec(any::<u8>(), 0..4096)) {
            load_bytes_no_panic(&bytes);
        }

        /// GGUF magic + arbitrary tail must never panic.
        #[test]
        fn fuzz_gguf_load_with_magic(tail in prop::collection::vec(any::<u8>(), 0..2048)) {
            let mut bytes = b"GGUF".to_vec();
            bytes.extend_from_slice(&tail);
            load_bytes_no_panic(&bytes);
        }
    }
}
