use memmap2::Mmap;
use std::collections::HashMap;
use std::sync::Arc;

pub mod arena_games;
pub mod arena_judge;
pub mod constants;
pub mod corpus_trainer;
pub mod debate_trainer;
pub mod dspy;
pub mod ecc;
pub mod galore;
pub mod holographic_loss;
pub mod ldt_micro;
pub mod memory_bank;
pub mod memory_profiler;
pub mod rlvr;
pub mod routing;
pub mod sandbox;
pub mod slime;
pub mod slime_backward;
pub mod slime_forward;
pub mod rope_cache;
pub mod gemv_gpu;
pub mod smart_tokenizer;
pub mod slime_jepa;
pub mod stp_loss;
pub mod subagents;
pub mod trainer_ui;
pub mod workspace;
pub mod workspace_agent;
pub mod slime_x;
pub mod memory_continuity;
pub mod boot_indexer;
pub mod file_fusion;

/// Phase 15: Ash bare-metal QAT dispatcher (replaces VulkanQatDispatcher)
pub mod adam_state;
pub mod ash_qat_dispatcher;
pub mod cmud;
pub mod cmud_params;
pub mod cmud_thinking;
pub mod csa_indexer;
pub mod expert_bus;
pub mod ezop;
pub mod fenwick;
pub mod grad_checkpoint;
pub mod inference;

/// SPRINT-4 (4.1): corpus file collection utilities (extracted from `corpus_trainer`).
pub mod corpus;

/// SPRINT-4 (4.1): model checkpoint sync utilities (extracted from `corpus_trainer`).
pub mod checkpoint;
pub mod kv_context;
pub mod kv_compressed;
pub mod kv_dtype;
pub mod loss_cert;
pub mod moe_load;
pub mod moe_train;
pub mod muon;
pub mod optim;
pub mod p13;
pub mod pointer_audit;
pub mod error;
pub mod self_play;
pub mod sequence_pack;
pub mod slime_expert;
pub mod speculative;
/// MUD-WORM: Latent Neural Memory & Hot-Pluggable Synaptic Bank Architecture
pub mod worm;
#[cfg(test)]
mod tests;

/// MUD: Modular Understanding Dynamics
/// File version 1.0
pub const MUD_MAGIC: &[u8; 4] = b"MUD\x01";

// ── SEC-H1 loader caps (2026-07-26): parsed-from-file header counts are
// untrusted attacker surface. Real BitNet-class/SmolLM-class models have ~50
// metadata entries + ~700-1500 tensors; we leave a ~50× headroom. Refusing
// exotic counts (loaded into a HashMap/Vec that would balloon on first sync)
// bounds greedy-file DoS that previously could force multi-GB allocations from
// a 4-byte header field.
pub const MAX_META_ENTRIES: usize = 4_096;
pub const MAX_TENSOR_ENTRIES: usize = 65_536;
/// Single metadata key/value, tensor name, or string metadata payload.
/// 4 MiB is far above any legit tensor name (~30 bytes) or metadata string.
pub const MAX_STRING_LEN: usize = 4 * 1024 * 1024;
/// A tensor's `shape` vec lives in this many dims (real max: 4 for To-BitNet).
pub const MAX_SHAPE_DIMS: usize = 16;
/// R-31: cap de magnitud por dimensión de tensor (1<<28 ≈ 268M). Dimensiones
/// reales máximas: vocab ~200k, hidden/ffn ~16k. Un `u64 LE` arbitrario del
/// header (hasta 2^64-1) sin cota permitía `product`/`*` con wrap en release
/// (sin overflow-checks) → `expected_bytes` pequeño → OOB read. Con esta cota
/// + aritmética checked, dims absurdas devuelven `Err` en vez de hacer wrap.
pub const MAX_TENSOR_DIM: usize = 1 << 28;

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
    /// Offset of this tensor's data relative to the start of the data region
    /// (i.e. relative to `data_base`), as written in the file header.
    pub offset: usize,
    /// Absolute byte offset of the start of the data region within `mmap`.
    /// 0 for tensors that are not backed by an mmap (e.g. owned/created in code).
    pub data_base: usize,
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
            data_base: self.data_base,
            mmap: self.mmap.clone(),
            owned_data: self.owned_data.clone(),
        };
        if let Some(owned) = &cloned.owned_data {
            cloned.data_ptr = owned.as_ptr();
        } else if let Some(mmap) = &cloned.mmap {
            let data_start = (cloned.data_base + cloned.offset + 31) & !31;
            cloned.data_ptr = unsafe { mmap.as_ptr().add(data_start) };
        }
        cloned
    }
}

impl MudTensor {
    pub fn data_size(&self) -> usize {
        // R-31: saturating (nunca wrap). Un wrap a la baja/alta en release
        // podía inframedir `expected_bytes` → OOB read. Saturar como máximo
        // provoca un panic seguro en el slicing, jamás OOB.
        let elements: usize = self
            .shape
            .iter()
            .fold(1usize, |a, &d| a.saturating_mul(d));
        match self.t_type {
            MudTensorType::Ternary2Bit => elements.div_ceil(8).saturating_mul(4),
            MudTensorType::Float32 => elements.saturating_mul(4),
            MudTensorType::Float16 => elements.saturating_mul(2),
            MudTensorType::Int4 => elements.div_ceil(2),
            MudTensorType::Uint8 => elements,
        }
    }

    /// R-31: tamaño exacto con aritmética checked. `None` si hay overflow.
    pub fn checked_data_size(&self) -> Option<usize> {
        let mut elements = 1usize;
        for &d in &self.shape {
            elements = elements.checked_mul(d)?;
        }
        match self.t_type {
            MudTensorType::Ternary2Bit => elements.div_ceil(8).checked_mul(4),
            MudTensorType::Float32 => elements.checked_mul(4),
            MudTensorType::Float16 => elements.checked_mul(2),
            MudTensorType::Int4 => Some(elements.div_ceil(2)),
            MudTensorType::Uint8 => Some(elements),
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
    /// SPRINT-4 (4.1): source path on disk, used by the checkpoint helper to
    /// derive the canonical `weights/checkpoints/<stem>_latest.mud` filename.
    /// `None` for in-memory constructions (tests, file fusion).
    pub path: Option<String>,
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
                file_size,
                expected_min_size
            );
        }

        std::fs::rename(&temp_path, final_path)?;

        let final_metadata = std::fs::metadata(final_path)?;
        if final_metadata.len() != file_size {
            anyhow::bail!(
                "StreamingMudWriter::close: file size mismatch after rename (temp: {}, final: {})",
                file_size,
                final_metadata.len()
            );
        }

        Ok(())
    }
}

impl MudFile {
    /// Deep Configuration Incrustation: Process and parse the dynamically embedded configuration at load time.
    pub fn raw_config(&self) -> Option<serde_json::Value> {
        self.global_metadata
            .get("raw_config_json")
            .and_then(|s| serde_json::from_str(s).ok())
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
            // Defensive: a tensor left on mmap may have had its `data_ptr` reset to
            // null (e.g. after ecc_verify_all) or dangling (mmap dropped). Never
            // read through `data_ptr` here; the owned/mmap sources are authoritative.

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
            } else if let Some(mmap) = &tensor.mmap {
                // Read through the tensor's own Arc<Mmap> using its aligned offset,
                // NOT `data_ptr` (which may be null/dangling — e.g. after
                // ecc_verify_all reset it, or after the global mmap was dropped).
                let elements: usize = tensor.shape.iter().product();
                let s_expected = match tensor.t_type {
                    MudTensorType::Ternary2Bit => elements.div_ceil(8) * 4,
                    MudTensorType::Float32 => elements * 4,
                    MudTensorType::Float16 => elements * 2,
                    MudTensorType::Int4 => elements.div_ceil(2),
                    MudTensorType::Uint8 => elements,
                };
                // `tensor.offset` is relative to the start of the data region; the
                // absolute position in the mmap is `data_base + offset` (both already
                // 32-aligned by the writer, so the sum is too).
                let data_start = (tensor.data_base + tensor.offset + 31) & !31;
                let slice = &mmap[data_start..data_start + s_expected];
                file.write_all(slice)?;
                slice.len()
            } else {
                // No source data at all: write zeros rather than deref a null/dangling
                // pointer (defensive — would otherwise SIGSEGV on a degraded model).
                let elements: usize = tensor.shape.iter().product();
                let s_expected = match tensor.t_type {
                    MudTensorType::Ternary2Bit => elements.div_ceil(8) * 4,
                    MudTensorType::Float32 => elements * 4,
                    MudTensorType::Float16 => elements * 2,
                    MudTensorType::Int4 => elements.div_ceil(2),
                    MudTensorType::Uint8 => elements,
                };
                file.write_all(&vec![0u8; s_expected])?;
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
                written_size,
                final_size
            );
        }

        let verify_file = std::fs::File::open(path)?;
        let verify_mmap = unsafe { Mmap::map(&verify_file)? };
        if verify_mmap.len() != final_size as usize {
            anyhow::bail!(
                "MudFile::save: mmap size mismatch (mmap: {}, file: {})",
                verify_mmap.len(),
                final_size
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
                let mud_offset =
                    u64::from_le_bytes(mmap[offset_pos..offset_pos + 8].try_into()?) as usize;

                // SEC-H9: Validate that mud_offset is rational — not 0 (would
                // collide with the trailer itself) and not within the trailer
                // region (last 16 bytes: 8-byte offset + 8-byte "MUDEXEC\0").
                // Also must leave room for at least the 4-byte magic + header.
                let min_offset = 16; // minimum header size (magic + counts)
                let max_offset = mmap.len().saturating_sub(16); // exclude trailer
                if mud_offset >= min_offset
                    && mud_offset <= max_offset
                    && mud_offset + 4 <= mmap.len()
                    && &mmap[mud_offset..mud_offset + 4] == MUD_MAGIC
                {
                    start_offset = mud_offset;
                } else {
                    anyhow::bail!(
                        "SEC-H9: Invalid MUD offset {} in MUD-Executable trailer \
                         (must be in [{}, {}])",
                        mud_offset, min_offset, max_offset
                    );
                }
            } else {
                anyhow::bail!("Invalid MUD magic number and no MUDEXEC trailer found");
            }
        }

        let mut pos = start_offset + 4;

        let meta_count = u32::from_le_bytes(mmap[pos..pos + 4].try_into()?) as usize;
        // SEC-H1: refuse exotic meta_count before allocating the HashMap.
        if meta_count > MAX_META_ENTRIES {
            anyhow::bail!(
                "meta_count {meta_count} exceeds cap {MAX_META_ENTRIES} (SEC-H1)"
            );
        }
        pos += 4;
        let mut global_metadata = HashMap::with_capacity(meta_count);
        for _ in 0..meta_count {
            // SEC-H2 panic-fix (2026-07-27): bounds-check BEFORE slice read.
            // proptest fuzz uncovered that `mmap[pos..pos+4]` panics when
            // meta_count is within the SEC-H1 cap but the file body is
            // truncated. CVE-class CWE-125 (OOB read) turned into a DoS panic.
            if pos + 4 > mmap.len() {
                anyhow::bail!("truncated meta key header at pos {pos} (SEC-H2)");
            }
            let k_len = u32::from_le_bytes(mmap[pos..pos + 4].try_into()?) as usize;
            if k_len > MAX_STRING_LEN || pos + 4 + k_len > mmap.len() {
                anyhow::bail!("metadata key bogus len/bounds (SEC-H1)");
            }
            pos += 4;
            let key = String::from_utf8_lossy(&mmap[pos..pos + k_len]).into_owned();
            pos += k_len;

            if pos + 4 > mmap.len() {
                anyhow::bail!("truncated meta value header at pos {pos} (SEC-H2)");
            }
            let v_len = u32::from_le_bytes(mmap[pos..pos + 4].try_into()?) as usize;
            if v_len > MAX_STRING_LEN || pos + 4 + v_len > mmap.len() {
                anyhow::bail!("metadata value bogus len/bounds (SEC-H1)");
            }
            pos += 4;
            let val = String::from_utf8_lossy(&mmap[pos..pos + v_len]).into_owned();
            pos += v_len;
            global_metadata.insert(key, val);
        }

        if pos + 4 > mmap.len() {
            anyhow::bail!("truncated tensor_count header at pos {pos} (SEC-H2)");
        }
        let tensor_count = u32::from_le_bytes(mmap[pos..pos + 4].try_into()?) as usize;
        // SEC-H1: refuse exotic tensor_count before allocating the HashMap.
        if tensor_count > MAX_TENSOR_ENTRIES {
            anyhow::bail!(
                "tensor_count {tensor_count} exceeds cap {MAX_TENSOR_ENTRIES} (SEC-H1)"
            );
        }
        pos += 4;
        let mut tensors = HashMap::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            if pos + 4 > mmap.len() {
                anyhow::bail!("truncated tensor name header at pos {pos} (SEC-H2)");
            }
            let n_len = u32::from_le_bytes(mmap[pos..pos + 4].try_into()?) as usize;
            if n_len > MAX_STRING_LEN || pos + 4 + n_len > mmap.len() {
                anyhow::bail!("tensor name bogus len/bounds (SEC-H1)");
            }
            pos += 4;
            let name = String::from_utf8_lossy(&mmap[pos..pos + n_len]).into_owned();
            pos += n_len;

            if pos + 4 > mmap.len() {
                anyhow::bail!("truncated tensor t_type at pos {pos} (SEC-H2)");
            }
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

            if pos + 4 > mmap.len() {
                anyhow::bail!("truncated tensor shape rank at pos {pos} (SEC-H2)");
            }
            let s_len = u32::from_le_bytes(mmap[pos..pos + 4].try_into()?) as usize;
            if s_len > MAX_SHAPE_DIMS {
                anyhow::bail!("tensor shape rank {s_len} exceeds cap {MAX_SHAPE_DIMS} (SEC-H1)");
            }
            // SEC-H2: bounds-check for `s_len * 8` bytes (shape vector body).
            // R-31: checked (un `s_len` gigante hacía wrap en `s_len * 8`).
            let shape_bytes = s_len
                .checked_mul(8)
                .ok_or_else(|| anyhow::anyhow!("tensor shape body size overflow (R-31)"))?;
            let shape_end = pos
                .checked_add(4)
                .and_then(|v| v.checked_add(shape_bytes))
                .ok_or_else(|| anyhow::anyhow!("tensor shape body offset overflow (R-31)"))?;
            if shape_end > mmap.len() {
                anyhow::bail!("truncated tensor shape body (SEC-H2)");
            }
            pos += 4;
            let mut shape = Vec::with_capacity(s_len);
            for _ in 0..s_len {
                let dim = u64::from_le_bytes(mmap[pos..pos + 8].try_into()?) as usize;
                // R-31: cota de magnitud por dimensión (anti-wrap/OOM).
                if dim > MAX_TENSOR_DIM {
                    anyhow::bail!(
                        "tensor '{}' dim {dim} exceeds cap {MAX_TENSOR_DIM} (R-31)",
                        name
                    );
                }
                shape.push(dim);
                pos += 8;
            }

            if pos + 8 > mmap.len() {
                anyhow::bail!("truncated tensor offset at pos {pos} (SEC-H2)");
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
                    data_base: 0,
                    mmap: Some(mmap.clone()),
                    owned_data: None,
                },
            );
        }

        let data_start = (pos + 31) & !31; // Align start to 32-byte boundary (matching writer)
        let mmap_len = mmap.len();

        for tensor in tensors.values_mut() {
            tensor.data_base = data_start;
            // R-31: checked_add + bail (el `.expect` anterior era un panic
            // ante input no confiable; SEC-H2 exige solo `Err`).
            let ptr_offset = data_start.checked_add(tensor.offset).ok_or_else(|| {
                anyhow::anyhow!("load: data_start + tensor.offset overflow (R-31)")
            })?;

            // R-31: producto y tamaños con checked (release no tiene
            // overflow-checks: `product`/`*`/`+` con wrap eludían el gate OOB).
            let mut n_elements = 1usize;
            for &d in &tensor.shape {
                n_elements = n_elements.checked_mul(d).ok_or_else(|| {
                    anyhow::anyhow!(
                        "load: tensor '{}' shape product overflow (R-31)",
                        tensor.name
                    )
                })?;
            }
            let expected_bytes = match tensor.t_type {
                MudTensorType::Ternary2Bit => n_elements
                    .div_ceil(8)
                    .checked_mul(4)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "load: tensor '{}' byte size overflow (R-31)",
                            tensor.name
                        )
                    })?,
                MudTensorType::Float32 => n_elements.checked_mul(4).ok_or_else(|| {
                    anyhow::anyhow!("load: tensor '{}' byte size overflow (R-31)", tensor.name)
                })?,
                MudTensorType::Float16 => n_elements.checked_mul(2).ok_or_else(|| {
                    anyhow::anyhow!("load: tensor '{}' byte size overflow (R-31)", tensor.name)
                })?,
                MudTensorType::Int4 => n_elements.div_ceil(2),
                MudTensorType::Uint8 => n_elements,
            };

            let end = ptr_offset.checked_add(expected_bytes).ok_or_else(|| {
                anyhow::anyhow!("load: tensor '{}' end offset overflow (R-31)", tensor.name)
            })?;
            if end > mmap_len {
                anyhow::bail!(
                    "CORRUPTION DETECTED: Tensor '{}' (type {:?}) requires {} bytes at offset 0x{:x}, but mmap only has {} bytes remaining.",
                    tensor.name, tensor.t_type, expected_bytes, ptr_offset, mmap_len.saturating_sub(ptr_offset)
                );
            }

            // SAFETY: `end = ptr_offset + expected_bytes` (checked) ≤ mmap_len;
            // el slice [ptr_offset, end) está dentro del mmap.
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
        let this = Self {
            mmap: Some(mmap),
            skills,
            global_metadata,
            path: Some(path.to_string()),
        };
        // R-30: metadata↔shapeFail-fast: un header válido con dims incoherentes
        // (p.ej. `token_embd.weight` de 10 elementos con vocab=32000) antes
        // producía OOB read / OOM en `inference.rs` (`from_raw_parts` con
        // longitudes de metadata). Ahora es `Err` en carga.
        this.validate_model_shapes()?;
        Ok(this)
    }

    /// R-30: valida que cada tensor presente encaje con las dims declaradas en
    /// `global_metadata` (`token_embd == [vocab,hidden]`,
    /// `blk.N.attn_q == [n_heads*head_dim,hidden]`, …) con `checked_*`.
    ///
    /// Semántica:
    /// - Sin dims completas en metadata → `Ok(())` (nada que cotejar; el
    ///   forward exige metadata vía `expect_metadata` igualmente).
    /// - Tensor ausente → se omite (el forward tolera nulos en opcionales).
    /// - Tensor presente con shape incoherente → `Err` (no panic/OOB/OOM).
    pub fn validate_model_shapes(&self) -> anyhow::Result<()> {
        fn meta_usize(md: &std::collections::HashMap<String, String>, keys: &[&str]) -> Option<usize> {
            keys.iter()
                .filter_map(|k| md.get(*k))
                .filter_map(|v| v.parse::<usize>().ok())
                .next()
        }
        let md = &self.global_metadata;
        let (hidden, layers, n_heads, n_kv, ffn_mid, vocab) = match (
            meta_usize(md, &["hidden_size"]),
            meta_usize(md, &["num_hidden_layers", "num_layers"]),
            meta_usize(md, &["num_attention_heads", "num_heads"]),
            meta_usize(md, &["num_key_value_heads", "num_kv_heads"]),
            meta_usize(md, &["intermediate_size", "ffn_hidden"]),
            meta_usize(md, &["vocab_size"]),
        ) {
            (Some(h), Some(l), Some(nh), Some(nkv), Some(f), Some(v)) => (h, l, nh, nkv, f, v),
            _ => return Ok(()),
        };
        for (k, v) in [
            ("hidden_size", hidden),
            ("num_layers", layers),
            ("n_heads", n_heads),
            ("n_kv_heads", n_kv),
            ("ffn_mid", ffn_mid),
            ("vocab_size", vocab),
        ] {
            if v == 0 || v > MAX_TENSOR_DIM {
                anyhow::bail!("R-30: metadata dim {k}={v} fuera de rango");
            }
        }
        if n_heads == 0 || hidden % n_heads != 0 {
            anyhow::bail!("R-30: hidden_size={hidden} no divisible por n_heads={n_heads}");
        }
        let head_dim = hidden / n_heads;
        let q_rows = n_heads.checked_mul(head_dim).ok_or_else(|| {
            anyhow::anyhow!("R-30: q_rows overflow (checked_mul)")
        })?;
        let kv_rows = n_kv.checked_mul(head_dim).ok_or_else(|| {
            anyhow::anyhow!("R-30: kv_rows overflow (checked_mul)")
        })?;
        let core = match self.skills.get("core") {
            Some(c) => c,
            None => return Ok(()),
        };
        let check = |name: &str, expected: &[usize]| -> anyhow::Result<()> {
            if let Some(t) = core.tensors.get(name) {
                if t.shape != expected {
                    anyhow::bail!(
                        "R-30: tensor '{name}' shape {:?} incoherente con metadata (esperado {:?})",
                        t.shape,
                        expected
                    );
                }
            }
            Ok(())
        };
        // Embeddings / head.
        check("token_embd.weight", &[vocab, hidden])?;
        check("token_embd.prq_scale", &[vocab])?;
        check("output.weight", &[vocab, hidden])?;
        check("output.prq_scale", &[vocab])?;
        check("output_norm.weight", &[hidden])?;
        // Capas.
        for blk in 0..layers {
            let p = format!("blk.{blk}.");
            let w = |n: &str| format!("{p}{n}.weight");
            let s = |n: &str| format!("{p}{n}.prq_scale");
            check(&w("attn_q"), &[q_rows, hidden])?;
            check(&w("attn_k"), &[kv_rows, hidden])?;
            check(&w("attn_v"), &[kv_rows, hidden])?;
            check(&w("attn_o"), &[hidden, q_rows])?;
            check(&s("attn_q"), &[q_rows])?;
            check(&s("attn_k"), &[kv_rows])?;
            check(&s("attn_v"), &[kv_rows])?;
            check(&s("attn_o"), &[hidden])?;
            check(&w("attn_norm"), &[hidden])?;
            check(&w("ffn_norm"), &[hidden])?;
            // FFN denso (naming universal: si hay triple denso se valida;
            // si el modelo es MoE con expert.* se omite aquí).
            let has_dense = core.tensors.contains_key(&w("ffn_up"))
                || core.tensors.contains_key(&w("ffn_gate"))
                || core.tensors.contains_key(&w("ffn_down"));
            if has_dense {
                check(&w("ffn_up"), &[ffn_mid, hidden])?;
                check(&w("ffn_gate"), &[ffn_mid, hidden])?;
                check(&w("ffn_down"), &[hidden, ffn_mid])?;
                check(&s("ffn_up"), &[ffn_mid])?;
                check(&s("ffn_gate"), &[ffn_mid])?;
                check(&s("ffn_down"), &[hidden])?;
            }
            check(&format!("{p}mhc_alpha"), &[hidden])?;
            check(&format!("{p}mhc_beta"), &[hidden])?;
        }
        Ok(())
    }

    pub fn get_tensor_ternary(&self, skill: &str, name: &str) -> Option<*const u32> {
        self.skills
            .get(skill)?
            .tensors
            .get(name)
            .filter(|t| t.t_type == MudTensorType::Ternary2Bit)
            .map(|t| t.data_ptr as *const u32)
    }

    /// Copy every tensor off the read-only mmap into owned heap buffers.
    ///
    /// Required for STE QAT training: optimizer pack writes scales + packed
    /// weights in place. A plain `Mmap` is PROT_READ and will SIGSEGV on store.
    /// After this call, `data_ptr` points into `owned_data` (writable via cast).
    ///
    /// **RAM warning:** full materialize doubles working set (mmap pages + owned).
    /// Prefer [`materialize_for_ste_train`] when only last-N layers are thawed.
    pub fn materialize_writable(&mut self) {
        let _ = self.materialize_tensors_if(|_skill, _name, _t| true);
    }

    /// Selectively copy tensors matching `pred` off RO mmap into owned heap.
    ///
    /// Tensors that stay on mmap keep zero-copy reads (inference + frozen layers).
    /// Returns `(tensors_copied, bytes_copied)`. Shared `self.mmap` is dropped only
    /// when no tensor still references it.
    pub fn materialize_tensors_if<F>(&mut self, mut pred: F) -> (usize, usize)
    where
        F: FnMut(&str, &str, &MudTensor) -> bool,
    {
        let mut count = 0usize;
        let mut bytes = 0usize;
        let mut any_left_on_mmap = false;

        // Collect skill names first so we can mutate tensors without borrow fights.
        let skill_names: Vec<String> = self.skills.keys().cloned().collect();
        for skill_name in skill_names {
            let Some(skill) = self.skills.get_mut(&skill_name) else {
                continue;
            };
            let tensor_names: Vec<String> = skill.tensors.keys().cloned().collect();
            for t_name in tensor_names {
                let Some(tensor) = skill.tensors.get_mut(&t_name) else {
                    continue;
                };
                if tensor.owned_data.is_some() {
                    if let Some(ref owned) = tensor.owned_data {
                        tensor.data_ptr = owned.as_ptr();
                    }
                    continue;
                }
                if !pred(&skill_name, &t_name, tensor) {
                    if tensor.mmap.is_some() {
                        any_left_on_mmap = true;
                    }
                    continue;
                }
                let nbytes = tensor.data_size();
                if nbytes == 0 || tensor.data_ptr.is_null() {
                    continue;
                }
                let mut buf = vec![0u8; nbytes];
                // SAFETY: data_ptr was validated at load; nbytes == data_size().
                unsafe {
                    std::ptr::copy_nonoverlapping(tensor.data_ptr, buf.as_mut_ptr(), nbytes);
                }
                tensor.data_ptr = buf.as_ptr();
                tensor.owned_data = Some(buf);
                tensor.mmap = None;
                count += 1;
                bytes += nbytes;
            }
        }

        if !any_left_on_mmap {
            self.mmap = None;
        }
        (count, bytes)
    }

    /// RAM-first materialize for STE QAT: only thawed layer weights/scales (+ emb if not frozen).
    ///
    /// Frozen lower blocks and RMSNorms stay on RO mmap (zero-copy forward).
    /// `first_train_layer` is the first *trainable* block index (layers `[0, first)` frozen).
    pub fn materialize_for_ste_train(
        &mut self,
        first_train_layer: usize,
        freeze_emb: bool,
    ) -> (usize, usize) {
        let has_mmap = self.mmap.is_some();
        let (n, b) = self.materialize_tensors_if(|_skill, name, _t| {
            if !has_mmap {
                return true;
            }
            if name.starts_with("token_embd") || name.starts_with("output") {
                return !freeze_emb;
            }
            // blk.N.* — only STE-written weight + prq_scale for thawed layers
            if let Some(rest) = name.strip_prefix("blk.") {
                let layer = rest
                    .split('.')
                    .next()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                if layer < first_train_layer {
                    return false;
                }
                return name.contains(".weight") || name.contains(".prq_scale");
            }
            // norms / ecc / misc: never written by STE pack → keep mmap
            false
        });
        if std::env::var("MUD_CIRCUIT_TUI").is_err() {
            log::info!(
                "  \x1b[1;36m[RAM]\x1b[0m selective materialize: {} tensors, {:.1} MiB owned (first_train={}, freeze_emb={})",
                n,
                b as f64 / (1024.0 * 1024.0),
                first_train_layer,
                freeze_emb
            );
        }
        (n, b)
    }

    /// Load a .mud file and verify ECC parity for all ternary tensors.
    /// Single-bit errors are corrected; parity-bit errors are logged.
    pub fn load_verified(path: &str) -> anyhow::Result<Self> {
        let mut mud = Self::load(path)?;
        let (corrected, parity_err) = mud.ecc_verify_all();
        if corrected > 0 || parity_err > 0 {
            log::warn!(
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
                        let slice =
                            unsafe { std::slice::from_raw_parts(tensor.data_ptr, u32_count * 4) };
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
                        data_base: 0,
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
                        let start = ecc_tensor.data_base + ecc_tensor.offset;
                        mmap[start..start + u32_count].to_vec()
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
/// L-09: EZOP pack path (raw pointers for nibble writes).
pub fn pack_ternary_row(values: &[f32], delta: f32) -> Vec<u8> {
    let n = values.len();
    let u32_count = n.div_ceil(8);
    let mut packed = vec![0u32; u32_count];
    // SAFETY: values and packed sized for n / u32_count.
    unsafe {
        ezop::pack_ternary_into(values.as_ptr(), n, delta, packed.as_mut_ptr());
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

/// ELUT 4-bit → f32 decode table: index = nibble, value = {+1 if 0x1, -1 if 0xF, else 0}.
/// Branchless lookup replaces the per-element `match` in the GEMV decode hot path.
/// Shared by `dequantize_ternary_row` and `unpack_ternary2bit_to_f32`.
pub(crate) const TERNARY_LUT: [f32; 16] = [
    0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0,
];

/// Dequantizes a row of packed ternary weights (1.58-bit) to f32.
/// L-09: EZOP write path into `out`. Branchless nibble LUT lookup.
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

    let out_p = out.as_mut_ptr();
    let u32_count = n / 8;
    let remainder = n % 8;
    for i in 0..u32_count {
        let val = *packed.add(i);
        let base = i * 8;
        for j in 0..8 {
            let bits = ((val >> (j * 4)) & 0xF) as usize;
            *out_p.add(base + j) = TERNARY_LUT[bits];
        }
    }
    if remainder > 0 {
        let val = *packed.add(u32_count);
        let base = u32_count * 8;
        for j in 0..remainder {
            let bits = ((val >> (j * 4)) & 0xF) as usize;
            *out_p.add(base + j) = TERNARY_LUT[bits];
        }
    }
}
pub mod integral_threshold;
pub mod pcore_pool;
pub mod circuit_rpg;
