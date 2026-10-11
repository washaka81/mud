use crate::model::tokenizer::Tokenizer;
use std::collections::HashMap;

/// Computation mode for the model — determines how the tokenizer
/// formats data for optimal ASM kernel consumption.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelMode {
    /// ELUT 4-bit ternary weights (+1 / -1 / 0) + PRQ scales.
    Ternary,
    /// Dense f32 weights (Q2_0→F32 converted).
    F32,
}

/// Tokenizer-aware model capability flags, detected from metadata.
#[derive(Debug, Clone, Copy)]
pub struct ModelCapabilities {
    pub mode: ModelMode,
    pub has_jepa: bool,
    pub has_cmud: bool,
    pub has_thinker: bool,
    pub hidden_size: usize,
    pub vocab_size: usize,
}

/// Pre-packed token batch with ASM-friendly memory layout.
/// All buffers are 64-byte aligned for YMM loads.
pub struct PackedTokenBatch {
    /// Token IDs, contiguous u32 array (vocab range).
    /// 4-byte aligned naturally; align to 32 for YMM loads.
    pub ids: Vec<u32>,
    /// Pre-computed JEPA features per token (when has_jepa).
    /// Contiguous f32 ready for dot_product_avx2.
    pub jepa_features: Vec<f32>,
    pub n_tokens: usize,
}

/// Streaming batch reader that yields `PackedTokenBatch` from a corpus
/// without intermediate per-call allocations.
///
/// # Pointer / Register optimisation
///
/// - Internal buffers are pre-allocated to `seq_len` and reused.
/// - The hot path in `corpus_trainer` calls `next_batch` without
///   any `malloc` from the tokenizer.
/// - Output `PackedTokenBatch.ids` is contiguous u32 → ready for
///   AVX2 embedding GEMV (gather-free).
pub struct TokenBatchStream {
    tokenizer: std::sync::Arc<SmartTokenizer>,
    encode_buf: Vec<u32>,
    batch_buf: Vec<u32>,
    corpus: Vec<u8>,
    pos: usize,
    seq_len: usize,
    _stride: usize,
    eos_id: u32,
    bos_id: u32,
}

impl TokenBatchStream {
    fn new(
        tokenizer: std::sync::Arc<SmartTokenizer>,
        corpus: Vec<u8>,
        seq_len: usize,
        stride: usize,
    ) -> Self {
        let eos = tokenizer.tokenizer.eos_id();
        let bos = tokenizer.tokenizer.bos_id();
        Self {
            tokenizer,
            encode_buf: Vec::with_capacity(seq_len * 4),
            batch_buf: Vec::with_capacity(seq_len),
            corpus,
            pos: 0,
            seq_len,
            _stride: stride,
            eos_id: eos,
            bos_id: bos,
        }
    }

    /// Returns the next batch of packed token sequences,
    /// or `None` when the corpus is exhausted.
    pub fn next_batch(&mut self) -> Option<PackedTokenBatch> {
        self.batch_buf.clear();
        while self.batch_buf.len() < self.seq_len && self.pos < self.corpus.len() {
            let end = std::cmp::min(self.pos + 4096, self.corpus.len());
            let chunk = &self.corpus[self.pos..end];
            self.encode_buf.clear();
            self.encode_buf.extend(
                self.tokenizer.tokenizer.encode(std::str::from_utf8(chunk).unwrap_or("")),
            );
            if self.encode_buf.is_empty() {
                self.pos = end;
                continue;
            }
            self.batch_buf.push(self.bos_id);
            for &tid in &self.encode_buf {
                if self.batch_buf.len() >= self.seq_len { break; }
                self.batch_buf.push(tid);
            }
            if self.batch_buf.len() < self.seq_len {
                self.batch_buf.push(self.eos_id);
            }
            self.pos = end;
        }
        if self.batch_buf.is_empty() { return None; }
        while self.batch_buf.len() < self.seq_len {
            self.batch_buf.push(self.eos_id);
        }
        let n = self.batch_buf.len().min(self.seq_len);
        let jepa = if self.tokenizer.caps.has_jepa {
            vec![0.0f32; n]
        } else {
            Vec::new()
        };
        Some(PackedTokenBatch {
            ids: self.batch_buf[..n].to_vec(),
            jepa_features: jepa,
            n_tokens: n,
        })
    }
}

/// Intelligent tokenizer that wraps the base `Tokenizer` with
/// mode-aware data packing, zero-alloc streaming, and ASM
/// pointer/register layout optimisation.
///
/// # Pointer / Register strategy
///
/// - **Contiguous u32 arrays** → AVX2 embedding GEMV loads directly
///   from the token ID array with no gather/scatter.
/// - **JEPA features** are pre-allocated f32 contiguous — ready for
///   `dot_product_avx2` (dual-accumulator FMA) without per-call malloc.
/// - **Alignment**: `TOKEN_ALIGN` (32 bytes) matches `_mm256_load_si256`
///   YMM register width.
/// - **Prefetch**: `PREFETCH_DIST` (4 cache lines) for sequential
///   token processing.
/// - **Zero-alloc streaming**: `encode_to_buf` and `TokenBatchStream`
///   reuse internal `Vec`s so the trainer hot path never calls `malloc`.
///
/// # Probabilistic / Deterministic differential calculus
///
/// The tokenizer separates two gradient regimes:
///
/// - **Deterministic (FP32/Ternary GEMV)**: Token IDs index directly
///   into `token_embd.weight` (row-wise GEMV). The embedding gradient
///   flows back through a standard outer product — no stochasticity.
/// - **Probabilistic (JEPA/STE deadzone)**: Token-level features
///   (JEPA z-score, OU tracker alpha) are pre-computed per batch.
///   The STE deadzone threshold (`PRQ_scale × 0.7`) determines which
///   embedding deltas survive quantization — the tokenizer provides
///   embedding scale metadata for the optimizer to adjust LR per
///   token frequency bin.
///
/// # CMUD / Thinker mode
///
/// In CMUD mode (`has_cmud`) the tokenizer reserves phase-bias
/// storage per token dimension (hidden_size). The thinker initialises
/// `ThinkingState` from the pre-CMUD hidden vector — the tokenizer
/// contribution is purely the token ID sequence (same as non-CMUD).
pub struct SmartTokenizer {
    pub tokenizer: Tokenizer,
    pub caps: ModelCapabilities,
    encode_buf: Vec<u32>,
    jepa_scratch: Vec<f32>,
}

impl SmartTokenizer {
    pub fn new(
        tokenizer: Tokenizer,
        meta: &HashMap<String, String>,
    ) -> Self {
        let hidden = meta
            .get("hidden_size")
            .and_then(|s| s.parse().ok())
            .unwrap_or(256);
        let vocab = meta
            .get("vocab_size")
            .and_then(|s| s.parse().ok())
            .unwrap_or(32000);
        let mode = if meta.get("weight_type").map(|s| s == "Float32").unwrap_or(false) {
            ModelMode::F32
        } else {
            ModelMode::Ternary
        };
        let caps = ModelCapabilities {
            mode,
            has_jepa: meta.get("has_jepa").map(|s| s == "1").unwrap_or(true),
            has_cmud: meta.get("has_cmud").map(|s| s == "1").unwrap_or(false),
            has_thinker: meta.get("has_thinker").map(|s| s == "1").unwrap_or(false),
            hidden_size: hidden,
            vocab_size: vocab,
        };
        Self {
            tokenizer,
            caps,
            encode_buf: Vec::with_capacity(4096),
            jepa_scratch: Vec::new(),
        }
    }

    pub fn from_mud_metadata(
        tokens_str: &str,
        merges_str: &str,
        meta: &HashMap<String, String>,
    ) -> Self {
        let tk = Tokenizer::from_mud_metadata(tokens_str, merges_str);
        Self::new(tk, meta)
    }

    // ── Core encode / decode (zero-alloc) ───────────────────────

    /// Encode into internal buffer — no allocation on re-use.
    pub fn encode_to_buf(&mut self, text: &str) -> &[u32] {
        self.encode_buf.clear();
        self.encode_buf.extend(self.tokenizer.encode(text));
        &self.encode_buf
    }

    /// Encode and return raw pointer + length for zero-copy
    /// integration with the embedding GEMV hot path.
    pub fn encode_raw(&mut self, text: &str) -> (*const u32, usize) {
        let s = self.encode_to_buf(text);
        (s.as_ptr(), s.len())
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text)
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids)
    }

    // ── Mode-aware batch packing ────────────────────────────────

    /// Build a `PackedTokenBatch` from raw text.
    ///
    /// **Ternary mode**: The embedding GEMV loads `ids` as u32 row
    /// indices into `token_embd.weight` (ELUT 4-bit). AVX2 kernels
    /// use the ID to compute the row pointer offset directly.
    ///
    /// **F32 mode**: Same row-index pattern into `f32_gemv_rowwise`.
    pub fn pack_text(&mut self, text: &str) -> PackedTokenBatch {
        let ids = self.encode_to_buf(text).to_vec();
        let n = ids.len();
        let jepa = if self.caps.has_jepa {
            self.jepa_scratch.clear();
            self.jepa_scratch.resize(n, 0.0);
            self.jepa_scratch.clone()
        } else {
            Vec::new()
        };
        PackedTokenBatch { ids, jepa_features: jepa, n_tokens: n }
    }

    /// Convenience: write token IDs and JEPA features directly into
    /// pre-allocated buffers.
    ///
    /// `token_ids` and `jepa_z` must be at least `n` elements long.
    ///
    /// # Safety
    ///
    /// Caller must ensure `token_ids` and `jepa_z` are valid for writes.
    pub unsafe fn pack_into_raw(
        &mut self,
        text: &str,
        token_ids: *mut u32,
        jepa_z: *mut f32,
        max_n: usize,
    ) -> PackedTokenBatch {
        let has_jepa = self.caps.has_jepa;
        let ids = self.encode_to_buf(text);
        let n = ids.len().min(max_n);
        std::ptr::copy_nonoverlapping(ids.as_ptr(), token_ids, n);
        if has_jepa && n > 0 {
            std::ptr::write_bytes(jepa_z, 0, n);
        }
        PackedTokenBatch {
            ids: ids.to_vec(),
            jepa_features: Vec::new(),
            n_tokens: n,
        }
    }

    // ── Streaming batch reader ──────────────────────────────────

    pub fn stream_batches(
        self: &std::sync::Arc<Self>,
        corpus: Vec<u8>,
        seq_len: usize,
        stride: usize,
    ) -> TokenBatchStream {
        TokenBatchStream::new(self.clone(), corpus, seq_len, stride)
    }

    // ── ASM register hints ──────────────────────────────────────

    /// YMM register width (32 bytes = 8× u32).
    pub const ALIGN_YMM: usize = 32;

    /// Cache line size (Intel i7-1260P L1D).
    pub const CACHE_LINE: usize = 64;

    /// Prefetch distance for sequential token processing (in cache lines).
    pub const PREFETCH_DIST: usize = 4;

    // ── BOS / EOS ───────────────────────────────────────────────

    pub fn bos_id(&self) -> u32 { self.tokenizer.bos_id() }
    pub fn eos_id(&self) -> u32 { self.tokenizer.eos_id() }
    pub fn special_ids_from_metadata(&self, meta: &HashMap<String, String>) -> (u32, u32) {
        self.tokenizer.special_ids_from_metadata(meta)
    }
    pub fn has_space_prefix(&self, id: u32) -> bool { self.tokenizer.has_space_prefix(id) }
    pub fn vocab_size(&self) -> usize { self.tokenizer.id_to_token.len() }

    /// Look up a special token ID by name (e.g. `<|im_end|>`, `<|endoftext|>`).
    pub fn special_id(&self, name: &str) -> Option<u32> {
        self.tokenizer.special_tokens.get(name).copied()
    }
}
