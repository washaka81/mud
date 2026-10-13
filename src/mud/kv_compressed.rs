//! # Stream I: Lossless KV Cache Compression (Delta + LZ4)
//!
//! Compresión lossless del KV cache usando Delta encoding + LZ4.
//!
//! - **Delta encoding:** Los valores KV entre tokens consecutivos son muy similares.
//!   Almacenar deltas reduce la entropía, mejorando la compresión LZ4.
//! - **LZ4:** ~8 GB/s descompresión (AVX2), 2-3× compresión sobre f32.
//! - **Resultado:** 4-6× compresión lossless, mínimo costo CPU.
//!
//! ## Env vars
//! - `MUD_KV_COMPRESS=1` — activar compresión (default: 0)
//! - `MUD_KV_COMPRESS_DELTA=1` — delta encoding (default: 1 cuando compress=1)
//! - `MUD_KV_BLOCK_SIZE=256` — tokens por bloque (default: 256)
//! - `MUD_KV_COMPRESS_CELLV5=1` — **reserved** for PLAN_DIA_2 Fase-3 (CellV5
//!   pre-compression: F32 → CellV5 (4×) → Delta (2×) → LZ4 (2-3×) = 16-24× total).
//!   The flag is read + surfaced by the resolver but the Fase-3 compression
//!   pass itself is **not yet implemented** (deferred — requires CellV5
//!   pack/unpack integration that does not corrupt the lossless pipeline).
//!   Operators get a `log::warn!` warning so silent no-op is impossible.
//!
//! ## Pipeline overlap (D4)
//! Los bloques se descomprimen en background mientras la atención procesa
//! el bloque actual. Usa `PipelineOverlap` en `ash_qat_dispatcher.rs`.

use std::cell::RefCell;

thread_local! {
    /// P-01: thread-local delta buffer for `compress_block` (eliminates per-call Vec allocation).
    static KV_DELTA_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    /// BUG-1 + RIESGO-3 FIX (2026-08-03): thread-local byte buffer for LZ4
    /// `decompress_into` — avoids per-call Vec allocation in the KV hot path.
    /// Sized in `decompress_block` (block_elems * 4 bytes).
    static KV_DECOMPRESS_BUF: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    /// BUG-2-style alignment-safe f32 reinterpret buffer (mirrors miri fix 2026-07-28).
    /// `lz4_flex::decompress_into` writes bytes; we `copy_nonoverlapping` into this
    /// f32 buffer with guaranteed 4-byte alignment instead of `from_raw_parts` on Vec<u8>.
    static KV_DECOMPRESS_F32_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Re-export for tests / external callers that read the flag.
pub fn kv_compress_cellv5_reserved() -> bool {
    std::env::var("MUD_KV_COMPRESS_CELLV5").as_deref() == Ok("1")
}
#[derive(Clone, Copy, Debug)]
pub struct KvCompressConfig {
    pub enabled: bool,
    pub use_delta: bool,
    pub block_size: usize,
    /// Fase-3 reserved flag (not-yet-implemented CellV5 pre-compression).
    /// Emitting here lets operators see the env var is *recognized* so they
    /// don't expect a silent no-op when flipping it.
    pub cellv5_reserved: bool,
}

impl KvCompressConfig {
    pub fn resolve() -> Self {
        let enabled = std::env::var("MUD_KV_COMPRESS").as_deref() == Ok("1");
        let use_delta = if enabled {
            std::env::var("MUD_KV_COMPRESS_DELTA").as_deref() != Ok("0")
        } else {
            false
        };
        let block_size = std::env::var("MUD_KV_BLOCK_SIZE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(256)
            .max(16);
        let cellv5_reserved = std::env::var("MUD_KV_COMPRESS_CELLV5").as_deref() == Ok("1");
        if cellv5_reserved {
            log::warn!(
                "[kv_compressed] MUD_KV_COMPRESS_CELLV5=1 is RECOGNIZED but the Fase-3 \
                 pipeline (F32→CellV5→Delta→LZ4) is not yet implemented. Falling back to \
                 Delta+LZ4 only — see docs/planning/PLAN_DIA_2_2026-07-26.md Fase 3."
            );
        }
        Self {
            enabled,
            use_delta,
            block_size,
            cellv5_reserved,
        }
    }
}

/// Un bloque de KV cache comprimido.
#[derive(Clone)]
pub struct KvBlock {
    /// Datos LZ4 comprimidos.
    pub compressed: Vec<u8>,
    /// Tamaño original (descomprimido) en bytes.
    pub original_size: usize,
    /// Tokens válidos en este bloque.
    pub valid_tokens: usize,
    /// Valores base para delta reconstruction (primer token del bloque).
    /// Sólo usado cuando `use_delta` es true.
    pub base: Vec<f32>,
}

impl KvBlock {
    pub fn new() -> Self {
        Self {
            compressed: Vec::new(),
            original_size: 0,
            valid_tokens: 0,
            base: Vec::new(),
        }
    }
}

impl Default for KvBlock {
    fn default() -> Self {
        Self::new()
    }
}

/// KV cache comprimido con Delta + LZ4 (lossless).
///
/// Almacena bloques de tokens comprimidos. Cada bloque contiene
/// `block_size` tokens. Los bloques se comprimen al escribir y se
/// descomprimen bajo demanda durante la atención.
#[derive(Clone)]
pub struct CompressedKvCache {
    /// Bloques comprimidos.
    pub blocks: Vec<KvBlock>,
    /// Configuración.
    pub config: KvCompressConfig,
    /// Dimensión de cabeza (head_dim).
    pub head_dim: usize,
    /// Número de cabezas KV.
    pub n_kv_heads: usize,
}

impl CompressedKvCache {
    /// Crea un nuevo cache comprimido.
    pub fn new(config: KvCompressConfig, head_dim: usize, n_kv_heads: usize) -> Self {
        Self {
            blocks: Vec::new(),
            config,
            head_dim,
            n_kv_heads,
        }
    }

    /// Número de bloques necesarios para `num_tokens` tokens.
    pub fn num_blocks(&self, num_tokens: usize) -> usize {
        num_tokens.div_ceil(self.config.block_size)
    }

    /// Bloque que contiene el token `token_idx`.
    pub fn block_for_token(&self, token_idx: usize) -> usize {
        token_idx / self.config.block_size
    }

    /// Comprime un bloque de KV data (Delta + LZ4).
    ///
    /// `data` debe contener `valid_tokens * n_kv_heads * head_dim` elementos f32.
    /// El primer token se almacena como `base` para delta reconstruction.
    pub fn compress_block(
        &mut self,
        block_idx: usize,
        data: &[f32],
        valid_tokens: usize,
    ) {
        if !self.config.enabled {
            return;
        }

        let block_elems = self.config.block_size * self.n_kv_heads * self.head_dim;
        let actual_elems = valid_tokens * self.n_kv_heads * self.head_dim;
        let _ = block_elems; // reserved for future block-size validation

        // Asegurar que hay suficientes bloques
        while self.blocks.len() <= block_idx {
            self.blocks.push(KvBlock::new());
        }

        let block = &mut self.blocks[block_idx];
        block.valid_tokens = valid_tokens;
        block.original_size = actual_elems * 4; // f32 = 4 bytes

        if self.config.use_delta && valid_tokens > 0 {
            // Delta encoding: almacenar base (primer token) + deltas
            let head_dim = self.head_dim;
            let n_kv_heads = self.n_kv_heads;
            let first_token_elems = n_kv_heads * head_dim;

            // Guardar base (primer token)
            block.base.clear();
            block.base.extend_from_slice(&data[..first_token_elems]);

            // P-01: use thread-local delta buffer instead of allocating per-call.
            KV_DELTA_BUF.with(|cell| {
                let mut delta_buf = cell.borrow_mut();
                delta_buf.clear();
                delta_buf.resize(actual_elems, 0.0f32);

                // Calcular deltas: delta[t] = data[t] - data[t-1]
                // Primer token: delta = 0 (base already stored)
                for t in 1..valid_tokens {
                    let prev_start = (t - 1) * first_token_elems;
                    let curr_start = t * first_token_elems;
                    for i in 0..first_token_elems {
                        delta_buf[curr_start + i] = data[curr_start + i] - data[prev_start + i];
                    }
                }

                // Comprimir deltas con LZ4
                // SAFETY: `delta_buf` was just resized to exactly
                // `actual_elems` f32; viewing its storage as `actual_elems * 4`
                // bytes needs only alignment 1 (satisfied), the range covers
                // exactly the initialized vector, and the borrow keeps it alive
                // and unaliased for the LZ4 call.
                let delta_bytes = unsafe {
                    std::slice::from_raw_parts(
                        delta_buf.as_ptr() as *const u8,
                        actual_elems * 4,
                    )
                };
                block.compressed = lz4_flex::compress(delta_bytes);
            });
        } else {
            // Sin delta: comprimir directamente con LZ4
            // SAFETY: `compress_block`'s documented caller contract (doc above)
            // requires `data` to contain `valid_tokens * n_kv_heads * head_dim
            // == actual_elems` f32, so the `actual_elems * 4` byte range is
            // within the initialized slice; alignment 1 for u8 is satisfied and
            // the shared borrow outlives the compression.
            let data_bytes = unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr() as *const u8,
                    actual_elems * 4,
                )
            };
            block.compressed = lz4_flex::compress(data_bytes);
        }
    }

    /// Descomprime un bloque de KV (LZ4 + delta reconstruction).
    ///
    /// Escribe `valid_tokens * n_kv_heads * head_dim` elementos f32 en `out`.
    ///
    /// BUG-1 FIX (2026-08-03): valida `decompressed.len()` y `block.base.len()`
    /// ANTES de cualquier slice. Con datos corruptos, devuelve 0 y loguea (P-17
    /// fail-fast/skip+log), nunca panic. Usa `decompress_into` con thread-local
    /// buffer (P-01: cero alloc en hot path KV).
    pub fn decompress_block(&self, block_idx: usize, out: &mut [f32]) -> usize {
        if !self.config.enabled || block_idx >= self.blocks.len() {
            return 0;
        }

        let block = &self.blocks[block_idx];
        let valid_tokens = block.valid_tokens;
        let head_dim = self.head_dim;
        let n_kv_heads = self.n_kv_heads;
        let first_token_elems = n_kv_heads * head_dim;
        let actual_elems = valid_tokens * first_token_elems;

        if out.len() < actual_elems {
            return 0;
        }
        // P-17: guard against zero-sized / overflow multiplication
        if actual_elems == 0 {
            return 0;
        }

        let expected_bytes = actual_elems * 4; // f32 = 4 bytes

        // BUG-1 FIX: decompress into thread-local byte buffer (P-01, no alloc).
        // Validate the byte count BEFORE any slice reinterpretation.
        let decompressed_len = KV_DECOMPRESS_BUF.with(|buf_cell| {
            let mut buf = buf_cell.borrow_mut();
            buf.clear();
            buf.resize(expected_bytes, 0u8);
            match lz4_flex::decompress_into(&block.compressed, &mut buf) {
                Ok(n) => n,
                Err(e) => {
                    log::warn!(
                        "[kv_compressed] decompress_block {block_idx}: LZ4 error ({e}); skipping"
                    );
                    0 // P-17: skip+log, no panic
                }
            }
        });

        // Validate decompressed size matches expectation (corrupt data, version mismatch)
        if decompressed_len != expected_bytes {
            log::warn!(
                "[kv_compressed] decompress_block {block_idx}: size mismatch (got {decompressed_len} bytes, expected {expected_bytes}); skipping"
            );
            return 0;
        }

        // BUG-2-style alignment-safe reinterpret: copy bytes into TLS f32 buffer,
        // then do ALL reconstruction within the closure scope (slice cannot escape
        // the RefCell borrow). Mirrors the miri fix (2026-07-28) for CellV5/LZ4.
        // The nested `KV_DECOMPRESS_BUF.with` reads the persisted byte buffer
        // (same thread_local, data survives across sequential `with` calls).
        KV_DECOMPRESS_F32_BUF.with(|fbuf_cell| {
            let mut fbuf = fbuf_cell.borrow_mut();
            fbuf.clear();
            fbuf.resize(actual_elems, 0.0);

            KV_DECOMPRESS_BUF.with(|bb| {
                let bytes = bb.borrow();
                // SAFETY: expected_bytes <= buf.len() (validated == expected_bytes above);
                // fbuf has actual_elems f32 = expected_bytes bytes. No overlap (different allocs).
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        fbuf.as_mut_ptr() as *mut u8,
                        expected_bytes,
                    );
                }
            });

            if self.config.use_delta && valid_tokens > 0 {
                // Delta reconstruction: data[t] = base + sum(delta[1..t])
                // Primer token = base
                if block.base.len() < first_token_elems {
                    log::warn!(
                        "[kv_compressed] decompress_block {block_idx}: base too short ({}); skipping",
                        block.base.len()
                    );
                    return 0; // P-17: skip+log
                }
                out[..first_token_elems].copy_from_slice(&block.base[..first_token_elems]);

                // Reconstruir tokens t >= 1
                for t in 1..valid_tokens {
                    let prev_start = (t - 1) * first_token_elems;
                    let curr_start = t * first_token_elems;
                    for i in 0..first_token_elems {
                        out[curr_start + i] = out[prev_start + i] + fbuf[curr_start + i];
                    }
                }
            } else {
                // Sin delta: copiar directamente
                out[..actual_elems].copy_from_slice(&fbuf);
            }

            valid_tokens
        })
    }

    /// Ratio de compresión del bloque (0.0 = sin comprimir, 1.0 = 100% comprimido).
    pub fn compression_ratio(&self, block_idx: usize) -> f32 {
        if block_idx >= self.blocks.len() {
            return 0.0;
        }
        let block = &self.blocks[block_idx];
        if block.original_size == 0 {
            return 0.0;
        }
        1.0 - (block.compressed.len() as f32 / block.original_size as f32)
    }

    /// Tamaño total comprimido (bytes).
    pub fn total_compressed_bytes(&self) -> usize {
        self.blocks.iter().map(|b| b.compressed.len()).sum()
    }

    /// Tamaño total original (bytes).
    pub fn total_original_bytes(&self) -> usize {
        self.blocks.iter().map(|b| b.original_size).sum()
    }

    /// Ratio de compresión global.
    pub fn global_ratio(&self) -> f32 {
        let total_orig = self.total_original_bytes();
        if total_orig == 0 {
            return 0.0;
        }
        1.0 - (self.total_compressed_bytes() as f32 / total_orig as f32)
    }

    /// Limpia todos los bloques.
    pub fn clear(&mut self) {
        self.blocks.clear();
    }

    /// Número de bloques almacenados.
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Verdadero si no hay bloques.
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }
}

/// Verdadero si la compresión KV está habilitada (cached env, P-01).
pub fn kv_compress_enabled() -> bool {
    std::env::var("MUD_KV_COMPRESS").as_deref() == Ok("1")
}

/// Reset cache de env vars (para tests).
pub fn reset_kv_compress_cache() {
    // No hay cache thread_local en esta versión; los env vars se leen directamente.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_compress_config_resolve() {
        // Default: disabled
        std::env::remove_var("MUD_KV_COMPRESS");
        let cfg = KvCompressConfig::resolve();
        assert!(!cfg.enabled);
    }

    #[test]
    fn test_kv_compress_roundtrip_no_delta() {
        std::env::set_var("MUD_KV_COMPRESS", "1");
        std::env::set_var("MUD_KV_COMPRESS_DELTA", "0");
        let cfg = KvCompressConfig::resolve();
        assert!(cfg.enabled);
        assert!(!cfg.use_delta);

        let head_dim = 64;
        let n_kv_heads = 4;
        let _block_size = 16;
        let mut cache = CompressedKvCache::new(cfg, head_dim, n_kv_heads);

        // Generar datos de prueba (KV values)
        let valid_tokens = 16;
        let total_elems = valid_tokens * n_kv_heads * head_dim;
        let data: Vec<f32> = (0..total_elems).map(|i| (i as f32) * 0.001).collect();

        cache.compress_block(0, &data, valid_tokens);

        // Descomprimir
        let mut out = vec![0.0f32; total_elems];
        let recovered = cache.decompress_block(0, &mut out);
        assert_eq!(recovered, valid_tokens);

        // Verificar exactitud (lossless)
        for (i, (d, o)) in data.iter().zip(out.iter()).enumerate() {
            assert!(
                (d - o).abs() < 1e-5,
                "Mismatch at {}: expected {}, got {}",
                i, d, o
            );
        }
    }

    #[test]
    fn test_kv_compress_roundtrip_with_delta() {
        std::env::set_var("MUD_KV_COMPRESS", "1");
        std::env::set_var("MUD_KV_COMPRESS_DELTA", "1");
        let cfg = KvCompressConfig::resolve();
        assert!(cfg.enabled);
        assert!(cfg.use_delta);

        let head_dim = 32;
        let n_kv_heads = 2;
        let _block_size = 8;
        let mut cache = CompressedKvCache::new(cfg, head_dim, n_kv_heads);

        // Generar datos con alta correlación entre tokens (realista para KV cache)
        let valid_tokens = 8;
        let total_elems = valid_tokens * n_kv_heads * head_dim;
        let mut data = vec![0.0f32; total_elems];
        // Primer token: valores base
        let head_elems = n_kv_heads * head_dim;
        for (i, d) in data[..head_elems].iter_mut().enumerate() {
            *d = (i as f32) * 0.1;
        }
        // Tokens siguientes: pequeños deltas
        for t in 1..valid_tokens {
            let prev_start = (t - 1) * head_elems;
            let curr_start = t * head_elems;
            let (before, after) = data.split_at_mut(curr_start);
            let prev = &before[prev_start..prev_start + head_elems];
            let curr = &mut after[..head_elems];
            for (c, p) in curr.iter_mut().zip(prev.iter()) {
                *c = *p + 0.001;
            }
        }

        cache.compress_block(0, &data, valid_tokens);

        // Descomprimir
        let mut out = vec![0.0f32; total_elems];
        let recovered = cache.decompress_block(0, &mut out);
        assert_eq!(recovered, valid_tokens);

        // Verificar exactitud (lossless)
        for (i, (d, o)) in data.iter().zip(out.iter()).enumerate() {
            assert!(
                (d - o).abs() < 1e-4,
                "Delta roundtrip mismatch at {}: expected {}, got {}",
                i, d, o
            );
        }
    }

    #[test]
    fn test_kv_compress_ratio_with_delta() {
        std::env::set_var("MUD_KV_COMPRESS", "1");
        std::env::set_var("MUD_KV_COMPRESS_DELTA", "1");
        let cfg = KvCompressConfig::resolve();

        let head_dim = 64;
        let n_kv_heads = 4;
        let _block_size = 256;
        let mut cache = CompressedKvCache::new(cfg, head_dim, n_kv_heads);

        // Datos con alta correlación (realista)
        let valid_tokens = 256;
        let total_elems = valid_tokens * n_kv_heads * head_dim;
        let mut data = vec![0.0f32; total_elems];
        let head_elems = n_kv_heads * head_dim;
        for (i, d) in data[..head_elems].iter_mut().enumerate() {
            *d = (i as f32) * 0.01;
        }
        for t in 1..valid_tokens {
            let prev_start = (t - 1) * head_elems;
            let curr_start = t * head_elems;
            let (before, after) = data.split_at_mut(curr_start);
            let prev = &before[prev_start..prev_start + head_elems];
            let curr = &mut after[..head_elems];
            for (c, p) in curr.iter_mut().zip(prev.iter()) {
                *c = *p + 0.0001;
            }
        }

        cache.compress_block(0, &data, valid_tokens);

        let ratio = cache.compression_ratio(0);
        // Delta + LZ4 debería lograr al menos 50% de compresión para datos correlacionados
        assert!(ratio > 0.5, "Expected >50% compression, got {:.2}%", ratio * 100.0);
    }

    #[test]
    fn test_kv_compress_multiple_blocks() {
        std::env::set_var("MUD_KV_COMPRESS", "1");
        std::env::set_var("MUD_KV_COMPRESS_DELTA", "0");
        let cfg = KvCompressConfig::resolve();

        let head_dim = 16;
        let n_kv_heads = 2;
        let block_size = 4;
        let mut cache = CompressedKvCache::new(cfg, head_dim, n_kv_heads);

        // Comprimir 3 bloques
        for block_idx in 0..3 {
            let total_elems = block_size * n_kv_heads * head_dim;
            let data: Vec<f32> = (0..total_elems)
                .map(|i| (block_idx * 100 + i) as f32 * 0.01)
                .collect();
            cache.compress_block(block_idx, &data, block_size);
        }

        assert_eq!(cache.len(), 3);

        // Descomprimir cada bloque
        for block_idx in 0..3 {
            let total_elems = block_size * n_kv_heads * head_dim;
            let mut out = vec![0.0f32; total_elems];
            let recovered = cache.decompress_block(block_idx, &mut out);
            assert_eq!(recovered, block_size);
        }
    }

    #[test]
    fn test_kv_compress_disabled() {
        std::env::remove_var("MUD_KV_COMPRESS");
        let cfg = KvCompressConfig::resolve();
        assert!(!cfg.enabled);

        let mut cache = CompressedKvCache::new(cfg, 64, 4);
        let data = vec![1.0f32; 64 * 4];
        cache.compress_block(0, &data, 1);
        // Cuando está deshabilitado, no se comprime
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_kv_compress_block_size_config() {
        std::env::set_var("MUD_KV_COMPRESS", "1");
        std::env::set_var("MUD_KV_BLOCK_SIZE", "512");
        let cfg = KvCompressConfig::resolve();
        assert_eq!(cfg.block_size, 512);

        std::env::set_var("MUD_KV_BLOCK_SIZE", "32");
        let cfg = KvCompressConfig::resolve();
        assert_eq!(cfg.block_size, 32);

        std::env::remove_var("MUD_KV_BLOCK_SIZE");
        let cfg = KvCompressConfig::resolve();
        assert_eq!(cfg.block_size, 256); // default
    }

    #[test]
    fn test_kv_compress_global_ratio() {
        std::env::set_var("MUD_KV_COMPRESS", "1");
        std::env::set_var("MUD_KV_COMPRESS_DELTA", "1");
        let cfg = KvCompressConfig::resolve();

        let head_dim = 32;
        let n_kv_heads = 2;
        let block_size = 64;
        let mut cache = CompressedKvCache::new(cfg, head_dim, n_kv_heads);

        // Datos correlacionados
        let head_elems = n_kv_heads * head_dim;
        for block_idx in 0..2 {
            let total_elems = block_size * n_kv_heads * head_dim;
            let mut data = vec![0.0f32; total_elems];
            for (i, d) in data[..head_elems].iter_mut().enumerate() {
                *d = (i as f32) * 0.01;
            }
            for t in 1..block_size {
                let prev_start = (t - 1) * head_elems;
                let curr_start = t * head_elems;
                let (before, after) = data.split_at_mut(curr_start);
                let prev = &before[prev_start..prev_start + head_elems];
                let curr = &mut after[..head_elems];
                for (c, p) in curr.iter_mut().zip(prev.iter()) {
                    *c = *p + 0.0001;
                }
            }
            cache.compress_block(block_idx, &data, block_size);
        }

        let ratio = cache.global_ratio();
        assert!(ratio > 0.5, "Expected >50% global compression, got {:.2}%", ratio * 100.0);
    }
}
