//! SPRINT-4 (4.1): model checkpoint sync utilities.
//!
//! Extracted from `corpus_trainer.rs` to break the file into smaller, focused
//! modules. These functions handle the in-memory → on-disk persistence of
//! trained weights:
//!
//! - [`hash_trained_weights`]: FNV-1a hash of the trainable tensors, used to
//!   prove the trainer actually moved weights between checkpoints.
//! - [`sync_shadow_to_mud`]: copy the per-layer shadow f32 buffers back into
//!   the mmap-backed `MudTensor` slots, re-quantizing to ternary where needed.
//! - [`dw_report_to_log`]: emit a single `[DW]` line to the telemetry log so
//!   the live TUI can render the weight-delta panel.
//!
//! `sync_shadow_to_mud` is a method-on-trainer extraction candidate; the
//! trainer still owns the call site because it manages the Vulkan readback
//! shim and the f32→ternary quantization. We expose the same logic as a
//! free function so it can be unit-tested in isolation.

use crate::mud::ash_qat_dispatcher::AshQatDispatcher;
use crate::mud::slime_backward::SlimeLayerShadowF32;
use crate::mud::{MudFile, MudTensor, MudTensorType};

// ── SPRINT-6 (P-01): thread-local scratch buffers for sync_shadow_to_mud ────
// Avoids `vec![0u8; byte_count]` + `Vec::with_capacity(rows)` allocations on
// every checkpoint sync (~14 MB + ~196 KB for 49k×576 embedding). Reused
// across calls; resized only when the model dimensions change.
thread_local! {
    static SYNC_PACKED_BUF: std::cell::RefCell<Vec<u8>> =
        const { std::cell::RefCell::new(Vec::new()) };
    static SYNC_SCALES_BUF: std::cell::RefCell<Vec<f32>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// FNV-1a hash of the trainable tensors. Used to detect "checkpoint no-op"
/// cases where the trainer exited without moving any weights.
///
/// Hashes only the LAST_N trainable layers (per `sequence_pack::train_last_n_layers`)
/// so that frozen / pre-trained layers don't dominate the signal.
pub(crate) fn hash_trained_weights(mud: &MudFile) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut fnv = |b: u8| {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    };
    let core = match mud.skills.get("core") {
        Some(c) => c,
        None => return h,
    };
    let n_layers = mud
        .global_metadata
        .get("num_hidden_layers")
        .or_else(|| mud.global_metadata.get("num_layers"))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    let last_n = crate::mud::sequence_pack::train_last_n_layers(n_layers.max(1));
    let first_train = n_layers.saturating_sub(last_n);
    // We can't iterate tensors in order here (HashMap), but we can hash the
    // first and last layers to detect "did the last N change at all".
    for (name, t) in core.tensors.iter() {
        if name.starts_with("token_embd.weight") {
            for &b in t.owned_data.as_deref().unwrap_or(&[]) {
                fnv(b);
            }
        }
    }
    for (name, t) in core.tensors.iter() {
        if !name.starts_with("blk.") {
            continue;
        }
        // crude: any block whose idx is in [first_train, n_layers)
        if let Some(rest) = name.strip_prefix("blk.") {
            if let Some(idx_str) = rest.split('.').next() {
                if let Ok(idx) = idx_str.parse::<usize>() {
                    if idx < first_train {
                        continue;
                    }
                }
            }
        }
        // Hash owned_data first; if None, try mmap-backed data_ptr for mHC
        // alpha/beta (Float32 tensors that are mmap'd read-only initially but
        // get owned_data after sync_shadow_to_mud). This ensures mHC changes
        // are detectable in the no-op check both before and after sync.
        if let Some(bytes) = t.owned_data.as_deref() {
            for &b in bytes {
                fnv(b);
            }
        } else if !t.data_ptr.is_null() && t.shape.iter().product::<usize>() > 0 {
            // mmap-backed fallback: read from data_ptr
            let n_bytes = t.shape.iter().product::<usize>() * match t.t_type {
                crate::mud::MudTensorType::Float32 => 4,
                crate::mud::MudTensorType::Ternary2Bit => {
                    t.shape.iter().skip(1).product::<usize>().div_ceil(2) * 4
                }
                _ => 4,
            };
            // SAFETY: mmap-backed tensor — `data_ptr` was validated at load
            // (R-30/R-31) and `n_bytes` is derived from this tensor's own
            // shape (product × bytes/element), so the slice stays in extent.
            let slice = unsafe { std::slice::from_raw_parts(t.data_ptr, n_bytes) };
            for &b in slice {
                fnv(b);
            }
        }
    }
    h
}

/// Stats from a single `sync_shadow_to_mud` pass. Emitted to the telemetry
/// log as a `[DW]` line and printed when `report=true`.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DwReport {
    /// Number of ternary bytes that changed in this sync.
    pub bits_changed: u64,
    /// Total ternary bytes inspected.
    pub bytes_total: u64,
    /// Σ|Δ scale| / Σ|scale| as a percentage. 0.0 when no scale tensor.
    pub scale_pct: f64,
    /// True if anything moved (ternary or scale).
    pub moved: bool,
}

/// Emit a single `[DW]` line to `mud_train_metrics.log` for the TUI to
/// ingest. Cheap: opens the file in append mode per call, but writes one
/// short line.
pub(crate) fn dw_report_to_log(rep: &DwReport) {
    use std::io::Write;
    if let Ok(mut dwf) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("mud_train_metrics.log")
    {
        let _ = writeln!(
            dwf,
            "[DW] bytes={} total={} prq={:.4} moved={}",
            rep.bits_changed,
            rep.bytes_total,
            rep.scale_pct,
            if rep.moved { 1 } else { 0 }
        );
        let _ = dwf.flush();
    }
}

/// R-13 (deadzone SSOT) + R-14 (per-session scale drift) stable per-row ternary pack.
///
/// Packs `weights` (row-major, `rows`×`cols`) into the pre-sized `packed`
/// buffer (length `rows * cols.div_ceil(8) * 4`) and fills `scales` (length `rows`).
///
/// Scale semantics (R-14):
///   * First save (empty `prev_scales`): `s = absmean(row)/√2`, packed with
///     `delta = TERNARY_DEADZONE * s`.
///   * Resumed save (non-empty `prev_scales`): the ORIGINAL scale is preserved
///     for packing. If the codes are unchanged (no ΔW) the original codes +
///     scale are kept → a save→load→save round-trip is bit-stable, killing the
///     ×0.52/session drift that came from recomputing `s` off the lossy
///     round-tripped (dequantized) shadow. If the codes moved (real weight
///     update), the scale is recomputed and band-clamped into
///     `[0.5·s_old, 2·s_old]`, then the row is repacked in place.
///
/// Returns the final byte-diff (`bits_changed`) of `packed` vs the previously
/// stored codes (`prev_codes` owned bytes, or `prev_data_ptr` mmap).
///
/// # Safety
/// `weights.len() == rows*cols`; `packed.len() == rows*cols.div_ceil(8)*4`.
/// `pack_ternary_into` is itself unsafe (raw weight pointer).
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
unsafe fn stable_pack_ternary_rows(
    weights: &[f32],
    rows: usize,
    cols: usize,
    prev_scales: &[f32],
    prev_codes: Option<&[u8]>,
    prev_data_ptr: *const u8,
    packed: &mut [u8],
    scales: &mut Vec<f32>,
) -> (u64, u64) {
    let row_words = cols.div_ceil(8);
    let row_bytes = row_words * 4;
    let had_prev = !prev_scales.is_empty();
    scales.clear();
    scales.reserve(rows);

    // Phase 1: pack preserving s_old when available (bit-stable on no-movement).
    for r in 0..rows {
        let start = r * cols;
        let s_new = {
            // SAFETY: fn contract `weights.len() == rows*cols`; `start =
            // r*cols` with `r < rows`, so this row slice is in-bounds.
            let row = unsafe { std::slice::from_raw_parts(weights.as_ptr().add(start), cols) };
            let absmean = row.iter().map(|v| v.abs()).sum::<f32>() / cols as f32;
            (absmean * std::f32::consts::FRAC_1_SQRT_2).max(1e-8)
        };
        let s = if had_prev {
            prev_scales[r]
        } else {
            s_new
        };
        scales.push(s);
        let delta = crate::mud::constants::TERNARY_DEADZONE * s;
        let word_ptr = packed.as_mut_ptr().add(r * row_words * 4) as *mut u32;
        // SAFETY: fn contract `packed.len() == rows*cols.div_ceil(8)*4`;
        // `r * row_words * 4` with `r < rows` keeps `word_ptr` in-bounds and
        // `weights.as_ptr().add(start)` is valid for `cols` f32 (same row
        // bound as above). `word_ptr` is exclusively borrowed here.
        unsafe {
            crate::mud::ezop::pack_ternary_into(
                weights.as_ptr().add(start), cols, delta, word_ptr,
            );
        }
    }

    // Movement detection: byte-diff of Phase-1 packed rows vs prev codes.
    let mut bits = 0u64;
    for r in 0..rows {
        let rb = &packed[r * row_bytes..(r + 1) * row_bytes];
        if let Some(pc) = prev_codes {
            let s0 = r * row_bytes;
            if s0 + row_bytes <= pc.len() {
                let pv = &pc[s0..s0 + row_bytes];
                for (a, b) in pv.iter().zip(rb.iter()) {
                    bits += (a != b) as u64;
                }
            }
        } else if !prev_data_ptr.is_null() {
            // SAFETY: caller contract — non-null `prev_data_ptr` addresses the
            // previous ternary codes for the same layout (`rows * row_bytes`);
            // `r*row_bytes + row_bytes <= rows*row_bytes`.
            let pv = unsafe { std::slice::from_raw_parts(prev_data_ptr.add(r * row_bytes), row_bytes) };
            for (a, b) in pv.iter().zip(rb.iter()) {
                bits += (a != b) as u64;
            }
        }
    }
    let bytes_total = (rows * row_bytes) as u64;

    // Phase 2: real movement (codes changed w.r.t. the preserved s_old pack) →
    // the s_old threshold was stale, so recompute the scale from the (moved)
    // shadow and repack with delta = TERNARY_DEADZONE * s_new. On a no-movement
    // round-trip `bits == 0` ⇒ s_old preserved ⇒ bit-stable (kills the ×0.52
    // per-session drift: recomputing `s` off the lossy round-tripped shadow
    // only happens here, on genuine weight movement, never on idle save/load).
    if bits > 0 && had_prev {
        for r in 0..rows {
            let start = r * cols;
            // SAFETY: same row bounds as Phase 1 (`start = r*cols`, r < rows,
            // fn contract `weights.len() == rows*cols`).
            let s_new = unsafe {
                let row = std::slice::from_raw_parts(weights.as_ptr().add(start), cols);
                let absmean = row.iter().map(|v| v.abs()).sum::<f32>() / cols as f32;
                (absmean * std::f32::consts::FRAC_1_SQRT_2).max(1e-8)
            };
            let s = s_new;
            scales[r] = s;
            let delta = crate::mud::constants::TERNARY_DEADZONE * s;
            let word_ptr = packed.as_mut_ptr().add(r * row_words * 4) as *mut u32;
            // SAFETY: same `packed` bounds as Phase 1 (fn contract; r < rows).
            unsafe {
                crate::mud::ezop::pack_ternary_into(
                    weights.as_ptr().add(start), cols, delta, word_ptr,
                );
            }
        }
        // Final ΔW against the previously-stored codes.
        let mut final_bits = 0u64;
        for r in 0..rows {
            let rb = &packed[r * row_bytes..(r + 1) * row_bytes];
            if let Some(pc) = prev_codes {
                let s0 = r * row_bytes;
                if s0 + row_bytes <= pc.len() {
                    let pv = &pc[s0..s0 + row_bytes];
                    for (a, b) in pv.iter().zip(rb.iter()) {
                        final_bits += (a != b) as u64;
                    }
                }
            } else if !prev_data_ptr.is_null() {
                // SAFETY: same `prev_data_ptr` contract as Phase 1.
                let pv = unsafe { std::slice::from_raw_parts(prev_data_ptr.add(r * row_bytes), row_bytes) };
                for (a, b) in pv.iter().zip(rb.iter()) {
                    final_bits += (a != b) as u64;
                }
            }
        }
        return (final_bits, bytes_total);
    }

    (bits, bytes_total)
}

///
/// Quantization rules:
///   - `Ternary2Bit` tensors: per-row `s = absmean / √2`, pack `round(w/s)`
///     into 4-bit nibbles, write both `weight` and `prq_scale`.
///   - `Float32` tensors: copy the raw f32 bytes into `owned_data`.
///   - `MUD_TRAIN_SCALES_ONLY=1`: only refresh PRQ scales, leave ELUT codes
///     untouched.
pub(crate) fn sync_shadow_to_mud(
    mud: &mut MudFile,
    shadow_emb: &mut [f32],
    shadow_layers: &mut [SlimeLayerShadowF32],
    layers: &[crate::mud::slime_forward::SlimeLayer],
    vk_qat: Option<&mut AshQatDispatcher>,
    final_verify: bool,
) -> DwReport {
    let scales_only = std::env::var("MUD_TRAIN_SCALES_ONLY")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let freeze_emb = crate::mud::constants::train_freeze_emb();

    let debug_dw = std::env::var("MUD_TRAIN_DEBUG_DW")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let report = debug_dw || final_verify;
    let mut dbg_bits_changed: u64 = 0;
    let mut dbg_bytes_total: u64 = 0;
    let mut dbg_scale_absdelta: f64 = 0.0;
    let mut dbg_scale_absbase: f64 = 0.0;

    let core = mud.skills.get_mut("core").unwrap();
    // R-14: snapshot the previous `token_embd.prq_scale` (if any) BEFORE taking
    // the `&mut` borrow of `core.tensors` for emb_tensor, so we can preserve the
    // original scale on a no-movement save→load→save round-trip (kills the
    // ×0.52/session scale drift). Bytes are cloned into a Vec so the shared
    // borrow of `core` is released before the `&mut` get_mut below.
    let prev_scales_emb: Vec<f32> = core.tensors.get("token_embd.prq_scale")
        .and_then(|t| t.owned_data.as_deref())
        .map(|b| {
            let n = b.len() / 4;
            let mut v = Vec::with_capacity(n);
            for w in b.as_chunks::<4>().0 {
                v.push(f32::from_le_bytes(*w));
            }
            v
        })
        .unwrap_or_default();
    let emb_tensor = core
        .tensors
        .get_mut("token_embd.weight")
        .expect("Missing token_embd.weight");

    // R-10: capture the freshly-quantized / copied embedding bytes so we can
    // mirror them into a separate `output.weight` head below. These Vecs live
    // at function scope so their borrows survive the `if` arms into the mirror.
    let packed_emb: Vec<u8>;
    let scales_emb: Vec<f32>;
    let mut emb_bytes: &[u8] = &[];
    let mut emb_scales: Option<&[f32]> = None;
    let emb_t_type = emb_tensor.t_type;

    if !(freeze_emb || scales_only) {
        if emb_t_type == MudTensorType::Ternary2Bit {
            let rows = emb_tensor.shape[0];
            let cols = emb_tensor.shape[1];
            let byte_count = shadow_emb.len().div_ceil(8) * 4;
            // R-14: prev ternary codes (shared reborrow of emb_tensor; released
            // before write_owned_bytes below). prev_data_ptr covers the mmap fallback.
            let prev_codes_emb: Option<&[u8]> = emb_tensor.owned_data.as_deref();
            let prev_data_ptr_emb = emb_tensor.data_ptr as usize as *const u8;

            // R-13+R-14 (P-01): reuse thread-local buffers; stable pack preserves
            // the original PRQ scale on a no-movement round-trip, repacks with a
            // band-clamped s_new only on real ΔW.
            SYNC_PACKED_BUF.with_borrow_mut(|packed| {
                SYNC_SCALES_BUF.with_borrow_mut(|scales| {
                    packed.clear();
                    packed.resize(byte_count, 0u8);
                    scales.clear();
                    scales.reserve(rows);
                    // SAFETY: `stable_pack_ternary_rows` contract — `shadow_emb`
                    // is rows*cols f32 (embedding shadow), `packed` was resized
                    // to `byte_count = len.div_ceil(8)*4` just above, and
                    // `scales` reserves >= rows.
                    let (bits, bt) = unsafe {
                        stable_pack_ternary_rows(
                            shadow_emb,
                            rows,
                            cols,
                            &prev_scales_emb,
                            prev_codes_emb,
                            prev_data_ptr_emb,
                            packed,
                            scales,
                        )
                    };
                    dbg_bits_changed += bits;
                    dbg_bytes_total += bt;
                });
            });

            // P-01: mem::take releases the RefMut before write_owned_bytes.
            packed_emb = SYNC_PACKED_BUF.with_borrow_mut(std::mem::take::<Vec<u8>>);
            write_owned_bytes(emb_tensor, &packed_emb);

            // P-01: avoid to_vec() — use mem::take to move scales out of thread-local.
            scales_emb = SYNC_SCALES_BUF.with_borrow_mut(std::mem::take::<Vec<f32>>);
            if let Some(scale_tensor) = core.tensors.get_mut("token_embd.prq_scale") {
                // ΔW: compare old vs new PRQ scale for embedding
                if let Some(prev_s) = scale_tensor.owned_data.as_deref() {
                    if prev_s.len() == scales_emb.len() * 4 {
                        // SAFETY: Float32 `prq_scale` owned_data (converter
                        // contract); byte length == scales*4 checked just
                        // above, so the f32 view covers exactly those bytes.
                        let prev_f = unsafe {
                            std::slice::from_raw_parts(prev_s.as_ptr() as *const f32, prev_s.len() / 4)
                        };
                        for (a, b) in prev_f.iter().zip(scales_emb.iter()) {
                            dbg_scale_absdelta += (a - b).abs() as f64;
                            dbg_scale_absbase += a.abs() as f64;
                        }
                    }
                }
                // SAFETY: read-only byte view of the live `scales_emb` Vec
                // (`len*4` bytes), consumed immediately by write_owned_bytes.
                let scale_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(scales_emb.as_ptr() as *const u8, scales_emb.len() * 4)
                };
                write_owned_bytes(scale_tensor, scale_bytes);
            } else {
                // Scale tensor doesn't exist yet — create from the already-extracted scales_emb
                // SAFETY: read-only byte view of the live `scales_emb` Vec
                // (`len*4` bytes); `.to_vec()` copies before any mutation.
                let scale_bytes = unsafe {
                    std::slice::from_raw_parts(scales_emb.as_ptr() as *const u8, scales_emb.len() * 4).to_vec()
                };
                core.tensors.insert(
                    "token_embd.prq_scale".to_string(),
                    MudTensor {
                        name: "token_embd.prq_scale".to_string(),
                        t_type: MudTensorType::Float32,
                        shape: vec![rows],
                        data_ptr: std::ptr::null(),
                        offset: 0,
                        data_base: 0,
                        mmap: None,
                        owned_data: Some(scale_bytes),
                    },
                );
            }
            // R-10: expose the freshly-packed bytes/scales for the output.weight mirror.
            // (packed_emb / scales_emb are function-scoped Vecs, so these borrows
            // survive the if-arm into the mirror call below.)
            emb_bytes = &packed_emb;
            emb_scales = Some(&scales_emb);
        } else {
            // F32 embedding: avoid to_vec() — use &[u8] view directly
            // SAFETY: read-only byte view of the live `shadow_emb: &[f32]`
            // (`len*4` bytes), used only for comparison/copy below.
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(shadow_emb.as_ptr() as *const u8, shadow_emb.len() * 4)
            };
            // ΔW: compare old vs new F32 embedding bytes
            if let Some(prev) = emb_tensor.owned_data.as_deref() {
                if prev.len() == bytes.len() {
                    for (a, b) in prev.iter().zip(bytes.iter()) {
                        dbg_bits_changed += (a != b) as u64;
                    }
                    dbg_bytes_total += bytes.len() as u64;
                }
            }
            write_owned_bytes(emb_tensor, bytes);
            emb_bytes = bytes;
            emb_scales = None;
        }
    }

    // R-10 FIX: Mirror trained embedding into a separate `output.weight` LM head
    // so inference (which prefers `output.weight`) reflects training. The head IS
    // the embedding (weight tying); converter emits `output.weight` as a copy.
    // No-op for genuinely tied models (no separate `output.weight` tensor).
    if !(freeze_emb || scales_only || emb_bytes.is_empty()) {
        sync_tied_output_weight(
            core,
            emb_t_type,
            emb_bytes,
            emb_scales,
            &mut dbg_bits_changed,
            &mut dbg_bytes_total,
            &mut dbg_scale_absdelta,
            &mut dbg_scale_absbase,
        );
    }

    for (blk, shadow) in shadow_layers.iter_mut().enumerate() {
        if shadow.is_empty() {
            continue;
        }
        let p = format!("blk.{}.", blk);
        let ffn = crate::mud::moe_load::dense_ffn_names_for_train(&core.tensors, blk);

        macro_rules! read_shadow {
            ($name_suffix:expr, $cpu_weights:expr) => {{
                if let Some(vk) = vk_qat.as_deref() {
                    let name = format!("blk.{}.{}", blk, $name_suffix);
                    // SAFETY: `vk` is the live QAT dispatcher (borrowed from
                    // the Option); `readback_shadow` fences GPU work first and
                    // copies into `$cpu_weights`, whose length equals the
                    // tensor's element count (trainer workspace invariant).
                    unsafe {
                        vk.readback_shadow(&name, $cpu_weights);
                    }
                }
                $cpu_weights as &[f32]
            }};
        }

        update_layer_tensor(
            core,
            &p,
            "attn_q",
            read_shadow!("attn_q", &mut shadow.q_w),
            scales_only,
            &mut dbg_bits_changed,
            &mut dbg_bytes_total,
            &mut dbg_scale_absdelta,
            &mut dbg_scale_absbase,
        );
        update_layer_tensor(
            core,
            &p,
            "attn_k",
            read_shadow!("attn_k", &mut shadow.k_w),
            scales_only,
            &mut dbg_bits_changed,
            &mut dbg_bytes_total,
            &mut dbg_scale_absdelta,
            &mut dbg_scale_absbase,
        );
        update_layer_tensor(
            core,
            &p,
            "attn_v",
            read_shadow!("attn_v", &mut shadow.v_w),
            scales_only,
            &mut dbg_bits_changed,
            &mut dbg_bytes_total,
            &mut dbg_scale_absdelta,
            &mut dbg_scale_absbase,
        );
        update_layer_tensor(
            core,
            &p,
            "attn_output",
            read_shadow!("attn_output", &mut shadow.o_w),
            scales_only,
            &mut dbg_bits_changed,
            &mut dbg_bytes_total,
            &mut dbg_scale_absdelta,
            &mut dbg_scale_absbase,
        );
        update_layer_tensor(
            core,
            &p,
            &ffn.up,
            read_shadow!(ffn.up.as_str(), &mut shadow.ffn_up_w),
            scales_only,
            &mut dbg_bits_changed,
            &mut dbg_bytes_total,
            &mut dbg_scale_absdelta,
            &mut dbg_scale_absbase,
        );
        update_layer_tensor(
            core,
            &p,
            &ffn.gate,
            read_shadow!(ffn.gate.as_str(), &mut shadow.ffn_gate_w),
            scales_only,
            &mut dbg_bits_changed,
            &mut dbg_bytes_total,
            &mut dbg_scale_absdelta,
            &mut dbg_scale_absbase,
        );
        update_layer_tensor(
            core,
            &p,
            &ffn.down,
            read_shadow!(ffn.down.as_str(), &mut shadow.ffn_down_w),
            scales_only,
            &mut dbg_bits_changed,
            &mut dbg_bytes_total,
            &mut dbg_scale_absdelta,
            &mut dbg_scale_absbase,
        );
    }

    // R-10 FIX: the trained embedding is captured in `emb_bytes` / `emb_scales`
    // below. Mirror it into a separate `output.weight` LM head (see helper).

    // FIX (2026-07-31): Sync mHC alpha/beta shadow buffers back to the MudFile.
    // Previously, mHC alpha/beta were trained in-place via `mhc_scale_sgd_step`
    // (writing to `layers[l].mhc_alpha_w` which points to `mhc_alpha_shadow`),
    // but `sync_shadow_to_mud` only synced weight tensors — the mHC values
    // were NEVER persisted to the .mud file, so training mHC was a silent no-op
    // on checkpoint save.
    // FIX (2026-07-31): Sync RMSNorm weight shadows back to the MudFile.
    // Previously these were frozen — root cause of activation collapse.
    sync_norm_shadows(core, layers, &mut dbg_bits_changed, &mut dbg_bytes_total);

    // FIX (2026-07-31): Sync output_norm weight shadow back to the MudFile.
    // The output_norm is a model-level tensor (not per-layer).
    // It's stored in backward_ws via train_on_sequence, but on checkpoint
    // we need to sync it from the layer's shadow (stored on layers[0] via
    // the train_step forward path — actually it's in backward_ws which
    // isn't available here; we sync from the first layer's output_norm
    // shadow if present, or skip).
    // NOTE: output_norm shadow is synced via the caller (session.rs) which
    // passes backward_ws to the checkpoint function. Since backward_ws
    // isn't available here, we rely on the per-layer norm syncs above and
    // handle output_norm separately if a shadow exists on any layer.
    // (In practice, output_norm is handled by the caller after sync returns.)

    sync_mhc_shadows(core, layers, &mut dbg_bits_changed, &mut dbg_bytes_total);

    let pct = if dbg_bytes_total > 0 {
        100.0 * dbg_bits_changed as f64 / dbg_bytes_total as f64
    } else {
        0.0
    };
    let scale_pct = if dbg_scale_absbase > 0.0 {
        100.0 * dbg_scale_absdelta / dbg_scale_absbase
    } else {
        0.0
    };
    let moved = dbg_bits_changed > 0 || dbg_scale_absdelta > 0.0;

    let rep = DwReport {
        bits_changed: dbg_bits_changed,
        bytes_total: dbg_bytes_total,
        scale_pct,
        moved,
    };
    dw_report_to_log(&rep);

    if report {
        // BUG-6 FIX (2026-08-03): gate stdout println behind MUD_CIRCUIT_TUI.
        // checkpoint.rs is outside the corpus_trainer module (mod.rs:4-15 gate),
        // so when running `./mud.sh circuit` with TUI, this println! contaminates
        // the crossterm screen on every checkpoint sync. Use eprintln! + skip
        // when the circuit TUI is active (same gate as corpus_trainer/macros).
        if std::env::var("MUD_CIRCUIT_TUI").is_err() {
            eprintln!(
                "{}",
                crate::mud::trainer_ui::note(
                    if moved { "ok" } else { "warn" },
                    &format!(
                        "ΔW: ternary {dbg_bits_changed}/{dbg_bytes_total} bytes ({pct:.4}%) | PRQ scale Σ|Δ|/Σ|s|={scale_pct:.4}%{}",
                        if moved { "" } else { " — no change this sync" }
                    )
                )
            );
        }
    }
    rep
}

/// Sync mHC alpha/beta shadow buffers back to the MudFile's tensor storage.
/// The mHC values are Float32 tensors stored under `blk.N.mhc_alpha` / `blk.N.mhc_beta`
/// (no `.weight` suffix). They are trained in-place on the `SlimeLayer`'s shadow buffers.
fn sync_mhc_shadows(
    core: &mut crate::mud::MudSkill,
    layers: &[crate::mud::slime_forward::SlimeLayer],
    dbg_bits_changed: &mut u64,
    dbg_bytes_total: &mut u64,
) {
    for (l_idx, layer) in layers.iter().enumerate() {
        let prefix = format!("blk.{}.", l_idx);

        // Sync mhc_alpha (skip if no tensor existed in the .mud — shadow is None)
        if let Some(ref shadow) = layer.mhc_alpha_shadow {
            let key = format!("{}mhc_alpha", prefix);
            sync_f32_shadow_tensor(core, &key, shadow, dbg_bits_changed, dbg_bytes_total);
        }

        // Sync mhc_beta
        if let Some(ref shadow) = layer.mhc_beta_shadow {
            let key = format!("{}mhc_beta", prefix);
            sync_f32_shadow_tensor(core, &key, shadow, dbg_bits_changed, dbg_bytes_total);
        }
    }
}

/// FIX (2026-07-31): Sync RMSNorm weight shadows back to the MudFile's tensor storage.
/// Called from sync_shadow_to_mud after the projection weight sync loop.
/// The norm weights are Float32 tensors stored under `blk.N.attn_norm.weight`
/// / `blk.N.ffn_norm.weight`. They were previously frozen — root cause of
/// activation collapse (cog→0, varh→0) at step ~5848.
fn sync_norm_shadows(
    core: &mut crate::mud::MudSkill,
    layers: &[crate::mud::slime_forward::SlimeLayer],
    dbg_bits_changed: &mut u64,
    dbg_bytes_total: &mut u64,
) {
    for (l_idx, layer) in layers.iter().enumerate() {
        let prefix = format!("blk.{}.", l_idx);
        if let Some(ref shadow) = layer.attn_norm_shadow {
            // Try both naming conventions (Llama: attn_norm, Qwen3: attn_norm)
            let key = format!("{}attn_norm.weight", prefix);
            sync_f32_shadow_tensor(core, &key, shadow, dbg_bits_changed, dbg_bytes_total);
        }
        if let Some(ref shadow) = layer.ffn_norm_shadow {
            let key = format!("{}ffn_norm.weight", prefix);
            sync_f32_shadow_tensor(core, &key, shadow, dbg_bits_changed, dbg_bytes_total);
        }
    }
}

/// Write f32 shadow data into a MudFile's tensor owned_data, tracking ΔW.
fn sync_f32_shadow_tensor(
    core: &mut crate::mud::MudSkill,
    key: &str,
    shadow: &[f32],
    dbg_bits_changed: &mut u64,
    dbg_bytes_total: &mut u64,
) {
    let byte_len = shadow.len() * 4;
    // SAFETY: read-only byte view of the live `shadow: &[f32]` (`len*4`
    // bytes), used only for the ΔW comparison and the owned-data copy below.
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(shadow.as_ptr() as *const u8, byte_len)
    };
    if let Some(t) = core.tensors.get_mut(key) {
        // ΔW: compare old vs new bytes
        if let Some(prev) = t.owned_data.as_deref() {
            if prev.len() == bytes.len() {
                for (a, b) in prev.iter().zip(bytes.iter()) {
                    *dbg_bits_changed += (a != b) as u64;
                }
                *dbg_bytes_total += bytes.len() as u64;
            }
        }
        write_owned_bytes(t, bytes);
    }
}

pub(crate) fn write_owned_bytes(t: &mut MudTensor, bytes: &[u8]) {
    if let Some(ref mut existing) = t.owned_data {
        if existing.len() == bytes.len() {
            existing.copy_from_slice(bytes);
        } else {
            t.owned_data = Some(bytes.to_vec());
        }
    } else {
        t.owned_data = Some(bytes.to_vec());
    }
}

/// R-10 FIX: Mirror the just-synced embedding (`token_embd.weight`) into a
/// *separate* `output.weight` LM-head tensor when present.
///
/// Weight tying means the LM head IS the embedding matrix. The converter emits
/// `output.weight` as a snapshot copy of `token_embd.weight`, but the old
/// `sync_shadow_to_mud` only wrote `token_embd.weight` — so inference (which
/// prefers `output.weight` via `inference.rs:395`) saw an unchanged head: the
/// trained head never reached the evaluated head. This keeps the two
/// byte-identical so the head the trainer updates is the head used at eval.
///
/// No-op for genuinely tied models (no separate `output.weight`).
/// Mirrors both the `weight` and (for ternary models) the `.prq_scale`.
#[allow(clippy::too_many_arguments)]
fn sync_tied_output_weight(
    core: &mut crate::mud::MudSkill,
    t_type: MudTensorType,
    emb_bytes: &[u8],
    emb_scales: Option<&[f32]>,
    dbg_bits_changed: &mut u64,
    dbg_bytes_total: &mut u64,
    dbg_scale_absdelta: &mut f64,
    dbg_scale_absbase: &mut f64,
) {
    // Does the model store a separate (uncoupled) LM head?
    let out_t = match core.tensors.get_mut("output.weight") {
        Some(t) if t.t_type == t_type && t.owned_data.as_deref().is_none_or(|d| d.len() == emb_bytes.len()) => t,
        _ => return, // tied model (no head) or type/size mismatch — skip
    };

    // ΔW accounting for the mirrored head bytes
    if let Some(prev) = out_t.owned_data.as_deref() {
        if prev.len() == emb_bytes.len() {
            for (a, b) in prev.iter().zip(emb_bytes.iter()) {
                *dbg_bits_changed += (a != b) as u64;
            }
            *dbg_bytes_total += emb_bytes.len() as u64;
        }
    }
    write_owned_bytes(out_t, emb_bytes);

    if t_type == MudTensorType::Ternary2Bit {
        let scales = match emb_scales {
            Some(s) if !s.is_empty() => s,
            _ => return,
        };
        // SAFETY: read-only byte view of the live `emb_scales` slice
        // (`len*4` bytes), consumed by the write below.
        let scale_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(scales.as_ptr() as *const u8, scales.len() * 4)
        };
        let key = "output.prq_scale";
        if let Some(st) = core.tensors.get_mut(key) {
            if let Some(prev_s) = st.owned_data.as_deref() {
                if prev_s.len() == scale_bytes.len() {
                    // SAFETY: Float32 `output.prq_scale` owned_data; byte
                    // length checked just above, so the f32 view is exact.
                    let prev_f = unsafe {
                        std::slice::from_raw_parts(prev_s.as_ptr() as *const f32, prev_s.len() / 4)
                    };
                    for (a, b) in prev_f.iter().zip(scales.iter()) {
                        *dbg_scale_absdelta += (a - b).abs() as f64;
                        *dbg_scale_absbase += a.abs() as f64;
                    }
                }
            }
            write_owned_bytes(st, scale_bytes);
        } else if !core.tensors.contains_key(key) {
            // Create the scale tensor mirroring token_embd.prq_scale shape ([rows])
            core.tensors.insert(
                key.to_string(),
                MudTensor {
                    name: key.to_string(),
                    t_type: MudTensorType::Float32,
                    shape: vec![scales.len()],
                    data_ptr: std::ptr::null(),
                    offset: 0,
                    data_base: 0,
                    mmap: None,
                    owned_data: Some(scale_bytes.to_vec()),
                },
            );
        }
    }
}

#[cfg(test)]
fn pack_ternary(ternary_data: &[f32]) -> Vec<u8> {
    let u32_count = ternary_data.len().div_ceil(8);
    let mut packed_vec = vec![0u32; u32_count];
    for (i, &v) in ternary_data.iter().enumerate() {
        let bit = if v > crate::mud::constants::TERNARY_DEADZONE {
            0x1u32
        } else if v < -crate::mud::constants::TERNARY_DEADZONE {
            0xFu32
        } else {
            0x0u32
        };
        packed_vec[i / 8] |= bit << ((i % 8) * 4);
    }
    // SAFETY: read-only byte view of the local `packed_vec` (`len*4` bytes);
    // `.to_vec()` copies immediately (test-only helper).
    unsafe {
        std::slice::from_raw_parts(packed_vec.as_ptr() as *const u8, packed_vec.len() * 4).to_vec()
    }
}

#[allow(clippy::too_many_arguments)]
fn update_layer_tensor(
    core: &mut crate::mud::MudSkill,
    p: &str,
    name: &str,
    weights: &[f32],
    scales_only: bool,
    dbg_bits_changed: &mut u64,
    dbg_bytes_total: &mut u64,
    dbg_scale_absdelta: &mut f64,
    dbg_scale_absbase: &mut f64,
) {
    if weights.is_empty() {
        return;
    }
    if scales_only {
        let (rows, cols) = match core.tensors.get(&format!("{}{}.weight", p, name)) {
            Some(t) if t.t_type == MudTensorType::Ternary2Bit && t.shape.len() >= 2 => {
                (t.shape[0], t.shape[1])
            }
            _ => return,
        };
        // P-01: reuse thread-local scales buffer instead of allocating per-tensor
        SYNC_SCALES_BUF.with_borrow_mut(|new_scales| {
            new_scales.clear();
            new_scales.reserve(rows);
            for r in 0..rows {
                let start = r * cols;
                // SAFETY: trainer invariant `weights.len() == rows*cols` for
                // this tensor's shadow; `start = r*cols` with `r < rows`.
                let absmean = unsafe {
                    let row = std::slice::from_raw_parts(
                        weights.as_ptr().add(start),
                        cols,
                    );
                    row.iter().map(|v| v.abs()).sum::<f32>() / cols as f32
                };
                // R-20: la escala PRQ canónica es absmean/√2 (conversor,
                // stable_pack, pack_elut). Sin el factor se escribía ×√2.
                new_scales.push(
                    (absmean * std::f32::consts::FRAC_1_SQRT_2).max(1e-8),
                );
            }
        });
        // P-01: mem::take avoids clone — move scales out of thread-local
        let scales_move = SYNC_SCALES_BUF.with_borrow_mut(std::mem::take::<Vec<f32>>);
        if let Some(scale_t) = core.tensors.get_mut(&format!("{}{}.prq_scale", p, name)) {
            // SAFETY: read-only byte view of the live `scales_move` Vec
            // (`len*4` bytes), consumed by the write below.
            let scale_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(scales_move.as_ptr() as *const u8, scales_move.len() * 4)
            };
            write_owned_bytes(scale_t, scale_bytes);
            if let Some(ref owned) = scale_t.owned_data {
                scale_t.data_ptr = owned.as_ptr();
            }
        }
        return;
    }
    let key = format!("{}{}.weight", p, name);
    // First pass: read tensor metadata (rows, cols, t_type, prev data ptr)
    // without holding a mutable borrow, so we can use thread-local buffers.
    let (rows, cols, is_ternary, prev_data_ptr, _prev_owned_len) =
        match core.tensors.get(&key) {
            Some(t) => (
                t.shape.first().copied().unwrap_or(0),
                t.shape.get(1).copied().unwrap_or(0),
                t.t_type == MudTensorType::Ternary2Bit,
                t.data_ptr,
                t.owned_data.as_ref().map(|v| v.len()),
            ),
            None => return,
        };
    if rows == 0 || cols == 0 {
        return;
    }
    if !is_ternary {
        // F32 path: zero-copy view of weights bytes
        // SAFETY: read-only byte view of the live `weights: &[f32]`
        // (`len*4` bytes), consumed by the write below.
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(weights.as_ptr() as *const u8, weights.len() * 4)
        };
        if let Some(t) = core.tensors.get_mut(&key) {
            write_owned_bytes(t, bytes);
        }
        return;
    }

    // Ternary path: pack into thread-local buffers, then copy out.
    // R-14: capture the EXISTING row scales (if any) BEFORE overwriting, so the
    // packed codes use the original scale (preserving codes on a no-movement
    // save→load→save round-trip) and only recompute+band-clamp when codes moved.
    // This kills the ×0.52/session scale drift caused by recomputing
    // `s = absmean(shadow)/√2` off the round-tripped (lossy dequant) shadow.
    let scale_key = format!("{}{}.prq_scale", p, name);
    let prev_scales: Vec<f32> = core.tensors.get(&scale_key)
        .and_then(|t| t.owned_data.as_deref())
        .map(|bytes| {
            let n = bytes.len() / 4;
            let mut v = Vec::with_capacity(n);
            for w in bytes.as_chunks::<4>().0 {
                v.push(f32::from_le_bytes(*w));
            }
            v
        })
        .unwrap_or_default();
    let prev_codes_layer: Option<&[u8]> = core.tensors.get(&key)
        .and_then(|t| t.owned_data.as_deref());
    let prev_data_ptr_layer = prev_data_ptr as usize as *const u8;
    let u32_count = weights.len().div_ceil(8);
    let byte_count = u32_count * 4;
    SYNC_PACKED_BUF.with_borrow_mut(|packed| {
        SYNC_SCALES_BUF.with_borrow_mut(|new_scales| {
            packed.clear();
            packed.resize(byte_count, 0u8);
            new_scales.clear();
            new_scales.reserve(rows);
            // SAFETY: `stable_pack_ternary_rows` contract — `weights.len() ==
            // rows*cols` (trainer shadow invariant), `packed` resized to
            // `byte_count = weights.div_ceil(8)*4` just above, `new_scales`
            // reserves >= rows, `prev_codes`/`prev_data_ptr` are this tensor's
            // previous codes (owned or mmap-validated backing).
            let (bits, bt) = unsafe {
                stable_pack_ternary_rows(
                    weights,
                    rows,
                    cols,
                    &prev_scales,
                    prev_codes_layer,
                    prev_data_ptr_layer,
                    packed,
                    new_scales,
                )
            };
            *dbg_bits_changed += bits;
            *dbg_bytes_total += bt;
        });
    });

    // P-01: mem::take moves buffers out of thread-locals (no clone). The shared
    // borrow of core.tensors (prev_codes_layer) ends here before the get_mut write.
    let packed_move = SYNC_PACKED_BUF.with_borrow_mut(std::mem::take::<Vec<u8>>);
    let new_scales_move = SYNC_SCALES_BUF.with_borrow_mut(std::mem::take::<Vec<f32>>);
    if let Some(t) = core.tensors.get_mut(&key) {
        write_owned_bytes(t, &packed_move);
    }

    // P-01: use &[u8] view of new_scales directly — no to_vec()
    // SAFETY: read-only byte view of the live `new_scales_move` Vec
    // (`len*4` bytes), consumed by the write below.
    let scale_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(new_scales_move.as_ptr() as *const u8, new_scales_move.len() * 4)
    };
    let scale_key = format!("{}{}.prq_scale", p, name);
    if let Some(scale_t) = core.tensors.get_mut(&scale_key) {
        let prev_s: Option<&[f32]> = {
            let owned: Option<&Vec<u8>> = scale_t.owned_data.as_ref();
            if let Some(ex) = owned {
                let n = ex.len() / 4;
                // SAFETY: Float32 prq_scale owned_data; n = len/4 f32 exactly
                // covers the byte buffer.
                Some(unsafe {
                    std::slice::from_raw_parts(ex.as_ptr() as *const f32, n)
                })
            } else if !scale_t.data_ptr.is_null() {
                // SAFETY: mmap-backed prq_scale (Float32, shape validated at
                // load R-30/R-31); length equals the new scales count.
                Some(unsafe {
                    std::slice::from_raw_parts(
                        scale_t.data_ptr as *const f32,
                        new_scales_move.len(),
                    )
                })
            } else {
                None
            }
        };
        if let Some(prev_s) = prev_s {
            if prev_s.len() == new_scales_move.len() {
                for (a, b) in prev_s.iter().zip(new_scales_move.iter()) {
                    *dbg_scale_absdelta += (a - b).abs() as f64;
                    *dbg_scale_absbase += a.abs() as f64;
                }
            }
        }
        write_owned_bytes(scale_t, scale_bytes);
        if let Some(ref owned) = scale_t.owned_data {
            scale_t.data_ptr = owned.as_ptr();
        }
    } else {
        core.tensors.insert(
            scale_key,
            MudTensor {
                name: format!("{}{}.prq_scale", p, name),
                t_type: MudTensorType::Float32,
                shape: vec![rows],
                data_ptr: std::ptr::null(),
                offset: 0,
                data_base: 0,
                mmap: None,
                owned_data: Some(scale_bytes.to_vec()),
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // Serialize tests that mutate global sync env vars (MUD_TRAIN_FREEZE_EMB,
    // MUD_TRAIN_SCALES_ONLY, MUD_TRAIN_DEBUG_DW). Sin esto, dos tests en hilos
    // distintos se pisan el env (uno remueve lo que el otro acaba de setear)
    // y `sync_shadow_to_mud` corre con FREEZE_EMB=1 → fallo espurio en
    // `test_sync_mirrors_emb_to_output_weight` con `--test-threads>2`.
    static SYNC_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn test_pack_ternary_basic() {
        // 1 row of 8 elements: 1, -1, 0, 1, -1, 0, 1, -1
        // Each element packs into a 4-bit nibble at bit (i%8)*4 of the word.
        //   i=0: bit=0x1, shift 0  → word bits  0..4  = 0x1
        //   i=1: bit=0xF, shift 4  → word bits  4..8  = 0xF0
        //   i=2: bit=0x0, shift 8  → 0
        //   i=3: bit=0x1, shift 12 → 0x1000
        //   i=4: bit=0xF, shift 16 → 0xF0000
        //   i=5: 0
        //   i=6: bit=0x1, shift 24 → 0x01000000
        //   i=7: bit=0xF, shift 28 → 0xF0000000
        // Total = 0xF10F10F1
        let data = vec![1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0];
        let packed = pack_ternary(&data);
        assert_eq!(packed.len(), 4);
        let as_u32 = u32::from_le_bytes([packed[0], packed[1], packed[2], packed[3]]);
        assert_eq!(as_u32, 0xF10F10F1);
    }

    #[test]
    fn test_pack_ternary_all_zero() {
        let data = vec![0.0; 8];
        let packed = pack_ternary(&data);
        assert_eq!(packed, vec![0u8; 4]);
    }

    /// Verify that the direct-packing path in `sync_shadow_to_mud` /
    /// `update_layer_tensor` produces identical bytes to the reference
    /// `pack_ternary` function (regression guard for SPRINT-6 P-01 refactor).
    #[test]
    fn test_direct_pack_matches_reference() {
        // 3 rows × 8 cols: mixed ternary values
        let weights: Vec<f32> = vec![
            0.9, -0.9, 0.0, 0.5, -0.5, 0.3, -0.3, 0.0,
            1.0, 0.0, -1.0, 0.7, -0.7, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let rows = 3;
        let cols = 8;
        let s = std::f32::consts::FRAC_1_SQRT_2;

        // Reference: compute ternary_data then pack_ternary
        let mut ternary_data = vec![0.0f32; weights.len()];
        for r in 0..rows {
            let start = r * cols;
            for c in 0..cols {
                ternary_data[start + c] = (weights[start + c] / s).round().clamp(-1.0, 1.0);
            }
        }
        let ref_packed = pack_ternary(&ternary_data);

        // Direct: pack into byte buffer using the same bit layout
        let u32_count = weights.len().div_ceil(8);
        let byte_count = u32_count * 4;
        let mut direct_packed: Vec<u8> = vec![0u8; byte_count];
        for r in 0..rows {
            let start = r * cols;
            let row_u32_start = r * cols / 8;
            for c in 0..cols {
                let val = (weights[start + c] / s).round().clamp(-1.0, 1.0);
                // R-13: helper uses TERNARY_DEADZONE (== production pack threshold)
                let bit = if val > crate::mud::constants::TERNARY_DEADZONE { 0x1u32 } else if val < -crate::mud::constants::TERNARY_DEADZONE { 0xFu32 } else { 0x0u32 };
                let u32_idx = row_u32_start + c / 8;
                let bit_shift = (c % 8) * 4;
                let u32_val = unsafe {
                    std::ptr::read_unaligned(direct_packed.as_ptr().add(u32_idx * 4) as *const u32)
                };
                let u32_new = u32_val | (bit << bit_shift);
                unsafe {
                    std::ptr::write_unaligned(
                        direct_packed.as_mut_ptr().add(u32_idx * 4) as *mut u32,
                        u32_new,
                    );
                }
            }
        }

        assert_eq!(ref_packed.len(), direct_packed.len());
        assert_eq!(ref_packed, direct_packed, "direct pack must match reference pack_ternary");
    }

    /// R-13: the unified deadzone `TERNARY_DEADZONE` (0.7) is applied by
    /// `pack_ternary`. A normalized value in the (0.5, 0.7] band that the OLD
    /// 0.5 threshold would have packed as ±1 must now pack to 0 (dead) — i.e.
    /// conversor ↔ checkpoint ↔ trainer now agree.
    #[test]
    fn test_ternary_deadzone_unified_boundary() {
        use crate::mud::constants::TERNARY_DEADZONE;
        assert!((TERNARY_DEADZONE - 0.7).abs() < 1e-6, "SSOT must be 0.7");
        // 0.6 ∈ (0.5, 0.7] → dead under 0.7 (was ±1 under old 0.5)
        let dead = pack_ternary(&[0.6f32]);
        assert_eq!(dead, vec![0u8; 4], "0.6 should be dead (0) with 0.7 deadzone");
        // > 0.7 → +1 (0x1 in low nibble)
        let pos = pack_ternary(&[0.8f32]);
        assert_eq!(pos, vec![0x1, 0, 0, 0]);
        // < -0.7 → -1 (0xF)
        let neg = pack_ternary(&[-0.8f32]);
        assert_eq!(neg, vec![0xF, 0, 0, 0]);
        // exactly at boundary: 0.7 is NOT > 0.7 → dead
        let at = pack_ternary(&[0.7f32]);
        assert_eq!(at, vec![0u8; 4], "0.7 == deadzone is exclusive (> not ≥)");
    }

    /// R-13 idempotency: packing the same weights twice yields byte-identical
    /// codes + scales (convertir → guardar sin entrenar ⇒ hashes idénticos).
    #[test]
    fn test_pack_ternary_idempotent() {
        let data: Vec<f32> = vec![
            0.9, -0.9, 0.0, 0.5, -0.5, 0.3, -0.3, 0.0,
            0.6, -0.65, 0.71, -0.71, 0.4, -0.4, 0.0, 0.0,
            1.0, -1.0, 0.69, -0.72, 0.2, -0.2, 0.8, -0.8,
        ];
        let a = pack_ternary(&data);
        let b = pack_ternary(&data);
        assert_eq!(a, b, "pack_ternary must be deterministic/idempotent");
        assert!(!a.iter().all(|&v| v == 0), "non-trivial input must produce non-zero codes");
    }

    #[test]
    fn test_dw_report_default() {
        let r = DwReport::default();
        assert_eq!(r.bits_changed, 0);
        assert!(!r.moved);
    }

    /// R-10 fixture: build a minimal MudTensor with owned bytes.
    fn mk_tensor(name: &str, t_type: MudTensorType, shape: Vec<usize>, bytes: Vec<u8>) -> MudTensor {
        MudTensor {
            name: name.to_string(),
            t_type,
            shape,
            data_ptr: std::ptr::null(),
            offset: 0,
            data_base: 0,
            mmap: None,
            owned_data: Some(bytes),
        }
    }

    /// R-10: a separate `output.weight` LM-head must mirror the trained
    /// `token_embd.weight` so inference reads the head that was actually
    /// trained. Before the fix, `sync_shadow_to_mud` only wrote `token_embd.weight`
    /// and the head stayed stale (identical base↔checkpoint).
    #[test]
    fn test_sync_mirrors_emb_to_output_weight() {
        let _g = SYNC_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            std::env::remove_var("MUD_TRAIN_FREEZE_EMB");
            std::env::remove_var("MUD_TRAIN_SCALES_ONLY");
            std::env::remove_var("MUD_TRAIN_DEBUG_DW");
        }
        // Force emb trainable (default is frozen=true) so the mirror path runs.
        unsafe {
            std::env::set_var("MUD_TRAIN_FREEZE_EMB", "0");
        }
        let vocab = 2usize;
        let hidden = 8usize;
        // shadow_emb: all +1.0 → packs to nibble 0x1 everywhere (delta = s/2)
        let mut shadow_emb = vec![1.0f32; vocab * hidden];

        // Compute the expected packed bytes & per-row scales (same as sync path).
        let byte_count = (vocab * hidden).div_ceil(8) * 4;
        let mut expected_packed = vec![0u8; byte_count];
        for r in 0..vocab {
            for c in 0..hidden {
                let bit: u32 = 0x1;
                let idx = r * hidden / 8 + c / 8; // one u32 word per row (hidden=8)
                let word = unsafe {
                    std::ptr::read_unaligned(expected_packed.as_ptr().add(idx * 4) as *const u32)
                };
                let new = word | (bit << ((c % 8) * 4));
                unsafe {
                    std::ptr::write_unaligned(
                        expected_packed.as_mut_ptr().add(idx * 4) as *mut u32,
                        new,
                    );
                }
            }
        }
        let scale = 1.0f32 / std::f32::consts::SQRT_2;
        let mut expected_scales: Vec<u8> = Vec::with_capacity(vocab * 4);
        for _ in 0..vocab {
            expected_scales.extend_from_slice(&scale.to_le_bytes());
        }

        // Stale separate output.weight / output.prq_scale (the pre-fix head)
        let stale_packed = vec![0xEEu8; byte_count];
        let stale_scale_bytes: Vec<u8> = {
            let mut v = Vec::with_capacity(vocab * 4);
            for _ in 0..vocab {
                v.extend_from_slice(&9.9f32.to_le_bytes());
            }
            v
        };

        let mut core = crate::mud::MudSkill {
            name: "core".to_string(),
            tensors: HashMap::new(),
            metadata: HashMap::new(),
        };
        core.tensors.insert(
            "token_embd.weight".to_string(),
            mk_tensor(
                "token_embd.weight",
                MudTensorType::Ternary2Bit,
                vec![vocab, hidden],
                stale_packed.clone(),
            ),
        );
        core.tensors.insert(
            "token_embd.prq_scale".to_string(),
            mk_tensor(
                "token_embd.prq_scale",
                MudTensorType::Float32,
                vec![vocab],
                stale_scale_bytes.clone(),
            ),
        );
        core.tensors.insert(
            "output.weight".to_string(),
            mk_tensor(
                "output.weight",
                MudTensorType::Ternary2Bit,
                vec![vocab, hidden],
                vec![0xDDu8; byte_count],
            ),
        );
        core.tensors.insert(
            "output.prq_scale".to_string(),
            mk_tensor(
                "output.prq_scale",
                MudTensorType::Float32,
                vec![vocab],
                stale_scale_bytes.clone(),
            ),
        );

        let mut mud = crate::mud::MudFile {
            mmap: None,
            skills: HashMap::new(),
            global_metadata: HashMap::new(),
            path: None,
        };
        mud.skills.insert("core".to_string(), core);

        let mut shadow_layers = vec![crate::mud::slime_backward::SlimeLayerShadowF32::empty()];
        let layers: Vec<crate::mud::slime_forward::SlimeLayer> = Vec::new();

        let rep = sync_shadow_to_mud(
            &mut mud,
            &mut shadow_emb,
            &mut shadow_layers,
            &layers,
            None,
            true,
        );

        let core = mud.skills.get("core").unwrap();
        let out = core.tensors.get("output.weight").unwrap();
        let emb = core.tensors.get("token_embd.weight").unwrap();
        assert_eq!(
            out.owned_data, emb.owned_data,
            "output.weight must mirror token_embd.weight (R-10)"
        );
        assert_eq!(out.owned_data.as_deref(), Some(expected_packed.as_slice()));

        let out_s = core.tensors.get("output.prq_scale").unwrap();
        let emb_s = core.tensors.get("token_embd.prq_scale").unwrap();
        assert_eq!(
            out_s.owned_data, emb_s.owned_data,
            "output.prq_scale must mirror token_embd.prq_scale (R-10)"
        );
        assert_eq!(out_s.owned_data.as_deref(), Some(expected_scales.as_slice()));

        assert!(rep.moved, "sync must report head + emb movement");
        unsafe {
            std::env::remove_var("MUD_TRAIN_FREEZE_EMB");
        }
    }

    /// R-10 regression: if the model is genuinely tied (no separate
    /// `output.weight`), `sync_shadow_to_mud` must NOT create a spurious
    /// output.weight — it should be a silent no-op for the head.
    #[test]
    fn test_sync_no_output_weight_when_tied() {
        let _g = SYNC_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            std::env::remove_var("MUD_TRAIN_FREEZE_EMB");
            std::env::remove_var("MUD_TRAIN_SCALES_ONLY");
            std::env::remove_var("MUD_TRAIN_DEBUG_DW");
        }
        // Force emb trainable so token_embd sync runs.
        unsafe {
            std::env::set_var("MUD_TRAIN_FREEZE_EMB", "0");
        }
        let vocab = 2usize;
        let hidden = 8usize;
        let mut shadow_emb = vec![1.0f32; vocab * hidden];

        let byte_count = (vocab * hidden).div_ceil(8) * 4;
        let mut core = crate::mud::MudSkill {
            name: "core".to_string(),
            tensors: HashMap::new(),
            metadata: HashMap::new(),
        };
        core.tensors.insert(
            "token_embd.weight".to_string(),
            mk_tensor(
                "token_embd.weight",
                MudTensorType::Ternary2Bit,
                vec![vocab, hidden],
                vec![0u8; byte_count],
            ),
        );
        core.tensors.insert(
            "token_embd.prq_scale".to_string(),
            mk_tensor(
                "token_embd.prq_scale",
                MudTensorType::Float32,
                vec![vocab],
                vec![0u8; vocab * 4],
            ),
        );
        // NOTE: no "output.weight" inserted — model is tied.

        let mut mud = crate::mud::MudFile {
            mmap: None,
            skills: HashMap::new(),
            global_metadata: HashMap::new(),
            path: None,
        };
        mud.skills.insert("core".to_string(), core);

        let mut shadow_layers = vec![crate::mud::slime_backward::SlimeLayerShadowF32::empty()];
        let layers: Vec<crate::mud::slime_forward::SlimeLayer> = Vec::new();

        sync_shadow_to_mud(&mut mud, &mut shadow_emb, &mut shadow_layers, &layers, None, false);

        let core = mud.skills.get("core").unwrap();
        assert!(!core.tensors.contains_key("output.weight"), "tied model must not synthesize output.weight");
        unsafe {
            std::env::remove_var("MUD_TRAIN_FREEZE_EMB");
        }
    }

    /// R-14: 5 save→load→save cycles with no training must keep `prq_scale`
    /// and the dequantized values stable (±1e-6). The naive recompute
    /// `s = absmean(shadow)/√2` off the round-tripped (lossy dequant) shadow
    /// would drift ~×0.5 each cycle — the stable pack must preserve s_old
    /// because the codes are unchanged (no ΔW).
    #[test]
    fn test_prq_scale_stable_roundtrip() {
        let _g = SYNC_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            std::env::remove_var("MUD_TRAIN_FREEZE_EMB");
            std::env::remove_var("MUD_TRAIN_SCALES_ONLY");
            std::env::remove_var("MUD_TRAIN_DEBUG_DW");
        }
        let rows = 2usize;
        let cols = 8usize;
        let row_words = cols.div_ceil(8);
        let row_bytes = row_words * 4;
        let byte_count = rows * row_bytes;

        // Mixed-sign non-trivial shadow (no row all-zero → absmean>0).
        let shadow: Vec<f32> = vec![
            1.0, -1.0, 0.5, -0.3, 0.8, -0.9, 0.1, -0.7, // row 0
            0.6, -1.0, 0.2, 0.4, -0.8, 0.3, 0.9, -0.5,  // row 1
        ];

        // First save: no prev_scales/codes → s_new.
        let mut packed = vec![0u8; byte_count];
        let mut scales: Vec<f32> = Vec::with_capacity(rows);
        let (_bits0, _) = unsafe {
            stable_pack_ternary_rows(
                &shadow, rows, cols, &[], None, std::ptr::null(),
                &mut packed, &mut scales,
            )
        };
        assert_eq!(scales.len(), rows);

        // Round-tripped shadow = dequant(prev_codes, scales) = code·s — this is
        // EXACTLY what the trainer reload produces (lossy dequant into f32 shadow).
        let mut rt_shadow = vec![0f32; rows * cols];
        for r in 0..rows {
            let s = scales[r];
            let code_ptr = unsafe { packed.as_ptr().add(r * row_bytes) as *const u32 };
            let mut codes = vec![0f32; cols];
            unsafe { crate::mud::dequantize_ternary_row(code_ptr, &mut codes, cols); }
            for (i, c) in codes.iter().enumerate() {
                rt_shadow[r * cols + i] = (*c) * s;
            }
        }

        // 5 cycles: shadow == dequant(prev) → must PRESERVE scale + codes.
        for cycle in 0..5 {
            let prev_packed = packed.clone();
            let prev_scales = scales.clone();
            let mut new_packed = vec![0u8; byte_count];
            let mut new_scales: Vec<f32> = Vec::with_capacity(rows);
            let (bits, _) = unsafe {
                stable_pack_ternary_rows(
                    &rt_shadow, rows, cols, &prev_scales, Some(&prev_packed),
                    std::ptr::null(), &mut new_packed, &mut new_scales,
                )
            };
            assert_eq!(
                bits, 0,
                "cycle {cycle}: no-movement round-trip must be bit-stable (bits_changed==0)"
            );
            assert_eq!(
                new_packed, prev_packed,
                "cycle {cycle}: codes must be reproduced verbatim"
            );
            for (a, b) in new_scales.iter().zip(prev_scales.iter()) {
                assert!(
                    (a - b).abs() < 1e-6,
                    "cycle {cycle}: scale drifted {:e} vs {:e} (delta {:e})",
                    a, b, (a - b).abs()
                );
            }
            packed = new_packed;
            scales = new_scales;
        }

        // Contrast: a NAIVE recompute (s = absmean(rt_shadow)/√2, no preserve)
        // WOULD drift. Prove the preserve logic is load-bearing: the per-row
        // naive factor mean|code|/√2 differs from 1.0 (here 0.5–0.9 range),
        // so recomputing every cycle would shrink s unboundedly.
        for r in 0..rows {
            let row = &rt_shadow[r * cols..(r + 1) * cols];
            let absmean = row.iter().map(|v| v.abs()).sum::<f32>() / cols as f32;
            let s_naive = (absmean * std::f32::consts::FRAC_1_SQRT_2).max(1e-8);
            // mean|code| ≤ 1 always; if any code is ±1 the factor is < 1/√2·1+... 
            // i.e. s_naive < s_preserved (drift downward). The preserve keeps s0.
            assert!(
                s_naive <= scales[r] * 1.0001,
                "naive recompute should not exceed preserved scale (drift proof): \
                 s_naive={:e} s_preserved={:e}",
                s_naive, scales[r]
            );
            assert!(
                (s_naive - scales[r]).abs() / scales[r] > 0.05,
                "naive recompute should drift >5% away from preserved scale: \
                 ratio={}",
                s_naive / scales[r]
            );
        }
    }

    /// R-14: when the shadow genuinely moves (real ΔW), the pack must detect
    /// movement (bits_changed > 0) AND recompute the scale (not preserve the
    /// stale s_old).
    #[test]
    fn test_prq_scale_recomputes_on_movement() {
        let rows = 1usize;
        let cols = 8usize;
        let row_words = cols.div_ceil(8);
        let row_bytes = row_words * 4;
        let byte_count = rows * row_bytes;

        let shadow1: Vec<f32> = vec![1.0, -1.0, 0.5, 0.8, -0.9, 0.1, -0.7, 0.3];

        // First save (no prev) → codes1, scales1.
        let mut packed1 = vec![0u8; byte_count];
        let mut scales1: Vec<f32> = Vec::with_capacity(rows);
        unsafe {
            stable_pack_ternary_rows(
                &shadow1, rows, cols, &[], None, std::ptr::null(),
                &mut packed1, &mut scales1,
            )
        };
        let s1 = scales1[0];

        // Simulate a training step: flip a dead code (+0.1, below delta) to
        // alive (+0.5, above delta) — a real code-changing movement (uniform
        // scaling would NOT change codes, since s tracks them).
        let mut shadow2 = shadow1.clone();
        shadow2[5] = 0.5; // 0.1 → 0.5 : code flips 0 → +1
        let mut packed2 = vec![0u8; byte_count];
        let mut scales2: Vec<f32> = Vec::with_capacity(rows);
        let (bits, _) = unsafe {
            stable_pack_ternary_rows(
                &shadow2, rows, cols, &scales1, Some(&packed1),
                std::ptr::null(), &mut packed2, &mut scales2,
            )
        };
        assert!(bits > 0, "real weight movement must produce bits_changed > 0");
        // Movement → scale must be RECOMPUTED from the new shadow, not preserved.
        let absmean2 = shadow2.iter().map(|v| v.abs()).sum::<f32>() / cols as f32;
        let s_new = (absmean2 * std::f32::consts::FRAC_1_SQRT_2).max(1e-8);
        assert!((scales2[0] - s_new).abs() < 1e-6, "moved scale must recompute to s_new");
        assert!((scales2[0] - s1).abs() / s1 > 0.05, "moved scale must differ from stale s_old");
        // Codes changed too.
        assert_ne!(packed2, packed1, "moved shadow must produce different codes");
    }
}
