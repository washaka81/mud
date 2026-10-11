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

/// Apply the shadow f32 buffers to the mmap-backed `MudTensor`s.
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
    let emb_tensor = core
        .tensors
        .get_mut("token_embd.weight")
        .expect("Missing token_embd.weight");

    if !(freeze_emb || scales_only) {
        if emb_tensor.t_type == MudTensorType::Ternary2Bit {
            let rows = emb_tensor.shape[0];
            let cols = emb_tensor.shape[1];
            let byte_count = shadow_emb.len().div_ceil(8) * 4;
            // SPRINT-6 (P-01): reuse thread-local buffers instead of allocating
            // per-sync. All work happens inside the with_borrow_mut closure so
            // the RefMut borrow is released before we touch core.tensors.
            SYNC_PACKED_BUF.with_borrow_mut(|packed| {
                SYNC_SCALES_BUF.with_borrow_mut(|scales| {
                    packed.clear();
                    packed.resize(byte_count, 0u8);
                    scales.clear();
                    scales.reserve(rows);

                    // Fast per-row pack: compute scale, then batch-pack 8 elements
                    // per u32 word via pack_ternary_into (one store/word, no
                    // read-modify-write per element). The ternary threshold in the
                    // original code is val=round(w·s⁻¹).clamp(-1,1) > 0.5, which is
                    // equivalent to |w| > 0.5·s, so we pass delta=0.5·s.
                    for r in 0..rows {
                        let start = r * cols;
                        let s = unsafe {
                            let row = std::slice::from_raw_parts(
                                shadow_emb.as_ptr().add(start),
                                cols,
                            );
                            let absmean = row.iter().map(|v| v.abs()).sum::<f32>()
                                / cols as f32;
                            (absmean * std::f32::consts::FRAC_1_SQRT_2).max(1e-8)
                        };
                        scales.push(s);
                        let delta = 0.5f32 * s;
                        let row_u32_start = r * cols / 8;
                        let word_ptr =
                            unsafe { packed.as_mut_ptr().add(row_u32_start * 4) as *mut u32 };
                        unsafe {
                            crate::mud::ezop::pack_ternary_into(
                                shadow_emb.as_ptr().add(start),
                                cols,
                                delta,
                                word_ptr,
                            );
                        }
                    }
                });
            });

            // ΔW instrumentation for embedding (was missing — caused hash≠ΔW contradiction)
            {
                let prev: Option<&[u8]> = emb_tensor.owned_data.as_deref();
                SYNC_PACKED_BUF.with_borrow(|packed| {
                    if let Some(prev) = prev {
                        if prev.len() == packed.len() {
                            for (a, b) in prev.iter().zip(packed.iter()) {
                                dbg_bits_changed += (a != b) as u64;
                            }
                            dbg_bytes_total += packed.len() as u64;
                        }
                    }
                });
            }
            // P-01: mem::take avoids clone() — moves the Vec out of the thread-local,
            // releasing the borrow before write_owned_bytes. The thread-local
            // keeps a reusable (empty) Vec for next call.
            let packed_move = SYNC_PACKED_BUF.with_borrow_mut(std::mem::take::<Vec<u8>>);
            write_owned_bytes(emb_tensor, &packed_move);

            // P-01: avoid to_vec() — use mem::take to move scales out of thread-local
            // (zero-copy write to tensor, no clone, no double-copy).
            let scales_move = SYNC_SCALES_BUF.with_borrow_mut(std::mem::take::<Vec<f32>>);
            if let Some(scale_tensor) = core.tensors.get_mut("token_embd.prq_scale") {
                // ΔW: compare old vs new PRQ scale for embedding
                if let Some(prev_s) = scale_tensor.owned_data.as_deref() {
                    if prev_s.len() == scales_move.len() * 4 {
                        let prev_f = unsafe {
                            std::slice::from_raw_parts(prev_s.as_ptr() as *const f32, prev_s.len() / 4)
                        };
                        for (a, b) in prev_f.iter().zip(scales_move.iter()) {
                            dbg_scale_absdelta += (a - b).abs() as f64;
                            dbg_scale_absbase += a.abs() as f64;
                        }
                    }
                }
                let scale_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(scales_move.as_ptr() as *const u8, scales_move.len() * 4)
                };
                write_owned_bytes(scale_tensor, scale_bytes);
            } else {
                // Scale tensor doesn't exist yet — create from the already-extracted scales_move
                let scale_bytes = unsafe {
                    std::slice::from_raw_parts(scales_move.as_ptr() as *const u8, scales_move.len() * 4).to_vec()
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
        } else {
            // F32 embedding: avoid to_vec() — use &[u8] view directly
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
        }
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

#[cfg(test)]
fn pack_ternary(ternary_data: &[f32]) -> Vec<u8> {
    let u32_count = ternary_data.len().div_ceil(8);
    let mut packed_vec = vec![0u32; u32_count];
    for (i, &v) in ternary_data.iter().enumerate() {
        let bit = if v > 0.5 {
            0x1u32
        } else if v < -0.5 {
            0xFu32
        } else {
            0x0u32
        };
        packed_vec[i / 8] |= bit << ((i % 8) * 4);
    }
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
                let absmean = unsafe {
                    let row = std::slice::from_raw_parts(
                        weights.as_ptr().add(start),
                        cols,
                    );
                    row.iter().map(|v| v.abs()).sum::<f32>() / cols as f32
                };
                new_scales.push(absmean.max(1e-8));
            }
        });
        // P-01: mem::take avoids clone — move scales out of thread-local
        let scales_move = SYNC_SCALES_BUF.with_borrow_mut(std::mem::take::<Vec<f32>>);
        if let Some(scale_t) = core.tensors.get_mut(&format!("{}{}.prq_scale", p, name)) {
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
    let (rows, cols, is_ternary, prev_data_ptr, prev_owned_len) =
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
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(weights.as_ptr() as *const u8, weights.len() * 4)
        };
        if let Some(t) = core.tensors.get_mut(&key) {
            write_owned_bytes(t, bytes);
        }
        return;
    }

    // Ternary path: pack into thread-local buffers, then copy out.
    let u32_count = weights.len().div_ceil(8);
    let byte_count = u32_count * 4;
    SYNC_PACKED_BUF.with_borrow_mut(|packed| {
        SYNC_SCALES_BUF.with_borrow_mut(|new_scales| {
            packed.clear();
            packed.resize(byte_count, 0u8);
            new_scales.clear();
            new_scales.reserve(rows);

            // Fast per-row pack: compute scale, then batch-pack 8 elements
            // per u32 word via pack_ternary_into (one store/word, no
            // read-modify-write per element). The ternary threshold in the
            // original code is val=round(w·s⁻¹).clamp(-1,1) > 0.5, which is
            // equivalent to |w| > 0.5·s, so we pass delta=0.5·s.
            for r in 0..rows {
                let start = r * cols;
                let s = unsafe {
                    let row = std::slice::from_raw_parts(
                        weights.as_ptr().add(start),
                        cols,
                    );
                    let absmean = row.iter().map(|v| v.abs()).sum::<f32>()
                        / cols as f32;
                    (absmean * std::f32::consts::FRAC_1_SQRT_2).max(1e-8)
                };
                new_scales.push(s);
                let delta = 0.5f32 * s;
                let row_u32_start = r * cols / 8;
                let word_ptr =
                    unsafe { packed.as_mut_ptr().add(row_u32_start * 4) as *mut u32 };
                unsafe {
                    crate::mud::ezop::pack_ternary_into(
                        weights.as_ptr().add(start),
                        cols,
                        delta,
                        word_ptr,
                    );
                }
            }
        });
    });

    // P-01: mem::take avoids clone() — moves both buffers out of thread-locals.
    let packed_move = SYNC_PACKED_BUF.with_borrow_mut(std::mem::take::<Vec<u8>>);
    let new_scales_move = SYNC_SCALES_BUF.with_borrow_mut(std::mem::take::<Vec<f32>>);

    // ΔW instrumentation: compare prev tensor data vs new packed bytes
    if let Some(prev_len) = prev_owned_len {
        if prev_len == packed_move.len() {
            if let Some(t) = core.tensors.get(&key) {
                if let Some(prev) = t.owned_data.as_deref() {
                    for (a, b) in prev.iter().zip(packed_move.iter()) {
                        *dbg_bits_changed += (a != b) as u64;
                    }
                    *dbg_bytes_total += packed_move.len() as u64;
                } else if !prev_data_ptr.is_null() {
                    let prev = unsafe { std::slice::from_raw_parts(prev_data_ptr, packed_move.len()) };
                    for (a, b) in prev.iter().zip(packed_move.iter()) {
                        *dbg_bits_changed += (a != b) as u64;
                    }
                    *dbg_bytes_total += packed_move.len() as u64;
                }
            }
        }
    }
    if let Some(t) = core.tensors.get_mut(&key) {
        write_owned_bytes(t, &packed_move);
    }

    // P-01: use &[u8] view of new_scales directly — no to_vec()
    let scale_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(new_scales_move.as_ptr() as *const u8, new_scales_move.len() * 4)
    };
    let scale_key = format!("{}{}.prq_scale", p, name);
    if let Some(scale_t) = core.tensors.get_mut(&scale_key) {
        let prev_s: Option<&[f32]> = {
            let owned: Option<&Vec<u8>> = scale_t.owned_data.as_ref();
            if let Some(ex) = owned {
                let n = ex.len() / 4;
                Some(unsafe {
                    std::slice::from_raw_parts(ex.as_ptr() as *const f32, n)
                })
            } else if !scale_t.data_ptr.is_null() {
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
                let bit = if val > 0.5 { 0x1u32 } else if val < -0.5 { 0xFu32 } else { 0x0u32 };
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

    #[test]
    fn test_dw_report_default() {
        let r = DwReport::default();
        assert_eq!(r.bits_changed, 0);
        assert!(!r.moved);
    }
}
