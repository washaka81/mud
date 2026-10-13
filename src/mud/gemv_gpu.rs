//! GPU/GEMV dispatch infrastructure (extracted from `slime_forward.rs`).
//!
//! Contains: env-var caches, `GemvGpuCache` weight-cache, GPU ternary GEMV
//! dispatch (single + QKV), and the output-norm GPU path.
//! The CPU GEMV compute kernels (`ternary_gemv_rowwise`, `f32_gemv_rowwise`, etc.)
//! remain in `slime_forward.rs` and call `try_gpu_*` helpers here.

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

// ── PERF-2 (V25 audit): cache `MUD_GEMV_ROWS` env var. ──────────────────────
// Read once at first call (AtomicU64: 0/1). Hot path is `ternary_gemv_rowwise_submit`
// (7 mats × 30 layers × 1024 tokens = 215k times/chunk); env::var (~50 ns) → ~1 ns.
static MUD_GEMV_ROWS_8: AtomicU64 = AtomicU64::new(0);
/// Returns true when `MUD_GEMV_ROWS=8` (enables 8-row AVX2 kernel).
#[inline]
pub(crate) fn prefer_gemv_8_rows() -> bool {
    match MUD_GEMV_ROWS_8.load(Ordering::Relaxed) {
        1 => true,
        _ => {
            let v = std::env::var("MUD_GEMV_ROWS")
                .map(|s| s.trim() == "8")
                .unwrap_or(false);
            MUD_GEMV_ROWS_8.store(if v { 1 } else { 0 }, Ordering::Relaxed);
            v
        }
    }
}

// ── PERF-2 (V25 audit): cache `MUD_USE_VULKAN` env var. ─────────────────────
// Read once. Hot path: output_norm + try_gpu_dense_mha (30k calls/chunk).
// AGENTS §2 blacklists Iris Xe so on i7-1260P UMA this is effectively always
// false; cache saves the env::var trip on every call.
static MUD_USE_VULKAN_ON: AtomicU64 = AtomicU64::new(0);
/// Returns true when Vulkan dispatch is not explicitly disabled.
pub(crate) fn use_vulkan_cached() -> bool {
    match MUD_USE_VULKAN_ON.load(Ordering::Relaxed) {
        1 => true,
        _ => {
            let v = std::env::var("MUD_USE_VULKAN")
                .map(|s| s != "0" && !s.eq_ignore_ascii_case("false"))
                .unwrap_or(true);
            MUD_USE_VULKAN_ON.store(if v { 1 } else { 0 }, Ordering::Relaxed);
            v
        }
    }
}

// ── SPRINT-5 (5.2): GPU dispatch fast path. ────────────────────────────────
//
// `try_gpu_ternary_gemv` (and `_qkv`) lock a `Mutex<GemvGpuCache>` on every
// call. On the common path (CPU-only host, Iris Xe blacklisted, GPU
// disabled by env, calibration already settled, weight cache miss),
// the lock is wasted work.
//
// We use a tri-state `AtomicU8` to short-circuit the lock when the
// dispatch is known to be disabled:
//   0 = unknown, must probe
//   1 = enabled + available, full fast path
//   2 = disabled (no device / opt-out), skip the lock entirely
//
// The probe runs once on the first call (CAS 0→1 or 0→2). All subsequent
// calls take one Relaxed load + early return when the state is 2.
static GPU_GEMV_DISPATCH: AtomicU8 = AtomicU8::new(0);

/// Returns true if the GPU dispatch is **known to be available**.
/// On the first call this probes; subsequent calls are O(1) atomic.
pub(crate) fn gpu_gemv_known_available() -> bool {
    match GPU_GEMV_DISPATCH.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            // First call: probe once. We don't try to lock the cache here
            // (that requires &mut). Just attempt a cheap AshContext probe.
            let ok = crate::vulkan::ash_backend::AshContext::new()
                .map(|c| c.is_available())
                .unwrap_or(false);
            GPU_GEMV_DISPATCH.store(if ok { 1 } else { 2 }, Ordering::Relaxed);
            ok
        }
    }
}

/// Mark the GPU dispatch as permanently disabled (e.g. after a calibration
/// failure that published `no_device`). Avoids re-probing on every call.
pub(crate) fn mark_gpu_gemv_disabled() {
    GPU_GEMV_DISPATCH.store(2, Ordering::Relaxed);
}

// ── Phase B+ / Stream F: policy-driven GPU GEMV ───────────────────────────────

struct GemvGpuCache {
    ctx: Option<crate::vulkan::ash_backend::AshContext>,
    last_w_ptr: usize,
    last_w_words: usize,
    last_sc_ptr: usize,
    last_sc_n: usize,
    /// Stream F: last Q/K/V host pointers for weight-cache skip.
    last_q_ptr: usize,
    last_k_ptr: usize,
    last_v_ptr: usize,
    last_q_words: usize,
    last_k_words: usize,
    last_v_words: usize,
    last_q_sc: usize,
    last_k_sc: usize,
    last_v_sc: usize,
    last_n_q: usize,
    last_n_kv: usize,
    /// True after auto-calib attempt (success or no-device).
    calibrated: bool,
}

static GEMV_ASH: std::sync::OnceLock<std::sync::Mutex<GemvGpuCache>> = std::sync::OnceLock::new();

fn gemv_gpu_cache() -> &'static std::sync::Mutex<GemvGpuCache> {
    GEMV_ASH.get_or_init(|| {
        let ctx = match crate::vulkan::ash_backend::AshContext::new() {
            Ok(c) if c.is_available() => Some(c),
            _ => None,
        };
        let no_dev = ctx.is_none();
        if no_dev {
            crate::vulkan::gemv_policy::publish_no_device();
            // SPRINT-5 (5.2): make the fast path skip the lock forever.
            mark_gpu_gemv_disabled();
        }
        std::sync::Mutex::new(GemvGpuCache {
            ctx,
            last_w_ptr: 0,
            last_w_words: 0,
            last_sc_ptr: 0,
            last_sc_n: 0,
            last_q_ptr: 0,
            last_k_ptr: 0,
            last_v_ptr: 0,
            last_q_words: 0,
            last_k_words: 0,
            last_v_words: 0,
            last_q_sc: 0,
            last_k_sc: 0,
            last_v_sc: 0,
            last_n_q: 0,
            last_n_kv: 0,
            calibrated: no_dev,
        })
    })
}

/// Whether policy allows *considering* GPU (not Off / vulkan disabled).
/// Prefer [`crate::vulkan::gemv_policy::should_try_gpu`] for shape-aware checks.
pub(crate) fn gpu_gemv_policy_active() -> bool {
    use crate::vulkan::gemv_policy::{parse_gemv_mode, vulkan_not_disabled, GemvGpuMode};
    if !vulkan_not_disabled() {
        return false;
    }
    !matches!(parse_gemv_mode(), GemvGpuMode::Off)
}

/// Try GPU tiled ternary GEMV. Returns true on success.
///
/// # Safety
/// Same pointer contracts as [`crate::mud::slime_forward::ternary_gemv_rowwise`].
pub(crate) unsafe fn try_gpu_ternary_gemv(
    acts_f32: &[f32],
    w_u8: *const u8,
    out_f32: &mut [f32],
    scales: *const f32,
    n_out: usize,
    n_in: usize,
) -> bool {
    use crate::vulkan::gemv_policy;

    // SPRINT-5 (5.2): fast path — skip the Mutex entirely when the GPU is
    // known to be disabled (no device, Iris Xe blacklisted, opt-out). The
    // AtomicU8 load is one Relaxed op; on CPU-only hosts this saves a
    // lock acquisition on every GEMV call.
    if !gpu_gemv_known_available() {
        return false;
    }
    if !gpu_gemv_policy_active() {
        return false;
    }
    if w_u8.is_null() || scales.is_null() || acts_f32.len() < n_in || out_f32.len() < n_out {
        return false;
    }
    // Fast shape reject before lock (except Auto pending calib — still needs ≥64²)
    if !gemv_policy::should_try_gpu(n_in, n_out) {
        return false;
    }

    let Ok(mut guard) = gemv_gpu_cache().lock() else {
        return false;
    };
    if guard.ctx.as_ref().map(|c| c.is_available()) != Some(true) {
        gemv_policy::publish_no_device();
        return false;
    }

    // Auto: one-shot profile on this device (uses same AshContext as inference).
    if !guard.calibrated {
        let ctx = guard.ctx.as_mut().expect("checked available");
        gemv_policy::ensure_calibrated(ctx);
        guard.calibrated = true;
        // Re-check threshold after calibration (may have become NEVER or high).
        if !gemv_policy::should_try_gpu(n_in, n_out) {
            return false;
        }
    } else if !gemv_policy::should_try_gpu(n_in, n_out) {
        return false;
    }

    let blocks = n_in.div_ceil(8);
    let w_words = n_out * blocks;
    let packed = std::slice::from_raw_parts(w_u8 as *const u32, w_words);
    let sc = std::slice::from_raw_parts(scales, n_out);

    let w_ptr = w_u8 as usize;
    let sc_ptr = scales as usize;
    let upload_w = guard.last_w_ptr != w_ptr || guard.last_w_words != w_words;
    let upload_sc = guard.last_sc_ptr != sc_ptr || guard.last_sc_n != n_out;

    let ok = {
        let ctx = guard.ctx.as_mut().expect("checked available");
        ctx.dispatch_gemv_host_sync_ex(
            &acts_f32[..n_in],
            packed,
            sc,
            &mut out_f32[..n_out],
            n_in,
            n_out,
            false,
            upload_w,
            upload_sc,
        )
        .is_ok()
    };
    if ok {
        if upload_w {
            guard.last_w_ptr = w_ptr;
            guard.last_w_words = w_words;
        }
        if upload_sc {
            guard.last_sc_ptr = sc_ptr;
            guard.last_sc_n = n_out;
        }
    }
    ok
}

/// Phase B: submit GEMV row tasks without waiting (caller must `pool.wait_all()`).
/// Used to overlap Q/K/V projections on the same PCorePool.
///
/// # Safety
/// Same as [`crate::mud::slime_forward::ternary_gemv_rowwise`]. Do not free buffers until after `wait_all`.
pub(crate) unsafe fn ternary_gemv_rowwise_submit(
    acts_f32: &[f32],
    w_u8: *const u8,
    out_f32: &mut [f32],
    scales: *const f32,
    n_out: usize,
    n_in: usize,
) {
    // R-35: row stride is u32 words per PADDED row (canonical R-15 layout:
    // rows*div_ceil(cols,8)*4 bytes). Floor (n_in/8) mis-strided every row
    // past the first when n_in % 8 != 0.
    let row_u32s = n_in.div_ceil(8);
    let w_u32 = w_u8 as *const u32;

    // FIX (2026-08-03): Small-matrix fast path — inline execution avoids
    // PCorePool dispatch overhead (~0.5ms per dispatch) which dominates
    // the compute (~1μs) when n_out ≤ 256 (Q/K/V projections on pico/nano
    // models: hidden=128 → n_out=128, 8 threads × 16 rows each = 0.8ms overhead
    // for 16 rows of compute that takes ~2μs). This gives ~6× speedup on
    // small-model token throughput.
    const SMALL_GEMV_THRESHOLD: usize = 256;
    if n_out < SMALL_GEMV_THRESHOLD {
        let prefer_8 = prefer_gemv_8_rows();
        let w_ptr = w_u32;
        let out_ptr = out_f32.as_mut_ptr();
        let acts_ptr = acts_f32.as_ptr();
        let sc_ptr = scales;

        let mut row = 0;
        if prefer_8 {
            while row + 8 <= n_out {
                crate::asm::ternary_gemv_8rows(n_in, acts_ptr, w_ptr.add(row * row_u32s), out_ptr.add(row), 1.0, row_u32s);
                crate::asm::apply_row_scales_8_avx2(out_ptr.add(row), sc_ptr.add(row));
                row += 8;
            }
        }
        while row + 4 <= n_out {
            crate::asm::ternary_gemv_4rows(n_in, acts_ptr, w_ptr.add(row * row_u32s), out_ptr.add(row), 1.0, row_u32s);
            for r in 0..4 {
                let s = (*sc_ptr.add(row + r)).clamp(-1e8, 1e8);
                *out_ptr.add(row + r) *= if s.is_finite() { s } else { 0.0 };
            }
            row += 4;
        }
        for r in row..n_out {
            let s = (*sc_ptr.add(r)).clamp(-1e8, 1e8);
            crate::asm::ternary_gemv(n_in, acts_ptr, w_ptr.add(r * row_u32s), out_ptr.add(r), if s.is_finite() { s } else { 0.0 });
        }
        return;
    }

    let pool = crate::mud::pcore_pool::get_pool();
    let n_tasks = pool.num_threads().max(1);
    // T11: `MUD_GEMV_ROWS=8` enables 8-row kernel. Default 4 — microbench on
    // i7-1260P (hidden=2048, warm L2) shows 4-row ≥ 8-row; 8-row helps more when
    // x is cold / multi-thread thrash (opt-in until host re-calib).
    let prefer_8 = prefer_gemv_8_rows();
    let align = if prefer_8 { 8 } else { 4 };
    let raw_chunk = (n_out / n_tasks).max(align);
    let rows_per_task = (raw_chunk / align * align).max(align);

    let acts_p = acts_f32.as_ptr() as usize;
    let w_p = w_u32 as usize;
    let out_p = out_f32.as_mut_ptr() as usize;
    let scales_p = scales as usize;

    for i in 0..n_tasks {
        let start_row = i * rows_per_task;
        let end_row = if i + 1 == n_tasks {
            n_out
        } else {
            (start_row + rows_per_task).min(n_out)
        };
        if start_row >= end_row {
            break;
        }

        pool.execute(move || {
            let acts = acts_p as *const f32;
            let w = w_p as *const u32;
            let out = out_p as *mut f32;
            let sc = scales_p as *const f32;

            let mut row = start_row;
            if prefer_8 {
                while row + 8 <= end_row {
                    // SAFETY: this closure owns the disjoint row range
                    // [start_row, end_row) ⊆ [0, n_out); `acts` covers n_in,
                    // `w` covers n_out * row_u32s u32 words, `out`/`sc` cover
                    // n_out (caller contract of `ternary_gemv_rowwise_submit`).
                    // Pointers were captured as `usize` to cross the thread
                    // boundary; each worker touches only its own rows.
                    unsafe {
                        crate::asm::ternary_gemv_8rows(
                            n_in,
                            acts,
                            w.add(row * row_u32s),
                            out.add(row),
                            1.0,
                            row_u32s,
                        );
                        // AVX2 batch scale application (replaces scalar loop)
                        // Processes 8 rows simultaneously using packed f32
                        crate::asm::apply_row_scales_8_avx2(
                            out.add(row),
                            sc.add(row),
                        );
                    }
                    row += 8;
                }
            }
            while row + 4 <= end_row {
                // SAFETY: same disjoint-row contract as the 8-row block above;
                // `row + 4 <= end_row` keeps every accessed row < n_out.
                unsafe {
                    crate::asm::ternary_gemv_4rows(
                        n_in,
                        acts,
                        w.add(row * row_u32s),
                        out.add(row),
                        1.0,
                        row_u32s,
                    );
                    // Scalar scale for 4-row remainder (AVX2 needs 8 elements)
                    for r in 0..4 {
                        let s = (*sc.add(row + r)).clamp(-1e8, 1e8);
                        *out.add(row + r) *= if s.is_finite() { s } else { 0.0 };
                    }
                }
                row += 4;
            }

            for r in row..end_row {
                // SAFETY: scalar tail of the same partition: `r < end_row ≤
                // n_out`, so `sc[r]`, `w[r*row_u32s ..]` and `out[r]` are all
                // in-bounds.
                unsafe {
                    let s = (*sc.add(r)).clamp(-1e8, 1e8);
                    crate::asm::ternary_gemv(
                        n_in,
                        acts,
                        w.add(r * row_u32s),
                        out.add(r),
                        if s.is_finite() { s } else { 0.0 },
                    );
                }
            }
        });
    }
}

/// R-40: ternary LM-head logits — `out[v] = prq_scale[v] * dot(acts, ELUT_row_v)`.
///
/// Batched 4-/8-row ELUT kernels over `PCorePool` with per-row PRQ scales; the
/// dense `vocab × hidden` f32 head is **never materialized** (on `ale.mud` the
/// dequantized head was ~77.6 MB read per token, ≈99 % of weight traffic).
/// GPU is tried first via the usual policy (`MUD_GPU_GEMV`), falling back to
/// the AVX2 pool.
///
/// # Safety
/// `acts` is valid for `n_in` f32; `w_u8` points to `n_out` ELUT-packed rows in
/// the canonical R-15 layout (`n_out * n_in.div_ceil(8) * 4` bytes); `scales`
/// is valid for `n_out` f32; `out` is valid for `n_out` f32. Buffers must not
/// alias (GEMV semantics).
pub unsafe fn ternary_head_logits(
    acts: &[f32],
    w_u8: *const u8,
    out: &mut [f32],
    scales: *const f32,
    n_out: usize,
    n_in: usize,
) {
    if try_gpu_ternary_gemv(acts, w_u8, out, scales, n_out, n_in) {
        return;
    }
    ternary_gemv_rowwise_submit(acts, w_u8, out, scales, n_out, n_in);
    crate::mud::pcore_pool::get_pool().wait_all();
}

/// Phase B + Stream F: Q, K, V projections.
///
/// 1. **GPU path (F):** one ash command buffer, three GEMV dispatches, one fence
///    (activations uploaded once; weight upload cached by host pointer).
/// 2. **CPU path (B):** three `ternary_gemv_rowwise_submit` + single `wait_all`.
///
/// # Safety
/// All weight/scale pointers and output buffers must be valid and non-aliasing.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn ternary_gemv_qkv_parallel(
    acts_f32: &[f32],
    q_w: *const u8,
    k_w: *const u8,
    v_w: *const u8,
    q_out: &mut [f32],
    k_out: &mut [f32],
    v_out: &mut [f32],
    q_scales: *const f32,
    k_scales: *const f32,
    v_scales: *const f32,
    n_q: usize,
    n_kv: usize,
    n_in: usize,
) {
    if try_gpu_ternary_gemv_qkv(
        acts_f32, q_w, k_w, v_w, q_out, k_out, v_out, q_scales, k_scales, v_scales, n_q, n_kv, n_in,
    ) {
        return;
    }
    ternary_gemv_rowwise_submit(acts_f32, q_w, q_out, q_scales, n_q, n_in);
    ternary_gemv_rowwise_submit(acts_f32, k_w, k_out, k_scales, n_kv, n_in);
    ternary_gemv_rowwise_submit(acts_f32, v_w, v_out, v_scales, n_kv, n_in);
    crate::mud::pcore_pool::get_pool().wait_all();
}

/// Stream F: GPU QKV multi-dispatch one CB. Returns true on success.
///
/// # Safety
/// Same as [`ternary_gemv_qkv_parallel`].
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn try_gpu_ternary_gemv_qkv(
    acts_f32: &[f32],
    q_w: *const u8,
    k_w: *const u8,
    v_w: *const u8,
    q_out: &mut [f32],
    k_out: &mut [f32],
    v_out: &mut [f32],
    q_scales: *const f32,
    k_scales: *const f32,
    v_scales: *const f32,
    n_q: usize,
    n_kv: usize,
    n_in: usize,
) -> bool {
    use crate::vulkan::gemv_policy;

    // SPRINT-5 (5.2): fast path — same as `try_gpu_ternary_gemv`.
    if !gpu_gemv_known_available() {
        return false;
    }
    if !gpu_gemv_policy_active() {
        return false;
    }
    if q_w.is_null()
        || k_w.is_null()
        || v_w.is_null()
        || q_scales.is_null()
        || k_scales.is_null()
        || v_scales.is_null()
        || n_q == 0
        || n_kv == 0
        || n_in == 0
        || !n_in.is_multiple_of(8)
        || acts_f32.len() < n_in
        || q_out.len() < n_q
        || k_out.len() < n_kv
        || v_out.len() < n_kv
    {
        return false;
    }
    // Gate on largest projection (Q); K/V ride in the same CB for free.
    if !gemv_policy::should_try_gpu(n_in, n_q.max(n_kv)) {
        return false;
    }

    let Ok(mut guard) = gemv_gpu_cache().lock() else {
        return false;
    };
    if guard.ctx.as_ref().map(|c| c.is_available()) != Some(true) {
        gemv_policy::publish_no_device();
        return false;
    }
    if !guard.calibrated {
        let ctx = guard.ctx.as_mut().expect("checked");
        gemv_policy::ensure_calibrated(ctx);
        guard.calibrated = true;
        if !gemv_policy::should_try_gpu(n_in, n_q.max(n_kv)) {
            return false;
        }
    } else if !gemv_policy::should_try_gpu(n_in, n_q.max(n_kv)) {
        return false;
    }

    let blocks = n_in.div_ceil(8);
    let q_words = n_q * blocks;
    let k_words = n_kv * blocks;
    let v_words = n_kv * blocks;
    let q_p = std::slice::from_raw_parts(q_w as *const u32, q_words);
    let k_p = std::slice::from_raw_parts(k_w as *const u32, k_words);
    let v_p = std::slice::from_raw_parts(v_w as *const u32, v_words);
    let q_sc = std::slice::from_raw_parts(q_scales, n_q);
    let k_sc = std::slice::from_raw_parts(k_scales, n_kv);
    let v_sc = std::slice::from_raw_parts(v_scales, n_kv);

    let q_ptr = q_w as usize;
    let k_ptr = k_w as usize;
    let v_ptr = v_w as usize;
    let q_sc_p = q_scales as usize;
    let k_sc_p = k_scales as usize;
    let v_sc_p = v_scales as usize;
    let upload_q = guard.last_q_ptr != q_ptr
        || guard.last_q_words != q_words
        || guard.last_q_sc != q_sc_p
        || guard.last_n_q != n_q;
    let upload_k = guard.last_k_ptr != k_ptr
        || guard.last_k_words != k_words
        || guard.last_k_sc != k_sc_p
        || guard.last_n_kv != n_kv;
    let upload_v = guard.last_v_ptr != v_ptr
        || guard.last_v_words != v_words
        || guard.last_v_sc != v_sc_p
        || guard.last_n_kv != n_kv;

    let ok = {
        let ctx = guard.ctx.as_mut().expect("checked");
        ctx.dispatch_gemv_qkv_host_sync(
            &acts_f32[..n_in],
            q_p,
            k_p,
            v_p,
            q_sc,
            k_sc,
            v_sc,
            &mut q_out[..n_q],
            &mut k_out[..n_kv],
            &mut v_out[..n_kv],
            n_in,
            n_q,
            n_kv,
            upload_q,
            upload_k,
            upload_v,
        )
        .is_ok()
    };
    if ok {
        if upload_q {
            guard.last_q_ptr = q_ptr;
            guard.last_q_words = q_words;
            guard.last_q_sc = q_sc_p;
            guard.last_n_q = n_q;
        }
        if upload_k {
            guard.last_k_ptr = k_ptr;
            guard.last_k_words = k_words;
            guard.last_k_sc = k_sc_p;
            guard.last_n_kv = n_kv;
        }
        if upload_v {
            guard.last_v_ptr = v_ptr;
            guard.last_v_words = v_words;
            guard.last_v_sc = v_sc_p;
            guard.last_n_kv = n_kv;
        }
    }
    ok
}

// ── Output normalization (GPU path) ─────────────────────────────────────────────

use crate::mud::slime::SlimeWorkspace;

/// Modifies ws.registers in place. No iscale needed (f16 self-scaled).
///
/// **L-06:** when `hidden >= RMS_GPU_MIN_HIDDEN` and Vulkan is available, uses
/// `rms_norm.comp` (seq_len=1). Falls back to the CPU path otherwise.
///
/// **L-14:** if `MUD_CMUD_THINK=1`, runs complex thinking stub + wave collapse
/// on the normalized activations (research path; default off).
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn apply_output_norm(ws: &mut SlimeWorkspace, output_norm_w: *const f32, eps: f32) {
    let hidden = ws.hidden_size;
    if hidden >= crate::vulkan::ash_backend::RMS_GPU_MIN_HIDDEN
        && try_gpu_rms_output_norm(ws, output_norm_w, eps)
    {
        maybe_cmud_think(ws);
        return;
    }
    apply_output_norm_cpu(ws, output_norm_w, eps);
    maybe_cmud_think(ws);
}

/// L-14: optional complex thinking + collapse into registers (uses `ffn_out_f32` scratch).
///
/// When `MUD_CMUD_CELLV5=1`, packs C-MUD complex state into CellV5 (u32 8:8:8:8)
/// for 4× memory compression (32 KB → 16 KB, fits L1 cache). The think step
/// runs on f32 (P-02 SSOT), then results are packed back to CellV5.
/// P-02 safe — CellV5 is opt-in, never replaces f32 production path.
fn maybe_cmud_think(ws: &mut SlimeWorkspace) {
    if !crate::mud::cmud::cmud_think_enabled() {
        return;
    }
    let hidden = ws.hidden_size;
    let use_cellv5 = crate::mud::cmud::cmud_cellv5_enabled() && ws.cmud_cells.len() >= hidden;

    let scratch = &mut ws.ffn_out_f32[..hidden];
    for (i, reg) in ws.registers.iter().take(hidden).enumerate() {
        scratch[i] = reg.read_accum();
    }

    // Clean NaN/Inf from scratch before C-MUD processing (P-17: fail-fast / skip+log nulls)
    for val in scratch.iter_mut().take(hidden) {
        if !val.is_finite() {
            *val = 0.0;
        }
    }

    // Optional: pack current state to CellV5 (4× compression, L1 cache fit)
    if use_cellv5 {
        let cells = &mut ws.cmud_cells[..hidden];
        for (i, cell) in cells.iter_mut().enumerate().take(hidden) {
            let h = crate::mud::cmud::ComplexF32::from_real(scratch[i]);
            *cell = crate::mud::cmud::pack_complexf32_to_cellv5(h);
        }
    }

    let _steps = crate::mud::cmud::maybe_think_collapse(scratch, ws.mhc_radius);

    // Clean NaN/Inf after think collapse (the collapse step can introduce NaN)
    for val in scratch.iter_mut().take(hidden) {
        if !val.is_finite() {
            *val = 0.0;
        }
    }

    // Optional: unpack from CellV5 (preserves packed state for next step)
    if use_cellv5 {
        let cells = &ws.cmud_cells[..hidden];
        for (i, cell) in cells.iter().enumerate().take(hidden) {
            let h = crate::mud::cmud::unpack_cellv5_to_complexf32(*cell);
            let val = h.re;
            scratch[i] = if val.is_finite() { val } else { 0.0 };
        }
    }

    for (i, reg) in ws.registers.iter_mut().take(hidden).enumerate() {
        let val = scratch[i];
        reg.write_accum(if val.is_finite() { val } else { 0.0 });
    }
}

fn apply_output_norm_cpu(ws: &mut SlimeWorkspace, output_norm_w: *const f32, eps: f32) {
    let hidden = ws.hidden_size;
    // L-09: pointer walk over registers (no intermediate heap alloc).
    let mut sum_sq = 0.0f32;
    for reg in ws.registers.iter().take(hidden) {
        let x = reg.read_accum();
        sum_sq += x * x;
    }
    let mean_sq = sum_sq / hidden as f32;
    let rms_inv = if mean_sq.is_finite() {
        1.0f32 / (mean_sq + eps).sqrt()
    } else {
        0.0
    };
    // SAFETY: output_norm_w valid for `hidden` (caller contract).
    unsafe {
        let mut all_zero = true;
        for i in 0..hidden {
            let xn = ws.registers[i].read_accum() * rms_inv * *output_norm_w.add(i);
            let xn = if xn.is_finite() { xn } else { 0.0 };
            ws.registers[i].write_accum(xn);
            if xn != 0.0 {
                all_zero = false;
            }
        }
        if all_zero {
            log::warn!("APPLY_OUTPUT_NORM PRODUCED ALL ZEROS!");
        }
    }
}

/// Lazy ash context for optional GPU RMSNorm (L-06). Independent of QAT dispatcher.
static RMS_ASH: std::sync::OnceLock<
    std::sync::Mutex<Option<crate::vulkan::ash_backend::AshContext>>,
> = std::sync::OnceLock::new();

fn rms_ash_slot() -> &'static std::sync::Mutex<Option<crate::vulkan::ash_backend::AshContext>> {
    RMS_ASH.get_or_init(|| {
        let ctx = match crate::vulkan::ash_backend::AshContext::new() {
            Ok(c) if c.is_available() => Some(c),
            _ => None,
        };
        std::sync::Mutex::new(ctx)
    })
}

fn try_gpu_rms_output_norm(ws: &mut SlimeWorkspace, output_norm_w: *const f32, eps: f32) -> bool {
    if !use_vulkan_cached() {
        return false;
    }
    let Ok(mut guard) = rms_ash_slot().lock() else {
        return false;
    };
    let Some(ctx) = guard.as_mut() else {
        return false;
    };
    if !ctx.is_available() {
        return false;
    }
    let hidden = ws.hidden_size;
    // P-01: reuse workspace buffers instead of per-call Vec allocation.
    // This function is called per-layer; Vec allocations were hot-loop allocs.
    // ws.q_f32 / ws.k_f32 are not needed during output-norm phase.
    // SAFETY: `ws` is exclusively borrowed, so `q_f32`/`k_f32` are distinct
    // `AlignedArray<f32>` allocations of ≥ hidden elements; the `as *mut`
    // casts are the only writers inside this block (registers are read before
    // `y` is written). `output_norm_w` is the model's `output_norm.weight`
    // (shape [hidden], R-30-validated); x/w/y lengths are documented on
    // `dispatch_rms_norm_sync`, which fences before readback.
    unsafe {
        let x_ptr = ws.q_f32.as_ptr() as *mut f32;
        let y_ptr = ws.k_f32.as_ptr() as *mut f32;
        for (i, reg) in ws.registers.iter().take(hidden).enumerate() {
            *x_ptr.add(i) = reg.read_accum();
        }
        let w = std::slice::from_raw_parts(output_norm_w, hidden);
        let x_slice = std::slice::from_raw_parts(x_ptr, hidden);
        let ok = ctx.dispatch_rms_norm_sync(x_slice, w, std::slice::from_raw_parts_mut(y_ptr, hidden), hidden, 1, eps).is_ok();
        if !ok {
            return false;
        }
        for (i, reg) in ws.registers.iter_mut().take(hidden).enumerate() {
            let v = *y_ptr.add(i);
            let v = if v.is_finite() { v } else { 0.0 };
            reg.write_accum(v);
        }
    }
    true
}

/// L-06: dense multi-head causal attention via `mha.comp` when work is large enough.
/// Layout: q [seq, n_head, head_dim], k/v [seq, n_kv_head, head_dim].
/// Returns false if GPU unavailable, seq>64, or work below `MHA_GPU_MIN_WORK`.
///
/// Main decode path keeps CPU+HCA attention; this is for short prefills / tests.
#[allow(clippy::too_many_arguments)]
pub fn try_gpu_dense_mha(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    seq_len: usize,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
) -> bool {
    use crate::vulkan::ash_backend::MHA_GPU_MIN_WORK;
    if seq_len == 0 || seq_len > 64 || seq_len * n_head < MHA_GPU_MIN_WORK {
        return false;
    }
    if !use_vulkan_cached() {
        return false;
    }
    let Ok(mut guard) = rms_ash_slot().lock() else {
        return false;
    };
    let Some(ctx) = guard.as_mut() else {
        return false;
    };
    if !ctx.is_available() {
        return false;
    }
    // SAFETY: `seq_len ≤ 64` was checked above; callers pass q/out of
    // `seq*n_head*head_dim` and k/v of `seq*n_kv_head*head_dim` f32, matching
    // the layout required by `dispatch_mha_sync`'s own `# Safety`.
    unsafe {
        ctx.dispatch_mha_sync(q, k, v, out, seq_len, n_head, n_kv_head, head_dim)
            .is_ok()
    }
}

// ============================================================
// R-40: kernel-vs-escalar del head ternario (sin materializar f32)
// ============================================================
#[cfg(test)]
mod tests {
    use super::*;

    /// ELUT 4-bit pack of ternary values (nibble = value+1 in {-1,0,1} →
    /// {0x0,0x1,0x2}), matching `pack_ternary_into`'s code semantics used by
    /// the dequant LUT (0x1 → +1, 0xF → −1 in canonical rows; here we build
    /// codes through the same TERNARY_LUT path as the converter).
    fn pack_row(vals: &[i8]) -> Vec<u32> {
        let mut words = vec![0u32; vals.len().div_ceil(8)];
        for (i, &v) in vals.iter().enumerate() {
            let code = match v {
                1 => 0x1u32,
                -1 => 0xFu32,
                _ => 0x0u32,
            };
            words[i / 8] |= code << ((i % 8) * 4);
        }
        words
    }

    #[test]
    fn test_r40_ternary_head_matches_scalar() {
        let n_out = 40usize; // crosses 8/4/1-row tails
        let n_in = 24usize; // multiple of 8 and a non-multiple case below
        let mut rng = rand::rng();
        use rand::RngExt;
        let acts: Vec<f32> = (0..n_in).map(|_| rng.random_range(-2.0..2.0)).collect();
        let scales: Vec<f32> = (0..n_out).map(|_| rng.random_range(0.05..1.5)).collect();

        let mut raw: Vec<Vec<i8>> = Vec::new();
        let mut packed: Vec<u32> = Vec::new();
        let stride = n_in.div_ceil(8);
        for _ in 0..n_out {
            let row: Vec<i8> = (0..n_in).map(|_| rng.random_range(-1..=1)).collect();
            let mut pw = pack_row(&row);
            pw.resize(stride, 0);
            packed.extend_from_slice(&pw);
            raw.push(row);
        }

        let mut got = vec![0.0f32; n_out];
        unsafe {
            ternary_gemv_rowwise_submit(
                &acts,
                packed.as_ptr() as *const u8,
                &mut got,
                scales.as_ptr(),
                n_out,
                n_in,
            );
            crate::mud::pcore_pool::get_pool().wait_all();
        }

        let t = crate::mud::TERNARY_LUT;
        for v in 0..n_out {
            let expected: f32 = scales[v]
                * raw[v]
                    .iter()
                    .zip(acts.iter())
                    .map(|(&w, &x)| {
                        let code = if w == 1 {
                            1
                        } else if w == -1 {
                            15
                        } else {
                            0
                        };
                        t[code] * x
                    })
                    .sum::<f32>();
            let diff = (got[v] - expected).abs();
            let tol = 1e-4 * expected.abs().max(1.0);
            assert!(
                diff <= tol,
                "R-40 row {v}: kernel={} scalar={} diff={diff}",
                got[v],
                expected
            );
        }
    }

    #[test]
    fn test_r40_ternary_head_non_multiple_of_8() {
        // R-35 tail path: n_in = 8k+r with r != 0 must be exact too.
        let n_out = 9usize;
        let n_in = 19usize; // r = 3
        let acts: Vec<f32> = (0..n_in).map(|i| (i as f32).sin()).collect();
        let scales: Vec<f32> = (0..n_out).map(|i| 0.1 + (i as f32) * 0.05).collect();
        let stride = n_in.div_ceil(8);
        let mut raw: Vec<Vec<i8>> = Vec::new();
        let mut packed: Vec<u32> = Vec::new();
        for r in 0..n_out {
            let row: Vec<i8> = (0..n_in).map(|i| ((i + r) % 3) as i8 - 1).collect();
            let mut pw = pack_row(&row);
            pw.resize(stride, 0);
            packed.extend_from_slice(&pw);
            raw.push(row);
        }
        let mut got = vec![0.0f32; n_out];
        unsafe {
            ternary_gemv_rowwise_submit(
                &acts,
                packed.as_ptr() as *const u8,
                &mut got,
                scales.as_ptr(),
                n_out,
                n_in,
            );
            crate::mud::pcore_pool::get_pool().wait_all();
        }
        let t = crate::mud::TERNARY_LUT;
        for v in 0..n_out {
            let expected: f32 = scales[v]
                * raw[v]
                    .iter()
                    .zip(acts.iter())
                    .map(|(&w, &x)| t[if w == 1 { 1 } else if w == -1 { 15 } else { 0 }] * x)
                    .sum::<f32>();
            assert!(
                (got[v] - expected).abs() <= 1e-4 * expected.abs().max(1.0),
                "R-40 tail row {v}: kernel={} scalar={}",
                got[v],
                expected
            );
        }
    }
}

