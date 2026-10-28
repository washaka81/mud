//! SAFETY (module contract, R-36) — the `unsafe` sites in this module are
//! one of the following patterns; anything else carries its own adjacent
//! `// SAFETY:` comment:
//!
//! (a) Dereferences of raw `SlimeLayer` pointers (`q_w`/`k_w`/`v_w`/`o_w`,
//!     `ffn_up_w`/`ffn_gate_w`/`ffn_down_w`, their `*_scales`,
//!     `attn_norm_w`/`ffn_norm_w`/`attn_sub_norm_w`/`ffn_sub_norm_w`,
//!     `q_norm_w`/`k_norm_w`, `mhc_alpha_w`/`mhc_beta_w`/`mhc_radius_w`).
//!     Each points either into the payload of a `MudTensor` validated at load
//!     (R-30/R-31) or into an owned `Box<[f32]>` shadow of the layer, so the
//!     pointee outlives the forward call. Null pointers are checked before
//!     use, every index stays below the model dims passed to the kernel
//!     (`hidden`/`ffn_mid`/`head_d`/`n_out`/`n_in`), and the kernels receive
//!     element counts equal to those dims.
//! (b) Raw views / pointer arithmetic over caller-owned `SlimeWorkspace`
//!     buffers (`registers`, `gemv_accum`, `norm_i8`, `q_f32`/`k_f32`/`v_f32`,
//!     `o_act_f32`/`o_act_i8`, `ffn_up_f32`/`ffn_gate_f32`/`ffn_mid_f32`/
//!     `ffn_out_f32`, `scores`, `hca_kv_cache`, `jepa_z`, `slimex_*_scratch`,
//!     `csa_*_scratch`). The workspace is owned by the caller for the whole
//!     forward and each buffer is allocated for the model dims at
//!     construction; views (`as_ptr()`/`as_mut_ptr()`, `.add(offset)`,
//!     `[..n]`) stay within `0..buffer.len()`. The two in-place mHC residual
//!     sites (aliasing views of `registers`) carry their own `// SAFETY:` note.
//! (c) AVX2/ASM kernel calls (`ezop::*`, `asm::*`,
//!     `forge_autograd::avx_math::*`, `csa_indexer::*`). The only supported
//!     target is x86_64 with AVX2/FMA (i7-1260P) and the crate is built with
//!     `-C target-cpu=native` (`.cargo/config.toml`, `mud.sh`), so those
//!     instructions are guaranteed for this binary; dispatching sites
//!     additionally check `is_x86_feature_detected!`. Buffer/length arguments
//!     are exactly the (a)/(b) extents.
//!
//! Sites outside (a)-(c) (`unsafe impl Send/Sync`, `ExpertBus::forward`, the
//! CSA `Vec::from_raw_parts` views) carry their own adjacent `// SAFETY:`
//! comments.

use crate::mud::slime::{SlimeRegister, SlimeWorkspace};
use crate::mud::slime_jepa::jepa_stabilizer;
use crate::mud::ezop;
use std::sync::atomic::{AtomicU64, Ordering};

// ── BUG-2 (V25 audit): NaN/zero RMSNorm hot-path counters. ──────────────────
// Replaces 7 `panic!` sites that aborted the process with SIGABRT. Now they
// log once per ~64k events and increment an AtomicU64 so the engine survives
// a corrupt .mud without killing the inference / training session.
static NAN_RMSNORM_ATTN_COUNT: AtomicU64 = AtomicU64::new(0);
static DEAD_RMSNORM_ATTN_COUNT: AtomicU64 = AtomicU64::new(0);
static NAN_BEFORE_STEP5_COUNT: AtomicU64 = AtomicU64::new(0);
static NAN_ACT_SCALE_STEP5_COUNT: AtomicU64 = AtomicU64::new(0);
static NAN_AFTER_O_GEMV_COUNT: AtomicU64 = AtomicU64::new(0);
static NAN_BEFORE_STEP7_COUNT: AtomicU64 = AtomicU64::new(0);

#[inline]
fn nan_skip(counter: &AtomicU64, label: &'static str, layer_idx: usize, pos: usize) {
    let c = counter.fetch_add(1, Ordering::Relaxed) + 1;
    // Throttle: log first 4, then every 64k to avoid TUI spam.
    // SEC-H6 (2026-07-27): migrated from eprintln! to log::warn! so the
    // NaN-guard diagnostics flow through the structured logger. When no
    // logger is installed (tests, library usage) these are no-ops; in
    // production the env_logger in main.rs surfaces them at WARN level.
    if c <= 4 || c & 0xFFFF == 0 {
        log::warn!(
            "[NaN-GUARD] {} layer={} pos={} count={} (continuing with degraded output)",
            label, layer_idx, pos, c
        );
    }
}

// ── PERF-2 (V25 audit) + SPRINT-5 (5.2): env caches + GPU dispatch.
// Moved to gemv_gpu module. Internal helpers + re-exports below.
use crate::mud::gemv_gpu::{try_gpu_ternary_gemv};
pub use crate::mud::gemv_gpu::{apply_output_norm, try_gpu_dense_mha};
pub(crate) use crate::mud::gemv_gpu::{ternary_gemv_qkv_parallel, ternary_gemv_rowwise_submit};

// ── PERF-1 (V25 audit): precompute inv_freq. Moved to rope_cache module. ───────
pub use crate::mud::rope_cache::{compute_rope_inv_freq, RopeCache};

fn slime_rmsnorm_i8(
    regs: &[SlimeRegister],
    gemv_accum: &mut [f32],
    weights: *const f32,
    out_i8: &mut [i8],
    eps: f32,
) -> f32 {
    let n = regs.len();
    // Phase 1: extract all f32 accum values from registers into gemv_accum.
    // read_accum() uses F16C (fast) — one register read per element.
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        gemv_accum[i] = regs[i].read_accum();
    }
    // Phase 2: AVX2 sum_sq (8 f32/iter via FMA) — replaces scalar f64 loop.
    let sum_sq = if n >= 8 {
        unsafe { ezop::sum_sq(gemv_accum.as_ptr(), n) }
    } else {
        let mut s = 0.0f32;
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let v = gemv_accum[i];
            s += v * v;
        }
        s
    };
    let rms_inv = 1.0f32 / ((sum_sq / n as f32) + eps).sqrt();

    // Phase 3: scale each value by rms_inv * weight, track peak abs (AVX2).
    // Reuses gemv_accum as both input and output (in-place scaling + peak find).
    #[cfg(target_arch = "x86_64")]
    let peak = if n >= 8 && is_x86_feature_detected!("avx2") {
        unsafe {
            use std::arch::x86_64::*;
            let gp = gemv_accum.as_mut_ptr();
            let wp = weights;
            let mut max_v = 0.0f32;
            let mut i = 0;
            let rms_inv_v = _mm256_set1_ps(rms_inv);
            let sign_mask = _mm256_set1_ps(-0.0);
            while i + 8 <= n {
                let gv = _mm256_loadu_ps(gp.add(i));
                let wv = _mm256_loadu_ps(wp.add(i));
                let scaled = _mm256_mul_ps(gv, _mm256_mul_ps(rms_inv_v, wv));
                _mm256_storeu_ps(gp.add(i), scaled);
                // abs via sign-bit clear
                let absv = _mm256_andnot_ps(sign_mask, scaled);
                // Horizontal max: 256-bit → 128-bit → scalar
                let hi = _mm256_extractf128_ps(absv, 1);
                let lo = _mm256_castps256_ps128(absv);
                let m128 = _mm_max_ps(lo, hi); // 4 elements
                let hi64 = _mm_movehl_ps(m128, m128); // upper 2 → lower 2
                let m128 = _mm_max_ps(m128, hi64);
                let lo64 = _mm_shuffle_ps(m128, m128, 1); // swap elements 0↔1
                let m128 = _mm_max_ps(m128, lo64);
                max_v = max_v.max(_mm_cvtss_f32(m128));
                i += 8;
            }
            // Scalar remainder
            #[allow(clippy::needless_range_loop)]
            for j in i..n {
                let xn = gemv_accum[j] * rms_inv * *wp.add(j);
                gemv_accum[j] = xn;
                let av = xn.abs();
                if av > max_v { max_v = av; }
            }
            max_v
        }
    } else {
        let mut peak = 0.0f32;
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let xn = gemv_accum[i] * rms_inv * unsafe { *weights.add(i) };
            gemv_accum[i] = xn;
            let av = xn.abs();
            if av > peak { peak = av; }
        }
        peak
    };

    #[cfg(not(target_arch = "x86_64"))]
    let peak = {
        let mut peak = 0.0f32;
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let xn = gemv_accum[i] * rms_inv * unsafe { *weights.add(i) };
            gemv_accum[i] = xn;
            let av = xn.abs();
            if av > peak { peak = av; }
        }
        peak
    };

    let peak = peak.max(crate::mud::constants::EPSILON_FLOOR);
    let inv_peak = 127.0 / peak;
    for i in 0..n {
        out_i8[i] = (gemv_accum[i] * inv_peak).clamp(-127.0, 127.0) as i8;
    }
    peak / 127.0
}

// Manual Clone impl below (parking_lot::Mutex doesn't impl Clone).
pub struct SlimeLayer {
    pub q_w: *const u8,
    pub k_w: *const u8,
    pub v_w: *const u8,
    pub o_w: *const u8,
    pub q_scales: *const f32,
    pub k_scales: *const f32,
    pub v_scales: *const f32,
    pub o_scales: *const f32,

    pub ffn_up_w: *const u8,
    pub ffn_gate_w: *const u8,
    pub ffn_down_w: *const u8,
    pub ffn_up_scales: *const f32,
    pub ffn_gate_scales: *const f32,
    pub ffn_down_scales: *const f32,

    pub attn_norm_w: *const f32,
    pub ffn_norm_w: *const f32,
    pub attn_sub_norm_w: *const f32,
    pub ffn_sub_norm_w: *const f32,

    /// Qwen3-style per-head RMSNorm on Q (shape `[head_dim]`); null = skip.
    pub q_norm_w: *const f32,
    /// Qwen3-style per-head RMSNorm on K (shape `[head_dim]`); null = skip.
    pub k_norm_w: *const f32,

    pub mhc_alpha_w: *const f32,
    pub mhc_beta_w: *const f32,
    pub mhc_radius_w: *const f32,
    /// OWNED writable shadow for mHC alpha/beta (mhc_alpha_w/mhc_beta_w point
    /// into these when the source is mmap'd read-only). Without this, the
    /// optimizer's `mhc_scale_sgd_step` writes UB (segfault) to the mmap
    /// region. Radius is read-only (never updated) so no shadow needed.
    pub mhc_alpha_shadow: Option<Box<[f32]>>,
    pub mhc_beta_shadow: Option<Box<[f32]>>,

    /// OWNED writable shadow for RMSNorm weight vectors (attn/ffn).
    /// When the source is mmap'd read-only, `attn_norm_w`/`ffn_norm_w` point
    /// into these shadows so the optimizer can write updated F32 values
    /// without SIGSEGV on the read-only mmap region.
    /// FIX (2026-07-31): Previously these were frozen at pretrained values —
    /// the root cause of activation collapse (cog→0, varh→0) after ~5000 steps.
    pub attn_norm_shadow: Option<Box<[f32]>>,
    pub ffn_norm_shadow: Option<Box<[f32]>>,

    pub n_kv_heads: usize,
    pub ffn_mid: usize,
    /// RoPE base frequency (0.0 = disable RoPE).
    pub rope_theta: f32,
    /// PERF-1: precomputed `inv_freq[i] = rope_theta^(-2i/head_d)` for the recurrence
    /// `theta[i+1] = theta[i] * inv_freq[i]`. Length = head_d / 2. Empty when RoPE disabled.
    /// Computed once in `compute_rope_inv_freq` at layer construction.
    pub rope_inv_freq: Vec<f32>,
    /// PERF-1: per-position RoPE cos/sin cache (16-entry ring). Pre-computed on miss
    /// (O(half_d) sin_cos), reused for all heads at the same pos. Lock-free access
    /// (SlimeLayer is not shared across threads during a single forward).
    pub rope_cache: parking_lot::Mutex<RopeCache>,
    /// When true, q_w/k_w/v_w/o_w/ffn_*_w point to F32 data (not ELUT ternary).
    /// Scales pointers are ignored. Used for Q2_0→F32 converted models.
    pub is_f32_weights: bool,
}

// SAFETY (SEC-H3 audit 2026-07-27): SlimeLayer holds 20 `*const` weight/scale
// pointers that are immutable shared views into mmap-backed or owned tensor
// data (kept alive by the parent MudFile for the layer's lifetime). The only
// interior-mutability field is `rope_cache: parking_lot::Mutex<RopeCache>`,
// and Mutex<T> is already Send+Sync when T: Send. All pointer fields are
// `*const` (read-only); mutation of weights happens in the optimizer which
// operates on separate shadow tensors (SlimeLayerShadowF32), never through
// `&SlimeLayer`. A forward pass uses `&SlimeLayer` from a single task; the
// Send+Sync impl allows moving the layer across threads (e.g. PCorePool
// worker) but does not introduce data races because every pointer is const
// and the Mutex is sound. Therefore Send+Sync is sound.
// SAFETY: see the SEC-H3 note above — every pointer field is `*const` and its
// pointee is kept alive by the parent `MudFile` (or is a layer-owned shadow)
// for the layer's lifetime; the only interior mutability is the
// `parking_lot::Mutex` (Send+Sync when its contents are), and no safe API
// hands out `&mut` to the weight data through `&SlimeLayer`, so moving or
// sharing the layer across threads cannot cause a data race.
unsafe impl Send for SlimeLayer {}
unsafe impl Sync for SlimeLayer {}

// Manual Clone: parking_lot::Mutex<T> doesn't derive Clone.
impl Clone for SlimeLayer {
    fn clone(&self) -> Self {
        Self {
            q_w: self.q_w,
            k_w: self.k_w,
            v_w: self.v_w,
            o_w: self.o_w,
            q_scales: self.q_scales,
            k_scales: self.k_scales,
            v_scales: self.v_scales,
            o_scales: self.o_scales,
            ffn_up_w: self.ffn_up_w,
            ffn_gate_w: self.ffn_gate_w,
            ffn_down_w: self.ffn_down_w,
            ffn_up_scales: self.ffn_up_scales,
            ffn_gate_scales: self.ffn_gate_scales,
            ffn_down_scales: self.ffn_down_scales,
            attn_norm_w: self.attn_norm_w,
            ffn_norm_w: self.ffn_norm_w,
            attn_sub_norm_w: self.attn_sub_norm_w,
            ffn_sub_norm_w: self.ffn_sub_norm_w,
            q_norm_w: self.q_norm_w,
            k_norm_w: self.k_norm_w,
            mhc_alpha_w: self.mhc_alpha_w,
            mhc_beta_w: self.mhc_beta_w,
            mhc_radius_w: self.mhc_radius_w,
            mhc_alpha_shadow: None,
            mhc_beta_shadow: None,
            attn_norm_shadow: None,
            ffn_norm_shadow: None,
            n_kv_heads: self.n_kv_heads,
            ffn_mid: self.ffn_mid,
            rope_theta: self.rope_theta,
            rope_inv_freq: self.rope_inv_freq.clone(),
            // Fresh per-layer RopeCache (no shared mutable state).
            rope_cache: parking_lot::Mutex::new(RopeCache::new()),
            is_f32_weights: self.is_f32_weights,
        }
    }
}

/// Shared ternary GEMV (ELUT 4-bit + PRQ) for dense layers and Mini MoE experts (L-11).
///
/// **Stream C / Phase B+:** GPU path is policy-driven ([`crate::vulkan::gemv_policy`]):
/// - `MUD_GPU_GEMV=0` → always CPU
/// - `MUD_GPU_GEMV=1` → GPU when work ≥ min (`GEMV_GPU_MIN_WORK` or `MUD_GPU_GEMV_MIN`)
/// - `MUD_GPU_GEMV=auto` / unset → one-shot CPU vs GPU micro-bench; GPU only past break-even
///
/// Weight upload is cached by host pointer. Falls back to AVX2 on any GPU failure.
///
/// # Safety
/// `w_u8` / `scales` must cover `n_out` rows × `n_in` inputs (8 weights/u32).
pub(crate) unsafe fn ternary_gemv_rowwise(
    acts_f32: &[f32],
    w_u8: *const u8,
    out_f32: &mut [f32],
    scales: *const f32,
    n_out: usize,
    n_in: usize,
) {
    if try_gpu_ternary_gemv(acts_f32, w_u8, out_f32, scales, n_out, n_in) {
        return;
    }
    ternary_gemv_rowwise_submit(acts_f32, w_u8, out_f32, scales, n_out, n_in);
    crate::mud::pcore_pool::get_pool().wait_all();
}

/// F32 dense GEMV for models with dequantized F32 weights (Q2_0→F32 conversion).
/// out[row] = dot(acts, w_row) for each row.
///
/// # Safety
/// `w_f32_ptr` must point to `n_out * n_in` f32 values (row-major).
pub(crate) unsafe fn f32_gemv_rowwise(
    acts_f32: &[f32],
    w_f32_ptr: *const u8,
    out_f32: &mut [f32],
    n_out: usize,
    n_in: usize,
) {
    f32_gemv_submit(acts_f32, w_f32_ptr, out_f32, n_out, n_in);
    crate::mud::pcore_pool::get_pool().wait_all();
}

/// F32 dense GEMV — submit only (does NOT call `wait_all`).
/// Caller must call `pcore_pool::get_pool().wait_all()` after all submissions.
///
/// Fix (2026-08-03): Allows overlapping up+gate FFN projections by submitting
/// both GEMVs before waiting once (matching the ternary path's Phase B overlap).
/// For F32 models: 2× dispatch→wait cycles become 1× dispatch×2→1× wait.
///
/// # Safety
/// Same as [`f32_gemv_rowwise`]. Do not free buffers until after `wait_all`.
pub(crate) unsafe fn f32_gemv_submit(
    acts_f32: &[f32],
    w_f32_ptr: *const u8,
    out_f32: &mut [f32],
    n_out: usize,
    n_in: usize,
) {
    let w = w_f32_ptr as *const f32;

    // Small-matrix fast path — inline for n_out < 256 to avoid PCorePool
    // dispatch overhead (~0.5ms) that dominates compute (~2μs) on small models
    // (pico/nano: Q/K/V projections at n_out=128).
    const SMALL_GEMV_THRESHOLD: usize = 256;
    if n_out < SMALL_GEMV_THRESHOLD {
        for r in 0..n_out {
            let row_ptr = w.add(r * n_in);
            let mut val = 0.0f32;
            for (c, a) in acts_f32.iter().enumerate().take(n_in) {
                val += a * *row_ptr.add(c);
            }
            *out_f32.as_mut_ptr().add(r) = val;
        }
        return;
    }

    let pool = crate::mud::pcore_pool::get_pool();
    let n_tasks = pool.num_threads().max(1);
    let rows_per_task = (n_out / n_tasks).max(1);

    let acts_p = acts_f32.as_ptr() as usize;
    let w_p = w as usize;
    let out_p = out_f32.as_mut_ptr() as usize;

    for i in 0..n_tasks {
        let start = i * rows_per_task;
        let end = if i + 1 == n_tasks { n_out } else { (start + rows_per_task).min(n_out) };
        if start >= end { break; }

        pool.execute(move || {
            let acts = acts_p as *const f32;
            let w = w_p as *const f32;
            let out = out_p as *mut f32;

            for r in start..end {
                let row_ptr = unsafe { w.add(r * n_in) };
                let mut val = 0.0f32;
                for c in 0..n_in {
                    val += unsafe { *acts.add(c) } * unsafe { *row_ptr.add(c) };
                }
                unsafe { *out.add(r) = val };
            }
        });
    }
}

/// F32 GEMM backward (dequantized Q2_0→F32 models).
///
/// Computes both the weight gradient (outer product, dL/dW = grad_y ⊗ x^T)
/// and the input gradient (GEMV, dL/dx = grad_y^T @ W), accumulating into
/// the provided grad buffers.
///
/// # Safety
/// `w` must point to `n_out × n_in` f32 values (row-major).
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn f32_gemv_backward(
    grad_y: &[f32],
    x: &[f32],
    w: *const f32,
    grad_w: &mut [f32],
    grad_x: &mut [f32],
    scratch: &mut [f32],
    n_out: usize,
    n_in: usize,
) {
    let _ = scratch; // reserved for future batched GEMM tiling
    let _ = n_out; // derivable from grad_y.len()
    let _ = n_in; // derivable from x.len() / grad_w row stride
    let w_ptr = w;

    // dL/dW = grad_y ⊗ x^T (outer product). Each grad_w[out] += grad_y[out] * x[in].
    for (out, &gy) in grad_y.iter().enumerate() {
        if gy == 0.0 {
            continue;
        }
        let gw_row = grad_w.as_mut_ptr().add(out * n_in);
        let x_ptr = x.as_ptr();
        for c in 0..n_in {
            *gw_row.add(c) += gy * *x_ptr.add(c);
        }
    }

    // dL/dx = grad_y^T @ W: grad_x[in] += sum over out of grad_y[out] * w[out*n_in + in].
    // Callers zero grad_x beforehand; this function accumulates.
    let gx_ptr = grad_x.as_mut_ptr();
    for (out, &gy) in grad_y.iter().enumerate() {
        if gy == 0.0 {
            continue;
        }
        let w_row = w_ptr.add(out * n_in);
        for c in 0..n_in {
            *gx_ptr.add(c) += gy * *w_row.add(c);
        }
    }
}

/// Batched F32 GEMV backward for Q/K/V projections (dL/dx = grad_y^T @ W).
/// Accumulates grad_x from all three projections.
///
/// # Safety
/// Same pointer contracts as [`f32_gemv_backward`].
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn f32_gemv_backward_batch(
    grad_q_y: &[f32],
    grad_k_y: &[f32],
    grad_v_y: &[f32],
    x: &[f32],
    q_w: *const f32,
    k_w: *const f32,
    v_w: *const f32,
    grad_q_w: &mut [f32],
    grad_k_w: &mut [f32],
    grad_v_w: &mut [f32],
    grad_x: &mut [f32],
    scratch: &mut [f32],
    n_q: usize,
    n_kv: usize,
    n_in: usize,
) {
    // Q backward
    f32_gemv_backward(grad_q_y, x, q_w, grad_q_w, grad_x, scratch, n_q, n_in);
    // K backward (accumulate grad_x)
    f32_gemv_backward(grad_k_y, x, k_w, grad_k_w, grad_x, scratch, n_kv, n_in);
    // V backward (accumulate grad_x)
    f32_gemv_backward(grad_v_y, x, v_w, grad_v_w, grad_x, scratch, n_kv, n_in);
}

// ── Phase B+ / Stream C: GPU GEMV dispatch moved to gemv_gpu module. ────────────
// Internal helpers re-exported via gemv_gpu::* imports.

/// Phase 1-3: Manifold-Constrained Hyper-Connections (mHC)
/// Geometrically bounds the residual stream: h_next = proj_manifold(α·gate·h + (1-gate)·β·f(h), radius)
/// Gate = sigmoid(JEPA_integral) — smooth, low-pass filtered controller.
/// Guarantees ||h|| ≤ radius structurally.
#[inline(always)]
fn mhc_residual(
    h_in: &[SlimeRegister],
    h_out: &mut [SlimeRegister],
    f_h: &[f32],
    radius: f32,
    alpha_w: *const f32,
    beta_w: *const f32,
) {
    let n = h_in.len();

    // Phase 1: compute val = alpha·h_in + (1-gate)·beta·f_h per element.
    // Phase 2: track max_abs for projection.
    // Phase 3: mHC radius projection (always applied — P-17 safety bound).
    // FIX (2026-07-30): projection was skipped entirely when alpha_w was null
    // (base model path), causing unbounded h_in + f_h growth → f16 overflow.
    let has_mhc = !alpha_w.is_null();
    let mut max_abs = 0.0f32;

    // Branchless dispatch: hoist `has_mhc` outside the per-element loop.
    // Eliminates 3 conditional branches per element (alpha/beta/gate).
    // For n=2048, that's 6144 eliminated branches per mHC call.
    if has_mhc {
        // mHC-enabled path: read alpha/beta/gate per element from pointers
        for i in 0..n {
            let alpha = unsafe { *alpha_w.add(i) };
            let beta = unsafe { *beta_w.add(i) };
            let gate = h_in[i].gate();
            let val = alpha * h_in[i].read_accum() + (1.0 - gate) * beta * f_h[i];
            // P-17: fail-fast on non-finite
            let val = if val.is_finite() { val } else { 0.0 };
            h_out[i].write_accum(val);
            h_out[i].write_integral(h_in[i].read_integral());
            let av = val.abs();
            if av > max_abs {
                max_abs = av;
            }
        }
    } else {
        // Base model path: alpha=1.0, beta=1.0, gate=0.0 (all constants,
        // compiler can fully optimize the arithmetic — no pointer loads)
        for i in 0..n {
            let val = h_in[i].read_accum() + f_h[i];
            let val = if val.is_finite() { val } else { 0.0 };
            h_out[i].write_accum(val);
            h_out[i].write_integral(h_in[i].read_integral());
            let av = val.abs();
            if av > max_abs {
                max_abs = av;
            }
        }
    }

    // Phase 3: mHC projection — cap L2 norm to `radius`.
    // ALWAYS applied (even base models) to prevent unbounded residual growth.
    // For base models the fallback radius is ws.mhc_radius (≈ 1.5·√hidden),
    // passed in from the call site.
    if max_abs > 0.0 && max_abs.is_finite() {
        let mut sum_sq = 0.0f32;
        let scale_down = 1.0 / max_abs;
        for h in h_out.iter().take(n) {
            let v = h.read_accum() * scale_down;
            sum_sq += v * v;
        }
        let norm = (sum_sq.sqrt() * max_abs).max(1e-8);
        if norm > radius {
            let scale = radius / norm;
            for h in h_out.iter_mut() {
                let scaled = h.read_accum() * scale;
                // Guard against overflow in the second write_accum
                let scaled = if scaled.is_finite() { scaled } else { 0.0 };
                h.write_accum(scaled);
            }
        }
    }
}

pub fn layer_is_valid(layer: &SlimeLayer) -> bool {
    if layer.is_f32_weights {
        // F32 path: scales are null (no PRQ tensors), only weights + norms needed
        !layer.q_w.is_null()
            && !layer.k_w.is_null()
            && !layer.v_w.is_null()
            && !layer.o_w.is_null()
            && !layer.ffn_up_w.is_null()
            && !layer.ffn_gate_w.is_null()
            && !layer.ffn_down_w.is_null()
            && !layer.attn_norm_w.is_null()
            && !layer.ffn_norm_w.is_null()
    } else {
        !layer.q_w.is_null()
            && !layer.k_w.is_null()
            && !layer.v_w.is_null()
            && !layer.o_w.is_null()
            && !layer.q_scales.is_null()
            && !layer.k_scales.is_null()
            && !layer.v_scales.is_null()
            && !layer.o_scales.is_null()
            && !layer.ffn_up_w.is_null()
            && !layer.ffn_gate_w.is_null()
            && !layer.ffn_down_w.is_null()
            && !layer.ffn_up_scales.is_null()
            && !layer.ffn_gate_scales.is_null()
            && !layer.ffn_down_scales.is_null()
            && !layer.attn_norm_w.is_null()
            && !layer.ffn_norm_w.is_null()
    }
}

/// One transformer layer: RMSNorm → QKV → Attention → O → Residual+JEPA → FFN → Residual+JEPA
/// Uses FP32 ternary_gemv (AVX2) via dequantized i8→f32 activations.
/// Zero-allocation via SlimeWorkspace pre-allocated buffers.
/// Dense SwiGLU FFN (steps 9–12) shared by evaluate path and MoE fallback.
fn dense_ffn_swiglu(
    layer: &SlimeLayer,
    ws: &mut SlimeWorkspace,
    hidden: usize,
    ffn_mid: usize,
    tape: &mut Option<&mut crate::mud::slime_backward::SlimeLayerTape>,
    eps: f32,
) {
    // Input already in ws.ffn_out_f32[..hidden]
    // Phase B: overlap up + gate GEMVs (one pool barrier instead of two)
    // Fix (2026-08-03): F32 path was sequential (each call dispatched + waited).
    // Now uses f32_gemv_submit (no-wait) to overlap both projections, matching
    // the ternary path's Phase B pattern. Single wait_all after both submits.
    unsafe {
        if layer.is_f32_weights {
            f32_gemv_submit(
                &ws.ffn_out_f32[..hidden], layer.ffn_up_w, &mut ws.ffn_up_f32, ffn_mid, hidden
            );
            f32_gemv_submit(
                &ws.ffn_out_f32[..hidden], layer.ffn_gate_w, &mut ws.ffn_gate_f32, ffn_mid, hidden
            );
            crate::mud::pcore_pool::get_pool().wait_all();
        } else {
            ternary_gemv_rowwise_submit(
                &ws.ffn_out_f32[..hidden],
                layer.ffn_up_w,
                &mut ws.ffn_up_f32,
                layer.ffn_up_scales,
                ffn_mid,
                hidden,
            );
            ternary_gemv_rowwise_submit(
                &ws.ffn_out_f32[..hidden],
                layer.ffn_gate_w,
                &mut ws.ffn_gate_f32,
                layer.ffn_gate_scales,
                ffn_mid,
                hidden,
            );
            crate::mud::pcore_pool::get_pool().wait_all();
        }
    }

    if let Some(t) = tape.as_mut() {
        t.ffn_up_f32.copy_from_slice(&ws.ffn_up_f32[..ffn_mid]);
        t.ffn_gate_f32.copy_from_slice(&ws.ffn_gate_f32[..ffn_mid]);
    }

    unsafe {
        crate::asm::silu_vectorial_avx2(
            ffn_mid,
            ws.ffn_gate_f32.as_ptr(),
            ws.ffn_mid_f32.as_mut_ptr(),
        );
    }
    {
        let mid = &mut ws.ffn_mid_f32[..ffn_mid];
        let up = &ws.ffn_up_f32[..ffn_mid];
        // AVX2 element-wise multiply: mid[i] *= up[i] (8 f32/iter)
        if ffn_mid >= 8 && std::is_x86_feature_detected!("avx2") {
            unsafe {
                crate::mud::ezop::vmul_inplace_avx2(up.as_ptr(), mid.as_mut_ptr(), ffn_mid);
            }
        } else {
            for i in 0..ffn_mid {
                mid[i] *= up[i];
            }
        }
    }

    if let Some(t) = tape.as_mut() {
        t.ffn_mid_f32
            .copy_from_slice(&ws.ffn_mid_f32[..layer.ffn_mid]);
    }

    if !layer.ffn_sub_norm_w.is_null() {
        unsafe {
            let mid_ptr = ws.ffn_mid_f32.as_mut_ptr();
            // AVX2 sum_sq (8 f32/iter via FMA)
            let sum_sq = if ffn_mid >= 8 {
                ezop::sum_sq(mid_ptr, ffn_mid)
            } else {
                let mut s = 0.0f32;
                for i in 0..ffn_mid {
                    let x = *mid_ptr.add(i);
                    s += x * x;
                }
                s
            };
            let rms = (sum_sq / ffn_mid as f32 + eps).sqrt();
            let inv_rms = 1.0 / rms;
            // AVX2 scale with per-element weight: mid[i] *= inv_rms * w[i]
            if ffn_mid >= 8 {
                ezop::scale_with_weight_avx2(mid_ptr, layer.ffn_sub_norm_w, inv_rms, ffn_mid);
            } else {
                for i in 0..ffn_mid {
                    *mid_ptr.add(i) = *mid_ptr.add(i) * inv_rms * *layer.ffn_sub_norm_w.add(i);
                }
            }
        }
    }

    let ffn_hid_peak = {
        // AVX2 peak abs (8 f32/iter)
        let p = if ffn_mid >= 8 {
            unsafe { ezop::peak_abs_avx2(ws.ffn_mid_f32.as_ptr(), ffn_mid) }
        } else {
            let mut peak = 0.0f32;
            for i in 0..ffn_mid {
                peak = peak.max(ws.ffn_mid_f32[i].abs());
            }
            peak
        };
        (p / 127.0f32).max(1e-8f32)
    };
    // FIX (2026-08-03): Vectorize i8 quantization (8 lanes/iter via AVX2+FMA).
    // Replaces scalar loop that dominated the FFN mid computation on small models.
    if ffn_mid >= 8 && std::is_x86_feature_detected!("avx2") {
        unsafe {
            ezop::quantize_i8_avx2(
                ws.ffn_mid_f32.as_ptr(),
                ws.o_act_i8.as_mut_ptr(),
                ffn_hid_peak,
                ffn_mid,
            );
        }
    } else {
        for i in 0..ffn_mid {
            ws.o_act_i8[i] =
                (ws.ffn_mid_f32[i] / ffn_hid_peak.clamp(1e-8, 1e8)).clamp(-127.0, 127.0) as i8;
        }
    }

    // Dequantize o_act_i8 → ffn_gate_f32 (AVX2, 8 i8→f32/iter)
    if is_x86_feature_detected!("avx2") && ffn_mid >= 8 {
        unsafe {
            ezop::scale_i8_to_f32_avx2(
                ws.o_act_i8.as_ptr(),
                ws.ffn_gate_f32.as_mut_ptr(),
                ffn_hid_peak,
                ffn_mid,
            );
        }
    } else {
        for i in 0..ffn_mid {
            ws.ffn_gate_f32[i] = ws.o_act_i8[i] as f32 * ffn_hid_peak;
        }
    }
    unsafe {
        if layer.is_f32_weights {
            f32_gemv_rowwise(&ws.ffn_gate_f32[..ffn_mid], layer.ffn_down_w, &mut ws.ffn_out_f32, hidden, ffn_mid);
        } else {
            ternary_gemv_rowwise(
                &ws.ffn_gate_f32[..ffn_mid],
                layer.ffn_down_w,
                &mut ws.ffn_out_f32,
                layer.ffn_down_scales,
                hidden,
                ffn_mid,
            );
        }
    }
}

/// Dense FFN by default. For Mini MoE (L-11) use [`evaluate_slime_block_moe`].
pub fn evaluate_slime_block(
    layer: &SlimeLayer,
    layer_idx: usize,
    ws: &mut SlimeWorkspace,
    pos: usize,
    eps: f32,
    tape: Option<&mut crate::mud::slime_backward::SlimeLayerTape>,
    gpu: Option<&mut crate::mud::ash_qat_dispatcher::AshQatDispatcher>,
) -> crate::mud::slime_jepa::TensorDiagnostics {
    evaluate_slime_block_moe(layer, layer_idx, ws, pos, eps, tape, None, None, gpu)
}

/// L-11: same as [`evaluate_slime_block`] with optional ExpertBus FFN.
/// When `moe` is None/empty → dense layer FFN (C7 compatibility).
#[allow(clippy::too_many_arguments, clippy::needless_option_as_deref)]
pub fn evaluate_slime_block_moe(
    layer: &SlimeLayer,
    layer_idx: usize,
    ws: &mut SlimeWorkspace,
    pos: usize,
    eps: f32,
    mut tape: Option<&mut crate::mud::slime_backward::SlimeLayerTape>,
    moe: Option<&crate::mud::expert_bus::ExpertBus>,
    moe_scratch: Option<&mut crate::mud::expert_bus::ExpertScratch>,
    mut gpu: Option<&mut crate::mud::ash_qat_dispatcher::AshQatDispatcher>,
) -> crate::mud::slime_jepa::TensorDiagnostics {
    debug_assert!(layer_is_valid(layer));
    let hidden = ws.hidden_size;
    let n_heads = ws.n_heads;
    let n_kv_heads = layer.n_kv_heads;
    let head_d = ws.head_dim;
    // L-13: dense ring uses dense_kv_cap, not full logical max_pos
    debug_assert!(pos < ws.max_pos, "pos {pos} >= max_pos {}", ws.max_pos);
    let dense_cap = ws.dense_kv_cap.max(1);
    let kv_offset = dense_cap * head_d;
    let layer_offset = layer_idx * (n_kv_heads * kv_offset);

    // HCA compressed history slots (L-13)
    let hca_max_pos = ws.hca_slots.max(1);
    let hca_kv_offset = hca_max_pos * head_d;
    let hca_layer_offset = layer_idx * (n_kv_heads * hca_kv_offset);
    // ── Step 1: RMSNorm → f32 (R-43: no i8 round-trip) ──
    // slime_rmsnorm_i8 still runs to compute the normalized f32 in gemv_accum
    // (and the i8 quantization is discarded). We copy gemv_accum → ffn_out_f32
    // directly, skipping the i8→f32 dequantization.
    let act_scale = slime_rmsnorm_i8(
        &ws.registers[..hidden],
        &mut ws.gemv_accum,
        layer.attn_norm_w,
        &mut ws.norm_i8,
        eps,
    );

    // R-43: store f32 directly, skip i8 round-trip.
    ws.ffn_out_f32[..hidden].copy_from_slice(&ws.gemv_accum[..hidden]);

    if let Some(t) = tape.as_mut() {
        t.norm_f32_attn[..hidden].copy_from_slice(&ws.ffn_out_f32[..hidden]);
        t.attn_act_scale = 1.0; // no quantization
    }

    if !act_scale.is_finite() {
        nan_skip(&NAN_RMSNORM_ATTN_COUNT, "NaN act_scale from RMSNorm attn", layer_idx, pos);
    }
    // R-43: compute peak from f32 gemv_accum (no i8).
    let peak_f32 = ws.gemv_accum[..hidden]
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, f32::max);
    if peak_f32 < 1e-7 && act_scale < 1e-7 {
        // Diagnostic scan — values used only for the throttled log inside `nan_skip`.
        let mut _nz_regs: i32 = 0;
        let mut _first_nz: usize = hidden;
        for i in 0..hidden {
            if ws.registers[i].read_accum() != 0.0 {
                _nz_regs += 1;
                if _first_nz == hidden {
                    _first_nz = i;
                }
            }
        }
        nan_skip(
            &DEAD_RMSNORM_ATTN_COUNT,
            "Dead RMSNorm (peak_norm=0)",
            layer_idx,
            pos,
        );
    }

    // ── Step 2: ffn_out_f32 already populated (R-43: no i8 round-trip) ──
    // The i8→f32 dequantization is eliminated; ffn_out_f32 holds the RMSNorm
    // result directly from gemv_accum.
    unsafe {
        if layer.is_f32_weights {
            // F32 path: overlap Q/K/V projections (submit 3, wait once).
            // Fix (2026-08-03): was 3 sequential f32_gemv_rowwise calls
            // (each dispatches + waits). Now f32_gemv_submit overlaps all 3.
            f32_gemv_submit(&ws.ffn_out_f32[..hidden], layer.q_w, &mut ws.q_f32, hidden, hidden);
            f32_gemv_submit(&ws.ffn_out_f32[..hidden], layer.k_w, &mut ws.k_f32, n_kv_heads * head_d, hidden);
            f32_gemv_submit(&ws.ffn_out_f32[..hidden], layer.v_w, &mut ws.v_f32, n_kv_heads * head_d, hidden);
            crate::mud::pcore_pool::get_pool().wait_all();
            if std::env::var("MUD_DEBUG_LAYERS").as_deref() == Ok("1") && layer_idx == 0 {
                let qn: f32 = ws.q_f32[..hidden].iter().map(|v| v*v).sum::<f32>().sqrt();
                let kn: f32 = ws.k_f32[..n_kv_heads*head_d].iter().map(|v| v*v).sum::<f32>().sqrt();
                let vn: f32 = ws.v_f32[..n_kv_heads*head_d].iter().map(|v| v*v).sum::<f32>().sqrt();
                eprintln!("[DBG] L0 QKV q_norm={:.4} k_norm={:.4} v_norm={:.4}", qn, kn, vn);
            }
        } else {
            ternary_gemv_qkv_parallel(
                &ws.ffn_out_f32[..hidden],
                layer.q_w,
                layer.k_w,
                layer.v_w,
                &mut ws.q_f32,
                &mut ws.k_f32,
                &mut ws.v_f32,
                layer.q_scales,
                layer.k_scales,
                layer.v_scales,
                hidden,
                n_kv_heads * head_d,
                hidden,
            );
        }
    }


    // ── Step 2a: Qwen3 q_norm / k_norm (per-head RMSNorm) before RoPE ──
    // Weights are `[head_dim]`; applied independently to each head slice.
    // AVX2: sum_sq via FMA (8 f32/iter), scale via weighted multiply (8 f32/iter).
    if !layer.q_norm_w.is_null() {
        unsafe {
            for h in 0..n_heads {
                let base = h * head_d;
                let q_ptr = ws.q_f32.as_mut_ptr().add(base);
                // AVX2 sum_sq (8 f32/iter via FMA)
                let sum_sq = if head_d >= 8 {
                    ezop::sum_sq(q_ptr, head_d)
                } else {
                    let mut s = 0.0f32;
                    for i in 0..head_d {
                        let x = *q_ptr.add(i);
                        s += x * x;
                    }
                    s
                };
                let inv_rms = 1.0 / (sum_sq / head_d as f32 + eps).sqrt();
                // AVX2 scale with per-element weight (8 f32/iter)
                if head_d >= 8 {
                    ezop::scale_with_weight_avx2(q_ptr, layer.q_norm_w, inv_rms, head_d);
                } else {
                    for i in 0..head_d {
                        *q_ptr.add(i) *= inv_rms * *layer.q_norm_w.add(i);
                    }
                }
            }
        }
    }
    if !layer.k_norm_w.is_null() {
        unsafe {
            for h in 0..n_kv_heads {
                let base = h * head_d;
                let k_ptr = ws.k_f32.as_mut_ptr().add(base);
                let sum_sq = if head_d >= 8 {
                    ezop::sum_sq(k_ptr, head_d)
                } else {
                    let mut s = 0.0f32;
                    for i in 0..head_d {
                        let x = *k_ptr.add(i);
                        s += x * x;
                    }
                    s
                };
                let inv_rms = 1.0 / (sum_sq / head_d as f32 + eps).sqrt();
                if head_d >= 8 {
                    ezop::scale_with_weight_avx2(k_ptr, layer.k_norm_w, inv_rms, head_d);
                } else {
                    for i in 0..head_d {
                        *k_ptr.add(i) *= inv_rms * *layer.k_norm_w.add(i);
                    }
                }
            }
        }
    }

    // ── Step 2b: RoPE on Q (all heads) and K (kv heads), in-place ──
    // PERF-1: precomputed `inv_freq` + per-pos cos/sin ring cache.
    // Old code: `powf` (~50 ns) + `sin_cos` (~30 ns) per (pos, head, i).
    // New: 1 cache lookup per (pos, layer); all heads reuse the same cos/sin arrays.
    // P-01: borrowed slices (zero-clone on cache hit) — guard held for the duration.
    if layer.rope_theta > 0.0 && !layer.rope_inv_freq.is_empty() {
        let half_d = head_d / 2;
        let mut guard = layer.rope_cache.lock();
        let (cos_tbl, sin_tbl) = guard.get_or_compute(pos, &layer.rope_inv_freq);
        let cos_ptr = cos_tbl.as_ptr();
        let sin_ptr = sin_tbl.as_ptr();
        let use_avx = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && half_d >= 8;
        // AVX2 RoPE: complex multiply rotation (8 elements/iter via FMA).
        // x0'[i] = x0[i]*cos[i] - x1[i]*sin[i]  (fmsub)
        // x1'[i] = x1[i]*cos[i] + x0[i]*sin[i]  (fma)
        for h in 0..n_heads {
            let base = h * head_d;
            unsafe {
                let q_ptr = ws.q_f32.as_mut_ptr().add(base);
                if use_avx {
                    ezop::rope_rotate_avx2(
                        q_ptr, q_ptr.add(half_d),
                        cos_ptr, sin_ptr, half_d,
                    );
                } else {
                    for i in 0..half_d {
                        let cos = *cos_ptr.add(i);
                        let sin = *sin_ptr.add(i);
                        let x0 = *q_ptr.add(i);
                        let x1 = *q_ptr.add(i + half_d);
                        *q_ptr.add(i) = x0 * cos - x1 * sin;
                        *q_ptr.add(i + half_d) = x1 * cos + x0 * sin;
                    }
                }
            }
        }
        for h in 0..n_kv_heads {
            let base = h * head_d;
            unsafe {
                let k_ptr = ws.k_f32.as_mut_ptr().add(base);
                if use_avx {
                    ezop::rope_rotate_avx2(
                        k_ptr, k_ptr.add(half_d),
                        cos_ptr, sin_ptr, half_d,
                    );
                } else {
                    for i in 0..half_d {
                        let cos = *cos_ptr.add(i);
                        let sin = *sin_ptr.add(i);
                        let x0 = *k_ptr.add(i);
                        let x1 = *k_ptr.add(i + half_d);
                        *k_ptr.add(i) = x0 * cos - x1 * sin;
                        *k_ptr.add(i + half_d) = x1 * cos + x0 * sin;
                    }
                }
            }
        }
    }

    // ── Step 3: Store K, V in dense ring (L-13; stream I f16 packs when enabled) ──
    let pos_slot = ws.dense_slot(pos);
    for kv_h in 0..n_kv_heads {
        let cache_base = layer_offset + kv_h * kv_offset + pos_slot * head_d;
        let mut k_row = [0.0f32; 256];
        let mut v_row = [0.0f32; 256];
        let hd = head_d.min(256);
        k_row[..hd].copy_from_slice(&ws.k_f32[kv_h * head_d..kv_h * head_d + hd]);
        v_row[..hd].copy_from_slice(&ws.v_f32[kv_h * head_d..kv_h * head_d + hd]);
        ws.store_dense_k(cache_base, &k_row[..hd]);
        ws.store_dense_v(cache_base, &v_row[..hd]);
    }

    // HCA: mean-pool a block of tokens that just left the sliding window
    let hist_token_idx = pos.saturating_sub(ws.hca_window);
    if pos >= ws.hca_window
        && hist_token_idx % ws.hca_compression_ratio == ws.hca_compression_ratio - 1
    {
        let comp_t = hist_token_idx / ws.hca_compression_ratio;
        if comp_t < hca_max_pos {
            // Stack-friendly: head_dim ≤ 256 in product models
            let mut mean_k = [0.0f32; 256];
            let mut mean_v = [0.0f32; 256];
            let hd = head_d.min(256);
            for kv_h in 0..n_kv_heads {
                let hca_cache_base = hca_layer_offset + kv_h * hca_kv_offset + comp_t * head_d;
                mean_k[..hd].fill(0.0);
                mean_v[..hd].fill(0.0);
                for i in 0..ws.hca_compression_ratio {
                    let t_old = hist_token_idx - i;
                    let slot = ws.dense_slot(t_old);
                    let old_cache_base = layer_offset + kv_h * kv_offset + slot * head_d;
                    // load_dense_* share scratch — accumulate immediately
                    {
                        let kl = ws.load_dense_k(old_cache_base);
                        for d in 0..hd {
                            mean_k[d] += kl[d];
                        }
                    }
                    {
                        let vl = ws.load_dense_v(old_cache_base);
                        for d in 0..hd {
                            mean_v[d] += vl[d];
                        }
                    }
                }
                let inv_r = 1.0 / (ws.hca_compression_ratio as f32);
                for d in 0..hd {
                    mean_k[d] *= inv_r;
                    mean_v[d] *= inv_r;
                }
                ws.store_hca_k(hca_cache_base, &mean_k[..hd]);
                ws.store_hca_v(hca_cache_base, &mean_v[..hd]);
            }
        }
    }

    // ── Step 4: Attention over HCA history + dense recent ring ──
    // Stream E: CSA lightning top-k over HCA when history is large (inference only).
    let inv_sqrt_d = 1.0 / (head_d as f32).sqrt();
    let gqa_scale = n_heads / n_kv_heads;
    let csa_pol = crate::mud::csa_indexer::CsaPolicy::resolve();
    // Training tapes need full HCA mass for correct grads — skip sparse path.
    let csa_active = tape.is_none() && csa_pol.enabled;

    // SPRINT-5 (5.1): use pre-allocated workspace scratch (P-01).
    //
    // The csa_indexer API takes `&mut Vec<usize>` and `&mut Vec<f32>`. To
    // avoid per-call allocations while keeping the API stable, we build
    // transient `Vec` *views* over the workspace storage using
    // `Vec::from_raw_parts`. The capacity is the workspace slice length
    // and we never push more items than the workspace pre-allocated
    // (top_k + tail + 8 ≤ 256 by default; num_comp_tokens ≤ hca_max_pos).
    //
    // SAFETY:
    //   - The `Vec`s view the workspace memory only.
    //   - We `mem::forget` the Vecs before any other code touches the
    //     workspace memory again (e.g. another call to
    //     `evaluate_slime_block_moe` would overlap, but that never
    //     happens — the workspace is owned by the caller).
    //   - We never allow the Vecs to grow beyond their workspace cap.
    // SAFETY: `ws.csa_blocks_scratch` is caller-owned workspace storage that
    // outlives this borrow and is not aliased while these Vec views exist
    // (see contract above); the Vec is rebuilt over the same allocation.
    let mut csa_blocks: Vec<usize> = unsafe {
        Vec::from_raw_parts(
            ws.csa_blocks_scratch.as_mut_ptr(),
            0,
            ws.csa_blocks_scratch.len(),
        )
    };
    let scratch_cap = ws.csa_index_scratch.len().min(hca_max_pos);
    let mut csa_index_scratch: Vec<f32> = if csa_active {
        // SAFETY: same workspace-view contract as `csa_blocks` above;
        // `scratch_cap` is min(capacity, hca_max_pos), so any push stays
        // within the allocation (never reallocated).
        unsafe { Vec::from_raw_parts(ws.csa_index_scratch.as_mut_ptr(), 0, scratch_cap) }
    } else {
        Vec::new()
    };

    for h in 0..n_heads {
        let kv_h = h / gqa_scale;
        let q_off = h * head_d;

        let hist_end_idx = pos.saturating_sub(ws.hca_window);
        let num_comp_tokens = (hist_end_idx / ws.hca_compression_ratio).min(hca_max_pos);
        let recent_start = num_comp_tokens * ws.hca_compression_ratio;

        // CSA lightning needs contiguous f32 HCA K — only when dtype is f32.
        let use_csa = csa_active && csa_pol.should_sparse(num_comp_tokens) && !ws.kv_dtype.is_f16();
        if use_csa {
            let hca_k_head = hca_layer_offset + kv_h * hca_kv_offset;
            unsafe {
                crate::mud::csa_indexer::index_hca_blocks(
                    ws.q_f32.as_ptr().add(q_off),
                    ws.hca_kv_cache.as_ptr().add(hca_k_head),
                    head_d,
                    num_comp_tokens,
                    csa_pol,
                    None,
                    inv_sqrt_d,
                    &mut csa_index_scratch,
                    &mut csa_blocks,
                );
            }
        } else {
            csa_blocks.clear();
            csa_blocks.extend(0..num_comp_tokens);
        }
        let n_hca_sel = csa_blocks.len();
        let n_dense = pos.saturating_sub(recent_start).saturating_add(1);
        debug_assert!(ws.scores.len() >= n_hca_sel + n_dense);

        let mut max_score = f32::NEG_INFINITY;
        let mut score_idx = 0;
        // Local Q copy so load_* can mut-borrow workspace without aliasing q_f32
        let mut q_local = [0.0f32; 256];
        let qd = head_d.min(256);
        q_local[..qd].copy_from_slice(&ws.q_f32[q_off..q_off + qd]);
        let q_ptr = q_local.as_ptr();

        // 1. Selected compressed historical blocks (all HCA or CSA top-k ∪ tail)
        for &comp_t in &csa_blocks {
            let hca_k_base = hca_layer_offset + kv_h * hca_kv_offset + comp_t * head_d;
            let k_row = ws.load_hca_k(hca_k_base);
            let mut s = unsafe { crate::asm::dot_product_avx2(qd, q_ptr, k_row.as_ptr()) };
            s *= inv_sqrt_d;
            ws.scores[score_idx] = s;
            if s > max_score {
                max_score = s;
            }
            score_idx += 1;
        }

        // 2. Recent high-fidelity tokens (dense ring) — always full
        for t in recent_start..=pos {
            let slot = ws.dense_slot(t);
            let k_pos_base = layer_offset + kv_h * kv_offset + slot * head_d;
            let k_row = ws.load_dense_k(k_pos_base);
            let mut s = unsafe { crate::asm::dot_product_avx2(qd, q_ptr, k_row.as_ptr()) };
            s *= inv_sqrt_d;
            ws.scores[score_idx] = s;
            if s > max_score {
                max_score = s;
            }
            score_idx += 1;
        }
        let total_attn_elements = score_idx;

        // AVX2 softmax: subtract max → exp → HCA scale → sum → normalize.
        // Vectorized exp (8 f32/iter via fast exp2 poly) + vectorized sum.
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && total_attn_elements >= 8 {
            unsafe {
                let scores_ptr = ws.scores.as_mut_ptr();
                for i in 0..total_attn_elements {
                    ws.scores[i] -= max_score;
                }
                // Step 2: exp in-place (AVX2, 8 f32/iter)
                crate::mud::ezop::exp_avx2(scores_ptr, scores_ptr, total_attn_elements);
                // Step 3: HCA scale for first n_hca_sel elements
                if n_hca_sel > 0 {
                    let hca_ratio = ws.hca_compression_ratio as f32;
                    crate::mud::ezop::scale_vec_avx2(scores_ptr, hca_ratio, n_hca_sel);
                }
                // Step 4: vectorized sum
                let sum_exp = crate::mud::ezop::sum_avx2(scores_ptr, total_attn_elements);
                let inv_sum = 1.0 / (sum_exp + 1e-10);
                // Step 5: normalize (AVX2 scale)
                crate::mud::ezop::scale_vec_avx2(scores_ptr, inv_sum, total_attn_elements);
            }
        } else {
            let mut sum_exp = 0.0;
            for i in 0..total_attn_elements {
                ws.scores[i] = (ws.scores[i] - max_score).exp();
                if i < n_hca_sel {
                    ws.scores[i] *= ws.hca_compression_ratio as f32;
                }
                sum_exp += ws.scores[i];
            }
            let inv_sum = 1.0 / (sum_exp + 1e-10);
            for i in 0..total_attn_elements {
                ws.scores[i] *= inv_sum;
            }
        }

        ws.o_act_f32[q_off..q_off + head_d].fill(0.0);
        let mut s_idx = 0;

        for &comp_t in &csa_blocks {
            let sv = ws.scores[s_idx];
            let hca_v_base = hca_layer_offset + kv_h * hca_kv_offset + comp_t * head_d;
            // Copy V row out of scratch before axpy (scratch reused)
            let v_tmp: [f32; 256] = {
                let vr = ws.load_hca_v(hca_v_base);
                let mut a = [0.0f32; 256];
                a[..head_d.min(256)].copy_from_slice(&vr[..head_d.min(256)]);
                a
            };
            unsafe {
                forge_autograd::avx_math::axpy_avx2(
                    &mut ws.o_act_f32[q_off..q_off + head_d],
                    sv,
                    &v_tmp[..head_d.min(256)],
                );
            }
            s_idx += 1;
        }

        for t in recent_start..=pos {
            let sv = ws.scores[s_idx];
            let slot = ws.dense_slot(t);
            let v_pos_base = layer_offset + kv_h * kv_offset + slot * head_d;
            let v_tmp: [f32; 256] = {
                let vr = ws.load_dense_v(v_pos_base);
                let mut a = [0.0f32; 256];
                a[..head_d.min(256)].copy_from_slice(&vr[..head_d.min(256)]);
                a
            };
            unsafe {
                forge_autograd::avx_math::axpy_avx2(
                    &mut ws.o_act_f32[q_off..q_off + head_d],
                    sv,
                    &v_tmp[..head_d.min(256)],
                );
            }
            s_idx += 1;
        }
    }

    if let Some(t) = tape.as_mut() {
        // Copy the softmax scores up to the current position
        t.scores[..=pos].copy_from_slice(&ws.scores[..=pos]);
        t.o_act_f32.copy_from_slice(&ws.o_act_f32[..hidden]);
        t.pos = pos;
    }

    // ── Step 5: Re-quantize attn output → i8 for O projection ──
    let attn_peak = {
        for i in 0..hidden {
            if !ws.o_act_f32[i].is_finite() {
                nan_skip(&NAN_BEFORE_STEP5_COUNT, "NaN in o_act_f32 before Step 5", layer_idx, pos);
                break;
            }
        }

        // AVX2 peak abs via horizontal max reduction (8 f32/iter)
        let p = if is_x86_feature_detected!("avx2") && hidden >= 8 {
            unsafe { ezop::peak_abs_avx2(ws.o_act_f32.as_ptr(), hidden) }
        } else {
            let mut peak = 0.0f32;
            for i in 0..hidden {
                peak = peak.max(ws.o_act_f32[i].abs());
            }
            peak
        };
        p.max(1e-8f32)
    };
    let inv_attn_peak = 127.0f32 / attn_peak;
    for i in 0..hidden {
        ws.o_act_i8[i] = (ws.o_act_f32[i] * inv_attn_peak).clamp(-127.0, 127.0) as i8;
    }

    let attn_act_scale = attn_peak / 127.0f32;
    if !attn_act_scale.is_finite() {
        nan_skip(&NAN_ACT_SCALE_STEP5_COUNT, "NaN attn_act_scale step 5", layer_idx, pos);
    }

    // Dequantize o_act_i8 → ffn_out_f32 (AVX2, 8 i8→f32/iter)
    if is_x86_feature_detected!("avx2") && hidden >= 8 {
        unsafe {
            ezop::scale_i8_to_f32_avx2(
                ws.o_act_i8.as_ptr(),
                ws.ffn_out_f32.as_mut_ptr(),
                attn_act_scale,
                hidden,
            );
        }
    } else {
        for i in 0..hidden {
            ws.ffn_out_f32[i] = ws.o_act_i8[i] as f32 * attn_act_scale;
        }
    }
    unsafe {
        if layer.is_f32_weights {
            f32_gemv_rowwise(&ws.ffn_out_f32[..hidden], layer.o_w, &mut ws.o_act_f32, hidden, hidden);
        } else {
            ternary_gemv_rowwise(
                &ws.ffn_out_f32[..hidden],
                layer.o_w,
                &mut ws.o_act_f32,
                layer.o_scales,
                hidden,
                hidden,
            );
        }
        for i in 0..hidden {
            if !ws.o_act_f32[i].is_finite() {
                nan_skip(&NAN_AFTER_O_GEMV_COUNT, "NaN in o_act_f32 after O GEMV", layer_idx, pos);
                break;
            }
        }
    }

    // ── Step 6: attn_sub_norm (optional) ──
    if !layer.attn_sub_norm_w.is_null() {
        unsafe {
            let o_ptr = ws.o_act_f32.as_mut_ptr();
            // AVX2 sum_sq (8 f32/iter via FMA)
            let sum_sq = if hidden >= 8 {
                ezop::sum_sq(o_ptr, hidden)
            } else {
                let mut s = 0.0f32;
                for i in 0..hidden {
                    let x = *o_ptr.add(i);
                    s += x * x;
                }
                s
            };
            let rms = (sum_sq / hidden as f32 + eps).sqrt();
            let inv_rms = 1.0 / rms;
            // AVX2 scale with per-element weight: x[i] *= inv_rms * w[i]
            if hidden >= 8 {
                ezop::scale_with_weight_avx2(o_ptr, layer.attn_sub_norm_w, inv_rms, hidden);
            } else {
                for i in 0..hidden {
                    *o_ptr.add(i) = *o_ptr.add(i) * inv_rms * *layer.attn_sub_norm_w.add(i);
                }
            }
        }
    }

    // Scale attention output to prevent residual explosion (DeepNet-style).
    // The ternary attention block can produce ~50x amplification on the first
    // layer; scaling by 0.5 keeps the residual stream bounded and stable.
    for i in 0..hidden {
        ws.o_act_f32[i] *= 0.25;
    }

    // ── Step 7: Attention JEPA + Residual ──
    for i in 0..hidden {
        if !ws.o_act_f32[i].is_finite() {
            nan_skip(&NAN_BEFORE_STEP7_COUNT, "NaN in o_act_f32 before Step 7", layer_idx, pos);
            break;
        }
    }

    let attn_jepa_idx = 2 * layer_idx;
    let tape_attn = tape.as_mut().map(|t| &mut t.attn_v_jepa[..hidden]);
    let attn_z_start = attn_jepa_idx * ws.hidden_size;
    let attn_z = &mut ws.jepa_z[attn_z_start..attn_z_start + ws.hidden_size];
    let delta_attn = jepa_stabilizer(
        &mut ws.o_act_f32[..hidden],
        &mut ws.registers,
        &mut ws.jepa_mu[attn_jepa_idx],
        &mut ws.jepa_inv_sigma[attn_jepa_idx],
        &mut ws.jepa_var_ema[attn_jepa_idx],
        attn_z,
        tape_attn,
        gpu.as_deref_mut(),
    );
    ws.jepa_integral = ws.jepa_integral * 0.99 + delta_attn;

    // Phase 1 & 2 & 3: Manifold-Constrained Hyper-Connections (mHC)
    // Geometric bounding instead of clipping or adaptive dt
    // FIX (2026-07-30): fallback radius was 1000.0 (loose) — use workspace
    // mhc_radius (≈ 1.5·√hidden) so base models without mhc_radius tensor
    // still get a bounded residual. The old 1000.0 allowed f16 overflow.
    let layer_radius = if layer.mhc_radius_w.is_null() {
        ws.mhc_radius
    } else {
        unsafe { *layer.mhc_radius_w }
    };
    // Phase 1 (mHC trainable): capture h_in for the post-attn mHC site.
    // f_h for this site is o_act_f32 (already tape'd).
    if let Some(t) = tape.as_mut() {
        for i in 0..hidden {
            t.mhc_attn_h_in[i] = ws.registers[i].read_accum();
        }
    }
    // SAFETY: mhc_residual reads h_in[i] → writes h_out[i] independently per index.
    // In-place alias (h_in == h_out on registers) is safe.
    let regs_ptr = ws.registers.as_mut_ptr();
    let o_act_ptr = ws.o_act_f32.as_ptr();
    unsafe {
        mhc_residual(
            std::slice::from_raw_parts(regs_ptr.cast_const(), hidden),
            std::slice::from_raw_parts_mut(regs_ptr, hidden),
            std::slice::from_raw_parts(o_act_ptr, hidden),
            layer_radius,
            layer.mhc_alpha_w,
            layer.mhc_beta_w,
        );
    }

    // ── Step 8: FFN RMSNorm → i8 ──
    let _ffn_act_scale = slime_rmsnorm_i8(
        &ws.registers[..hidden],
        &mut ws.gemv_accum,
        layer.ffn_norm_w,
        &mut ws.norm_i8,
        eps,
    );

    // R-43: store f32 directly, skip i8 round-trip.
    ws.ffn_out_f32[..hidden].copy_from_slice(&ws.gemv_accum[..hidden]);

    if let Some(t) = tape.as_mut() {
        t.norm_f32_ffn[..hidden].copy_from_slice(&ws.ffn_out_f32[..hidden]);
        t.ffn_act_scale = 1.0; // no quantization
    }

    // ── Step 9–12: FFN (dense or L-11 Mini MoE) ──
    let ffn_mid = layer.ffn_mid;
    // R-43: ffn_out_f32 already populated (no i8 round-trip).

    let use_moe = moe
        .map(|b| b.mounted_count() > 0 && b.hidden == hidden)
        .unwrap_or(false);

    if use_moe {
        // L-11: ExpertBus replaces dense SwiGLU FFN
        let bus = moe.expect("use_moe implies Some");
        // Need mutable scratch — if missing, allocate temporary (rare path)
        let mut local_scratch;
        let scratch = if let Some(s) = moe_scratch {
            s
        } else {
            local_scratch =
                crate::mud::expert_bus::ExpertScratch::new(hidden, ffn_mid, bus.capacity());
            &mut local_scratch
        };
        // SAFETY: expert weight pointers owned by bus/model
        let seed = (layer_idx as u32)
            .wrapping_mul(0x9E37_79B9)
            .wrapping_add(pos as u32);
        if let Err(e) = unsafe {
            bus.forward(
                &ws.ffn_out_f32[..hidden],
                &mut ws.o_act_f32[..hidden],
                scratch,
                seed,
            )
        } {
            // Fall back to dense on bus error
            log::warn!("[L-11 MoE] forward failed ({e}); dense FFN fallback");
            dense_ffn_swiglu(layer, ws, hidden, ffn_mid, &mut tape, eps);
        } else {
            ws.ffn_out_f32[..hidden].copy_from_slice(&ws.o_act_f32[..hidden]);
            if let Some(t) = tape.as_mut() {
                // Tape mid/up/gate not meaningful for multi-expert; zero mid
                t.ffn_mid_f32[..ffn_mid].fill(0.0);
            }
        }
    } else {
        dense_ffn_swiglu(layer, ws, hidden, ffn_mid, &mut tape, eps);
    }

    // Scale FFN output to prevent residual explosion.
    for i in 0..hidden {
        ws.ffn_out_f32[i] *= 0.25;
    }

    // ── Step 13: FFN JEPA + Residual ──
    let ffn_jepa_idx = 2 * layer_idx + 1;
    let tape_ffn = tape.as_mut().map(|t| &mut t.ffn_v_jepa[..hidden]);
    let ffn_z_start = ffn_jepa_idx * ws.hidden_size;
    let ffn_z = &mut ws.jepa_z[ffn_z_start..ffn_z_start + ws.hidden_size];
    let delta_ffn = jepa_stabilizer(
        &mut ws.ffn_out_f32[..hidden],
        &mut ws.registers,
        &mut ws.jepa_mu[ffn_jepa_idx],
        &mut ws.jepa_inv_sigma[ffn_jepa_idx],
        &mut ws.jepa_var_ema[ffn_jepa_idx],
        ffn_z,
        tape_ffn,
        gpu.as_deref_mut(),
    );
    ws.jepa_integral = ws.jepa_integral * 0.99 + delta_ffn;

    let mut all_zero = true;

    // Phase 1 & 2 & 3: Manifold-Constrained Hyper-Connections (mHC)
    // FIX (2026-07-30): fallback radius was 1000.0 — use ws.mhc_radius so the
    // mHC projection is always active even when mhc_radius_w is null.
    let layer_radius = if layer.mhc_radius_w.is_null() {
        ws.mhc_radius
    } else {
        unsafe { *layer.mhc_radius_w }
    };
    // Phase 1 (mHC trainable): capture h_in and f_h for the post-ffn mHC site.
    if let Some(t) = tape.as_mut() {
        for i in 0..hidden {
            t.mhc_ffn_h_in[i] = ws.registers[i].read_accum();
            t.mhc_ffn_f_h[i] = ws.ffn_out_f32[i];
        }
    }
    // SAFETY: in-place mHC (same buffer for h_in and h_out).
    let regs_ptr = ws.registers.as_mut_ptr();
    let ffn_out_ptr = ws.ffn_out_f32.as_ptr();
    unsafe {
        mhc_residual(
            std::slice::from_raw_parts(regs_ptr.cast_const(), hidden),
            std::slice::from_raw_parts_mut(regs_ptr, hidden),
            std::slice::from_raw_parts(ffn_out_ptr, hidden),
            layer_radius,
            layer.mhc_alpha_w,
            layer.mhc_beta_w,
        );
    }

    for i in 0..hidden {
        if ws.registers[i].read_accum() != 0.0 {
            all_zero = false;
        }
    }
    if all_zero {
        // FIX (2026-07-31): Dead-activation recovery — inject deterministic noise
        // into registers to break the all-zero cycle. Without this, zero activations
        // → zero gradients → permanent dead neurons. The noise is small enough to
        // not disturb healthy forward passes (only triggers when ALL registers
        // are exactly 0.0).
        static DEAD_RECOVERY_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let recovery_idx = DEAD_RECOVERY_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Deterministic pseudo-noise from position + layer + counter.
        // Magnitude ~1e-3 — large enough to survive RMSNorm + quantization.
        for i in 0..hidden.min(8) {
            let seed = ((layer_idx as u64).wrapping_mul(2654435761).wrapping_add(pos as u64).wrapping_add(recovery_idx).wrapping_add(i as u64)) & 0xFFFF;
            let noise = (seed as f32 / 65536.0 - 0.5) * 2e-3;
            ws.registers[i].write_accum(noise);
        }
        all_zero = ws.registers.iter().any(|r| r.read_accum() != 0.0); // recheck after injection
        // FIX (2026-07-31): Only compute debug if recovery FAILED (all_zero still true).
        // If recovery succeeded, skip debug entirely to prevent log spam.
        if all_zero {
            let a_first = if layer.mhc_alpha_w.is_null() {
                1.0
            } else {
                unsafe { *layer.mhc_alpha_w }
            };
            let b_first = if layer.mhc_beta_w.is_null() {
                1.0
            } else {
                unsafe { *layer.mhc_beta_w }
            };
            let r = if layer.mhc_radius_w.is_null() {
                ws.mhc_radius
            } else {
                unsafe { *layer.mhc_radius_w }
            };
            let v0 = ws.registers[0].read_accum();
            let g0 = ws.registers[0].read_integral();
            let mut nz_regs = 0i32;
            let mut sum_abs_regs = 0f64;
            for i in 0..hidden {
                if ws.registers[i].read_accum() != 0.0 {
                    nz_regs += 1;
                }
                sum_abs_regs += ws.registers[i].read_accum().abs() as f64;
            }
            let mean_abs = sum_abs_regs / hidden as f64;
            let g_mean: f64 = (0..hidden)
                .map(|i| ws.registers[i].read_integral() as f64)
                .sum::<f64>()
                / hidden as f64;
            let mut nan_kv = false;
            for i in 0..(ws.dense_kv_elems().min(16)) {
                // Skip NaN scan when f16-packed (checked on write path)
                if ws.kv_dtype.is_f16() {
                    break;
                }
                if !ws.kv_cache[i].is_finite() {
                    nan_kv = true;
                    break;
                }
            }
            for i in 0..(ws.v_cache.len().min(16)) {
                if !ws.v_cache[i].is_finite() {
                    nan_kv = true;
                    break;
                }
            }
            log::warn!("LAYER L{} POS {} ALLZERO (a0={:.4} b0={:.4} R={:.2} regs[0]={:.4} integral0={:.4} nz_regs={} mean_abs={:.2e} integral_mean={:.4} NaN_KV={})",
                layer_idx, pos, a_first, b_first, r, v0, g0, nz_regs, mean_abs, g_mean, nan_kv);
        }
    }

    // L-15: mark activation tape valid for backward / recompute accounting
    if let Some(t) = tape.as_mut() {
        t.valid = true;
        t.pos = pos;
    }

    // SX-01: Snapshot registers → slimex_cells (u32 8:8:8:8 cell buffer).
    // No-op when MUD_SLIMEX=0 (ws.slimex_enabled=false). Covers all callers:
    // corpus_trainer, cmud_train (forward_last_hidden_and_head), inference.
    //
    // SPRINT-2: Use the workspace's pre-allocated `slimex_jepa_scratch` instead of
    // allocating a fresh `Vec<f32>` per block/layer (P-01). The scratch is sized to
    // `hidden_size` once at workspace creation. This restores zero-alloc in the
    // slimex path (PERF-2 V25 audit).
    if ws.slimex_enabled {
        let ffn_z_snap_start = (2 * layer_idx + 1) * ws.hidden_size;
        let hidden = ws.hidden_size;
        let jepa_len = ws.jepa_z.len();
        let copy_len = hidden.min(jepa_len.saturating_sub(ffn_z_snap_start));
        // SAFETY: scratch is `hidden_size` f32; copy_len ≤ hidden.
        unsafe {
            std::ptr::copy_nonoverlapping(
                ws.jepa_z.as_ptr().add(ffn_z_snap_start),
                ws.slimex_jepa_scratch.as_mut_ptr(),
                copy_len,
            );
            // Zero the tail (if any) so the snapshot path sees a stable buffer.
            if copy_len < hidden {
                std::ptr::write_bytes(
                    ws.slimex_jepa_scratch.as_mut_ptr().add(copy_len),
                    0u8,
                    (hidden - copy_len) * std::mem::size_of::<f32>(),
                );
            }
        }
        // SAFETY: scratch is exactly `hidden` f32 — matches `snapshot_to_slimex` contract.
        let scratch_slice = unsafe {
            std::slice::from_raw_parts(ws.slimex_jepa_scratch.as_ptr(), hidden)
        };
        ws.snapshot_to_slimex(scratch_slice);
    }

    // ── Diagnostic Health Check ──
    let diag = crate::mud::slime_jepa::check_tensor_health(&ws.registers, 1.0);

    // SPRINT-5 (5.1): prevent the workspace-viewing `Vec`s from being
    // dropped (they would free the workspace memory). We forget the
    // length only — the actual backing storage lives in `ws`.
    std::mem::forget(csa_blocks);
    std::mem::forget(csa_index_scratch);

    diag
}
// ── Output norm (GPU path) moved to gemv_gpu module. ──────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slime_forward_structure() {
        let hidden = 32;
        let mut ws = SlimeWorkspace::new(hidden, 32, 1, 1, 32, hidden, 1, 128.0);
        let row_sz = hidden / 8 * 4;
        let q_w = vec![0x00u8; hidden * row_sz];
        let k_w = q_w.clone();
        let v_w = q_w.clone();
        let o_w = q_w.clone();
        let ffn_up = q_w.clone();
        let ffn_gate = q_w.clone();
        let ffn_down = q_w.clone();

        let scales = vec![0.01f32; hidden];
        let norm_w = vec![1.0f32; hidden];

        let layer = SlimeLayer {
            q_w: q_w.as_ptr(),
            k_w: k_w.as_ptr(),
            v_w: v_w.as_ptr(),
            o_w: o_w.as_ptr(),
            q_scales: scales.as_ptr(),
            k_scales: scales.as_ptr(),
            v_scales: scales.as_ptr(),
            o_scales: scales.as_ptr(),
            ffn_up_w: ffn_up.as_ptr(),
            ffn_gate_w: ffn_gate.as_ptr(),
            ffn_down_w: ffn_down.as_ptr(),
            ffn_up_scales: scales.as_ptr(),
            ffn_gate_scales: scales.as_ptr(),
            ffn_down_scales: scales.as_ptr(),
            attn_norm_w: norm_w.as_ptr(),
            ffn_norm_w: norm_w.as_ptr(),
            attn_sub_norm_w: norm_w.as_ptr(),
            ffn_sub_norm_w: norm_w.as_ptr(),
            q_norm_w: std::ptr::null(),
            k_norm_w: std::ptr::null(),
            mhc_alpha_w: std::ptr::null(),
            mhc_beta_w: std::ptr::null(),
            mhc_alpha_shadow: None,
            mhc_beta_shadow: None,
            attn_norm_shadow: None,
            ffn_norm_shadow: None,
            mhc_radius_w: std::ptr::null(),
            n_kv_heads: 1,
            ffn_mid: hidden,
            rope_theta: 0.0,
            rope_inv_freq: Vec::new(),
            rope_cache: parking_lot::Mutex::new(RopeCache::new()),
            is_f32_weights: false,
        };

        ws.registers[0].write_accum(100.0);
        ws.jepa_z[0] = 1.0; // layer 0 attention head, dim 0: initial z = 1.0
        ws.jepa_z[ws.hidden_size] = 1.0; // layer 0 FFN head, dim 0: initial z = 1.0
        evaluate_slime_block(&layer, 0, &mut ws, 0, 1e-6, None, None);
        assert!(
            ws.jepa_mu[0].is_finite(),
            "attention JEPA mu must be finite"
        );
        assert!(ws.jepa_mu[1].is_finite(), "FFN JEPA mu must be finite");
        assert!(
            ws.jepa_mu[0] > 0.0,
            "attention mu should be initialized from batch_mu_z"
        );
    }

    #[test]
    fn test_f32_gemv_backward_matches_reference() {
        let n_out = 4;
        let n_in = 8;
        let grad_y: Vec<f32> = (0..n_out).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let x: Vec<f32> = (0..n_in).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let w: Vec<f32> = (0..(n_out * n_in)).map(|i| (i as f32 + 1.0) * 0.01).collect();

        let mut grad_w = vec![0.0f32; n_out * n_in];
        let mut grad_x = vec![0.0f32; n_in];
        let scratch = &mut [0.0f32; 8][..];

        unsafe {
            f32_gemv_backward(&grad_y, &x, w.as_ptr(), &mut grad_w, &mut grad_x, scratch, n_out, n_in);
        }

        for out in 0..n_out {
            for c in 0..n_in {
                let expected = grad_y[out] * x[c];
                assert!((grad_w[out * n_in + c] - expected).abs() < 1e-5,
                    "grad_w[{}*{} + {}] = {}, expected {}", out, n_in, c, grad_w[out * n_in + c], expected);
            }
        }
        for c in 0..n_in {
            let mut expected = 0.0f32;
            for out in 0..n_out {
                expected += grad_y[out] * w[out * n_in + c];
            }
            assert!((grad_x[c] - expected).abs() < 1e-5,
                "grad_x[{}] = {}, expected {}", c, grad_x[c], expected);
        }
    }

    #[test]
    fn test_f32_gemv_backward_accumulates() {
        let n_out = 4;
        let n_in = 8;
        let grad_y: Vec<f32> = (0..n_out).map(|i| i as f32 + 1.0).collect();
        let x: Vec<f32> = (0..n_in).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let w: Vec<f32> = (0..(n_out * n_in)).map(|i| (i as f32 + 1.0) * 0.01).collect();

        let mut grad_x = vec![1.0f32; n_in];
        let mut grad_w = vec![0.0f32; n_out * n_in];
        let scratch = &mut [0.0f32; 8][..];

        unsafe {
            f32_gemv_backward(&grad_y, &x, w.as_ptr(), &mut grad_w, &mut grad_x, scratch, n_out, n_in);
        }

        for c in 0..n_in {
            let mut ref_val = 1.0f32;
            for out in 0..n_out {
                ref_val += grad_y[out] * w[out * n_in + c];
            }
            assert!((grad_x[c] - ref_val).abs() < 1e-5,
                "grad_x[{}] = {}, expected {}", c, grad_x[c], ref_val);
        }
        for out in 0..n_out {
            for c in 0..n_in {
                let expected = grad_y[out] * x[c];
                assert!((grad_w[out * n_in + c] - expected).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_gpu_gemv_policy_off_inactive() {
        // Mode parse is cached in OnceLock — only assert pure helpers that don't need env.
        use crate::vulkan::gemv_policy::{should_try_gpu, GemvGpuMode};
        assert_eq!(GemvGpuMode::Off, GemvGpuMode::Off);
        assert!(!should_try_gpu(0, 256));
        assert!(!should_try_gpu(7, 256));
    }

    #[test]
    fn test_phase_bplus_gpu_gemv_matches_cpu_if_available() {
        // Force On for this test (if OnceLock already set by another test, try_gpu may no-op).
        unsafe {
            std::env::set_var("MUD_GPU_GEMV", "1");
            std::env::set_var("MUD_USE_VULKAN", "1");
        }
        if !crate::mud::gemv_gpu::gpu_gemv_policy_active() {
            return;
        }

        // Work must exceed GEMV_GPU_MIN_WORK (256*256) for On mode
        let n_in = 256usize;
        let n_out = 256usize;
        let blocks = n_in / 8;
        let x: Vec<f32> = (0..n_in).map(|i| (i as f32 * 0.01).sin()).collect();
        let packed = vec![0x1111_1111u32; n_out * blocks];
        let scales = vec![1.0f32; n_out];
        let mut y_cpu = vec![0.0f32; n_out];
        let mut y_gpu = vec![0.0f32; n_out];

        // CPU path without going through GPU gate
        unsafe {
            ternary_gemv_rowwise_submit(
                &x,
                packed.as_ptr() as *const u8,
                &mut y_cpu,
                scales.as_ptr(),
                n_out,
                n_in,
            );
            crate::mud::pcore_pool::get_pool().wait_all();
        }

        let gpu_ok = unsafe {
            try_gpu_ternary_gemv(
                &x,
                packed.as_ptr() as *const u8,
                &mut y_gpu,
                scales.as_ptr(),
                n_out,
                n_in,
            )
        };

        if !gpu_ok {
            return; // no device or policy threshold
        }
        let mut max_diff = 0.0f32;
        for i in 0..n_out {
            max_diff = max_diff.max((y_cpu[i] - y_gpu[i]).abs());
        }
        assert!(
            max_diff < 1e-2,
            "GPU/CPU GEMV divergence {max_diff} (tolerance 1e-2 for UMA float)"
        );
    }

    #[test]
    fn test_f32_gemv_rowwise_matches_scalar() {
        let n_out = 7;
        let n_in = 13;
        let acts: Vec<f32> = (0..n_in).map(|i| (i as f32 + 1.0) * 0.25).collect();
        let w: Vec<f32> = (0..(n_out * n_in)).map(|i| (i as f32 + 1.0) * 0.01 - 0.5).collect();
        let mut out = vec![0.0f32; n_out];
        unsafe {
            f32_gemv_rowwise(&acts, w.as_ptr() as *const u8, &mut out, n_out, n_in);
        }
        for r in 0..n_out {
            let mut refv = 0.0f32;
            for c in 0..n_in {
                refv += w[r * n_in + c] * acts[c];
            }
            assert!((out[r] - refv).abs() < 1e-5,
                "row {} = {} expected {}", r, out[r], refv);
        }
    }
}
