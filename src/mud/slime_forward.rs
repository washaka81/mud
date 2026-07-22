use crate::mud::slime::{SlimeRegister, SlimeWorkspace};
use crate::mud::slime_jepa::jepa_stabilizer;

fn slime_rmsnorm_i8(
    regs: &[SlimeRegister],
    gemv_accum: &mut [f32],
    weights: *const f32,
    out_i8: &mut [i8],
    eps: f32,
) -> f32 {
    let n = regs.len();
    let mut sum_sq: f64 = 0.0;
    for reg in regs {
        // SlimeRegister v2: read_accum() returns f32 directly from f16 (no iscale)
        let x = reg.read_accum() as f64;
        sum_sq += x * x;
    }
    let rms_inv = 1.0f32 / ((sum_sq / n as f64) as f32 + eps).sqrt();

    let mut peak = 0.0f32;
    for i in 0..n {
        let xn = regs[i].read_accum() * rms_inv * unsafe { *weights.add(i) };
        gemv_accum[i] = xn;
        peak = peak.max(xn.abs());
    }
    peak = peak.max(crate::mud::constants::EPSILON_FLOOR);
    let inv_peak = 127.0 / peak;
    for i in 0..n {
        out_i8[i] = (gemv_accum[i] * inv_peak).clamp(-127.0, 127.0) as i8;
    }
    peak / 127.0
}

#[derive(Clone, Copy)]
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

    pub n_kv_heads: usize,
    pub ffn_mid: usize,
    /// RoPE base frequency (0.0 = disable RoPE).
    pub rope_theta: f32,
}

unsafe impl Send for SlimeLayer {}
unsafe impl Sync for SlimeLayer {}

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

// ── Phase B+ / Stream C: policy-driven GPU GEMV ───────────────────────────────

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
fn gpu_gemv_policy_active() -> bool {
    use crate::vulkan::gemv_policy::{parse_gemv_mode, vulkan_not_disabled, GemvGpuMode};
    if !vulkan_not_disabled() {
        return false;
    }
    !matches!(parse_gemv_mode(), GemvGpuMode::Off)
}

/// Try GPU tiled ternary GEMV. Returns true on success.
///
/// # Safety
/// Same pointer contracts as [`ternary_gemv_rowwise`].
unsafe fn try_gpu_ternary_gemv(
    acts_f32: &[f32],
    w_u8: *const u8,
    out_f32: &mut [f32],
    scales: *const f32,
    n_out: usize,
    n_in: usize,
) -> bool {
    use crate::vulkan::gemv_policy;

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

    let blocks = n_in / 8;
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
/// Same as [`ternary_gemv_rowwise`]. Do not free buffers until after `wait_all`.
pub(crate) unsafe fn ternary_gemv_rowwise_submit(
    acts_f32: &[f32],
    w_u8: *const u8,
    out_f32: &mut [f32],
    scales: *const f32,
    n_out: usize,
    n_in: usize,
) {
    let row_u32s = n_in / 8;
    let w_u32 = w_u8 as *const u32;

    let pool = crate::mud::pcore_pool::get_pool();
    let n_tasks = pool.num_threads().max(1);
    // T11: `MUD_GEMV_ROWS=8` enables 8-row kernel. Default 4 — microbench on
    // i7-1260P (hidden=2048, warm L2) shows 4-row ≥ 8-row; 8-row helps more when
    // x is cold / multi-thread thrash (opt-in until host re-calib).
    let prefer_8 = std::env::var("MUD_GEMV_ROWS")
        .map(|v| v.trim() == "8")
        .unwrap_or(false);
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
                    unsafe {
                        crate::asm::ternary_gemv_8rows(
                            n_in,
                            acts,
                            w.add(row * row_u32s),
                            out.add(row),
                            1.0,
                            row_u32s,
                        );
                        for r in 0..8 {
                            let s = (*sc.add(row + r)).clamp(-1e8, 1e8);
                            *out.add(row + r) *= if s.is_finite() { s } else { 0.0 };
                        }
                    }
                    row += 8;
                }
            }
            while row + 4 <= end_row {
                unsafe {
                    crate::asm::ternary_gemv_4rows(
                        n_in,
                        acts,
                        w.add(row * row_u32s),
                        out.add(row),
                        1.0,
                        row_u32s,
                    );
                    for r in 0..4 {
                        let s = (*sc.add(row + r)).clamp(-1e8, 1e8);
                        *out.add(row + r) *= if s.is_finite() { s } else { 0.0 };
                    }
                }
                row += 4;
            }

            for r in row..end_row {
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
unsafe fn try_gpu_ternary_gemv_qkv(
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

    let blocks = n_in / 8;
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
    let mut max_abs = 0.0f32;
    for i in 0..h_in.len() {
        let alpha = if alpha_w.is_null() {
            1.0
        } else {
            unsafe { *alpha_w.add(i) }
        };
        let beta = if beta_w.is_null() {
            1.0
        } else {
            unsafe { *beta_w.add(i) }
        };
        // For base models not trained with JEPA, gate must be 0 so (1-gate) = 1.0
        let gate = if alpha_w.is_null() && beta_w.is_null() {
            0.0
        } else {
            h_in[i].gate()
        };
        let val = alpha * h_in[i].read_accum() + (1.0 - gate) * beta * f_h[i];

        h_out[i].write_accum(val);
        // Propagate integral to output register (carry state forward)
        h_out[i].jepa_energy = h_in[i].jepa_energy;
        max_abs = max_abs.max(val.abs());
    }

    // mHC projection: if ||h|| > radius, scale down to radius
    // Skip if base model (alpha_w is null), as base models were not trained with bounded residual streams
    if max_abs > 0.0 && !alpha_w.is_null() {
        let mut sum_sq = 0.0f32;
        let scale_down = 1.0 / max_abs;
        for h in h_out.iter_mut().take(h_in.len()) {
            let v = h.read_accum() * scale_down;
            sum_sq += v * v;
        }
        let norm = (sum_sq.sqrt() * max_abs).max(1e-8);
        if norm > radius {
            let scale = radius / norm;
            for h in h_out.iter_mut() {
                let scaled = h.read_accum() * scale;
                h.write_accum(scaled);
            }
        }
    }
}

pub fn layer_is_valid(layer: &SlimeLayer) -> bool {
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
    // Phase B: overlap up + gate GEMVs (one pool barrier)
    unsafe {
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
        let mut i = 0;
        while i + 8 <= ffn_mid {
            for k in 0..8 {
                mid[i + k] *= up[i + k];
            }
            i += 8;
        }
        while i < ffn_mid {
            mid[i] *= up[i];
            i += 1;
        }
    }

    if let Some(t) = tape.as_mut() {
        t.ffn_mid_f32
            .copy_from_slice(&ws.ffn_mid_f32[..layer.ffn_mid]);
    }

    if !layer.ffn_sub_norm_w.is_null() {
        unsafe {
            let mut sum_sq = 0.0f32;
            for i in 0..ffn_mid {
                let x = ws.ffn_mid_f32[i];
                sum_sq += x * x;
            }
            let rms = (sum_sq / ffn_mid as f32 + eps).sqrt();
            for i in 0..ffn_mid {
                ws.ffn_mid_f32[i] = ws.ffn_mid_f32[i] / rms * *layer.ffn_sub_norm_w.add(i);
            }
        }
    }

    let ffn_hid_peak = {
        let mut p = 0.0f32;
        for i in 0..ffn_mid {
            p = p.max(ws.ffn_mid_f32[i].abs());
        }
        (p / 127.0f32).max(1e-8f32)
    };
    for i in 0..ffn_mid {
        ws.o_act_i8[i] =
            (ws.ffn_mid_f32[i] / ffn_hid_peak.clamp(1e-8, 1e8)).clamp(-127.0, 127.0) as i8;
    }

    for i in 0..ffn_mid {
        ws.ffn_gate_f32[i] = ws.o_act_i8[i] as f32 * ffn_hid_peak;
    }
    unsafe {
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

/// Dense FFN by default. For Mini MoE (L-11) use [`evaluate_slime_block_moe`].
pub fn evaluate_slime_block(
    layer: &SlimeLayer,
    layer_idx: usize,
    ws: &mut SlimeWorkspace,
    pos: usize,
    eps: f32,
    tape: Option<&mut crate::mud::slime_backward::SlimeLayerTape>,
) -> crate::mud::slime_jepa::TensorDiagnostics {
    evaluate_slime_block_moe(layer, layer_idx, ws, pos, eps, tape, None, None)
}

/// L-11: same as [`evaluate_slime_block`] with optional ExpertBus FFN.
/// When `moe` is None/empty → dense layer FFN (C7 compatibility).
#[allow(clippy::too_many_arguments)]
pub fn evaluate_slime_block_moe(
    layer: &SlimeLayer,
    layer_idx: usize,
    ws: &mut SlimeWorkspace,
    pos: usize,
    eps: f32,
    mut tape: Option<&mut crate::mud::slime_backward::SlimeLayerTape>,
    moe: Option<&crate::mud::expert_bus::ExpertBus>,
    moe_scratch: Option<&mut crate::mud::expert_bus::ExpertScratch>,
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
    // ── Step 1: RMSNorm → i8 ──
    let act_scale = slime_rmsnorm_i8(
        &ws.registers[..hidden],
        &mut ws.gemv_accum,
        layer.attn_norm_w,
        &mut ws.norm_i8,
        eps,
    );

    if let Some(t) = tape.as_mut() {
        t.norm_i8_attn.copy_from_slice(&ws.norm_i8);
        t.attn_act_scale = act_scale;
    }

    if !act_scale.is_finite() {
        panic!(
            "NaN act_scale={} from RMSNorm attn (regs[0]={})",
            act_scale,
            ws.registers[0].read_accum()
        );
    }
    let mut peak_norm = 0i8;
    for &v in ws.norm_i8.iter() {
        if v.abs() > peak_norm {
            peak_norm = v.abs();
        }
    }
    if peak_norm == 0 && act_scale < 1e-7 {
        let mut nz_regs = 0i32;
        let mut first_nz = hidden;
        for i in 0..hidden {
            if ws.registers[i].read_accum() != 0.0 {
                nz_regs += 1;
                if first_nz == hidden {
                    first_nz = i;
                }
            }
        }
        panic!(
            "Dead RMSNorm L{}: peak_norm=0 act_scale={:.2e} nz_regs={} first_nz={}",
            layer_idx, act_scale, nz_regs, first_nz
        );
    }

    // ── Step 2: Dequantize i8→f32, then Q/K/V GEMV (Phase B: single barrier) ──
    for i in 0..hidden {
        ws.ffn_out_f32[i] = ws.norm_i8[i] as f32 * act_scale;
    }
    unsafe {
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

    // ── Step 2a: Qwen3 q_norm / k_norm (per-head RMSNorm) before RoPE ──
    // Weights are `[head_dim]`; applied independently to each head slice.
    if !layer.q_norm_w.is_null() {
        unsafe {
            for h in 0..n_heads {
                let base = h * head_d;
                let mut sum_sq = 0.0f32;
                for i in 0..head_d {
                    let x = ws.q_f32[base + i];
                    sum_sq += x * x;
                }
                let inv_rms = 1.0 / (sum_sq / head_d as f32 + eps).sqrt();
                for i in 0..head_d {
                    let w = *layer.q_norm_w.add(i);
                    ws.q_f32[base + i] *= inv_rms * w;
                }
            }
        }
    }
    if !layer.k_norm_w.is_null() {
        unsafe {
            for h in 0..n_kv_heads {
                let base = h * head_d;
                let mut sum_sq = 0.0f32;
                for i in 0..head_d {
                    let x = ws.k_f32[base + i];
                    sum_sq += x * x;
                }
                let inv_rms = 1.0 / (sum_sq / head_d as f32 + eps).sqrt();
                for i in 0..head_d {
                    let w = *layer.k_norm_w.add(i);
                    ws.k_f32[base + i] *= inv_rms * w;
                }
            }
        }
    }

    // ── Step 2b: RoPE on Q (all heads) and K (kv heads), in-place ──
    if layer.rope_theta > 0.0 {
        let half_d = head_d / 2;
        for h in 0..n_heads {
            let base = h * head_d;
            for i in 0..half_d {
                let theta = pos as f32 * layer.rope_theta.powf(-2.0 * i as f32 / head_d as f32);
                let (sin, cos) = theta.sin_cos();
                let x0 = ws.q_f32[base + i];
                let x1 = ws.q_f32[base + i + half_d];
                ws.q_f32[base + i] = x0 * cos - x1 * sin;
                ws.q_f32[base + i + half_d] = x1 * cos + x0 * sin;
            }
        }
        for h in 0..n_kv_heads {
            let base = h * head_d;
            for i in 0..half_d {
                let theta = pos as f32 * layer.rope_theta.powf(-2.0 * i as f32 / head_d as f32);
                let (sin, cos) = theta.sin_cos();
                let x0 = ws.k_f32[base + i];
                let x1 = ws.k_f32[base + i + half_d];
                ws.k_f32[base + i] = x0 * cos - x1 * sin;
                ws.k_f32[base + i + half_d] = x1 * cos + x0 * sin;
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
    let mut csa_blocks: Vec<usize> =
        Vec::with_capacity(csa_pol.top_k.saturating_add(csa_pol.tail).saturating_add(8));
    let mut csa_index_scratch: Vec<f32> = if csa_active {
        vec![0.0f32; hca_max_pos]
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

        let mut sum_exp = 0.0;
        for i in 0..total_attn_elements {
            ws.scores[i] = (ws.scores[i] - max_score).exp();

            // Weight each compressed block's attention mass by pool size
            if i < n_hca_sel {
                ws.scores[i] *= ws.hca_compression_ratio as f32;
            }

            sum_exp += ws.scores[i];
        }
        let inv_sum = 1.0 / (sum_exp + 1e-10);
        for i in 0..total_attn_elements {
            ws.scores[i] *= inv_sum;
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
                panic!("NaN before Step 5");
            }
        }

        let mut p = 0.0f32;
        for i in 0..hidden {
            p = p.max(ws.o_act_f32[i].abs());
        }
        p.max(1e-8f32)
    };
    let inv_attn_peak = 127.0f32 / attn_peak;
    for i in 0..hidden {
        ws.o_act_i8[i] = (ws.o_act_f32[i] * inv_attn_peak).clamp(-127.0, 127.0) as i8;
    }

    let attn_act_scale = attn_peak / 127.0f32;
    if !attn_act_scale.is_finite() {
        panic!("NaN in attn_act_scale step 5");
    }

    for i in 0..hidden {
        ws.ffn_out_f32[i] = ws.o_act_i8[i] as f32 * attn_act_scale;
    }
    unsafe {
        ternary_gemv_rowwise(
            &ws.ffn_out_f32[..hidden],
            layer.o_w,
            &mut ws.o_act_f32,
            layer.o_scales,
            hidden,
            hidden,
        );
        for i in 0..hidden {
            if !ws.o_act_f32[i].is_finite() {
                panic!("NaN after O GEMV step 5");
            }
        }
    }

    // ── Step 6: attn_sub_norm (optional) ──
    if !layer.attn_sub_norm_w.is_null() {
        unsafe {
            let mut sum_sq = 0.0f32;
            for i in 0..hidden {
                let x = ws.o_act_f32[i];
                sum_sq += x * x;
            }
            let rms = (sum_sq / hidden as f32 + eps).sqrt();
            for i in 0..hidden {
                ws.o_act_f32[i] = ws.o_act_f32[i] / rms * *layer.attn_sub_norm_w.add(i);
            }
        }
    }

    // ── Step 7: Attention JEPA + Residual ──
    for i in 0..hidden {
        if !ws.o_act_f32[i].is_finite() {
            panic!("NaN before Step 7");
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
    );
    ws.jepa_integral = ws.jepa_integral * 0.99 + delta_attn;

    // Phase 1 & 2 & 3: Manifold-Constrained Hyper-Connections (mHC)
    // Geometric bounding instead of clipping or adaptive dt
    let layer_radius = if layer.mhc_radius_w.is_null() {
        1000.0
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
    mhc_residual(
        &ws.registers[..hidden],
        &mut ws.registers_tmp[..hidden],
        &ws.o_act_f32[..hidden],
        layer_radius,
        layer.mhc_alpha_w,
        layer.mhc_beta_w,
    );

    // ── Step 8: FFN RMSNorm → i8 ──
    let ffn_act_scale = slime_rmsnorm_i8(
        &ws.registers_tmp[..hidden],
        &mut ws.gemv_accum,
        layer.ffn_norm_w,
        &mut ws.norm_i8,
        eps,
    );

    if let Some(t) = tape.as_mut() {
        t.norm_i8_ffn.copy_from_slice(&ws.norm_i8);
        t.ffn_act_scale = ffn_act_scale;
    }

    // ── Step 9–12: FFN (dense or L-11 Mini MoE) ──
    let ffn_mid = layer.ffn_mid;
    for i in 0..hidden {
        ws.ffn_out_f32[i] = ws.norm_i8[i] as f32 * ffn_act_scale;
    }

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
            eprintln!("[L-11 MoE] forward failed ({e}); dense FFN fallback");
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

    // ── Step 13: FFN JEPA + Residual ──
    let ffn_jepa_idx = 2 * layer_idx + 1;
    let tape_ffn = tape.as_mut().map(|t| &mut t.ffn_v_jepa[..hidden]);
    let ffn_z_start = ffn_jepa_idx * ws.hidden_size;
    let ffn_z = &mut ws.jepa_z[ffn_z_start..ffn_z_start + ws.hidden_size];
    let delta_ffn = jepa_stabilizer(
        &mut ws.ffn_out_f32[..hidden],
        &mut ws.registers_tmp,
        &mut ws.jepa_mu[ffn_jepa_idx],
        &mut ws.jepa_inv_sigma[ffn_jepa_idx],
        &mut ws.jepa_var_ema[ffn_jepa_idx],
        ffn_z,
        tape_ffn,
    );
    ws.jepa_integral = ws.jepa_integral * 0.99 + delta_ffn;

    let mut all_zero = true;

    // Phase 1 & 2 & 3: Manifold-Constrained Hyper-Connections (mHC)
    let layer_radius = if layer.mhc_radius_w.is_null() {
        1000.0
    } else {
        unsafe { *layer.mhc_radius_w }
    };
    // Phase 1 (mHC trainable): capture h_in and f_h for the post-ffn mHC site.
    if let Some(t) = tape.as_mut() {
        for i in 0..hidden {
            t.mhc_ffn_h_in[i] = ws.registers_tmp[i].read_accum();
            t.mhc_ffn_f_h[i] = ws.ffn_out_f32[i];
        }
    }
    mhc_residual(
        &ws.registers_tmp[..hidden],
        &mut ws.registers[..hidden],
        &ws.ffn_out_f32[..hidden],
        layer_radius,
        layer.mhc_alpha_w,
        layer.mhc_beta_w,
    );

    for i in 0..hidden {
        if ws.registers[i].read_accum() != 0.0 {
            all_zero = false;
        }
    }
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
        let v0 = ws.registers_tmp[0].read_accum();
        let g0 = ws.registers_tmp[0].jepa_energy;
        let mut nz_regs = 0i32;
        let mut nz_tmp = 0i32;
        let mut sum_abs_regs = 0f64;
        for i in 0..hidden {
            if ws.registers[i].read_accum() != 0.0 {
                nz_regs += 1;
            }
            if ws.registers_tmp[i].read_accum() != 0.0 {
                nz_tmp += 1;
            }
            sum_abs_regs += ws.registers[i].read_accum().abs() as f64;
        }
        let mean_abs = sum_abs_regs / hidden as f64;
        let g_mean: f64 = (0..hidden)
            .map(|i| ws.registers_tmp[i].jepa_energy as f64)
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
        println!("LAYER L{} POS {} ALLZERO (α₀={:.4} β₀={:.4} R={:.2} regs_tmp[0]={:.4} integral₀={:.4} nz_regs={} nz_tmp={} mean_abs_regs={:.2e} integral_mean={:.4} NaN_KV={})",
            layer_idx, pos, a_first, b_first, r, v0, g0, nz_regs, nz_tmp, mean_abs, g_mean, nan_kv);
    }

    // L-15: mark activation tape valid for backward / recompute accounting
    if let Some(t) = tape.as_mut() {
        t.valid = true;
        t.pos = pos;
    }

    // ── Diagnostic Health Check ──
    crate::mud::slime_jepa::check_tensor_health(&ws.registers, 1.0)
}

/// Apply output_norm to registers in-place (final RMSNorm before LM head).
/// `output_norm_w` must be a valid pointer to [hidden] f32 weights.
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
fn maybe_cmud_think(ws: &mut SlimeWorkspace) {
    if !crate::mud::cmud::cmud_think_enabled() {
        return;
    }
    let hidden = ws.hidden_size;
    let scratch = &mut ws.ffn_out_f32[..hidden];
    for (i, reg) in ws.registers.iter().take(hidden).enumerate() {
        scratch[i] = reg.read_accum();
    }
    let _steps = crate::mud::cmud::maybe_think_collapse(scratch, ws.mhc_radius);
    for (i, reg) in ws.registers.iter_mut().take(hidden).enumerate() {
        reg.write_accum(scratch[i]);
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
            println!("APPLY_OUTPUT_NORM PRODUCED ALL ZEROS!");
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
    let use_vk = std::env::var("MUD_USE_VULKAN")
        .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(true);
    if !use_vk {
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
    let mut x = vec![0.0f32; hidden];
    let mut y = vec![0.0f32; hidden];
    for (xi, reg) in x.iter_mut().zip(ws.registers.iter()) {
        *xi = reg.read_accum();
    }
    let w = unsafe { std::slice::from_raw_parts(output_norm_w, hidden) };
    // SAFETY: buffers sized to hidden; ctx exclusively locked.
    let ok = unsafe {
        ctx.dispatch_rms_norm_sync(&x, w, &mut y, hidden, 1, eps)
            .is_ok()
    };
    if !ok {
        return false;
    }
    for (yi, reg) in y.iter().zip(ws.registers.iter_mut()) {
        let v = if yi.is_finite() { *yi } else { 0.0 };
        reg.write_accum(v);
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
    let use_vk = std::env::var("MUD_USE_VULKAN")
        .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(true);
    if !use_vk {
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
    unsafe {
        ctx.dispatch_mha_sync(q, k, v, out, seq_len, n_head, n_kv_head, head_dim)
            .is_ok()
    }
}

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
            mhc_radius_w: std::ptr::null(),
            n_kv_heads: 1,
            ffn_mid: hidden,
            rope_theta: 0.0,
        };

        ws.registers[0].write_accum(100.0);
        ws.jepa_z[0] = 1.0; // layer 0 attention head, dim 0: initial z = 1.0
        ws.jepa_z[ws.hidden_size] = 1.0; // layer 0 FFN head, dim 0: initial z = 1.0
        evaluate_slime_block(&layer, 0, &mut ws, 0, 1e-6, None);
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
        if !gpu_gemv_policy_active() {
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
}
