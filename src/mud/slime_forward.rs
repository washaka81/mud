use crate::mud::slime::{SlimeRegister, SlimeWorkspace};
use crate::mud::slime_jepa::jepa_stabilizer;


fn slime_rmsnorm_i8(regs: &[SlimeRegister], gemv_accum: &mut [f32], weights: *const f32, out_i8: &mut [i8], eps: f32) -> f32 {
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
    peak = peak.max(1e-8);
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

unsafe fn ternary_gemv_rowwise(
    acts_f32: &[f32],
    w_u8: *const u8,
    out_f32: &mut [f32],
    scales: *const f32,
    n_out: usize,
    n_in: usize,
) {
    // Vulkan dispatch para matrices grandes (FFN: 6912×576)
    // Activado via MUD_USE_VULKAN=1, threshold: ~1M elementos
    if n_out * n_in >= 1_000_000
        && std::env::var("MUD_USE_VULKAN").as_deref() == Ok("1")
        && crate::vulkan::vulkan_backend::vb_gemm_forward(
            acts_f32.as_ptr(),
            w_u8 as *const u32,
            out_f32.as_mut_ptr(),
            1,
            n_in as u32,
            n_out as u32,
            scales,
            1,
        ) == 0
    {
        return; // Vulkan tuvo éxito, P-cores libres para siguiente token
    }

    // CPU path: batch de 4 rows via ASM AVX2 (ternary FP32)
    // Parallelized over 4 P-Cores for extreme DDR5 saturation
    let row_u32s = n_in / 8;
    let w_u32 = w_u8 as *const u32;
    
    let pool = crate::mud::pcore_pool::get_pool();
    let rows_per_task = ((n_out / 8) / 4 * 4).max(4); // Multiple of 4
    
    let acts_p = acts_f32.as_ptr() as usize;
    let w_p = w_u32 as usize;
    let out_p = out_f32.as_mut_ptr() as usize;
    let scales_p = scales as usize;
    
    for i in 0..8 {
        let start_row = i * rows_per_task;
        let end_row = if i == 7 { n_out } else { start_row + rows_per_task };
        if start_row >= end_row { break; }
        
        pool.execute(move || {
            let acts = acts_p as *const f32;
            let w = w_p as *const u32;
            let out = out_p as *mut f32;
            let sc = scales_p as *const f32;
            
            let mut row = start_row;
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
                    crate::asm::ternary_gemv(n_in, acts, w.add(r * row_u32s), out.add(r), if s.is_finite() { s } else { 0.0 });
                }
            }
        });
    }
    pool.wait_all();
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
        let alpha = if alpha_w.is_null() { 1.0 } else { unsafe { *alpha_w.add(i) } };
        let beta = if beta_w.is_null() { 1.0 } else { unsafe { *beta_w.add(i) } };
        // Gate from JEPA integral (I-controller): smooth, low-pass filtered
        let gate = h_in[i].gate();
        let var_norm = 1.0 / (gate * gate + (1.0 - gate) * (1.0 - gate)).max(1e-8).sqrt();
        let val = (gate * alpha * h_in[i].read_accum() + (1.0 - gate) * beta * f_h[i]) * var_norm;
        
        h_out[i].write_accum(val);
        // Propagate integral to output register (carry state forward)
        h_out[i].write_integral(h_in[i].read_integral());
        max_abs = max_abs.max(val.abs());
    }

    // mHC projection: if ||h|| > radius, scale down to radius
    if max_abs > 0.0 {
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
pub fn evaluate_slime_block(
    layer: &SlimeLayer,
    layer_idx: usize,
    ws: &mut SlimeWorkspace,
    pos: usize,
    eps: f32,
    mut tape: Option<&mut crate::mud::slime_backward::SlimeLayerTape>,
) -> crate::mud::slime_jepa::TensorDiagnostics {
    debug_assert!(layer_is_valid(layer));
    let hidden = ws.hidden_size;
    let n_heads = ws.n_heads;
    let n_kv_heads = layer.n_kv_heads;
    let head_d = ws.head_dim;
    let kv_offset = ws.max_pos * head_d;
    let layer_offset = layer_idx * (n_kv_heads * kv_offset);
    
    // Priority 51: HCA Offsets
    let hca_max_pos = (ws.max_pos / ws.hca_compression_ratio).max(1);
    let hca_kv_offset = hca_max_pos * head_d;
    let hca_layer_offset = layer_idx * (n_kv_heads * hca_kv_offset);
    // ── Step 1: RMSNorm → i8 ──
    let act_scale = slime_rmsnorm_i8(&ws.registers[..hidden], &mut ws.gemv_accum, layer.attn_norm_w, &mut ws.norm_i8, eps);

    if let Some(t) = tape.as_mut() {
        t.norm_i8_attn.copy_from_slice(&ws.norm_i8);
        t.attn_act_scale = act_scale;
    }

    if !act_scale.is_finite() { panic!("NaN act_scale={} from RMSNorm attn (regs[0]={})", act_scale, ws.registers[0].read_accum()); }
    let mut peak_norm = 0i8;
    for &v in ws.norm_i8.iter() { if v.abs() > peak_norm { peak_norm = v.abs(); } }
    if peak_norm == 0 && act_scale < 1e-7 {
        let mut nz_regs = 0i32;
        let mut first_nz = hidden;
        for i in 0..hidden {
            if ws.registers[i].read_accum() != 0.0 { nz_regs += 1; if first_nz == hidden { first_nz = i; } }
        }
        panic!("Dead RMSNorm L{}: peak_norm=0 act_scale={:.2e} nz_regs={} first_nz={}", layer_idx, act_scale, nz_regs, first_nz);
    }

    // ── Step 2: Dequantize i8→f32, then Ternary GEMV (FP32 AVX2) ──
    for i in 0..hidden {
        ws.ffn_out_f32[i] = ws.norm_i8[i] as f32 * act_scale;
    }
    unsafe {
        ternary_gemv_rowwise(&ws.ffn_out_f32[..hidden], layer.q_w, &mut ws.q_f32, layer.q_scales, hidden, hidden);
        ternary_gemv_rowwise(&ws.ffn_out_f32[..hidden], layer.k_w, &mut ws.k_f32, layer.k_scales, n_kv_heads * head_d, hidden);
        ternary_gemv_rowwise(&ws.ffn_out_f32[..hidden], layer.v_w, &mut ws.v_f32, layer.v_scales, n_kv_heads * head_d, hidden);
    }

    // ── Step 2b: RoPE on Q (all heads) and K (kv heads), in-place ──
    if layer.rope_theta > 0.0 {
        for h in 0..n_heads {
            let base = h * head_d;
            for i in 0..head_d / 2 {
                let idx = 2 * i;
                let theta = pos as f32 * layer.rope_theta.powf(-2.0 * idx as f32 / head_d as f32);
                let (sin, cos) = theta.sin_cos();
                let x0 = ws.q_f32[base + idx];
                let x1 = ws.q_f32[base + idx + 1];
                ws.q_f32[base + idx] = x0 * cos - x1 * sin;
                ws.q_f32[base + idx + 1] = x1 * cos + x0 * sin;
            }
        }
        for h in 0..n_kv_heads {
            let base = h * head_d;
            for i in 0..head_d / 2 {
                let idx = 2 * i;
                let theta = pos as f32 * layer.rope_theta.powf(-2.0 * idx as f32 / head_d as f32);
                let (sin, cos) = theta.sin_cos();
                let x0 = ws.k_f32[base + idx];
                let x1 = ws.k_f32[base + idx + 1];
                ws.k_f32[base + idx] = x0 * cos - x1 * sin;
                ws.k_f32[base + idx + 1] = x1 * cos + x0 * sin;
            }
        }
    }

    // ── Step 3: Store K, V in KV cache ──
    for kv_h in 0..n_kv_heads {
        let cache_base = layer_offset + kv_h * kv_offset + pos * head_d;
        for d in 0..head_d {
            ws.kv_cache[cache_base + d] = ws.k_f32[kv_h * head_d + d];
            ws.v_cache[cache_base + d] = ws.v_f32[kv_h * head_d + d];
        }
    }

    // Priority 51: HCA Compression (Mean Pooling of Historical Tokens)
    let hist_token_idx = pos.saturating_sub(ws.hca_window);
    if pos >= ws.hca_window && hist_token_idx % ws.hca_compression_ratio == ws.hca_compression_ratio - 1 {
        let comp_t = hist_token_idx / ws.hca_compression_ratio;
        if comp_t < hca_max_pos {
            for kv_h in 0..n_kv_heads {
                let hca_cache_base = hca_layer_offset + kv_h * hca_kv_offset + comp_t * head_d;
                for d in 0..head_d {
                    let mut sum_k = 0.0;
                    let mut sum_v = 0.0;
                    for i in 0..ws.hca_compression_ratio {
                        let t_old = hist_token_idx - i;
                        let old_cache_base = layer_offset + kv_h * kv_offset + t_old * head_d;
                        sum_k += ws.kv_cache[old_cache_base + d];
                        sum_v += ws.v_cache[old_cache_base + d];
                    }
                    let inv_r = 1.0 / (ws.hca_compression_ratio as f32);
                    ws.hca_kv_cache[hca_cache_base + d] = sum_k * inv_r;
                    ws.hca_v_cache[hca_cache_base + d] = sum_v * inv_r;
                }
            }
        }
    }

    // ── Step 4: Scaled Dot-Product Attention (f32) with HCA (Priority 51) ──
    let inv_sqrt_d = 1.0 / (head_d as f32).sqrt();
    let gqa_scale = n_heads / n_kv_heads;

    for h in 0..n_heads {
        let kv_h = h / gqa_scale;
        let q_off = h * head_d;

        let hist_end_idx = pos.saturating_sub(ws.hca_window);
        let num_comp_tokens = hist_end_idx / ws.hca_compression_ratio;
        let recent_start = num_comp_tokens * ws.hca_compression_ratio;

        let mut max_score = f32::NEG_INFINITY;
        let mut score_idx = 0;

        // 1. Attention over compressed historical tokens
        for comp_t in 0..num_comp_tokens {
            let hca_k_base = hca_layer_offset + kv_h * hca_kv_offset + comp_t * head_d;
            let mut s = 0.0;
            for d in 0..head_d {
                let qv = ws.q_f32[q_off + d];
                let kv = ws.hca_kv_cache[hca_k_base + d];
                if !qv.is_finite() { panic!("NaN q_f32[h={}][d={}] = {}", h, d, qv); }
                if !kv.is_finite() { panic!("NaN hca_kv_cache[kv_h={}][comp_t={}][d={}] = {}", kv_h, comp_t, d, kv); }
                s += qv * kv;
            }
            s *= inv_sqrt_d;
            ws.scores[score_idx] = s;
            if s > max_score {
                max_score = s;
            }
            score_idx += 1;
        }

        // 2. Attention over recent high-fidelity tokens
        for t in recent_start..=pos {
            let k_pos_base = layer_offset + kv_h * kv_offset + t * head_d;
            let mut s = 0.0;
            for d in 0..head_d {
                let qv = ws.q_f32[q_off + d];
                let kv = ws.kv_cache[k_pos_base + d];
                if !qv.is_finite() { panic!("NaN q_f32[h={}][d={}] = {}", h, d, qv); }
                if !kv.is_finite() { panic!("NaN kv_cache[kv_h={}][t={}][d={}] = {}", kv_h, t, d, kv); }
                s += qv * kv;
            }
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
            
            // Weight the compressed token's attention mass
            if i < num_comp_tokens {
                ws.scores[i] *= ws.hca_compression_ratio as f32;
            }
            
            sum_exp += ws.scores[i];
        }
        let inv_sum = 1.0 / (sum_exp + 1e-10);
        for i in 0..total_attn_elements {
            ws.scores[i] *= inv_sum;
        }

        for d in 0..head_d {
            let mut acc = 0.0;
            let mut s_idx = 0;
            
            for comp_t in 0..num_comp_tokens {
                let sv = ws.scores[s_idx];
                if !sv.is_finite() { panic!("NaN scores[{}] = {}", s_idx, sv); }
                let hca_v_base = hca_layer_offset + kv_h * hca_kv_offset + comp_t * head_d;
                let vv = ws.hca_v_cache[hca_v_base + d];
                if !vv.is_finite() { panic!("NaN hca_v_cache[kv_h={}][comp_t={}][d={}] = {}", kv_h, comp_t, d, vv); }
                acc += sv * vv;
                s_idx += 1;
            }
            
            for t in recent_start..=pos {
                let sv = ws.scores[s_idx];
                if !sv.is_finite() { panic!("NaN scores[{}] = {}", s_idx, sv); }
                let v_pos_base = layer_offset + kv_h * kv_offset + t * head_d;
                let vv = ws.v_cache[v_pos_base + d];
                if !vv.is_finite() { panic!("NaN v_cache[kv_h={}][t={}][d={}] = {}", kv_h, t, d, vv); }
                acc += sv * vv;
                s_idx += 1;
            }
            ws.o_act_f32[q_off + d] = acc;
            if !acc.is_finite() { panic!("NaN inside Step 4 attention acc"); }
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
    for i in 0..hidden { if !ws.o_act_f32[i].is_finite() { panic!("NaN before Step 5"); } }

        let mut p = 0.0f32;
        for i in 0..hidden { p = p.max(ws.o_act_f32[i].abs()); }
        p.max(1e-8f32)
    };
    let inv_attn_peak = 127.0f32 / attn_peak;
    for i in 0..hidden {
        ws.o_act_i8[i] = (ws.o_act_f32[i] * inv_attn_peak).clamp(-127.0, 127.0) as i8;
    }

    let attn_act_scale = attn_peak / 127.0f32;
    if !attn_act_scale.is_finite() { panic!("NaN in attn_act_scale step 5"); }

    for i in 0..hidden {
        ws.ffn_out_f32[i] = ws.o_act_i8[i] as f32 * attn_act_scale;
    }
    unsafe {
        ternary_gemv_rowwise(&ws.ffn_out_f32[..hidden], layer.o_w, &mut ws.o_act_f32, layer.o_scales, hidden, hidden);
        for i in 0..hidden { if !ws.o_act_f32[i].is_finite() { panic!("NaN after O GEMV step 5"); } }
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
    for i in 0..hidden { if !ws.o_act_f32[i].is_finite() { panic!("NaN before Step 7"); } }

    let attn_jepa_idx = 2 * layer_idx;
    let tape_attn = tape.as_mut().map(|t| &mut t.attn_v_jepa[..hidden]);
    let attn_z_start = attn_jepa_idx * ws.hidden_size;
    let attn_z = &mut ws.jepa_z[attn_z_start .. attn_z_start + ws.hidden_size];
    let delta_attn = jepa_stabilizer(&mut ws.o_act_f32[..hidden], &mut ws.registers, &mut ws.jepa_mu[attn_jepa_idx], &mut ws.jepa_inv_sigma[attn_jepa_idx], &mut ws.jepa_var_ema[attn_jepa_idx], attn_z, tape_attn);
    ws.jepa_integral = ws.jepa_integral * 0.99 + delta_attn;
    
    // Phase 1 & 2 & 3: Manifold-Constrained Hyper-Connections (mHC)
    // Geometric bounding instead of clipping or adaptive dt
    let layer_radius = if layer.mhc_radius_w.is_null() { 1000.0 } else { unsafe { *layer.mhc_radius_w } };
    mhc_residual(
        &ws.registers[..hidden],
        &mut ws.registers_tmp[..hidden],
        &ws.o_act_f32[..hidden],
        layer_radius,
        layer.mhc_alpha_w,
        layer.mhc_beta_w,
    );


    // ── Step 8: FFN RMSNorm → i8 ──
    let ffn_act_scale = slime_rmsnorm_i8(&ws.registers_tmp[..hidden], &mut ws.gemv_accum, layer.ffn_norm_w, &mut ws.norm_i8, eps);

    if let Some(t) = tape.as_mut() {
        t.norm_i8_ffn.copy_from_slice(&ws.norm_i8);
        t.ffn_act_scale = ffn_act_scale;
    }

    // ── Step 9: FFN up & gate GEMV ──
    let ffn_mid = layer.ffn_mid;
    for i in 0..hidden {
        ws.ffn_out_f32[i] = ws.norm_i8[i] as f32 * ffn_act_scale;
    }
    unsafe {
        ternary_gemv_rowwise(&ws.ffn_out_f32[..hidden], layer.ffn_up_w, &mut ws.ffn_up_f32, layer.ffn_up_scales, ffn_mid, hidden);
        ternary_gemv_rowwise(&ws.ffn_out_f32[..hidden], layer.ffn_gate_w, &mut ws.ffn_gate_f32, layer.ffn_gate_scales, ffn_mid, hidden);
    }

    if let Some(t) = tape.as_mut() {
        t.ffn_up_f32.copy_from_slice(&ws.ffn_up_f32[..ffn_mid]);
        t.ffn_gate_f32.copy_from_slice(&ws.ffn_gate_f32[..ffn_mid]);
    }

    // ── Step 10: relu2(gate) * up  (BitNet b1.58 uses ReLU², not SiLU) ──
    for i in 0..ffn_mid {
        let g = ws.ffn_gate_f32[i];
        ws.ffn_mid_f32[i] = (if g > 0.0 { g * g } else { 0.0 }) * ws.ffn_up_f32[i];
    }

    if let Some(t) = tape.as_mut() {
        t.ffn_mid_f32.copy_from_slice(&ws.ffn_mid_f32[..layer.ffn_mid]);
    }

    // ── Step 10b: ffn_sub_norm on intermediate (optional, [ffn_mid]) ──
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

    // ── Step 11: Re-quantize FFN hidden → i8 ──
    let ffn_hid_peak = {
        let mut p = 0.0f32;
        for i in 0..ffn_mid { p = p.max(ws.ffn_mid_f32[i].abs()); }
        (p / 127.0f32).max(1e-8f32)
    };
    for i in 0..ffn_mid {
        ws.o_act_i8[i] = (ws.ffn_mid_f32[i] / ffn_hid_peak.clamp(1e-8, 1e8)).clamp(-127.0, 127.0) as i8;
    }

    // ── Step 12: Ternary GEMV down projection ──
    for i in 0..ffn_mid {
        ws.ffn_gate_f32[i] = ws.o_act_i8[i] as f32 * ffn_hid_peak;
    }
    unsafe {
        ternary_gemv_rowwise(&ws.ffn_gate_f32[..ffn_mid], layer.ffn_down_w, &mut ws.ffn_out_f32, layer.ffn_down_scales, hidden, ffn_mid);
    }

    // ── Step 13: FFN JEPA + Residual ──
    let ffn_jepa_idx = 2 * layer_idx + 1;
    let tape_ffn = tape.as_mut().map(|t| &mut t.ffn_v_jepa[..hidden]);
    let ffn_z_start = ffn_jepa_idx * ws.hidden_size;
    let ffn_z = &mut ws.jepa_z[ffn_z_start .. ffn_z_start + ws.hidden_size];
    let delta_ffn = jepa_stabilizer(&mut ws.ffn_out_f32[..hidden], &mut ws.registers_tmp, &mut ws.jepa_mu[ffn_jepa_idx], &mut ws.jepa_inv_sigma[ffn_jepa_idx], &mut ws.jepa_var_ema[ffn_jepa_idx], ffn_z, tape_ffn);
    ws.jepa_integral = ws.jepa_integral * 0.99 + delta_ffn;

    let mut all_zero = true;
    
    // Phase 1 & 2 & 3: Manifold-Constrained Hyper-Connections (mHC)
    let layer_radius = if layer.mhc_radius_w.is_null() { 1000.0 } else { unsafe { *layer.mhc_radius_w } };
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
        let a_first = if layer.mhc_alpha_w.is_null() { 1.0 } else { unsafe { *layer.mhc_alpha_w } };
        let b_first = if layer.mhc_beta_w.is_null() { 1.0 } else { unsafe { *layer.mhc_beta_w } };
        let r = if layer.mhc_radius_w.is_null() { ws.mhc_radius } else { unsafe { *layer.mhc_radius_w } };
        let v0 = ws.registers_tmp[0].read_accum();
        let g0 = ws.registers_tmp[0].read_integral();
        let mut nz_regs = 0i32;
        let mut nz_tmp = 0i32;
        let mut sum_abs_regs = 0f64;
        for i in 0..hidden {
            if ws.registers[i].read_accum() != 0.0 { nz_regs += 1; }
            if ws.registers_tmp[i].read_accum() != 0.0 { nz_tmp += 1; }
            sum_abs_regs += ws.registers[i].read_accum().abs() as f64;
        }
        let mean_abs = sum_abs_regs / hidden as f64;
        let g_mean: f64 = (0..hidden).map(|i| ws.registers_tmp[i].read_integral() as f64).sum::<f64>() / hidden as f64;
        let mut nan_kv = false;
        for i in 0..(ws.kv_cache.len().min(16)) {
            if !ws.kv_cache[i].is_finite() { nan_kv = true; break; }
        }
        for i in 0..(ws.v_cache.len().min(16)) {
            if !ws.v_cache[i].is_finite() { nan_kv = true; break; }
        }
        println!("LAYER L{} POS {} ALLZERO (α₀={:.4} β₀={:.4} R={:.2} regs_tmp[0]={:.4} integral₀={:.4} nz_regs={} nz_tmp={} mean_abs_regs={:.2e} integral_mean={:.4} NaN_KV={})",
            layer_idx, pos, a_first, b_first, r, v0, g0, nz_regs, nz_tmp, mean_abs, g_mean, nan_kv);
    }

    // ── Diagnostic Health Check ──
    crate::mud::slime_jepa::check_tensor_health(&ws.registers, 1.0, ws.iscale)
}


/// Apply output_norm to registers in-place (final RMSNorm before LM head).
/// `output_norm_w` must be a valid pointer to [hidden] f32 weights.
/// Modifies ws.registers in place. No iscale needed (f16 self-scaled).
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn apply_output_norm(ws: &mut SlimeWorkspace, output_norm_w: *const f32, eps: f32) {
    let hidden = ws.hidden_size;
    let mut sum_sq = 0.0f32;
    for i in 0..hidden {
        let x = ws.registers[i].read_accum();
        sum_sq += x * x;
    }
    let rms_inv = 1.0f32 / ((sum_sq / hidden as f32) + eps).sqrt();
    unsafe {
        let mut all_zero = true;
        for i in 0..hidden {
            let xn = ws.registers[i].read_accum() * rms_inv * *output_norm_w.add(i);
            ws.registers[i].write_accum(xn);
            if ws.registers[i].read_accum() != 0.0 {
                all_zero = false;
            }
        }
        if all_zero {
            println!("APPLY_OUTPUT_NORM PRODUCED ALL ZEROS!");
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn test_slime_forward_structure() {
        let hidden = 32;
        let mut ws = SlimeWorkspace::new(hidden, 32, 1, 1, 32, hidden, 1, 128.0);
        let row_sz = hidden / 16 * 4;
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
            q_w: q_w.as_ptr(), k_w: k_w.as_ptr(), v_w: v_w.as_ptr(), o_w: o_w.as_ptr(),
            q_scales: scales.as_ptr(), k_scales: scales.as_ptr(),
            v_scales: scales.as_ptr(), o_scales: scales.as_ptr(),
            ffn_up_w: ffn_up.as_ptr(), ffn_gate_w: ffn_gate.as_ptr(), ffn_down_w: ffn_down.as_ptr(),
            ffn_up_scales: scales.as_ptr(), ffn_gate_scales: scales.as_ptr(), ffn_down_scales: scales.as_ptr(),
            attn_norm_w: norm_w.as_ptr(), ffn_norm_w: norm_w.as_ptr(),
            attn_sub_norm_w: norm_w.as_ptr(), ffn_sub_norm_w: norm_w.as_ptr(),
            mhc_alpha_w: std::ptr::null(), mhc_beta_w: std::ptr::null(), mhc_radius_w: std::ptr::null(),
            n_kv_heads: 1, ffn_mid: hidden,
            rope_theta: 0.0,
        };

        ws.registers[0].write_accum(100.0);
        ws.jepa_z[0] = 1.0; // layer 0 attention head, dim 0: initial z = 1.0
        ws.jepa_z[ws.hidden_size] = 1.0; // layer 0 FFN head, dim 0: initial z = 1.0
        evaluate_slime_block(&layer, 0, &mut ws, 0, 1e-6, None);
        assert!(ws.jepa_mu[0].is_finite(), "attention JEPA mu must be finite");
        assert!(ws.jepa_mu[1].is_finite(), "FFN JEPA mu must be finite");
        assert!(ws.jepa_mu[0] > 0.0, "attention mu should be initialized from batch_mu_z");
    }
}
