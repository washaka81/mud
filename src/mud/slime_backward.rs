use crate::mud::slime::SlimeWorkspace;
use crate::mud::slime_forward::SlimeLayer;

/// Gradients for the SlimeLayer weights (in FP32, accumulating before Adam/SGD).
pub struct SlimeLayerGradients {
    pub q_w_grad: Vec<f32>,
    pub k_w_grad: Vec<f32>,
    pub v_w_grad: Vec<f32>,
    pub o_w_grad: Vec<f32>,

    pub ffn_up_w_grad: Vec<f32>,
    pub ffn_gate_w_grad: Vec<f32>,
    pub ffn_down_w_grad: Vec<f32>,
    
    // RMSNorm gradients
    pub attn_norm_w_grad: Vec<f32>,
    pub ffn_norm_w_grad: Vec<f32>,
}

impl SlimeLayerGradients {
    pub fn new(hidden: usize, ffn_mid: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        let kv_dim = n_kv_heads * head_dim;
        Self {
            q_w_grad: vec![0.0; hidden * hidden],
            k_w_grad: vec![0.0; kv_dim * hidden],
            v_w_grad: vec![0.0; kv_dim * hidden],
            o_w_grad: vec![0.0; hidden * hidden],
            ffn_up_w_grad: vec![0.0; ffn_mid * hidden],
            ffn_gate_w_grad: vec![0.0; ffn_mid * hidden],
            ffn_down_w_grad: vec![0.0; hidden * ffn_mid],
            attn_norm_w_grad: vec![0.0; hidden],
            ffn_norm_w_grad: vec![0.0; hidden],
        }
    }

    pub fn reset(&mut self) {
        self.q_w_grad.fill(0.0);
        self.k_w_grad.fill(0.0);
        self.v_w_grad.fill(0.0);
        self.o_w_grad.fill(0.0);
        self.ffn_up_w_grad.fill(0.0);
        self.ffn_gate_w_grad.fill(0.0);
        self.ffn_down_w_grad.fill(0.0);
        self.attn_norm_w_grad.fill(0.0);
        self.ffn_norm_w_grad.fill(0.0);
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OptimizerStrategy {
    Muon { ns_iters: usize },
    GaLore { rank: usize, update_freq: usize },
    ChunkedAdam { chunk_cols: usize },
    SparseAdam { only_active_rows: bool },
    Adam,
}

pub fn select_optimizer(rows: usize, cols: usize) -> OptimizerStrategy {
    let ratio = rows as f32 / cols as f32;
    let total = rows * cols;

    if total < 100_000 && (0.5..=2.0).contains(&ratio) {
        OptimizerStrategy::Muon { ns_iters: 5 }
    } else if (0.5..=2.0).contains(&ratio) {
        // BitNet-2B size square matrices also use Muon
        OptimizerStrategy::Muon { ns_iters: 5 }
    } else if ratio > 2.5 {
        let rank = (cols / 4).max(8);
        OptimizerStrategy::GaLore { rank, update_freq: 100 }
    } else if ratio < 0.4 {
        let chunk = 512;
        OptimizerStrategy::ChunkedAdam { chunk_cols: chunk }
    } else if total > 10_000_000 {
        OptimizerStrategy::SparseAdam { only_active_rows: true }
    } else {
        OptimizerStrategy::Adam
    }
}

/// Persistent F32 shadow buffers for QAT updates.
/// Since AVX2 QAT works on f32 buffers, we keep the decompressed weights across the batch.
pub struct SlimeLayerShadowF32 {
    pub q_w: Vec<f32>,
    pub k_w: Vec<f32>,
    pub v_w: Vec<f32>,
    pub o_w: Vec<f32>,
    pub ffn_up_w: Vec<f32>,
    pub ffn_gate_w: Vec<f32>,
    pub ffn_down_w: Vec<f32>,
    
    // Optimizers
    pub q_opt: OptimizerStrategy,
    pub k_opt: OptimizerStrategy,
    pub v_opt: OptimizerStrategy,
    pub o_opt: OptimizerStrategy,
    pub ffn_up_opt: OptimizerStrategy,
    pub ffn_gate_opt: OptimizerStrategy,
    pub ffn_down_opt: OptimizerStrategy,
}

impl SlimeLayerShadowF32 {
    pub fn new(hidden: usize, ffn_mid: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        let kv_dim = n_kv_heads * head_dim;
        Self {
            q_w: vec![0.0; hidden * hidden],
            k_w: vec![0.0; kv_dim * hidden],
            v_w: vec![0.0; kv_dim * hidden],
            o_w: vec![0.0; hidden * hidden],
            ffn_up_w: vec![0.0; ffn_mid * hidden],
            ffn_gate_w: vec![0.0; ffn_mid * hidden],
            ffn_down_w: vec![0.0; hidden * ffn_mid],
            q_opt: select_optimizer(hidden, hidden),
            k_opt: select_optimizer(kv_dim, hidden),
            v_opt: select_optimizer(kv_dim, hidden),
            o_opt: select_optimizer(hidden, hidden),
            ffn_up_opt: select_optimizer(ffn_mid, hidden),
            ffn_gate_opt: select_optimizer(ffn_mid, hidden),
            ffn_down_opt: select_optimizer(hidden, ffn_mid),
        }
    }
}

/// Unpacks Ternary2Bit (ELUT 4-bit nibbles, 8 weights per u32) into a flat f32 vector using PRQ scales.
pub fn unpack_ternary2bit_to_f32(packed: &[u8], scales: &[f32], cols: usize, out: &mut [f32]) {
    let rows = scales.len();
    assert_eq!(out.len(), rows * cols);
    assert_eq!(packed.len(), (rows * cols).div_ceil(8) * 4);

    let packed_u32 = unsafe {
        std::slice::from_raw_parts(packed.as_ptr() as *const u32, packed.len() / 4)
    };

    for (r, &scale) in scales.iter().enumerate().take(rows) {
        let row_start = r * cols;
        for c in 0..cols {
            let i = row_start + c;
            let u32_idx = i / 8;
            let shift = (i % 8) * 4;
            let val = (packed_u32[u32_idx] >> shift) & 0xF;
            
            // ELUT packing: we just care about the ternary value.
            // Usually val is mapped: 1 => +1, 2 => -1. Let's assume standard Ternary2Bit mapping in the lowest 2 bits.
            let w = match val & 0xF {
                0x1 => 1.0,
                0xF => -1.0,
                _ => 0.0,
            };
            out[i] = w * scale;
        }
    }
}

/// Memory structure for storing intermediate activations of a SINGLE layer for a SINGLE token.
/// These are required to compute exact derivatives during the Backward Pass (Activation Checkpointing).
pub struct SlimeLayerTape {
    pub norm_i8_attn: Vec<i8>, // i8 input to Attention Q, K, V projections
    pub norm_i8_ffn: Vec<i8>,  // i8 input to FFN Up, Gate projections
    pub q_f32: Vec<f32>,       // Output of Q projection
    pub k_f32: Vec<f32>,       // Output of K projection
    pub v_f32: Vec<f32>,       // Output of V projection
    pub scores: Vec<f32>,      // Softmax attention scores
    pub o_act_f32: Vec<f32>,   // Input to O projection
    pub ffn_up_f32: Vec<f32>,  // Output of FFN Up projection
    pub ffn_gate_f32: Vec<f32>,// Output of FFN Gate projection
    pub ffn_mid_f32: Vec<f32>, // Intermediate FFN activation: relu2(gate) * up
    pub attn_act_scale: f32,   // Scaling factor for Attn i8 input
    pub ffn_act_scale: f32,    // Scaling factor for FFN i8 input
    pub attn_v_jepa: Vec<f32>, // Latent state before Attn jepa_stabilizer
    pub ffn_v_jepa: Vec<f32>,  // Latent state before FFN jepa_stabilizer
    pub pos: usize,            // The sequence position of this tape
}

impl SlimeLayerTape {
    pub fn new(hidden: usize, ffn_mid: usize, n_kv_heads: usize, head_dim: usize, max_seq_len: usize, pos: usize) -> Self {
        let kv_dim = n_kv_heads * head_dim;
        Self {
            norm_i8_attn: vec![0; hidden],
            norm_i8_ffn: vec![0; hidden],
            q_f32: vec![0.0; hidden],
            k_f32: vec![0.0; kv_dim],
            v_f32: vec![0.0; kv_dim],
            scores: vec![0.0; max_seq_len],
            o_act_f32: vec![0.0; hidden],
            ffn_up_f32: vec![0.0; ffn_mid],
            ffn_gate_f32: vec![0.0; ffn_mid],
            ffn_mid_f32: vec![0.0; ffn_mid],
            attn_act_scale: 0.0,
            ffn_act_scale: 0.0,
            attn_v_jepa: vec![0.0; hidden],
            ffn_v_jepa: vec![0.0; hidden],
            pos,
        }
    }

    pub fn reset(&mut self) {
        self.norm_i8_attn.fill(0);
        self.norm_i8_ffn.fill(0);
        self.q_f32.fill(0.0);
        self.k_f32.fill(0.0);
        self.v_f32.fill(0.0);
        self.scores.fill(0.0);
        self.o_act_f32.fill(0.0);
        self.ffn_up_f32.fill(0.0);
        self.ffn_gate_f32.fill(0.0);
        self.ffn_mid_f32.fill(0.0);
        self.attn_act_scale = 0.0;
        self.ffn_act_scale = 0.0;
        self.attn_v_jepa.fill(0.0);
        self.ffn_v_jepa.fill(0.0);
    }
}

/// Helper: Compute gradients for a Ternary GEMV layer using the Straight-Through Estimator (STE).
/// Since we use ELUT 4-bit nibble packing, the weight unpacking will be needed for `grad_x`.
/// # Safety
/// Caller must ensure `w_u8` points to a valid 2-bit packed array of size `(n_in / 16) * 4 * n_out` bytes,
/// and `scales` points to an array of size `n_out`.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
pub unsafe fn ternary_gemv_backward(
    grad_y: &[f32],
    x_f32: &[f32],
    w_u8: *const u8,
    scales: *const f32,
    grad_w: &mut [f32],
    grad_x: &mut [f32],
    n_out: usize,
    n_in: usize,
) {
    // Phase 8: Thawing the Core (STE)
    // Clear grad_x since multiple rows will accumulate into it.
    // Wait, ternary_gemv_backward_avx2 uses `+ ` or `+=` for grad_x?
    // It must accumulate into grad_x! 
    // To parallelize accumulation into grad_x, we need local thread buffers or atomic locks.
    // However, if ternary_gemv_backward_avx2 processes ROWs of W, it processes ROWs of grad_w independently.
    // But EVERY row of W contributes to ALL elements of grad_x!
    // Since we can't easily multithread accumulation into grad_x without a local buffer,
    // we will run the backward pass sequentially for now, or parallelize grad_W only.
    crate::asm::ternary_gemv_backward_avx2(
        grad_y.as_ptr(),
        x_f32.as_ptr(),
        w_u8,
        scales,
        grad_w.as_mut_ptr(),
        grad_x.as_mut_ptr(),
        n_out,
        n_in
    );
}

/// Pre-allocated buffers for the backward pass to strictly satisfy P-01 (Zero-Allocation).
pub struct SlimeBackwardWorkspace {
    pub grad_ffn_up: Vec<f32>,
    pub grad_ffn_gate: Vec<f32>,
    pub grad_ffn_mid: Vec<f32>,
    pub ffn_hid_x: Vec<f32>,
    pub ffn_in_x: Vec<f32>,
    pub grad_ffn_in_up: Vec<f32>,
    pub grad_ffn_in_gate: Vec<f32>,
    pub grad_ffn_norm_out: Vec<f32>,
    pub grad_residual_ffn: Vec<f32>,
    pub grad_o_in: Vec<f32>,
    pub attn_in_x: Vec<f32>,
    pub grad_q_in: Vec<f32>,
    pub grad_k_in: Vec<f32>,
    pub grad_v_in: Vec<f32>,
}

impl SlimeBackwardWorkspace {
    pub fn new(hidden: usize, ffn_mid: usize, kv_dim: usize) -> Self {
        Self {
            grad_ffn_up: vec![0.0; ffn_mid],
            grad_ffn_gate: vec![0.0; ffn_mid],
            grad_ffn_mid: vec![0.0; ffn_mid],
            ffn_hid_x: vec![0.0; ffn_mid],
            ffn_in_x: vec![0.0; hidden],
            grad_ffn_in_up: vec![0.0; hidden],
            grad_ffn_in_gate: vec![0.0; hidden],
            grad_ffn_norm_out: vec![0.0; hidden],
            grad_residual_ffn: vec![0.0; hidden],
            grad_o_in: vec![0.0; hidden],
            attn_in_x: vec![0.0; hidden],
            grad_q_in: vec![0.0; hidden],
            grad_k_in: vec![0.0; kv_dim],
            grad_v_in: vec![0.0; kv_dim],
        }
    }
}

/// Backpropagate gradients through a single SlimeLayer using Straight-Through Estimator (STE).
/// `grad_in`: The gradient of the loss with respect to the output registers of this layer.
/// `grad_out`: The computed gradient of the loss with respect to the input registers of this layer.
#[allow(clippy::needless_range_loop)]
pub fn backward_slime_block(
    layer: &SlimeLayer,
    ws: &SlimeWorkspace,
    b_ws: &mut SlimeBackwardWorkspace,
    tape: &SlimeLayerTape,
    grads: &mut SlimeLayerGradients,
    grad_in: &[f32],
    grad_out: &mut [f32]
) {
    let hidden = ws.hidden_size;
    let ffn_mid = layer.ffn_mid;

    // Priority 35: SlimeBackward (Thawing the Core)
    // --- SPLIT-GRADIENT LATENT SPACE (FFN) ---
    let mut grad_ffn_out_buf = vec![0.0; hidden];
    // No res_scale (1/num_layers) — that was killing gradients (0.042/30 ≈ 0.0014 per layer).
    // No 0.9 JEPA modulation factor — 0.9^30 ≈ 0.042 compounds across 30 layers.
    // mHC radius constrains ||h|| ≤ radius, so stability is preserved without damping gradients.
    let kinetic_lambda = 0.005f32;

    for i in 0..hidden {
        let spring_force = tape.ffn_v_jepa[i];

        // STE: full gradient passes through the residual branch (identity for JEPA gate).
        // kinetic_grad regularizes large deviations from mu_ctx without damping the main signal.
        let kinetic_grad = -2.0 * kinetic_lambda * spring_force;

        grad_ffn_out_buf[i] = grad_in[i] + kinetic_grad;
        grad_out[i] = grad_in[i]; // Residual flows unchanged
    }
    


    // 5. Output Projection Backward
    
    // 1. FFN Down Projection Backward (Ternary STE)
    let grad_ffn_out = &grad_ffn_out_buf;
    
    // Reconstruct the exact quantized input to FFN down
    let ffn_hid_peak = {
        let mut p = 0.0f32;
        for i in 0..ffn_mid { p = p.max(tape.ffn_mid_f32[i].abs()); }
        (p / 127.0f32).max(1e-8f32)
    };
    
    for i in 0..ffn_mid {
        let q = (tape.ffn_mid_f32[i] / ffn_hid_peak).clamp(-127.0, 127.0) as i8;
        b_ws.ffn_hid_x[i] = q as f32 * ffn_hid_peak; // dequantized input
    }

    // Zero out grad_ffn_mid before accumulation since ternary_gemv_backward adds to it.
    b_ws.grad_ffn_mid.fill(0.0);

    unsafe {
        ternary_gemv_backward(
            grad_ffn_out,
            &b_ws.ffn_hid_x,
            layer.ffn_down_w,
            layer.ffn_down_scales,
            &mut grads.ffn_down_w_grad,
            &mut b_ws.grad_ffn_mid,
            hidden,
            ffn_mid,
        );
    }

    // 2. FFN ReLU² Activation Backward
    // Forward: ffn_mid_f32[i] = (if g > 0.0 { g * g } else { 0.0 }) * up[i]
    for i in 0..ffn_mid {
        let gy = b_ws.grad_ffn_mid[i];
        let g = tape.ffn_gate_f32[i];
        let u = tape.ffn_up_f32[i];

        if g > 0.0 {
            b_ws.grad_ffn_up[i] = gy * (g * g);
            b_ws.grad_ffn_gate[i] = gy * (2.0 * g * u);
        } else {
            b_ws.grad_ffn_up[i] = 0.0;
            b_ws.grad_ffn_gate[i] = 0.0;
        }
    }

    // 3. FFN Up & Gate Projection Backward (Ternary STE)
    // Both Up and Gate share the exact same input: norm_i8_ffn.
    // Dequantize input to exact f32 float using the tape's ffn_act_scale.
    for i in 0..hidden {
        b_ws.ffn_in_x[i] = tape.norm_i8_ffn[i] as f32 * tape.ffn_act_scale;
    }

    b_ws.grad_ffn_in_up.fill(0.0);
    unsafe {
        ternary_gemv_backward(
            &b_ws.grad_ffn_up,
            &b_ws.ffn_in_x,
            layer.ffn_up_w,
            layer.ffn_up_scales,
            &mut grads.ffn_up_w_grad,
            &mut b_ws.grad_ffn_in_up,
            ffn_mid,
            hidden,
        );
    }

    b_ws.grad_ffn_in_gate.fill(0.0);
    unsafe {
        ternary_gemv_backward(
            &b_ws.grad_ffn_gate,
            &b_ws.ffn_in_x,
            layer.ffn_gate_w,
            layer.ffn_gate_scales,
            &mut grads.ffn_gate_w_grad,
            &mut b_ws.grad_ffn_in_gate,
            ffn_mid,
            hidden,
        );
    }

    for i in 0..hidden {
        b_ws.grad_ffn_norm_out[i] = b_ws.grad_ffn_in_up[i] + b_ws.grad_ffn_in_gate[i];
    }

    // 4. FFN RMSNorm Backward (STE Pass-through)
    // Computes gradient w.r.t the norm weights and passes the signal to the residual stream
    for i in 0..hidden {
        let y_f32 = tape.norm_i8_ffn[i] as f32 * tape.ffn_act_scale;
        // SAFETY: ffn_norm_w is guaranteed by the engine to be valid for exactly `hidden` elements.
        let w = unsafe { *layer.ffn_norm_w.add(i) };
        
        let y_unscaled = if w.abs() > 1e-8 { y_f32 / w } else { 0.0 };
        grads.ffn_norm_w_grad[i] += b_ws.grad_ffn_norm_out[i] * y_unscaled;
        
        b_ws.grad_residual_ffn[i] = b_ws.grad_ffn_norm_out[i] * w;
        
        // Sum the FFN branch gradient back into the main residual stream
        grad_out[i] += b_ws.grad_residual_ffn[i];
    }

    // --- SPLIT-GRADIENT LATENT SPACE (ATTENTION) ---
    let mut grad_attn_out_buf = vec![0.0; hidden];
    for i in 0..hidden {
        let spring_force = tape.attn_v_jepa[i];

        // STE: full gradient propagation — removed 0.9 attenuation (0.9^30 ≈ 0.042)
        // and res_scale (1/num_layers) — mHC radius handles stability.
        let kinetic_grad = -2.0 * kinetic_lambda * spring_force;

        grad_attn_out_buf[i] = grad_out[i] + kinetic_grad;
        // grad_out[i] flows unchanged as the residual stream
    }


    // 5. Attention O Projection Backward (Ternary STE)
    // The gradient flowing into the Attention branch is the modulated grad_attn_out_buf.
    b_ws.grad_o_in.fill(0.0);
    unsafe {
        ternary_gemv_backward(
            &grad_attn_out_buf, // Gradient modulated by JEPA
            &tape.o_act_f32,
            layer.o_w,
            layer.o_scales,
            &mut grads.o_w_grad,
            &mut b_ws.grad_o_in,
            hidden,
            hidden, // o_w is [hidden, hidden]
        );
    }

    // 6. Scaled Dot-Product Attention Backward (Layer-wise STE Proxy)
    let head_d = hidden / ws.n_heads;
    let kv_dim = ws.n_kv_heads * head_d;
    let gqa_scale = ws.n_heads / ws.n_kv_heads;

    for h in 0..ws.n_heads {
        let kv_h = h / gqa_scale;
        let q_off = h * head_d;
        let kv_off = kv_h * head_d;
        
        for d in 0..head_d {
            let go = b_ws.grad_o_in[q_off + d];
            // STE Proxy: Route error through attention without full BPTT
            b_ws.grad_q_in[q_off + d] = go * tape.scores[tape.pos]; 
            b_ws.grad_k_in[kv_off + d] += go * tape.q_f32[q_off + d]; 
            b_ws.grad_v_in[kv_off + d] += go * tape.scores[tape.pos]; 
        }
    }

    // 7. Attention Q, K, V Projection Backward
    for i in 0..hidden {
        b_ws.attn_in_x[i] = tape.norm_i8_attn[i] as f32 * tape.attn_act_scale;
    }

    b_ws.grad_residual_ffn.fill(0.0); // Re-use buffer for accumulating QKV gradients
    unsafe {
        ternary_gemv_backward(
            &b_ws.grad_q_in, &b_ws.attn_in_x, layer.q_w, layer.q_scales,
            &mut grads.q_w_grad, &mut b_ws.grad_residual_ffn, hidden, hidden
        );
        ternary_gemv_backward(
            &b_ws.grad_k_in, &b_ws.attn_in_x, layer.k_w, layer.k_scales,
            &mut grads.k_w_grad, &mut b_ws.grad_residual_ffn, kv_dim, hidden
        );
        ternary_gemv_backward(
            &b_ws.grad_v_in, &b_ws.attn_in_x, layer.v_w, layer.v_scales,
            &mut grads.v_w_grad, &mut b_ws.grad_residual_ffn, kv_dim, hidden
        );
    }

    // 8. Attention RMSNorm Backward
    for i in 0..hidden {
        let y_f32 = b_ws.attn_in_x[i];
        // SAFETY: attn_norm_w is guaranteed to be valid for exactly `hidden` elements
        let w = unsafe { *layer.attn_norm_w.add(i) };
        let y_unscaled = if w.abs() > 1e-8 { y_f32 / w } else { 0.0 };
        
        grads.attn_norm_w_grad[i] += b_ws.grad_residual_ffn[i] * y_unscaled;
        grad_out[i] += b_ws.grad_residual_ffn[i] * w; // Final accumulation into the residual stream
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mud::slime_forward::SlimeLayer;
    use crate::mud::slime::SlimeWorkspace;

    #[allow(clippy::needless_range_loop)]
    #[test]
    #[ignore]
    fn test_ternary_gemv_backward() {
        let n_out = 1;
        let n_in = 16;
        let grad_y = vec![2.0f32];
        let x_f32 = vec![0.5f32; 16];
        
        // Ternary2Bit packing (now ELUT 4-bit nibbles: 8 weights per u32)
        // Pack +1 (0x1) for the first 8, and -1 (0xF) for the next 8.
        let mut w_u32_0: u32 = 0;
        let mut w_u32_1: u32 = 0;
        for i in 0..8 {
            w_u32_0 |= 0x1 << (i * 4);  // +1
        }
        for i in 0..8 {
            w_u32_1 |= 0xF << (i * 4);  // -1
        }
        let w_bytes = [w_u32_0, w_u32_1];
        let w_u8 = w_bytes.as_ptr() as *const u8;
        let scales = [1.5f32];

        let mut grad_w = vec![0.0f32; 16];
        let mut grad_x = vec![0.0f32; 16];

        unsafe {
            ternary_gemv_backward(
                &grad_y,
                &x_f32,
                w_u8,
                scales.as_ptr(),
                &mut grad_w,
                &mut grad_x,
                n_out,
                n_in,
            );
        }

        // Check grad_w: x^T * grad_y
        for i in 0..16 {
            assert_eq!(grad_w[i], 1.0);
        }

        // Check grad_x: grad_y * W_q * scale
        for i in 0..8 {
            assert_eq!(grad_x[i], 3.0);
        }
        for i in 8..16 {
            assert_eq!(grad_x[i], -3.0);
        }
    }

    #[test]
    fn test_backward_slime_block_structure() {
        let hidden = 32;
        let ffn_mid = 32;
        let ws = SlimeWorkspace::new(hidden, 32, 1, 1, 32, hidden, 30, 128.0);        let tape = SlimeLayerTape::new(hidden, ffn_mid, 1, hidden, 32, 0);
        let mut grads = SlimeLayerGradients::new(hidden, ffn_mid, 1, hidden);
        let kv_dim = hidden; // n_kv_heads * head_dim
        let mut b_ws = SlimeBackwardWorkspace::new(hidden, ffn_mid, kv_dim);
        
        let row_sz = hidden / 16 * 4;
        let q_w = vec![0x00u8; hidden * row_sz];
        let scales = vec![0.01f32; hidden];
        let norm_w = vec![1.0f32; hidden];
        
        let layer = SlimeLayer {
            q_w: q_w.as_ptr(), k_w: q_w.as_ptr(), v_w: q_w.as_ptr(), o_w: q_w.as_ptr(),
            q_scales: scales.as_ptr(), k_scales: scales.as_ptr(),
            v_scales: scales.as_ptr(), o_scales: scales.as_ptr(),
            ffn_up_w: q_w.as_ptr(), ffn_gate_w: q_w.as_ptr(), ffn_down_w: q_w.as_ptr(),
            ffn_up_scales: scales.as_ptr(), ffn_gate_scales: scales.as_ptr(), ffn_down_scales: scales.as_ptr(),
            attn_norm_w: norm_w.as_ptr(), ffn_norm_w: norm_w.as_ptr(),
            attn_sub_norm_w: std::ptr::null(), ffn_sub_norm_w: std::ptr::null(),
            mhc_alpha_w: std::ptr::null(), mhc_beta_w: std::ptr::null(), mhc_radius_w: std::ptr::null(),
            n_kv_heads: 1, ffn_mid,
            rope_theta: 0.0,
        };

        let grad_in = vec![1.0f32; hidden];
        let mut grad_out = vec![0.0f32; hidden];

        backward_slime_block(&layer, &ws, &mut b_ws, &tape, &mut grads, &grad_in, &mut grad_out);
        assert_eq!(grad_out[0], 1.0); // Because residual bypasses grad_in -> grad_out
    }
}
