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

    // Phase 1 (mHC trainable): per-element gradients for the dense f32 hyper-connection
    // scales alpha/beta. Accumulated over both mHC sites (post-attn, post-ffn).
    // Empty when the model has no mHC tensors (base model) or the layer is frozen.
    pub mhc_alpha_grad: Vec<f32>,
    pub mhc_beta_grad: Vec<f32>,
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
            mhc_alpha_grad: vec![0.0; hidden],
            mhc_beta_grad: vec![0.0; hidden],
        }
    }

    /// Zero-size grads for frozen layers (LAST_N / RAM-first seating).
    pub fn empty() -> Self {
        Self {
            q_w_grad: Vec::new(),
            k_w_grad: Vec::new(),
            v_w_grad: Vec::new(),
            o_w_grad: Vec::new(),
            ffn_up_w_grad: Vec::new(),
            ffn_gate_w_grad: Vec::new(),
            ffn_down_w_grad: Vec::new(),
            attn_norm_w_grad: Vec::new(),
            ffn_norm_w_grad: Vec::new(),
            mhc_alpha_grad: Vec::new(),
            mhc_beta_grad: Vec::new(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.q_w_grad.is_empty()
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
        if !self.mhc_alpha_grad.is_empty() {
            self.mhc_alpha_grad.fill(0.0);
            self.mhc_beta_grad.fill(0.0);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OptimizerStrategy {
    /// Plain SGD (no NS / GaLore) — default for quick smoke trains.
    Sgd,
    Muon {
        ns_iters: usize,
    },
    GaLore {
        rank: usize,
        update_freq: usize,
    },
    ChunkedAdam {
        chunk_cols: usize,
    },
    SparseAdam {
        only_active_rows: bool,
    },
    Adam,
}

/// Global optimizer policy from env (applied before shape dispatch).
///
/// `MUD_OPT` / `MUD_TRAIN_OPT`:
/// - `sgd` — always SGD
/// - `adam` — always Adam
/// - `muon` — always Muon (ns from [`crate::mud::muon::muon_ns_iters`])
/// - unset — shape dispatch; if `MUD_TRAIN_MAX_CHUNKS` set → SGD (smoke default)
pub fn optimizer_policy_override() -> Option<OptimizerStrategy> {
    let raw = std::env::var("MUD_OPT")
        .or_else(|_| std::env::var("MUD_TRAIN_OPT"))
        .unwrap_or_default();
    let key = raw.trim().to_ascii_lowercase();
    match key.as_str() {
        "sgd" | "plain" => Some(OptimizerStrategy::Sgd),
        "adam" => Some(OptimizerStrategy::Adam),
        "muon" => Some(OptimizerStrategy::Muon {
            ns_iters: crate::mud::muon::muon_ns_iters(),
        }),
        "galore" => Some(OptimizerStrategy::GaLore {
            rank: 64,
            update_freq: 100,
        }),
        "" => {
            // Quick sessions: skip expensive Muon NS unless user forced MUD_OPT
            if std::env::var("MUD_TRAIN_MAX_CHUNKS")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .filter(|&m| m > 0)
                .is_some()
            {
                Some(OptimizerStrategy::Sgd)
            } else {
                None
            }
        }
        _ => None,
    }
}

pub fn select_optimizer(rows: usize, cols: usize) -> OptimizerStrategy {
    if let Some(over) = optimizer_policy_override() {
        return over;
    }

    let ratio = rows as f32 / cols as f32;
    let total = rows * cols;
    let ns = crate::mud::muon::muon_ns_iters();

    if total < 100_000 && (0.5..=2.0).contains(&ratio) {
        OptimizerStrategy::Muon { ns_iters: ns }
    } else if (0.5..=2.0).contains(&ratio) {
        // BitNet-2B size square matrices also use Muon
        OptimizerStrategy::Muon { ns_iters: ns }
    } else if ratio > 2.5 {
        let rank = (cols / 4).max(8);
        OptimizerStrategy::GaLore {
            rank,
            update_freq: 100,
        }
    } else if ratio < 0.4 {
        let chunk = 512;
        OptimizerStrategy::ChunkedAdam { chunk_cols: chunk }
    } else if total > 10_000_000 {
        OptimizerStrategy::SparseAdam {
            only_active_rows: true,
        }
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

    // P0: Adam moments only when strategy is Adam / SparseAdam
    pub q_adam: Option<crate::mud::adam_state::AdamState>,
    pub k_adam: Option<crate::mud::adam_state::AdamState>,
    pub v_adam: Option<crate::mud::adam_state::AdamState>,
    pub o_adam: Option<crate::mud::adam_state::AdamState>,
    pub ffn_up_adam: Option<crate::mud::adam_state::AdamState>,
    pub ffn_gate_adam: Option<crate::mud::adam_state::AdamState>,
    pub ffn_down_adam: Option<crate::mud::adam_state::AdamState>,

    // Phase F+ SlimeX Bus
    pub slime_x: Option<crate::mud::slime_x::ShadowExpertBus>,
}

impl SlimeLayerShadowF32 {
    /// Empty shadow for frozen layers — zero heap (LAST_N seating / 1.7B RAM path).
    pub fn empty() -> Self {
        Self {
            q_w: Vec::new(),
            k_w: Vec::new(),
            v_w: Vec::new(),
            o_w: Vec::new(),
            ffn_up_w: Vec::new(),
            ffn_gate_w: Vec::new(),
            ffn_down_w: Vec::new(),
            q_opt: OptimizerStrategy::Sgd,
            k_opt: OptimizerStrategy::Sgd,
            v_opt: OptimizerStrategy::Sgd,
            o_opt: OptimizerStrategy::Sgd,
            ffn_up_opt: OptimizerStrategy::Sgd,
            ffn_gate_opt: OptimizerStrategy::Sgd,
            ffn_down_opt: OptimizerStrategy::Sgd,
            q_adam: None,
            k_adam: None,
            v_adam: None,
            o_adam: None,
            ffn_up_adam: None,
            ffn_gate_adam: None,
            ffn_down_adam: None,
            slime_x: None,
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.q_w.is_empty()
    }

    pub fn new(hidden: usize, ffn_mid: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        let kv_dim = n_kv_heads * head_dim;
        let q_opt = select_optimizer(hidden, hidden);
        let k_opt = select_optimizer(kv_dim, hidden);
        let v_opt = select_optimizer(kv_dim, hidden);
        let o_opt = select_optimizer(hidden, hidden);
        let ffn_up_opt = select_optimizer(ffn_mid, hidden);
        let ffn_gate_opt = select_optimizer(ffn_mid, hidden);
        let ffn_down_opt = select_optimizer(hidden, ffn_mid);
        use crate::mud::adam_state::AdamState;
        Self {
            q_w: vec![0.0; hidden * hidden],
            k_w: vec![0.0; kv_dim * hidden],
            v_w: vec![0.0; kv_dim * hidden],
            o_w: vec![0.0; hidden * hidden],
            ffn_up_w: vec![0.0; ffn_mid * hidden],
            ffn_gate_w: vec![0.0; ffn_mid * hidden],
            ffn_down_w: vec![0.0; hidden * ffn_mid],
            q_opt,
            k_opt,
            v_opt,
            o_opt,
            ffn_up_opt,
            ffn_gate_opt,
            ffn_down_opt,
            q_adam: AdamState::for_strategy(hidden * hidden, q_opt),
            k_adam: AdamState::for_strategy(kv_dim * hidden, k_opt),
            v_adam: AdamState::for_strategy(kv_dim * hidden, v_opt),
            o_adam: AdamState::for_strategy(hidden * hidden, o_opt),
            ffn_up_adam: AdamState::for_strategy(ffn_mid * hidden, ffn_up_opt),
            ffn_gate_adam: AdamState::for_strategy(ffn_mid * hidden, ffn_gate_opt),
            ffn_down_adam: AdamState::for_strategy(hidden * ffn_mid, ffn_down_opt),
            slime_x: None,
        }
    }
}

/// Unpacks Ternary2Bit (ELUT 4-bit nibbles, 8 weights per u32) into a flat f32 vector using PRQ scales.
pub fn unpack_ternary2bit_to_f32(packed: &[u8], scales: &[f32], cols: usize, out: &mut [f32]) {
    let rows = scales.len();
    assert_eq!(out.len(), rows * cols);
    assert_eq!(packed.len(), (rows * cols).div_ceil(8) * 4);

    let packed_u32 = packed.as_ptr() as *const u32;
    let out_p = out.as_mut_ptr();
    let lut = super::TERNARY_LUT;
    for (r, &scale) in scales.iter().enumerate() {
        let row_base = (r * cols) as isize;
        let u32_base = row_base / 8;
        let full_groups = cols / 8;
        let rem = cols % 8;
        for g in 0..full_groups {
            let val = unsafe { *packed_u32.offset(u32_base + g as isize) };
            let o = row_base + (g * 8) as isize;
            for j in 0..8 {
                let bits = ((val >> (j * 4)) & 0xF) as usize;
                unsafe { *out_p.offset(o + j as isize) = lut[bits] * scale };
            }
        }
        if rem > 0 {
            let val = unsafe { *packed_u32.offset(u32_base + full_groups as isize) };
            let o = row_base + (full_groups * 8) as isize;
            for j in 0..rem {
                let bits = ((val >> (j * 4)) & 0xF) as usize;
                unsafe { *out_p.offset(o + j as isize) = lut[bits] * scale };
            }
        }
    }
}

/// Memory structure for storing intermediate activations of a SINGLE layer for a SINGLE token.
/// These are required to compute exact derivatives during the Backward Pass (Activation Checkpointing).
pub struct SlimeLayerTape {
    pub norm_i8_attn: Vec<i8>,  // i8 input to Attention Q, K, V projections
    pub norm_i8_ffn: Vec<i8>,   // i8 input to FFN Up, Gate projections
    pub q_f32: Vec<f32>,        // Output of Q projection
    pub k_f32: Vec<f32>,        // Output of K projection
    pub v_f32: Vec<f32>,        // Output of V projection
    pub scores: Vec<f32>,       // Softmax attention scores
    pub o_act_f32: Vec<f32>,    // Input to O projection
    pub ffn_up_f32: Vec<f32>,   // Output of FFN Up projection
    pub ffn_gate_f32: Vec<f32>, // Output of FFN Gate projection
    pub ffn_mid_f32: Vec<f32>,  // Intermediate FFN activation: relu2(gate) * up
    pub attn_act_scale: f32,    // Scaling factor for Attn i8 input
    pub ffn_act_scale: f32,     // Scaling factor for FFN i8 input
    pub attn_v_jepa: Vec<f32>,  // Latent state before Attn jepa_stabilizer
    pub ffn_v_jepa: Vec<f32>,   // Latent state before FFN jepa_stabilizer
    // Phase 1 (mHC trainable): inputs needed to differentiate alpha/beta.
    // mHC forward: val = alpha*h_in + (1-gate)*beta*f_h.  f_h is o_act_f32 (attn) /
    // ffn branch output (ffn); gate = sigmoid(v_jepa) recomputed in backward.
    // We only need h_in (the residual register value entering each mHC site) and the
    // ffn branch f_h (o_act_f32 already covers the attn site).
    pub mhc_attn_h_in: Vec<f32>, // residual accum entering post-attn mHC
    pub mhc_ffn_h_in: Vec<f32>,  // residual accum entering post-ffn mHC
    pub mhc_ffn_f_h: Vec<f32>,   // ffn branch output (post-jepa) feeding post-ffn mHC
    pub pos: usize,              // The sequence position of this tape
    /// L-15: false after discard; recompute before backward when checkpointing.
    pub valid: bool,
}

impl SlimeLayerTape {
    pub fn new(
        hidden: usize,
        ffn_mid: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        pos: usize,
    ) -> Self {
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
            mhc_attn_h_in: vec![0.0; hidden],
            mhc_ffn_h_in: vec![0.0; hidden],
            mhc_ffn_f_h: vec![0.0; hidden],
            pos,
            valid: false,
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
        self.mhc_attn_h_in.fill(0.0);
        self.mhc_ffn_h_in.fill(0.0);
        self.mhc_ffn_f_h.fill(0.0);
        self.valid = false;
    }
}

/// Helper: Compute gradients for a Ternary GEMV layer using the Straight-Through Estimator (STE).
/// Since we use ELUT 4-bit nibble packing, the weight unpacking will be needed for `grad_x`.
/// # Safety
/// Caller must ensure `w_u8` points to a valid 4-bit ELUT packed array of size `(n_in / 8) * 4 * n_out` bytes,
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
    scratch_grad_x: &mut [f32],
    n_out: usize,
    n_in: usize,
) {
    grad_x.fill(0.0);
    scratch_grad_x.fill(0.0);

    let pool = crate::mud::pcore_pool::get_pool();
    let rows_per_task = ((n_out / 8) / 4 * 4).max(4);

    let gy_p = grad_y.as_ptr() as usize;
    let x_p = x_f32.as_ptr() as usize;
    let w_p = w_u8 as usize;
    let scales_p = scales as usize;
    let gw_p = grad_w.as_mut_ptr() as usize;
    let scratch_p = scratch_grad_x.as_mut_ptr() as usize;
    let w_u32_stride = n_in / 8;

    for i in 0..8 {
        let start_row = (i * rows_per_task).min(n_out);
        let end_row = if i == 7 {
            n_out
        } else {
            (start_row + rows_per_task).min(n_out)
        };
        if start_row >= end_row {
            break;
        }

        pool.execute(move || {
            let gy = gy_p as *const f32;
            let x_f32 = x_p as *const f32;
            let w_u32 = w_p as *const u32;
            let sc = scales_p as *const f32;
            let gw = gw_p as *mut f32;
            let local_gx_start = unsafe { (scratch_p as *mut f32).add(i * n_in) };

            unsafe {
                for r in start_row..end_row {
                    let g_y = *gy.add(r);
                    let scale = *sc.add(r);
                    let w_row = w_u32.add(r * w_u32_stride);

                    let gw_row = std::slice::from_raw_parts_mut(gw.add(r * n_in), n_in);
                    let x_slice = std::slice::from_raw_parts(x_f32, n_in);
                    forge_autograd::avx_math::axpy_avx2(gw_row, g_y, x_slice);

                    let scale_gy = scale * g_y;

                    for c in 0..n_in {
                        let u32_idx = c / 8;
                        let shift = (c % 8) * 4;
                        let val = (*w_row.add(u32_idx) >> shift) & 0xF;

                        let w_val = match val {
                            0x1 => 1.0,
                            0xF => -1.0,
                            _ => 0.0,
                        };

                        *local_gx_start.add(c) += w_val * scale_gy;
                    }
                }
            }
        });
    }
    pool.wait_all();

    for i in 0..8 {
        let local_gx_start = i * n_in;
        for c in 0..n_in {
            grad_x[c] += scratch_grad_x[local_gx_start + c];
        }
    }
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
    pub scratch_grad_x: Vec<f32>,
    /// L-09: preallocated branch grad (FFN out / attn out) — no per-step `vec!`.
    pub grad_branch: Vec<f32>,
}

impl SlimeBackwardWorkspace {
    pub fn new(hidden: usize, ffn_mid: usize, kv_dim: usize) -> Self {
        let max_in = hidden.max(kv_dim).max(ffn_mid);
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
            scratch_grad_x: vec![0.0; 8 * max_in],
            grad_branch: vec![0.0; hidden],
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
    grad_out: &mut [f32],
) {
    let hidden = ws.hidden_size;
    let ffn_mid = layer.ffn_mid;

    // Priority 35: SlimeBackward (Thawing the Core)
    // --- SPLIT-GRADIENT LATENT SPACE (FFN) ---
    // L-09: reuse preallocated grad_branch (no per-step allocation).
    let kinetic_lambda = 0.005f32;

    // Phase 1 (mHC trainable): gradient of the post-ffn hyper-connection scales.
    // Forward: out[i] = alpha[i]*h_in[i] + (1-gate[i])*beta[i]*f_h[i]
    //   dL/dalpha[i] = grad_out[i] * h_in[i]
    //   dL/dbeta[i]  = grad_out[i] * (1-gate[i]) * f_h[i]
    // grad wrt the mHC output here is grad_in (loss grad on block output).
    // Norm-projection clamp is approximated as a constant scale for this step (§3.4).
    if !grads.mhc_alpha_grad.is_empty() {
        for i in 0..hidden {
            let g = grad_in[i];
            let gate = 1.0f32 / (1.0f32 + (-tape.ffn_v_jepa[i]).exp());
            grads.mhc_alpha_grad[i] += g * tape.mhc_ffn_h_in[i];
            grads.mhc_beta_grad[i] += g * (1.0 - gate) * tape.mhc_ffn_f_h[i];
        }
    }

    // SAFETY: buffers sized to hidden at workspace construction.
    unsafe {
        let gi = grad_in.as_ptr();
        let go = grad_out.as_mut_ptr();
        let gb = b_ws.grad_branch.as_mut_ptr();
        let jepa = tape.ffn_v_jepa.as_ptr();
        for i in 0..hidden {
            let spring_force = *jepa.add(i);
            let kinetic_grad = -2.0 * kinetic_lambda * spring_force;
            let g = *gi.add(i);
            *gb.add(i) = g + kinetic_grad;
            *go.add(i) = g; // Residual flows unchanged
        }
    }

    // 1. FFN Down Projection Backward (Ternary STE)
    let grad_ffn_out = &b_ws.grad_branch[..hidden];

    // Reconstruct the exact quantized input to FFN down
    let ffn_hid_peak = {
        let mut p = 0.0f32;
        for i in 0..ffn_mid {
            p = p.max(tape.ffn_mid_f32[i].abs());
        }
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
            &mut b_ws.scratch_grad_x,
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
            &mut b_ws.scratch_grad_x,
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
            &mut b_ws.scratch_grad_x,
            ffn_mid,
            hidden,
        );
    }

    for i in 0..hidden {
        b_ws.grad_ffn_norm_out[i] = b_ws.grad_ffn_in_up[i] + b_ws.grad_ffn_in_gate[i];
    }

    // 4. FFN RMSNorm Backward (Orthogonalized STE Pass-through)
    // Computes gradient w.r.t the norm weights and passes the orthogonalized signal
    let mut sum_dy_xnorm = 0.0;
    for i in 0..hidden {
        let y_f32 = tape.norm_i8_ffn[i] as f32 * tape.ffn_act_scale;
        let w = unsafe { *layer.ffn_norm_w.add(i) };
        let y_unscaled = if w.abs() > 1e-8 { y_f32 / w } else { 0.0 }; // This is x_norm
        sum_dy_xnorm += b_ws.grad_ffn_norm_out[i] * w * y_unscaled;
    }
    let mean_dy_xnorm = sum_dy_xnorm / hidden as f32;

    for i in 0..hidden {
        let y_f32 = tape.norm_i8_ffn[i] as f32 * tape.ffn_act_scale;
        let w = unsafe { *layer.ffn_norm_w.add(i) };
        let y_unscaled = if w.abs() > 1e-8 { y_f32 / w } else { 0.0 }; // This is x_norm

        grads.ffn_norm_w_grad[i] += b_ws.grad_ffn_norm_out[i] * y_unscaled;

        // Orthogonalize gradient with respect to x_norm.
        // We omit the 1/rms factor (STE Pass-through) but keep the geometric projection
        b_ws.grad_residual_ffn[i] = b_ws.grad_ffn_norm_out[i] * w - y_unscaled * mean_dy_xnorm;

        // Sum the FFN branch gradient back into the main residual stream
        grad_out[i] += b_ws.grad_residual_ffn[i];
    }

    // Phase 1 (mHC trainable): gradient of the post-attn hyper-connection scales.
    // Forward: out[i] = alpha[i]*h_in[i] + (1-gate[i])*beta[i]*f_h[i], f_h = o_act_f32.
    // grad wrt the mHC output here is grad_out (accumulated after the FFN branch).
    if !grads.mhc_alpha_grad.is_empty() {
        for i in 0..hidden {
            let g = grad_out[i];
            let gate = 1.0f32 / (1.0f32 + (-tape.attn_v_jepa[i]).exp());
            grads.mhc_alpha_grad[i] += g * tape.mhc_attn_h_in[i];
            grads.mhc_beta_grad[i] += g * (1.0 - gate) * tape.o_act_f32[i];
        }
    }

    // --- SPLIT-GRADIENT LATENT SPACE (ATTENTION) ---
    // L-09: reuse grad_branch (FFN path finished with it).
    unsafe {
        let go = grad_out.as_ptr();
        let gb = b_ws.grad_branch.as_mut_ptr();
        let jepa = tape.attn_v_jepa.as_ptr();
        for i in 0..hidden {
            let kinetic_grad = -2.0 * kinetic_lambda * *jepa.add(i);
            *gb.add(i) = *go.add(i) + kinetic_grad;
        }
    }

    // 5. Attention O Projection Backward (Ternary STE)
    b_ws.grad_o_in.fill(0.0);
    b_ws.grad_k_in.fill(0.0);
    b_ws.grad_v_in.fill(0.0);
    unsafe {
        ternary_gemv_backward(
            &b_ws.grad_branch[..hidden],
            &tape.o_act_f32,
            layer.o_w,
            layer.o_scales,
            &mut grads.o_w_grad,
            &mut b_ws.grad_o_in,
            &mut b_ws.scratch_grad_x,
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
            &b_ws.grad_q_in,
            &b_ws.attn_in_x,
            layer.q_w,
            layer.q_scales,
            &mut grads.q_w_grad,
            &mut b_ws.grad_residual_ffn,
            &mut b_ws.scratch_grad_x,
            hidden,
            hidden,
        );
        ternary_gemv_backward(
            &b_ws.grad_k_in,
            &b_ws.attn_in_x,
            layer.k_w,
            layer.k_scales,
            &mut grads.k_w_grad,
            &mut b_ws.grad_residual_ffn,
            &mut b_ws.scratch_grad_x,
            kv_dim,
            hidden,
        );
        ternary_gemv_backward(
            &b_ws.grad_v_in,
            &b_ws.attn_in_x,
            layer.v_w,
            layer.v_scales,
            &mut grads.v_w_grad,
            &mut b_ws.grad_residual_ffn,
            &mut b_ws.scratch_grad_x,
            kv_dim,
            hidden,
        );
    }

    // 8. Attention RMSNorm Backward (Orthogonalized STE Pass-through)
    let mut sum_dy_xnorm = 0.0;
    for i in 0..hidden {
        let y_f32 = b_ws.attn_in_x[i];
        let w = unsafe { *layer.attn_norm_w.add(i) };
        let y_unscaled = if w.abs() > 1e-8 { y_f32 / w } else { 0.0 }; // This is x_norm
        sum_dy_xnorm += b_ws.grad_residual_ffn[i] * w * y_unscaled;
    }
    let mean_dy_xnorm = sum_dy_xnorm / hidden as f32;

    for i in 0..hidden {
        let y_f32 = b_ws.attn_in_x[i];
        // SAFETY: attn_norm_w is guaranteed to be valid for exactly `hidden` elements
        let w = unsafe { *layer.attn_norm_w.add(i) };
        let y_unscaled = if w.abs() > 1e-8 { y_f32 / w } else { 0.0 }; // This is x_norm

        grads.attn_norm_w_grad[i] += b_ws.grad_residual_ffn[i] * y_unscaled;

        let dx = b_ws.grad_residual_ffn[i] * w - y_unscaled * mean_dy_xnorm;
        grad_out[i] += dx; // Final accumulation into the residual stream
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mud::slime::SlimeWorkspace;
    use crate::mud::slime_forward::SlimeLayer;

    #[allow(clippy::needless_range_loop)]
    #[test]
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
            w_u32_0 |= 0x1 << (i * 4); // +1
        }
        for i in 0..8 {
            w_u32_1 |= 0xF << (i * 4); // -1
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
                &mut vec![0.0; 8 * n_in],
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

    /// Phase 1 (mHC trainable): verify the analytic alpha/beta gradient formula
    /// (dL/dalpha = g*h_in, dL/dbeta = g*(1-gate)*f_h) against finite differences.
    #[test]
    fn test_mhc_scale_grad_matches_finite_diff() {
        // mHC forward (single element, no norm clamp): out = alpha*h_in + (1-gate)*beta*f_h
        // Loss L = 0.5 * out^2  =>  dL/dout = out = g.
        let h_in = 0.7f32;
        let f_h = -0.4f32;
        let v_jepa = 0.3f32;
        let gate = 1.0f32 / (1.0f32 + (-v_jepa).exp());
        let alpha = 0.85f32;
        let beta = 0.15f32;

        let forward = |a: f32, b: f32| a * h_in + (1.0 - gate) * b * f_h;
        let out = forward(alpha, beta);
        let g = out; // dL/dout for L = 0.5*out^2

        // Analytic (matches backward_slime_block accumulation)
        let d_alpha = g * h_in;
        let d_beta = g * (1.0 - gate) * f_h;

        // Finite difference on L = 0.5*out^2
        let eps = 1e-3f32;
        let loss = |a: f32, b: f32| 0.5 * forward(a, b).powi(2);
        let fd_alpha = (loss(alpha + eps, beta) - loss(alpha - eps, beta)) / (2.0 * eps);
        let fd_beta = (loss(alpha, beta + eps) - loss(alpha, beta - eps)) / (2.0 * eps);

        assert!(
            (d_alpha - fd_alpha).abs() < 1e-3,
            "alpha grad {d_alpha} vs fd {fd_alpha}"
        );
        assert!(
            (d_beta - fd_beta).abs() < 1e-3,
            "beta grad {d_beta} vs fd {fd_beta}"
        );
    }

    /// Phase 1: the mHC SGD step must move params toward the negative gradient and clamp.
    #[test]
    fn test_mhc_scale_sgd_step_clamps() {
        use crate::mud::corpus_trainer::mhc_scale_sgd_step;
        let mut w = vec![0.85f32, 3.99f32];
        // grad positive -> param decreases; large lr*grad on w[1] must clamp at 4.0 lower bound side
        let grad = vec![1.0f32, -100.0f32];
        unsafe { mhc_scale_sgd_step(w.as_mut_ptr(), &grad, 0.1, 1.0) };
        assert!(w[0] < 0.85, "param should decrease with positive grad");
        assert!(
            w[1] <= 4.0 && w[1] >= 0.0,
            "param must stay clamped in [0,4]"
        );
    }

    #[test]
    fn test_backward_slime_block_structure() {
        let hidden = 32;
        let ffn_mid = 32;
        let ws = SlimeWorkspace::new(hidden, 32, 1, 1, 32, hidden, 30, 128.0);
        let tape = SlimeLayerTape::new(hidden, ffn_mid, 1, hidden, 32, 0);
        let mut grads = SlimeLayerGradients::new(hidden, ffn_mid, 1, hidden);
        let kv_dim = hidden; // n_kv_heads * head_dim
        let mut b_ws = SlimeBackwardWorkspace::new(hidden, ffn_mid, kv_dim);

        let row_sz = hidden / 8 * 4;
        let q_w = vec![0x00u8; hidden * row_sz];
        let scales = vec![0.01f32; hidden];
        let norm_w = vec![1.0f32; hidden];

        let layer = SlimeLayer {
            q_w: q_w.as_ptr(),
            k_w: q_w.as_ptr(),
            v_w: q_w.as_ptr(),
            o_w: q_w.as_ptr(),
            q_scales: scales.as_ptr(),
            k_scales: scales.as_ptr(),
            v_scales: scales.as_ptr(),
            o_scales: scales.as_ptr(),
            ffn_up_w: q_w.as_ptr(),
            ffn_gate_w: q_w.as_ptr(),
            ffn_down_w: q_w.as_ptr(),
            ffn_up_scales: scales.as_ptr(),
            ffn_gate_scales: scales.as_ptr(),
            ffn_down_scales: scales.as_ptr(),
            attn_norm_w: norm_w.as_ptr(),
            ffn_norm_w: norm_w.as_ptr(),
            attn_sub_norm_w: std::ptr::null(),
            ffn_sub_norm_w: std::ptr::null(),
            q_norm_w: std::ptr::null(),
            k_norm_w: std::ptr::null(),
            mhc_alpha_w: std::ptr::null(),
            mhc_beta_w: std::ptr::null(),
            mhc_radius_w: std::ptr::null(),
            n_kv_heads: 1,
            ffn_mid,
            rope_theta: 0.0,
        };

        let grad_in = vec![1.0f32; hidden];
        let mut grad_out = vec![0.0f32; hidden];

        backward_slime_block(
            &layer,
            &ws,
            &mut b_ws,
            &tape,
            &mut grads,
            &grad_in,
            &mut grad_out,
        );
        assert_eq!(grad_out[0], 1.0); // Because residual bypasses grad_in -> grad_out
    }
}
