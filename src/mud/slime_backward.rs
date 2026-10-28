//! SAFETY (module contract, R-36) — the `unsafe` sites in this module are
//! one of the following patterns; anything else carries its own adjacent
//! `// SAFETY:` comment:
//!
//! (a) Dereferences/passes of raw `SlimeLayer` pointers (`q_w`/`k_w`/`v_w`/
//!     `o_w`, `ffn_up_w`/`ffn_gate_w`/`ffn_down_w`, their `*_scales`,
//!     `attn_norm_w`/`ffn_norm_w`, `mhc_alpha_w`/`mhc_beta_w`, and the F32
//!     weight pointers of Q2_0-converted layers) into the STE/F32 backward
//!     kernels. Each points either into a `MudTensor` payload validated at
//!     load (R-30/R-31) or an owned `Box<[f32]>` shadow; the pointee outlives
//!     the backward call, indices stay below `hidden`/`ffn_mid`/`kv_dim`, and
//!     the packed byte count matches the R-35 per-row stride documented on
//!     `ternary_gemv_backward`.
//! (b) Raw views / pointer arithmetic over caller-owned buffers that live for
//!     the whole call: `SlimeBackwardWorkspace` Vecs (`grad_ffn_*`,
//!     `ffn_in_x`, `ffn_hid_x`, `attn_in_x`, `grad_o/q/k/v_in`,
//!     `grad_branch`, `grad_residual_ffn`, `scratch_grad_x`), `SlimeWorkspace`
//!     buffers, `SlimeLayerTape` buffers (`norm_i8_*`, `ffn_*_f32`,
//!     `attn_v_jepa`/`ffn_v_jepa`, `o_act_f32`, `mhc_*`), and the caller's
//!     `grad_in`/`grad_out`/`grads.*_grad` slices. All are sized to
//!     `hidden`/`ffn_mid`/`8*n_in` and are distinct allocations, except where
//!     the kernel documents in-place aliasing (`div_avx2` with `a == out`;
//!     `backward_rmsnorm_avx2` with `grad_y == grad_x`, where each 8-lane
//!     chunk is loaded before its store); views stay within `0..buffer.len()`.
//! (c) AVX2/ASM kernel calls (`ezop::*`, `asm::*`,
//!     `forge_autograd::avx_math::*`). The only supported target is x86_64
//!     with AVX2/FMA (i7-1260P) and the crate is built with
//!     `-C target-cpu=native` (`.cargo/config.toml`, `mud.sh`), so those
//!     instructions are guaranteed for this binary; dispatching sites
//!     additionally check `is_x86_feature_detected!`. Buffer/length arguments
//!     are exactly the (a)/(b) extents.
//!
//! The `unpack_ternary2bit_to_f32` raw word reads/writes are covered by length
//! assertions plus their own `// SAFETY:` notes.

use crate::mud::ezop;
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

// SPRINT-4: OptimizerStrategy + select_optimizer + optimizer_policy_override
// have been moved to `crate::mud::optim` to break the dependency cycle with
// `corpus_trainer`. Re-exported here for back-compat with existing call sites.
pub use crate::mud::optim::{optimizer_policy_override, select_optimizer, OptimizerStrategy};

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

    // RMSNorm weight shadows (F32, trainable). FIX (2026-07-31): previously frozen —
    // root cause of activation collapse via norm/weight scale mismatch.
    pub attn_norm_w: Vec<f32>,
    pub ffn_norm_w: Vec<f32>,

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
            attn_norm_w: Vec::new(),
            ffn_norm_w: Vec::new(),
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
            attn_norm_w: vec![0.0; hidden],
            ffn_norm_w: vec![0.0; hidden],
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
        }
    }
}

/// Unpacks Ternary2Bit (ELUT 4-bit nibbles, 8 weights per u32) into a flat f32 vector using PRQ scales.
/// R-15 canonical wire format: per-row padded `rows * cols.div_ceil(8) * 4`
/// bytes (same as converter `quantizer.rs` + `stable_pack_ternary_rows` +
/// `pack_ternary_into`). Each row starts at `r * row_words` u32 words.
pub fn unpack_ternary2bit_to_f32(packed: &[u8], scales: &[f32], cols: usize, out: &mut [f32]) {
    let rows = scales.len();
    let row_words = cols.div_ceil(8);
    assert_eq!(out.len(), rows * cols);
    assert_eq!(packed.len(), rows * row_words * 4);

    let packed_u32 = packed.as_ptr() as *const u32;
    let out_p = out.as_mut_ptr();
    let lut = super::TERNARY_LUT;
    for (r, &scale) in scales.iter().enumerate() {
        let row_base = (r * cols) as isize;
        let u32_base = (r * row_words) as isize;
        let full_groups = cols / 8;
        let rem = cols % 8;
        // SAFETY: `packed.len() == rows * row_words * 4` was asserted above,
        // so `u32_base + g < rows * row_words` for every `g < full_groups`;
        // read_unaligned is required since &[u8] has alignment 1. `out.len()
        // == rows * cols` (asserted) and `o + 8 <= rows * cols` for the same
        // reason, so the raw writes stay in-bounds.
        for g in 0..full_groups {
            // BUG-2 FIX (2026-08-03): use read_unaligned — `packed` is &[u8] (align 1),
            // so a plain *const u32 deref is UB if the pointer isn't 4-byte aligned.
            let val = unsafe { std::ptr::read_unaligned(packed_u32.offset(u32_base + g as isize)) };
            let o = row_base + (g * 8) as isize;
            // SAFETY: `j < 8` and `o + 8 <= rows*cols` (asserted len above).
            for j in 0..8 {
                let bits = ((val >> (j * 4)) & 0xF) as usize;
                unsafe { *out_p.offset(o + j as isize) = lut[bits] * scale };
            }
        }
        if rem > 0 {
            // SAFETY: the tail word (`u32_base + full_groups`) still belongs
            // to this row because `row_words = ceil(cols/8)`; same in-bounds
            // argument for `out` as above (writes `rem < 8` elements at
            // `row_base + full_groups*8`, ending exactly at row end).
            let val = unsafe { std::ptr::read_unaligned(packed_u32.offset(u32_base + full_groups as isize)) };
            let o = row_base + (full_groups * 8) as isize;
            for j in 0..rem {
                let bits = ((val >> (j * 4)) & 0xF) as usize;
                unsafe { *out_p.offset(o + j as isize) = lut[bits] * scale };
            }
        }
    }
}

/// R-42: Decode TernaryTQ (5 trits/byte) into f32, applying a per-row PRQ scale.
/// Each byte packs 5 ternary trits using base-3 encoding:
///   byte_value = t0 + 3*t1 + 9*t2 + 27*t3 + 81*t4  (range 0..243)
/// where ti ∈ {0,1,2} maps to {-1, 0, +1}.
/// `packed` must have at least `rows * ceil(cols/5)` bytes.
/// `scales` must have `rows` entries (one PRQ scale per row).
pub fn unpack_ternary_tq_to_f32(packed: &[u8], scales: &[f32], cols: usize, out: &mut [f32]) {
    let rows = scales.len();
    assert_eq!(out.len(), rows * cols);
    assert_eq!(packed.len(), rows * cols.div_ceil(5usize));

    for (r, &scale) in scales.iter().enumerate() {
        let row_base = r * cols;
        let byte_base = r * cols.div_ceil(5usize);
        for c in 0..cols {
            let byte_idx = byte_base + c / 5;
            let trit_idx = c % 5;
            let val = TQ_DECODE_LUT[packed[byte_idx] as usize * 5 + trit_idx] as f32;
            out[row_base + c] = val * scale;
        }
    }
}

/// R-42: LUT for fast TernaryTQ byte → 5 trits decoding.
/// TQ_DECODE_LUT[byte * 5 + i] = trit i of byte (0→-1, 1→0, 2→+1).
static TQ_DECODE_LUT: [i8; 256 * 5] = {
    let mut lut = [0i8; 256 * 5];
    let mut byte = 0usize;
    while byte < 256 {
        let mut v = byte;
        let mut i = 0;
        while i < 5 {
            let trit = (v % 3) as i8;
            lut[byte * 5 + i] = match trit {
                0 => -1,
                2 => 1,
                _ => 0,
            };
            v /= 3;
            i += 1;
        }
        byte += 1;
    }
    lut
};

/// R-42: AVX2-optimized GEMV for TernaryTQ (5 trits/byte) weights.
/// Uses LUT for byte→trit decoding and AVX2 for multiply-accumulate.
/// Falls back to scalar if AVX2 is not available.
#[cfg(target_arch = "x86_64")]
pub fn ternary_tq_gemv_avx2(
    packed: &[u8],
    scales: &[f32],
    x: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
) {
    use std::arch::x86_64::*;
    assert_eq!(packed.len(), rows * cols.div_ceil(5usize));
    assert_eq!(scales.len(), rows);
    assert_eq!(x.len(), cols);
    assert_eq!(out.len(), rows);

    if !is_x86_feature_detected!("avx2") {
        // Fallback to scalar
        ternary_tq_gemv(packed, scales, x, out, rows, cols);
        return;
    }

    for r in 0..rows {
        let byte_base = r * cols.div_ceil(5usize);
        let mut acc_vec = unsafe { _mm256_setzero_ps() };
        let mut c = 0usize;

        // Process 8 columns at a time
        while c + 8 <= cols {
            // Load 8 activations
            let x_vec = unsafe { _mm256_loadu_ps(x.as_ptr().add(c)) };

            // Decode 8 trits via LUT (scalar), then pack into AVX2.
            let mut trits = [0.0f32; 8];
            for (i, trit) in trits.iter_mut().enumerate() {
                let col = c + i;
                let byte_idx = byte_base + col / 5;
                let trit_idx = col % 5;
                *trit = TQ_DECODE_LUT[packed[byte_idx] as usize * 5 + trit_idx] as f32;
            }
            let w_vec = unsafe { _mm256_loadu_ps(trits.as_ptr()) };

            // Multiply and accumulate
            let prod = unsafe { _mm256_mul_ps(w_vec, x_vec) };
            acc_vec = unsafe { _mm256_add_ps(acc_vec, prod) };

            c += 8;
        }

        // Scalar remainder
        let mut acc_scalar = 0.0f32;
        // Extract accumulated sum from AVX2
        let mut tmp = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(tmp.as_mut_ptr(), acc_vec) };
        for v in tmp.iter() {
            acc_scalar += v;
        }
        while c < cols {
            let byte_idx = byte_base + c / 5;
            let trit_idx = c % 5;
            let w = TQ_DECODE_LUT[packed[byte_idx] as usize * 5 + trit_idx] as f32;
            acc_scalar += w * x[c];
            c += 1;
        }

        out[r] += acc_scalar * scales[r];
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub fn ternary_tq_gemv_avx2(
    packed: &[u8],
    scales: &[f32],
    x: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
) {
    ternary_tq_gemv(packed, scales, x, out, rows, cols);
}

/// R-42: GEMV for TernaryTQ (5 trits/byte) weights. Computes
///   out[row] = scale[row] * sum_c (trit(row, c) * x[c])
/// where trit maps 0→-1, 1→0, 2→+1.
/// `packed` holds `rows * ceil(cols/5)` bytes. `scales` holds `rows` f32.
/// `x` holds `cols` f32. `out` holds `rows` f32 (accumulated, not zeroed).
pub fn ternary_tq_gemv(
    packed: &[u8],
    scales: &[f32],
    x: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
) {
    assert_eq!(packed.len(), rows * cols.div_ceil(5usize));
    assert_eq!(scales.len(), rows);
    assert_eq!(x.len(), cols);
    assert_eq!(out.len(), rows);

    for r in 0..rows {
        let byte_base = r * cols.div_ceil(5usize);
        let mut acc = 0.0f32;
        for (c, &xv) in x.iter().enumerate() {
            let byte_idx = byte_base + c / 5;
            let trit_idx = c % 5;
            let w = TQ_DECODE_LUT[packed[byte_idx] as usize * 5 + trit_idx] as f32;
            acc += w * xv;
        }
        out[r] += acc * scales[r];
    }
}

/// Memory structure for storing intermediate activations of a SINGLE layer for a SINGLE token.
/// These are required to compute exact derivatives during the Backward Pass (Activation Checkpointing).
pub struct SlimeLayerTape {
    pub norm_f32_attn: Vec<f32>,  // f32 input to Attention Q, K, V projections (R-43: no i8 round-trip)
    pub norm_f32_ffn: Vec<f32>,   // f32 input to FFN Up, Gate projections (R-43: no i8 round-trip)
    pub q_f32: Vec<f32>,        // Output of Q projection
    pub k_f32: Vec<f32>,        // Output of K projection
    pub v_f32: Vec<f32>,        // Output of V projection
    pub scores: Vec<f32>,       // Softmax attention scores
    pub o_act_f32: Vec<f32>,    // Input to O projection
    pub ffn_up_f32: Vec<f32>,   // Output of FFN Up projection
    pub ffn_gate_f32: Vec<f32>, // Output of FFN Gate projection
    pub ffn_mid_f32: Vec<f32>,  // Intermediate FFN activation: relu2(gate) * up
    // R-43: scales removed (f32 stored directly, no i8 round-trip). Kept as
    // 1.0 for backward-compat with any code that still reads them.
    pub attn_act_scale: f32,
    pub ffn_act_scale: f32,
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
            norm_f32_attn: vec![0.0; hidden],
            norm_f32_ffn: vec![0.0; hidden],
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
        self.norm_f32_attn.fill(0.0);
        self.norm_f32_ffn.fill(0.0);
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

/// SPRINT-5 (5.3): branchless ELUT-4-bit → {-1.0, 0.0, +1.0} decode LUT.
/// Pre-computed once at process start; the table is read-only and lives in
/// the binary's `.rodata`, so the hot path is one indexed load per
/// 4-bit nibble. Replaces the previous `match val { 0x1 => 1.0, 0xF =>
/// -1.0, _ => 0.0 }` (3-branch compare + select). Only used in the
/// non-x86_64 scalar fallback path; x86_64 uses backward_ternary_gemv.s.
#[cfg_attr(target_arch = "x86_64", allow(dead_code))]
static TERNARY_NIBBLE_TO_F32: [f32; 16] = {
    let mut t = [0.0f32; 16];
    // ELUT 4-bit layout (BitNet ELUT, see MudTensorType::Ternary2Bit):
    //   0x1 → +1.0,  0xF → -1.0,  all other codes → 0.0 (deadzone)
    t[0x1] = 1.0;
    t[0xF] = -1.0;
    t
};

/// Helper: Compute gradients for a Ternary GEMV layer using the Straight-Through Estimator (STE).
/// Since we use ELUT 4-bit nibble packing, the weight unpacking will be needed for `grad_x`.
/// # Safety
/// Caller must ensure `w_u8` points to a valid 4-bit ELUT packed array of size `n_out * n_in.div_ceil(8) * 4` bytes (R-35: canonical per-row padded stride),
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
    // R-35: padded-row stride (canonical R-15 layout), not floor(n_in/8).
    let w_u32_stride = n_in.div_ceil(8);

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

            unsafe {
                #[cfg(target_arch = "x86_64")]
                {
                    let local_gx_ptr = (scratch_p as *mut f32).add(i * n_in);

                    // OPT-1: 8-row batched grad_x kernel (reduces grad_x load/store
                    // by 8× vs single-row). Only when n_in is multiple of 8 AND
                    // we have at least 8 rows to process.
                    if n_in.is_multiple_of(8) && (end_row - start_row) >= 8 {
                        let mut r = start_row;
                        while r + 8 <= end_row {
                            // OPT-4: Batched AXPY — loads x once per 8-element chunk,
                            // reuses across all 8 rows (FMA broadcast). Saves 7× x loads
                            // vs per-row axpy_avx2 calls.
                            crate::mud::ezop::axpy_8rows_avx2(
                                gy.add(r),
                                x_f32,
                                gw.add(r * n_in),
                                n_in,
                            );

                            // Build scale_gy[8] array (scale * grad_y per row) — AVX2 vectorized
                            let mut scale_gy_8: [f32; 8] = [0.0; 8];
                            {
                                use std::arch::x86_64::*;
                                let gy_v = _mm256_loadu_ps(gy.add(r));
                                let sc_v = _mm256_loadu_ps(sc.add(r));
                                let sg_v = _mm256_mul_ps(gy_v, sc_v);
                                _mm256_storeu_ps(scale_gy_8.as_mut_ptr(), sg_v);
                            }
                            let w_batch = w_u32.add(r * w_u32_stride);
                            // OPT-1: 8-row batched grad_x via ASM
                            crate::asm::backward_ternary_gemv_8rows_dx_avx2(
                                n_in,
                                scale_gy_8.as_ptr(),
                                w_batch,
                                local_gx_ptr,
                                w_u32_stride,
                            );
                            r += 8;
                        }
                        // Remainder rows: single-row kernel
                        for r in r..end_row {
                            let g_y = *gy.add(r);
                            let scale = *sc.add(r);
                            let w_row = w_u32.add(r * w_u32_stride);
                            let gw_row = std::slice::from_raw_parts_mut(
                                gw.add(r * n_in), n_in);
                            let x_slice = std::slice::from_raw_parts(x_f32, n_in);
                            forge_autograd::avx_math::axpy_avx2(gw_row, g_y, x_slice);
                            let scale_gy = scale * g_y;
                            crate::asm::backward_ternary_gemv_dx_avx2(
                                n_in, scale_gy, w_row, local_gx_ptr);
                        }
                    } else {
                        // Not enough rows or n_in not multiple of 8: single-row path
                        for r in start_row..end_row {
                            let g_y = *gy.add(r);
                            let scale = *sc.add(r);
                            let w_row = w_u32.add(r * w_u32_stride);
                            let gw_row = std::slice::from_raw_parts_mut(
                                gw.add(r * n_in), n_in);
                            let x_slice = std::slice::from_raw_parts(x_f32, n_in);
                            forge_autograd::avx_math::axpy_avx2(gw_row, g_y, x_slice);
                            let scale_gy = scale * g_y;
                            crate::asm::backward_ternary_gemv_dx_avx2(
                                n_in, scale_gy, w_row, local_gx_ptr);
                        }
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    let local_gx_ptr = (scratch_p as *mut f32).add(i * n_in);
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
                            let w_val = TERNARY_NIBBLE_TO_F32[val as usize];
                            *local_gx_ptr.add(c) += w_val * scale_gy;
                        }
                    }
                }
            }
        });
    }
    pool.wait_all();

    // 8-row gradient reduction: grad_x += scratch[0..8] (single fused merge)
    if is_x86_feature_detected!("avx2") && n_in >= 8 {
        unsafe {
            crate::mud::ezop::grad_x_reduce_8rows_avx2(
                grad_x.as_mut_ptr(),
                scratch_grad_x.as_ptr(),
                n_in,
            );
        }
    } else {
        for i in 0..8 {
            let local_gx_start = i * n_in;
            for c in 0..n_in {
                grad_x[c] += scratch_grad_x[local_gx_start + c];
            }
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
    // ── SPRINT-6 (P-01): pre-allocated train_on_sequence buffers ──────────────
    // Eliminates ~15 `vec![]` allocations per chunk (~100 KB). All sized at
    // workspace creation from hidden_size + num_neg (training config).
    pub x_data: Vec<f32>,
    pub pre_norm_x: Vec<f32>,
    pub final_x: Vec<f32>,
    pub class_embs_flat: Vec<f32>,
    pub class_logits: Vec<f32>,
    pub prob_q: Vec<f32>,
    pub d_logits: Vec<f32>,
    /// R-19: per-class ternary scales for sampled-softmax rows (target + negs).
    /// Same STE preprocessing for every row; per-class `scale_up` keeps logits
    /// coherently renormalized.
    pub class_scales: Vec<f32>,
    pub grad_in: Vec<f32>,
    pub grad_out: Vec<f32>,
    pub neg_ids: Vec<usize>,
    pub stp_hist: Vec<Vec<f32>>,
    pub stp_grad_t: Vec<f32>,
    pub stp_grad_r: Vec<f32>,
    pub stp_grad_s: Vec<f32>,
    /// SPRINT-6 (P-01): pre-allocated embedding checkpoint for grad recompute.
    /// Replaces `x_data[..hidden_size].to_vec()` in train_step.rs (was ~100 KB/chunk).
    pub emb_ckpt: Vec<f32>,
    /// FIX (2026-07-31): output_norm shadow buffer — the output_norm.weight tensor
    /// is mmap'd read-only; this owned copy lets the optimizer update it without
    /// SIGSEGV on the mmap region. Root cause of activation collapse.
    pub output_norm_shadow: Vec<f32>,
    /// Gradient w.r.t. output_norm.weight (pre-norm x_norm × grad_in).
    pub output_norm_w_grad: Vec<f32>,
}

impl SlimeBackwardWorkspace {
    /// `num_neg` — sampled-softmax negative count (training config; 0 for non-trainer use).
    pub fn new(hidden: usize, ffn_mid: usize, kv_dim: usize, num_neg: usize) -> Self {
        let max_in = hidden.max(kv_dim).max(ffn_mid);
        let num_classes = 1 + num_neg;
        const STP_HIST: usize = 4;
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
            // Pre-allocated train_on_sequence buffers (P-01)
            x_data: vec![0.0; hidden],
            pre_norm_x: vec![0.0; hidden],
            final_x: vec![0.0; hidden],
            class_embs_flat: vec![0.0; num_classes * hidden],
            class_logits: vec![0.0; num_classes],
            prob_q: vec![0.0; num_classes],
            d_logits: vec![0.0; num_classes],
            class_scales: vec![1.0; num_classes],
            grad_in: vec![0.0; hidden],
            grad_out: vec![0.0; hidden],
            neg_ids: vec![0; num_neg],
            stp_hist: (0..STP_HIST).map(|_| vec![0.0f32; hidden]).collect(),
            stp_grad_t: vec![0.0; hidden],
            stp_grad_r: vec![0.0; hidden],
            stp_grad_s: vec![0.0; hidden],
            emb_ckpt: vec![0.0; hidden],
            output_norm_shadow: Vec::new(),
            output_norm_w_grad: Vec::new(),
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

    // F32 weights (dequantized from Q2_0) — full backward support.
    // Uses f32_gemv_backward (dL/dW = grad_y ⊗ x^T) instead of ternary STE.
    // The input reconstruction is identical: dequantize the i8 tape values
    // to f32 using the stored act_scale (same as ternary backward).
    if layer.is_f32_weights {
        let kv_dim = ws.n_kv_heads * ws.head_dim;

        // Phase 1 (mHC trainable): gradient of the post-ffn hyper-connection scales.
        if !grads.mhc_alpha_grad.is_empty() {
            unsafe {
                crate::mud::ezop::mhc_backward_vec(
                    grad_in.as_ptr(),
                    tape.ffn_v_jepa.as_ptr(),
                    tape.mhc_ffn_h_in.as_ptr(),
                    tape.mhc_ffn_f_h.as_ptr(),
                    grads.mhc_alpha_grad.as_mut_ptr(),
                    grads.mhc_beta_grad.as_mut_ptr(),
                    hidden,
                );
            }
        }

        // Kinetic spring + mHC activation backward (R-12 FIX).
        // Forward: h_out = alpha·h_in + (1-gate)·beta·f_h + kinetic; scales now applied.
        let kinetic_lambda = 0.005f32;
        unsafe {
            crate::mud::ezop::mhc_activation_backward(
                grad_in.as_ptr(),
                tape.ffn_v_jepa.as_ptr(),
                layer.mhc_alpha_w,
                layer.mhc_beta_w,
                grad_out.as_mut_ptr(),
                b_ws.grad_branch.as_mut_ptr(),
                kinetic_lambda,
                hidden,
            );
        }

        // 1. FFN Down Projection Backward (F32)
        let grad_ffn_out = &b_ws.grad_branch[..hidden];

        // Reconstruct the exact quantized input to FFN down
        let ffn_hid_peak = {
            #[cfg(target_arch = "x86_64")]
            {
                let peak = unsafe { crate::mud::ezop::peak_abs_avx2(tape.ffn_mid_f32.as_ptr(), ffn_mid) };
                (peak / 127.0f32).max(1e-8f32)
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                let mut p = 0.0f32;
                for i in 0..ffn_mid {
                    p = p.max(tape.ffn_mid_f32[i].abs());
                }
                (p / 127.0f32).max(1e-8f32)
            }
        };

        for i in 0..ffn_mid {
            let q = (tape.ffn_mid_f32[i] / ffn_hid_peak).clamp(-127.0, 127.0) as i8;
            b_ws.ffn_hid_x[i] = q as f32 * ffn_hid_peak;
        }

        b_ws.grad_ffn_mid.fill(0.0);
        unsafe {
            crate::mud::slime_forward::f32_gemv_backward(
                grad_ffn_out,
                &b_ws.ffn_hid_x,
                layer.ffn_down_w as *const f32,
                &mut grads.ffn_down_w_grad,
                &mut b_ws.grad_ffn_mid,
                &mut b_ws.scratch_grad_x,
                hidden,
                ffn_mid,
            );
        }

        // 2. FFN ReLU² Activation Backward
        #[cfg(target_arch = "x86_64")]
        {
            unsafe {
                crate::mud::ezop::relu2_backward_avx2(
                    b_ws.grad_ffn_mid.as_ptr(),
                    tape.ffn_gate_f32.as_ptr(),
                    tape.ffn_up_f32.as_ptr(),
                    b_ws.grad_ffn_up.as_mut_ptr(),
                    b_ws.grad_ffn_gate.as_mut_ptr(),
                    ffn_mid,
                );
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
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
        }

        // 3. FFN Up & Gate Projection Backward (F32)
        // R-43: f32 stored directly, no i8 round-trip.
        b_ws.ffn_in_x[..hidden].copy_from_slice(&tape.norm_f32_ffn[..hidden]);

        b_ws.grad_ffn_in_up.fill(0.0);
        unsafe {
            crate::mud::slime_forward::f32_gemv_backward(
                &b_ws.grad_ffn_up,
                &b_ws.ffn_in_x,
                layer.ffn_up_w as *const f32,
                &mut grads.ffn_up_w_grad,
                &mut b_ws.grad_ffn_in_up,
                &mut b_ws.scratch_grad_x,
                ffn_mid,
                hidden,
            );
        }

        b_ws.grad_ffn_in_gate.fill(0.0);
        unsafe {
            crate::mud::slime_forward::f32_gemv_backward(
                &b_ws.grad_ffn_gate,
                &b_ws.ffn_in_x,
                layer.ffn_gate_w as *const f32,
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

        // 4. FFN RMSNorm Backward — VECTORIZED (AVX2 pipeline)
        // Pipeline:
        //   dequant_i8→f32 (scale_i8_to_f32_avx2) → div_avx2 (y_unscaled)
        //   → dot3_fma_avx2 (sum_dy_xnorm) → backward_rmsnorm_avx2 (ASM, FMA)
        // Eliminates 2 scalar loops (2048×2 = 4096 iterations) → ~256 AVX2 iters
        let y_f32_ptr = b_ws.scratch_grad_x.as_mut_ptr();
        let mean_dy_xnorm: f32;
        unsafe {
            if is_x86_feature_detected!("avx2") && hidden >= 8 {
                // Step 1: copy f32 norm → scratch_grad_x (R-43: no i8 round-trip)
                b_ws.scratch_grad_x[..hidden].copy_from_slice(&tape.norm_f32_ffn[..hidden]);
                // Step 2: y_unscaled = y_f32 / w (in-place: div_avx2 supports a==out)
                crate::mud::ezop::div_avx2(
                    y_f32_ptr, layer.ffn_norm_w, y_f32_ptr, 0.0, hidden,
                );
                // Step 3: sum_dy_xnorm = dot3_fma(grad, w, y_unscaled)
                let sum_dy_xnorm = crate::mud::ezop::dot3_fma_avx2(
                    b_ws.grad_ffn_norm_out.as_ptr(),
                    layer.ffn_norm_w,
                    y_f32_ptr,
                    hidden,
                );
                mean_dy_xnorm = sum_dy_xnorm / hidden as f32;
            } else {
                // Scalar fallback
                let mut sum_dy_xnorm = 0.0f32;
                for i in 0..hidden {
                    let y_f32 = tape.norm_f32_ffn[i];
                    let w = *layer.ffn_norm_w.add(i);
                    let y_unscaled = if w.abs() > 1e-8 { y_f32 / w } else { 0.0 };
                    b_ws.scratch_grad_x[i] = y_unscaled;
                    sum_dy_xnorm += b_ws.grad_ffn_norm_out[i] * w * y_unscaled;
                }
                mean_dy_xnorm = sum_dy_xnorm / hidden as f32;
            }
        }

        // Weight gradient (scalar, 2048 iters — negligible vs 32-layer GEMV)
        for i in 0..hidden {
            grads.ffn_norm_w_grad[i] += b_ws.grad_ffn_norm_out[i] * b_ws.scratch_grad_x[i];
        }

        // grad_x = grad_y * w - y_unscaled * mean  (ASM: backward_rmsnorm_avx2)
        // SAFETY: all arrays valid for hidden elements; scratch_grad_x has >= hidden pre-computed y_unscaled
        unsafe {
            crate::asm::backward_rmsnorm_avx2(
                hidden,
                b_ws.grad_ffn_norm_out.as_ptr(),
                b_ws.scratch_grad_x.as_ptr(),
                layer.ffn_norm_w,
                mean_dy_xnorm.to_bits(),
                b_ws.grad_residual_ffn.as_mut_ptr(),
            );
        }

        // Accumulate FFN residual gradient (was inside the old scalar loop)
        // AVX2: grad_out += grad_residual_ffn (8 f32/iter via FMA, a=1.0)
        if is_x86_feature_detected!("avx2") && hidden >= 8 {
            unsafe {
                crate::mud::ezop::axpy_avx2(
                    grad_out.as_mut_ptr(), 1.0,
                    b_ws.grad_residual_ffn.as_ptr(), hidden,
                );
            }
        } else {
            for i in 0..hidden {
                grad_out[i] += b_ws.grad_residual_ffn[i];
            }
        }

        // Phase 1 (mHC trainable): post-attn hyper-connection scales.
        if !grads.mhc_alpha_grad.is_empty() {
            unsafe {
                crate::mud::ezop::mhc_backward_vec(
                    grad_out.as_ptr(),
                    tape.attn_v_jepa.as_ptr(),
                    tape.mhc_attn_h_in.as_ptr(),
                    tape.o_act_f32.as_ptr(),
                    grads.mhc_alpha_grad.as_mut_ptr(),
                    grads.mhc_beta_grad.as_mut_ptr(),
                    hidden,
                );
            }
        }

        // --- ATTENTION ---
        // 5. Attention O Projection Backward (F32)
        b_ws.grad_o_in.fill(0.0);
        b_ws.grad_k_in.fill(0.0);
        b_ws.grad_v_in.fill(0.0);
        unsafe {
            crate::mud::slime_forward::f32_gemv_backward(
                &b_ws.grad_branch[..hidden],
                &tape.o_act_f32,
                layer.o_w as *const f32,
                &mut grads.o_w_grad,
                &mut b_ws.grad_o_in,
                &mut b_ws.scratch_grad_x,
                hidden,
                hidden,
            );
        }

        // 6. Scaled Dot-Product Attention Backward (STE Proxy)
        let head_d = hidden / ws.n_heads;
        let gqa_scale = ws.n_heads / ws.n_kv_heads;
        let scores_val = tape.scores[tape.pos];
        let use_avx = is_x86_feature_detected!("avx2") && head_d >= 8;

        if use_avx {
            unsafe {
                for h in 0..ws.n_heads {
                    let kv_h = h / gqa_scale;
                    let q_off = h * head_d;
                    let kv_off = kv_h * head_d;
                    let go_ptr = b_ws.grad_o_in.as_ptr().add(q_off);
                    let qf_ptr = tape.q_f32.as_ptr().add(q_off);
                    let gq_ptr = b_ws.grad_q_in.as_mut_ptr().add(q_off);
                    let gk_ptr = b_ws.grad_k_in.as_mut_ptr().add(kv_off);
                    let gv_ptr = b_ws.grad_v_in.as_mut_ptr().add(kv_off);

                    // K first: copy grad_o → grad_q, scale by q_f32, accumulate to grad_k
                    std::ptr::copy_nonoverlapping(go_ptr, gq_ptr, head_d);
                    crate::mud::ezop::vmul_inplace_avx2(qf_ptr, gq_ptr, head_d);
                    crate::mud::ezop::axpy_avx2(gk_ptr, 1.0, gq_ptr, head_d);

                    // Q+V: copy grad_o → grad_q, scale by scores_val, then V accumulates from grad_q
                    std::ptr::copy_nonoverlapping(go_ptr, gq_ptr, head_d);
                    crate::mud::ezop::scale_vec_avx2(gq_ptr, scores_val, head_d);
                    // grad_q now holds grad_o * scores = Q output
                    crate::mud::ezop::axpy_avx2(gv_ptr, 1.0, gq_ptr, head_d);
                }
            }
        } else {
            for h in 0..ws.n_heads {
                let kv_h = h / gqa_scale;
                let q_off = h * head_d;
                let kv_off = kv_h * head_d;
                for d in 0..head_d {
                    let go = b_ws.grad_o_in[q_off + d];
                    b_ws.grad_q_in[q_off + d] = go * tape.scores[tape.pos];
                    b_ws.grad_k_in[kv_off + d] += go * tape.q_f32[q_off + d];
                    b_ws.grad_v_in[kv_off + d] += go * tape.scores[tape.pos];
                }
            }
        }

        // 6b. Backward RoPE (un-rotate gradients before QKV projection backward)
        if layer.rope_theta > 0.0 && !layer.rope_inv_freq.is_empty() {
            let mut guard = layer.rope_cache.lock();
            let (cos_tbl, sin_tbl) = guard.get_or_compute(tape.pos, &layer.rope_inv_freq);
            for h in 0..ws.n_heads {
                let q_off = h * head_d;
                // SAFETY: grad_q_in has head_d elements at q_off; cos/sin have half_d elements
                unsafe {
                    crate::asm::backward_rope_asm(
                        head_d,
                        b_ws.grad_q_in[q_off..].as_mut_ptr(),
                        cos_tbl.as_ptr(),
                        sin_tbl.as_ptr(),
                    );
                }
            }
            for h in 0..ws.n_kv_heads {
                let kv_off = h * head_d;
                // SAFETY: grad_k_in has head_d elements at kv_off
                unsafe {
                    crate::asm::backward_rope_asm(
                        head_d,
                        b_ws.grad_k_in[kv_off..].as_mut_ptr(),
                        cos_tbl.as_ptr(),
                        sin_tbl.as_ptr(),
                    );
                }
            }
        }

        // 7. Attention Q, K, V Projection Backward (F32) — batched (no sync overhead)
        // R-43: f32 stored directly, no i8 round-trip.
        b_ws.attn_in_x[..hidden].copy_from_slice(&tape.norm_f32_attn[..hidden]);

        b_ws.grad_residual_ffn.fill(0.0);
        unsafe {
            crate::mud::slime_forward::f32_gemv_backward_batch(
                &b_ws.grad_q_in,
                &b_ws.grad_k_in,
                &b_ws.grad_v_in,
                &b_ws.attn_in_x,
                layer.q_w as *const f32,
                layer.k_w as *const f32,
                layer.v_w as *const f32,
                &mut grads.q_w_grad,
                &mut grads.k_w_grad,
                &mut grads.v_w_grad,
                &mut b_ws.grad_residual_ffn,
                &mut b_ws.scratch_grad_x,
                hidden,
                kv_dim,
                hidden,
            );
        }

        // 8. Attention RMSNorm Backward
        // P-01: pre-compute y_unscaled into scratch, use ASM kernel for grad_x
        let mut sum_dy_xnorm = 0.0f32;
        for i in 0..hidden {
            let y_f32 = b_ws.attn_in_x[i];
            let w = unsafe { *layer.attn_norm_w.add(i) };
            let y_unscaled = if w.abs() > 1e-8 { y_f32 / w } else { 0.0 };
            b_ws.scratch_grad_x[i] = y_unscaled;
            sum_dy_xnorm += b_ws.grad_residual_ffn[i] * w * y_unscaled;
        }
        let mean_dy_xnorm = sum_dy_xnorm / hidden as f32;

        // Weight gradient (scalar, per-element)
        for i in 0..hidden {
            grads.attn_norm_w_grad[i] += b_ws.grad_residual_ffn[i] * b_ws.scratch_grad_x[i];
        }

        // grad_x = grad_y * w - y_unscaled * mean  (ASM: backward_rmsnorm_avx2)
        // Output to scratch_grad_x (reuse as temp), then accumulate into grad_out
        // SAFETY: all arrays valid for hidden elements
        unsafe {
            crate::asm::backward_rmsnorm_avx2(
                hidden,
                b_ws.grad_residual_ffn.as_ptr(),
                b_ws.scratch_grad_x.as_ptr(),
                layer.attn_norm_w,
                mean_dy_xnorm.to_bits(),
                b_ws.scratch_grad_x.as_mut_ptr(), // overwrite temp with grad_x
            );
        }
        // AVX2 add: grad_out[i] += scratch_grad_x[i] (8 f32/iter via FMA)
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && hidden >= 8 {
            unsafe {
                ezop::axpy_avx2(grad_out.as_mut_ptr(), 1.0, b_ws.scratch_grad_x.as_ptr(), hidden);
            }
        } else {
            for i in 0..hidden {
                grad_out[i] += b_ws.scratch_grad_x[i];
            }
        }

        return;
    }

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
        unsafe {
            crate::mud::ezop::mhc_backward_vec(
                grad_in.as_ptr(),
                tape.ffn_v_jepa.as_ptr(),
                tape.mhc_ffn_h_in.as_ptr(),
                tape.mhc_ffn_f_h.as_ptr(),
                grads.mhc_alpha_grad.as_mut_ptr(),
                grads.mhc_beta_grad.as_mut_ptr(),
                hidden,
            );
        }
    }

    // SAFETY: buffers sized to hidden at workspace construction.
    // R-12 FIX: apply mHC scales (alpha, (1-gate)*beta) to the activation grads.
    // Forward: h_out = alpha·h_in + (1-gate)·beta·f_h (+ kinetic spring).
    //   grad_out  = grad_in · alpha                      → dL/d h_in (residual)
    //   grad_branch = (grad_in + kinetic)·(1-gate)·beta  → dL/d f_h  (grad_ffn_out)
    unsafe {
        crate::mud::ezop::mhc_activation_backward(
            grad_in.as_ptr(),
            tape.ffn_v_jepa.as_ptr(),
            layer.mhc_alpha_w,
            layer.mhc_beta_w,
            grad_out.as_mut_ptr(),
            b_ws.grad_branch.as_mut_ptr(),
            kinetic_lambda,
            hidden,
        );
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

    // 2. FFN SiLU Activation Backward (R-11 FIX: was ReLU², mismatched forward)
    // Forward: ffn_mid_f32[i] = silu(gate[i]) * up[i]
    //   silu(x) = x * σ(x);  silu'(x) = σ(x) * (1 + x*(1-σ(x)))
    //   grad_up[i]   = gy[i] * silu(gate[i])
    //   grad_gate[i] = gy[i] * up[i] * silu'(gate[i])
    // AVX2 (8 f32/iter) via ezop, with scalar fallback. Reuses the exp2
    // approximation of silu_vectorial_avx2 so backward==forward to ~1e-3.
    let gy_ptr = b_ws.grad_ffn_mid.as_ptr();
    let g_ptr = tape.ffn_gate_f32.as_ptr();
    let u_ptr = tape.ffn_up_f32.as_ptr();
    unsafe {
        crate::mud::ezop::silu_backward_avx2(
            gy_ptr,
            g_ptr,
            u_ptr,
            b_ws.grad_ffn_up.as_mut_ptr(),
            b_ws.grad_ffn_gate.as_mut_ptr(),
            ffn_mid,
        );
    }

    // 3. FFN Up & Gate Projection Backward (Ternary STE)
    // R-43: f32 stored directly, no i8 round-trip.
    b_ws.ffn_in_x[..hidden].copy_from_slice(&tape.norm_f32_ffn[..hidden]);

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

    // AVX2 add: grad_ffn_norm_out = grad_ffn_in_up + grad_ffn_in_gate (8 f32/iter)
    if is_x86_feature_detected!("avx2") && hidden >= 8 {
        unsafe {
            // Copy grad_ffn_in_up → grad_ffn_norm_out, then add grad_ffn_in_gate
            std::ptr::copy_nonoverlapping(
                b_ws.grad_ffn_in_up.as_ptr(),
                b_ws.grad_ffn_norm_out.as_mut_ptr(),
                hidden,
            );
            crate::mud::ezop::axpy_avx2(
                b_ws.grad_ffn_norm_out.as_mut_ptr(), 1.0,
                b_ws.grad_ffn_in_gate.as_ptr(), hidden,
            );
        }
    } else {
        for i in 0..hidden {
            b_ws.grad_ffn_norm_out[i] = b_ws.grad_ffn_in_up[i] + b_ws.grad_ffn_in_gate[i];
        }
    }

    // 4. FFN RMSNorm Backward (Orthogonalized STE Pass-through)
    // Computes gradient w.r.t the norm weights and passes the orthogonalized signal.
    // Uses pre-dequantized b_ws.ffn_in_x (computed at step 3) to avoid redundant dequant.
    let mut sum_dy_xnorm = 0.0f32;
    for i in 0..hidden {
        let y_f32 = b_ws.ffn_in_x[i];
        let w = unsafe { *layer.ffn_norm_w.add(i) };
        let y_unscaled = if w.abs() > 1e-8 { y_f32 / w } else { 0.0 };
        sum_dy_xnorm += b_ws.grad_ffn_norm_out[i] * w * y_unscaled;
    }
    let mean_dy_xnorm = sum_dy_xnorm / hidden as f32;

    for i in 0..hidden {
        let y_f32 = b_ws.ffn_in_x[i];
        let w = unsafe { *layer.ffn_norm_w.add(i) };
        let y_unscaled = if w.abs() > 1e-8 { y_f32 / w } else { 0.0 };

        grads.ffn_norm_w_grad[i] += b_ws.grad_ffn_norm_out[i] * y_unscaled;

        b_ws.grad_residual_ffn[i] = b_ws.grad_ffn_norm_out[i] * w - y_unscaled * mean_dy_xnorm;
    }

    // Sum the FFN branch gradient back into the main residual stream.
    // AVX2: grad_out += grad_residual_ffn (8 f32/iter via FMA, a=1.0)
    if is_x86_feature_detected!("avx2") && hidden >= 8 {
        unsafe {
            crate::mud::ezop::axpy_avx2(
                grad_out.as_mut_ptr(), 1.0,
                b_ws.grad_residual_ffn.as_ptr(), hidden,
            );
        }
    } else {
        for i in 0..hidden {
            grad_out[i] += b_ws.grad_residual_ffn[i];
        }
    }

    // Phase 1 (mHC trainable): gradient of the post-attn hyper-connection scales.
    // Forward: out[i] = alpha[i]*h_in[i] + (1-gate[i])*beta[i]*f_h[i], f_h = o_act_f32.
    // grad wrt the mHC output here is grad_out (accumulated after the FFN branch).
    if !grads.mhc_alpha_grad.is_empty() {
        unsafe {
            crate::mud::ezop::mhc_backward_vec(
                grad_out.as_ptr(),
                tape.attn_v_jepa.as_ptr(),
                tape.mhc_attn_h_in.as_ptr(),
                tape.o_act_f32.as_ptr(),
                grads.mhc_alpha_grad.as_mut_ptr(),
                grads.mhc_beta_grad.as_mut_ptr(),
                hidden,
            );
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
    // R-43: f32 stored directly, no i8 round-trip.
    b_ws.attn_in_x[..hidden].copy_from_slice(&tape.norm_f32_attn[..hidden]);

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

    // 8. Attention RMSNorm Backward — VECTORIZED (AVX2 pipeline)
    // Pipeline: div_avx2 (y_unscaled) → dot3_fma_avx2 (sum_dy_xnorm)
    //           → w_grad (scalar, 2048 iters) → ASM backward_rmsnorm (grad_x)
    // Eliminates 1 scalar loop (2048 iterations for y_unscaled+sum_dy_xnorm)
    // → ~256 AVX2 iterations. w_grad kept scalar (scratch_grad_x needed for ASM).
    let y_f32_ptr = b_ws.scratch_grad_x.as_mut_ptr();
    let mean_dy_xnorm: f32;
    unsafe {
        if is_x86_feature_detected!("avx2") && hidden >= 8 {
            // y_unscaled = attn_in_x / w → scratch_grad_x (read-only src, safe aliasing)
            crate::mud::ezop::div_avx2(
                b_ws.attn_in_x.as_ptr(),
                layer.attn_norm_w,
                y_f32_ptr,
                0.0,
                hidden,
            );
            // sum_dy_xnorm = dot3_fma(grad_residual, w, y_unscaled)
            let sum_dy_xnorm = crate::mud::ezop::dot3_fma_avx2(
                b_ws.grad_residual_ffn.as_ptr(),
                layer.attn_norm_w,
                y_f32_ptr,
                hidden,
            );
            mean_dy_xnorm = sum_dy_xnorm / hidden as f32;

            // w_grad += grad * y_unscaled (scalar — scratch_grad_x needed for ASM below)
            for i in 0..hidden {
                grads.attn_norm_w_grad[i] +=
                    b_ws.grad_residual_ffn[i] * b_ws.scratch_grad_x[i];
            }
        } else {
            // Scalar fallback (identical logic)
            let mut sum_dy_xnorm = 0.0f32;
            for i in 0..hidden {
                let y_f32 = b_ws.attn_in_x[i];
                let w = *layer.attn_norm_w.add(i);
                let y_unscaled = if w.abs() > 1e-8 { y_f32 / w } else { 0.0 };
                b_ws.scratch_grad_x[i] = y_unscaled;
                sum_dy_xnorm += b_ws.grad_residual_ffn[i] * w * y_unscaled;
                grads.attn_norm_w_grad[i] += b_ws.grad_residual_ffn[i] * y_unscaled;
            }
            mean_dy_xnorm = sum_dy_xnorm / hidden as f32;
        }
    }

    // grad_x = grad * w - y_unscaled * mean (ASM: backward_rmsnorm_avx2)
    // SAFETY: scratch_grad_x has y_unscaled, grad_residual_ffn/attn_norm_w valid for hidden.
    unsafe {
        crate::asm::backward_rmsnorm_avx2(
            hidden,
            b_ws.grad_residual_ffn.as_ptr(),
            b_ws.scratch_grad_x.as_ptr(),
            layer.attn_norm_w,
            mean_dy_xnorm.to_bits(),
            b_ws.grad_residual_ffn.as_mut_ptr(),
        );
        // Accumulate grad_x into the residual stream
        crate::mud::ezop::axpy_avx2(
            grad_out.as_mut_ptr(), 1.0,
            b_ws.grad_residual_ffn.as_ptr(), hidden,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mud::slime::SlimeWorkspace;
    use crate::mud::slime_forward::SlimeLayer;

    #[test]
    fn test_ternary_tq_gemv_avx2() {
        // Same test as scalar, but using AVX2 path.
        let vals_row0: Vec<i8> = vec![1, 1, 1, 1, 1, 1, 1, 1];
        let vals_row1: Vec<i8> = vec![1, -1, 0, 1, -1, 0, 1, -1];
        let cols: usize = 8;
        let rows: usize = 2;
        let nbytes = rows * cols.div_ceil(5usize);
        let mut packed = vec![0u8; nbytes];
        for (r, vals) in [vals_row0.as_slice(), vals_row1.as_slice()].iter().enumerate() {
            let byte_base = r * cols.div_ceil(5usize);
            for c in 0..cols {
                let trit = match vals[c] {
                    -1 => 0u8,
                    0 => 1u8,
                    1 => 2u8,
                    _ => 1u8,
                };
                packed[byte_base + c / 5] += trit * 3u8.pow((c % 5) as u32);
            }
        }
        let scales = [1.0f32, 2.0f32];
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut out = vec![0.0f32; rows];
        ternary_tq_gemv_avx2(&packed, &scales, &x, &mut out, rows, cols);
        // Row 0: sum = 1+2+3+4+5+6+7+8 = 36, scale 1.0 → 36.0
        assert!((out[0] - 36.0).abs() < 1e-5, "row0: {}", out[0]);
        // Row 1: 1-2+0+4-5+0+7-8 = -3, scale 2.0 → -6.0
        assert!((out[1] - (-6.0)).abs() < 1e-5, "row1: {}", out[1]);
    }

    #[test]
    fn test_ternary_tq_gemv() {
        // 2 rows, 5 cols. Row 0: all +1, Row 1: alternating.
        let vals_row0: Vec<i8> = vec![1, 1, 1, 1, 1];
        let vals_row1: Vec<i8> = vec![1, -1, 0, 1, -1];
        let cols: usize = 5;
        let rows: usize = 2;
        let nbytes = rows * cols.div_ceil(5usize); // 2 bytes (1 per row)
        let mut packed = vec![0u8; nbytes];
        for (r, vals) in [vals_row0.as_slice(), vals_row1.as_slice()].iter().enumerate() {
            let byte_base = r * cols.div_ceil(5usize);
            for c in 0..cols {
                let trit = match vals[c] {
                    -1 => 0u8,
                    0 => 1u8,
                    1 => 2u8,
                    _ => 1u8,
                };
                packed[byte_base] += trit * 3u8.pow((c % 5) as u32);
            }
        }
        let scales = [1.0f32, 2.0f32];
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut out = vec![0.0f32; rows];
        ternary_tq_gemv(&packed, &scales, &x, &mut out, rows, cols);
        // Row 0: sum = 1+2+3+4+5 = 15, scale 1.0 → 15.0
        assert!((out[0] - 15.0).abs() < 1e-6);
        // Row 1: 1*1 + (-1)*2 + 0*3 + 1*4 + (-1)*5 = 1-2+0+4-5 = -2, scale 2.0 → -4.0
        assert!((out[1] - (-4.0)).abs() < 1e-6);
    }

    #[test]
    fn test_unpack_ternary_tq_roundtrip() {
        // Encode 10 ternary values as base-3, decode, verify.
        let vals: Vec<i8> = vec![-1, 0, 1, -1, 1, 0, -1, 0, 1, -1];
        let scale = 2.5f32;
        let cols = vals.len();
        let rows = 1;
        let nbytes = rows * cols.div_ceil(5usize);
        let mut packed = vec![0u8; nbytes];
        // Encode
        for (c, &val) in vals.iter().enumerate() {
            let byte_idx = c / 5;
            let trit_idx = c % 5;
            let trit = match val {
                -1 => 0u8,
                0 => 1u8,
                1 => 2u8,
                _ => 1u8,
            };
            packed[byte_idx] += trit * 3u8.pow(trit_idx as u32);
        }
        let scales = [scale];
        let mut out = vec![0.0f32; rows * cols];
        unpack_ternary_tq_to_f32(&packed, &scales, cols, &mut out);
        for (i, &v) in vals.iter().enumerate() {
            let expected = v as f32 * scale;
            assert!(
                (out[i] - expected).abs() < 1e-6,
                "idx {}: got {} expected {}",
                i,
                out[i],
                expected
            );
        }
    }

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
        // SPRINT-4: `mhc_scale_sgd_step` moved to `crate::mud::optim` to break
        // the `corpus_trainer ↔ slime_backward` cycle. The function body is
        // byte-identical; just the import path changes.
        use crate::mud::optim::mhc_scale_sgd_step;
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
        let mut b_ws = SlimeBackwardWorkspace::new(hidden, ffn_mid, kv_dim, 0);

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
            mhc_alpha_shadow: None,
            mhc_beta_shadow: None,
            attn_norm_shadow: None,
            ffn_norm_shadow: None,
            mhc_radius_w: std::ptr::null(),
            n_kv_heads: 1,
            ffn_mid,
            rope_theta: 0.0,
            rope_inv_freq: Vec::new(),
            rope_cache: parking_lot::Mutex::new(crate::mud::slime_forward::RopeCache::new()),
            is_f32_weights: false,
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
