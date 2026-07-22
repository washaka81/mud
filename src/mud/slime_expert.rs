//! # L-11: SlimeExpert — ternary FFN expert (Mini MoE)
//!
//! Each expert is a drop-in SwiGLU FFN: `down(silu(gate(x)) * up(x))` with ELUT 4-bit
//! weights + PRQ scales (same wire format as dense `SlimeLayer` FFN).

use crate::mud::slime_forward::{ternary_gemv_rowwise, SlimeLayer};

/// One ternary expert (up / gate / down). Pointers are non-owning (P-00).
#[derive(Clone, Copy, Debug)]
pub struct SlimeExpert {
    pub id: u16,
    pub hidden: usize,
    pub ffn_mid: usize,
    pub up_w: *const u8,
    pub gate_w: *const u8,
    pub down_w: *const u8,
    pub up_scales: *const f32,
    pub gate_scales: *const f32,
    pub down_scales: *const f32,
}

// SAFETY: raw weight pointers are immutable shared tensors for the life of the model/bus.
unsafe impl Send for SlimeExpert {}
unsafe impl Sync for SlimeExpert {}

impl SlimeExpert {
    /// Build expert from explicit ELUT pointers (caller owns memory).
    #[allow(clippy::too_many_arguments)]
    pub fn from_ptrs(
        id: u16,
        hidden: usize,
        ffn_mid: usize,
        up_w: *const u8,
        gate_w: *const u8,
        down_w: *const u8,
        up_scales: *const f32,
        gate_scales: *const f32,
        down_scales: *const f32,
    ) -> Self {
        Self {
            id,
            hidden,
            ffn_mid,
            up_w,
            gate_w,
            down_w,
            up_scales,
            gate_scales,
            down_scales,
        }
    }

    /// Wrap the dense FFN of a `SlimeLayer` as expert slot (C7 backward-compat).
    pub fn from_dense_layer(id: u16, layer: &SlimeLayer, hidden: usize) -> Self {
        Self::from_ptrs(
            id,
            hidden,
            layer.ffn_mid,
            layer.ffn_up_w,
            layer.ffn_gate_w,
            layer.ffn_down_w,
            layer.ffn_up_scales,
            layer.ffn_gate_scales,
            layer.ffn_down_scales,
        )
    }

    pub fn is_valid(&self) -> bool {
        self.hidden > 0
            && self.ffn_mid > 0
            && !self.up_w.is_null()
            && !self.gate_w.is_null()
            && !self.down_w.is_null()
            && !self.up_scales.is_null()
            && !self.gate_scales.is_null()
            && !self.down_scales.is_null()
    }

    /// SwiGLU FFN forward into `out` (length `hidden`).
    /// Scratch: `up`, `gate`, `mid` must be ≥ `ffn_mid`; `out` ≥ `hidden`.
    ///
    /// # Safety
    /// Weight pointers valid; `x.len() >= hidden`.
    pub unsafe fn forward_swiglu(
        &self,
        x: &[f32],
        up: &mut [f32],
        gate: &mut [f32],
        mid: &mut [f32],
        out: &mut [f32],
    ) {
        debug_assert!(self.is_valid());
        let h = self.hidden;
        let m = self.ffn_mid;
        debug_assert!(x.len() >= h && up.len() >= m && gate.len() >= m && mid.len() >= m);
        debug_assert!(out.len() >= h);

        ternary_gemv_rowwise(&x[..h], self.up_w, &mut up[..m], self.up_scales, m, h);
        ternary_gemv_rowwise(&x[..h], self.gate_w, &mut gate[..m], self.gate_scales, m, h);

        crate::asm::silu_vectorial_avx2(m, gate.as_ptr(), mid.as_mut_ptr());
        // mid = silu(gate) * up
        {
            let mid_s = &mut mid[..m];
            let up_s = &up[..m];
            let mut i = 0;
            while i + 8 <= m {
                for k in 0..8 {
                    mid_s[i + k] *= up_s[i + k];
                }
                i += 8;
            }
            while i < m {
                mid_s[i] *= up_s[i];
                i += 1;
            }
        }

        ternary_gemv_rowwise(
            &mid[..m],
            self.down_w,
            &mut out[..h],
            self.down_scales,
            h,
            m,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_dense_nulls_invalid() {
        let layer = SlimeLayer {
            q_w: std::ptr::null(),
            k_w: std::ptr::null(),
            v_w: std::ptr::null(),
            o_w: std::ptr::null(),
            q_scales: std::ptr::null(),
            k_scales: std::ptr::null(),
            v_scales: std::ptr::null(),
            o_scales: std::ptr::null(),
            ffn_up_w: std::ptr::null(),
            ffn_gate_w: std::ptr::null(),
            ffn_down_w: std::ptr::null(),
            ffn_up_scales: std::ptr::null(),
            ffn_gate_scales: std::ptr::null(),
            ffn_down_scales: std::ptr::null(),
            attn_norm_w: std::ptr::null(),
            ffn_norm_w: std::ptr::null(),
            attn_sub_norm_w: std::ptr::null(),
            ffn_sub_norm_w: std::ptr::null(),
            q_norm_w: std::ptr::null(),
            k_norm_w: std::ptr::null(),
            mhc_alpha_w: std::ptr::null(),
            mhc_beta_w: std::ptr::null(),
            mhc_radius_w: std::ptr::null(),
            n_kv_heads: 1,
            ffn_mid: 32,
            rope_theta: 0.0,
        };
        let e = SlimeExpert::from_dense_layer(0, &layer, 16);
        assert!(!e.is_valid());
    }

    #[test]
    fn test_expert_ids() {
        let e = SlimeExpert::from_ptrs(
            3,
            8,
            16,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
        );
        assert_eq!(e.id, 3);
        assert_eq!(e.hidden, 8);
        assert_eq!(e.ffn_mid, 16);
    }
}
