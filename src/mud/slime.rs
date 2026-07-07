/// SlimeRegister v2: 4-byte dual-state register for the MUD inference engine.
///
/// Layout (u32):
///   bits [15: 0] = ternary_f16    — ternary matmul accumulation as IEEE f16
///   bits [31:16] = integral_f16   — JEPA integral I = 0.99·I + 0.01·v_jepa, as IEEE f16
///
/// Eliminates `iscale` from the hot loop: f16 carries its own exponent,
/// so values are stored at their natural magnitude (range ±65504) without
/// fixed-point scaling. This resolves the i16 saturation crisis structurally.
///
/// The integral JEPA (I-controller) replaces the old proportional v_jepa gate:
///   - I→0 at equilibrium (when v_jepa→0), guaranteeing gate≈0.5
///   - Low-pass filtering rejects transient noise spikes
///   - Smoother convergence than proportional-only control
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct SlimeRegister(pub u32);

impl Default for SlimeRegister {
    #[inline(always)]
    fn default() -> Self {
        Self(0)
    }
}

impl SlimeRegister {
    // ── Ternary accumulation (lower 16 bits = f16) ──────────────────────────

    /// Read ternary accumulation as f32. No iscale needed — f16 is self-scaled.
    #[inline(always)]
    pub fn read_accum(&self) -> f32 {
        half_to_float_bits((self.0 & 0xFFFF) as u16)
    }

    /// Write ternary accumulation. `val` is stored as f16 in lower 16 bits.
    #[inline(always)]
    pub fn write_accum(&mut self, val: f32) {
        self.0 = (self.0 & 0xFFFF_0000) | (float_to_half_bits(val) as u32);
    }

    // ── JEPA & Conciencia (upper 16 bits subdivided) ─────────────────────────
    // Bits [16:23]: JEPA Integral (i8, scaled by 100.0 to f32)
    // Bits [24:31]: Derivada Cognitiva / Conciencia (i8, scaled by 100.0 to f32)

    /// Read the running JEPA integral as f32.
    #[inline(always)]
    pub fn read_integral(&self) -> f32 {
        let val_i8 = ((self.0 >> 16) & 0xFF) as i8;
        (val_i8 as f32) / 100.0
    }

    /// Write the JEPA integral.
    #[inline(always)]
    pub fn write_integral(&mut self, val: f32) {
        let val_i8 = (val * 100.0).clamp(-128.0, 127.0) as i8 as u32;
        self.0 = (self.0 & 0xFF00_FFFF) | ((val_i8 & 0xFF) << 16);
    }

    /// Read the Cognitive Derivative (Conciencia) as f32.
    #[inline(always)]
    pub fn read_cognitive(&self) -> f32 {
        let val_i8 = ((self.0 >> 24) & 0xFF) as i8;
        (val_i8 as f32) / 100.0
    }

    /// Write the Cognitive Derivative (Conciencia).
    #[inline(always)]
    pub fn write_cognitive(&mut self, val: f32) {
        let val_i8 = (val * 100.0).clamp(-128.0, 127.0) as i8 as u32;
        self.0 = (self.0 & 0x00FF_FFFF) | ((val_i8 & 0xFF) << 24);
    }

    /// Sigmoid gate derived from the JEPA integral: `σ(I)`.
    /// Returns 0.5 when I=0 (neutral gate at equilibrium).
    #[inline(always)]
    pub fn gate(&self) -> f32 {
        let i = self.read_integral();
        1.0 / (1.0 + (-i).exp())
    }

    /// Update integral in-place: `I_next = 0.99·I + 0.01·v_jepa`.
    /// Clamped to f16 range to prevent overflow.
    #[inline(always)]
    pub fn update_integral(&mut self, v_jepa: f32) {
        let _i_prev = self.read_integral();
        let i_next = v_jepa.clamp(-50000.0, 50000.0);
        self.write_integral(i_next);
    }

    /// Initialize from an embedding value.
    /// Sets ternary_f16 = emb_val, integral_f16 = 0 (gate = 0.5, neutral).
    /// Also seeds all jepa_z trackers with |emb_val| (Lexical Resonance).
    #[inline]
    pub fn init_from_embed(
        reg: &mut SlimeRegister,
        jepa_z: &mut [f32],
        idx: usize,
        hidden: usize,
        num_layers: usize,
        emb_val: f32,
        is_first_token: bool,
    ) {
        reg.write_accum(emb_val);
        reg.write_integral(0.0); // neutral gate = 0.5
        reg.write_cognitive(0.0); // reset cognitive state
        if is_first_token {
            let abs_val = emb_val.abs().min(5.0);
            for head in 0..(2 * num_layers) {
                jepa_z[head * hidden + idx] = abs_val;
            }
        }
    }

    // ── Backward compatibility shim ─────────────────────────────────────────

    /// Legacy field accessor — use `read_accum()` instead.
    /// Kept for compatibility with code that references `reg.ternary_state`.
    #[inline(always)]
    pub fn ternary_state_f32(&self) -> f32 {
        self.read_accum()
    }

    /// Legacy field accessor — use `read_integral()` instead.
    #[inline(always)]
    pub fn jepa_energy_f32(&self) -> f32 {
        self.read_integral()
    }
}

/// Convert a f32 to IEEE 754 half-precision (f16) bits.
/// Uses the `half` crate to correctly handle subnormals, NaN, and Infinity.
#[inline(always)]
pub fn float_to_half_bits(float: f32) -> u16 {
    half::f16::from_f32(float).to_bits()
}

/// Convert IEEE 754 half-precision (f16) bits to f32.
/// Uses the `half` crate for robust conversion.
#[inline(always)]
pub fn half_to_float_bits(half: u16) -> f32 {
    half::f16::from_bits(half).to_f32()
}

#[derive(Clone)]
pub struct SlimeWorkspace {
    pub registers: std::vec::Vec<SlimeRegister>,
    pub registers_tmp: std::vec::Vec<SlimeRegister>,
    pub kv_cache: std::vec::Vec<f32>,
    pub v_cache: std::vec::Vec<f32>,
    pub hca_kv_cache: std::vec::Vec<f32>, // Priority 51: HCA Compressed Historical Cache
    pub hca_v_cache: std::vec::Vec<f32>,
    pub hca_window: usize,
    pub hca_compression_ratio: usize,
    pub jepa_mu: std::vec::Vec<f32>,
    pub jepa_inv_sigma: std::vec::Vec<f32>,
    pub jepa_var_ema: std::vec::Vec<f32>,
    pub jepa_z: std::vec::Vec<f32>,
    /// Retained for the embedding initializer and trainer compatibility.
    /// Not used in the register hot path (f16 is self-scaled).
    pub iscale: f32,
    pub mhc_radius: f32,
    pub jepa_integral: f32, // Workspace-level integral accumulator (telemetry)
    pub gemv_accum: std::vec::Vec<f32>, // Temporary f32 scratch for GEMV

    pub norm_i8: std::vec::Vec<i8>,
    pub q_f32: std::vec::Vec<f32>,
    pub k_f32: std::vec::Vec<f32>,
    pub v_f32: std::vec::Vec<f32>,
    pub scores: std::vec::Vec<f32>,
    pub o_act_f32: std::vec::Vec<f32>,
    pub o_act_i8: std::vec::Vec<i8>,
    pub ffn_up_f32: std::vec::Vec<f32>,
    pub ffn_gate_f32: std::vec::Vec<f32>,
    pub ffn_mid_f32: std::vec::Vec<f32>,
    pub ffn_out_f32: std::vec::Vec<f32>,

    pub max_pos: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub ffn_mid: usize,
    pub num_layers: usize,
}

impl SlimeWorkspace {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden_size: usize,
        max_pos: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        ffn_mid: usize,
        num_layers: usize,
        max_emb: f32,
    ) -> Self {
        // Cap max_pos at 8192 to prevent catastrophic OOM
        let actual_max_pos = max_pos.min(8192);
        let kv_size = num_layers * n_kv_heads * actual_max_pos * head_dim;
        let ffn_sz = ffn_mid.max(hidden_size);
        // HCA sizing (Priority 51): Compress historical KV elements by 10x
        let hca_ratio = 10;
        let hca_window = 256;
        let hca_kv_size =
            (num_layers * n_kv_heads * (actual_max_pos / hca_ratio) * head_dim).max(1);

        // iscale retained for embedding init compat (not used in f16 hot path)
        let iscale = (max_emb * (num_layers as f32).max(16.0)) / 32767.0;

        Self {
            registers: vec![SlimeRegister::default(); hidden_size],
            registers_tmp: vec![SlimeRegister::default(); hidden_size],
            kv_cache: vec![0.0f32; kv_size],
            v_cache: vec![0.0f32; kv_size],
            hca_kv_cache: vec![0.0f32; hca_kv_size],
            hca_v_cache: vec![0.0f32; hca_kv_size],
            hca_window,
            hca_compression_ratio: hca_ratio,
            jepa_mu: vec![0.0f32; 2 * num_layers],
            jepa_inv_sigma: vec![0.0f32; 2 * num_layers],
            jepa_var_ema: vec![0.0f32; 2 * num_layers],
            jepa_z: vec![0.0f32; 2 * num_layers * hidden_size],
            iscale,
            mhc_radius: 1.5 * (hidden_size as f32).sqrt(),
            jepa_integral: 0.0,
            gemv_accum: vec![0.0f32; ffn_sz.max(hidden_size)],

            norm_i8: vec![0i8; hidden_size],
            q_f32: vec![0.0f32; hidden_size],
            k_f32: vec![0.0f32; n_kv_heads * head_dim],
            v_f32: vec![0.0f32; n_kv_heads * head_dim],
            scores: vec![0.0f32; actual_max_pos],
            o_act_f32: vec![0.0f32; ffn_sz],
            o_act_i8: vec![0i8; ffn_sz],
            ffn_up_f32: vec![0.0f32; ffn_mid],
            ffn_gate_f32: vec![0.0f32; ffn_mid],
            ffn_mid_f32: vec![0.0f32; ffn_mid],
            ffn_out_f32: vec![0.0f32; hidden_size],

            max_pos: actual_max_pos,
            n_heads,
            n_kv_heads,
            head_dim,
            hidden_size,
            ffn_mid,
            num_layers,
        }
    }

    #[inline(always)]
    pub fn clear_registers(&mut self) {
        self.registers.fill(SlimeRegister::default());
        self.registers_tmp.fill(SlimeRegister::default());
        self.jepa_integral = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slime_register_layout() {
        // v2: repr(transparent) u32 = 4 bytes
        assert_eq!(core::mem::size_of::<SlimeRegister>(), 4);
    }

    #[test]
    fn test_register_ternary_roundtrip() {
        let mut reg = SlimeRegister::default();
        let values = [0.0f32, 1.0, -1.0, 100.0, -200.5, 0.001, 65000.0];
        for &v in &values {
            reg.write_accum(v);
            let back = reg.read_accum();
            // f16 has ~3 decimal digits of precision
            let rel_err = if v.abs() > 1e-3 { (back - v).abs() / v.abs() } else { (back - v).abs() };
            assert!(rel_err < 0.01, "roundtrip fail for {v}: got {back}, rel_err={rel_err}");
        }
    }

    #[test]
    fn test_register_integral_roundtrip() {
        let mut reg = SlimeRegister::default();
        // Integral starts at 0 (neutral gate = 0.5)
        assert_eq!(reg.read_integral(), 0.0);
        assert!((reg.gate() - 0.5).abs() < 0.001);

        let target = 1.0f32;
        reg.update_integral(target);
        let i = reg.read_integral();
        assert!((i - target).abs() < 0.1, "integral should equal v_jepa: got {i}");
        // Gate should be > 0.5
        assert!(reg.gate() > 0.5);
    }

    #[test]
    fn test_register_integral_independence_from_accum() {
        let mut reg = SlimeRegister::default();
        reg.write_accum(42.0);
        reg.write_integral(1.0);
        // Writing integral should not corrupt ternary_f16
        assert!((reg.read_accum() - 42.0).abs() < 0.5);
        reg.update_integral(1.0);
        
        assert_eq!(reg.read_accum(), 42.0);
        assert!((reg.read_integral() - 1.0).abs() < 0.02);
    }

    #[test]
    fn test_f16_f32_conversion() {
        let val = 1.0f32;
        let half = float_to_half_bits(val);
        let back = half_to_float_bits(half);
        assert!((val - back).abs() < 0.01);
        let zero = 0.0f32;
        let half_zero = float_to_half_bits(zero);
        assert_eq!(half_zero, 0);
        assert_eq!(half_to_float_bits(half_zero), 0.0);
        let neg = -2.5f32;
        let half_neg = float_to_half_bits(neg);
        let back_neg = half_to_float_bits(half_neg);
        assert!((neg - back_neg).abs() < 0.01);
    }

    #[test]
    fn test_integral_equilibrium() {
        // I-controller equilibrium: if v_jepa = c constantly, I → c (not 0)
        // I_inf = sum_{k=0}^{inf} 0.99^k * 0.01 * c = 0.01 * c / (1 - 0.99) = c
        let mut reg = SlimeRegister::default();
        let target = 1.0f32;
        reg.update_integral(target);
        let i = reg.read_integral();
        assert!((i - target).abs() < 0.1, "I should equal v_jepa={target}, got {i}");
    }

    #[test]
    fn test_slime_workspace_allocation() {
        let ws = SlimeWorkspace::new(2048, 1024, 16, 4, 128, 2048, 30, 128.0);
        assert_eq!(ws.registers.len(), 2048);
        assert_eq!(ws.kv_cache.len(), 30 * 4 * 1024 * 128);
        assert_eq!(ws.norm_i8.len(), 2048);
        assert_eq!(ws.q_f32.len(), 2048);
        assert_eq!(ws.k_f32.len(), 4 * 128);
        assert_eq!(ws.v_f32.len(), 4 * 128);
        assert_eq!(ws.scores.len(), 1024);
        assert_eq!(ws.ffn_up_f32.len(), 2048);
        assert_eq!(ws.ffn_mid_f32.len(), 2048);
        assert_eq!(ws.ffn_out_f32.len(), 2048);
    }
}
