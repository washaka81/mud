/// SlimeRegister v2: 8-byte dual-state register for the MUD inference engine.
///
/// Layout:
///   matmul_accum: f32   — ternary matmul accumulation as IEEE f32
///   jepa_energy: f32    — JEPA integral I, native f32
///
/// Eliminates `iscale` and all f16/f32 conversions from the hot loop.
#[derive(Copy, Clone, Debug)]
#[repr(C, align(8))]
pub struct SlimeRegister {
    /// Ternary matmul accumulator — native f32, zero conversion overhead
    pub matmul_accum: f32,

    /// JEPA integral I — native f32, full precision
    pub jepa_energy: f32,
}

impl Default for SlimeRegister {
    #[inline(always)]
    fn default() -> Self {
        Self {
            matmul_accum: 0.0,
            jepa_energy: 0.0,
        }
    }
}

impl SlimeRegister {
    // ── Ternary accumulation (f32) ──────────────────────────

    /// Read ternary accumulation. Native f32.
    #[inline(always)]
    pub fn read_accum(&self) -> f32 {
        self.matmul_accum
    }

    /// Write ternary accumulation.
    #[inline(always)]
    pub fn write_accum(&mut self, val: f32) {
        self.matmul_accum = val;
    }

    // ── JEPA ────────────────────────────────────────────────

    /// Read the running JEPA integral.
    #[inline(always)]
    pub fn read_integral(&self) -> f32 {
        self.jepa_energy
    }

    /// Write the JEPA integral.
    #[inline(always)]
    pub fn write_integral(&mut self, val: f32) {
        self.jepa_energy = val;
    }

    /// Sigmoid gate derived from the JEPA integral: `σ(I)`.
    /// Returns 0.5 when I=0 (neutral gate at equilibrium).
    #[inline(always)]
    pub fn gate(&self) -> f32 {
        let i = self.jepa_energy;
        1.0 / (1.0 + (-i).exp())
    }

    /// Update integral in-place: `I_next = 0.99·I + 0.01·v_jepa`.
    #[inline(always)]
    pub fn update_integral(&mut self, v_jepa: f32) {
        self.jepa_energy = v_jepa.clamp(-50000.0, 50000.0);
    }

    /// Initialize from an embedding value.
    /// Sets matmul_accum = emb_val, jepa_energy = 0.0 (gate = 0.5, neutral).
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
        reg.matmul_accum = emb_val;
        reg.jepa_energy = 0.0; // neutral gate = 0.5
        if is_first_token {
            let abs_val = emb_val.abs().min(5.0);
            for head in 0..(2 * num_layers) {
                jepa_z[head * hidden + idx] = abs_val;
            }
        }
    }

    // ── Backward compatibility shim ─────────────────────────────────────────

    /// Legacy field accessor — use `read_accum()` instead.
    #[inline(always)]
    pub fn ternary_state_f32(&self) -> f32 {
        self.matmul_accum
    }

    /// Legacy field accessor — use `read_integral()` instead.
    #[inline(always)]
    pub fn jepa_energy_f32(&self) -> f32 {
        self.jepa_energy
    }
}

#[derive(Clone)]
pub struct SlimeWorkspace {
    pub registers: std::vec::Vec<SlimeRegister>,
    pub registers_tmp: std::vec::Vec<SlimeRegister>,
    pub mai_bytes: std::vec::Vec<u8>,
    pub kv_cache: std::vec::Vec<f32>,
    pub v_cache: std::vec::Vec<f32>,
    pub hca_kv_cache: std::vec::Vec<f32>, // Priority 51 / L-13: HCA compressed history
    pub hca_v_cache: std::vec::Vec<f32>,
    /// Stream I: when F16, dense/HCA live in packs (f32 buffers empty).
    pub kv_dtype: crate::mud::kv_dtype::KvDtype,
    pub kv_cache_f16: std::vec::Vec<u8>,
    pub v_cache_f16: std::vec::Vec<u8>,
    pub hca_kv_f16: std::vec::Vec<u8>,
    pub hca_v_f16: std::vec::Vec<u8>,
    /// Scratch for one KV head row load (f16 → f32).
    pub kv_row_scratch: std::vec::Vec<f32>,
    pub hca_window: usize,
    pub hca_compression_ratio: usize,
    /// L-13: physical dense ring length (not necessarily == max_pos).
    pub dense_kv_cap: usize,
    /// L-13: number of HCA compressed slots.
    pub hca_slots: usize,
    pub jepa_mu: std::vec::Vec<f32>,
    pub jepa_inv_sigma: std::vec::Vec<f32>,
    pub jepa_var_ema: std::vec::Vec<f32>,
    pub jepa_z: std::vec::Vec<f32>,
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
        _max_emb: f32,
    ) -> Self {
        // L-13: 32k-ready policy — dense ring + HCA, not O(max_pos) full KV
        let policy = crate::mud::kv_context::KvContextPolicy::resolve(max_pos);
        let kv_size = num_layers * n_kv_heads * policy.dense_cap * head_dim;
        let ffn_sz = ffn_mid.max(hidden_size);
        let hca_kv_size = (num_layers * n_kv_heads * policy.hca_slots * head_dim).max(1);
        let kv_dt = crate::mud::kv_dtype::KvDtype::resolve();
        let (kv_f32, kv_f16, v_f32, v_f16, hca_k_f32, hca_k_f16, hca_v_f32, hca_v_f16) =
            if kv_dt.is_f16() {
                (
                    Vec::new(),
                    vec![0u8; kv_size * 2],
                    Vec::new(),
                    vec![0u8; kv_size * 2],
                    Vec::new(),
                    vec![0u8; hca_kv_size * 2],
                    Vec::new(),
                    vec![0u8; hca_kv_size * 2],
                )
            } else {
                (
                    vec![0.0f32; kv_size],
                    Vec::new(),
                    vec![0.0f32; kv_size],
                    Vec::new(),
                    vec![0.0f32; hca_kv_size],
                    Vec::new(),
                    vec![0.0f32; hca_kv_size],
                    Vec::new(),
                )
            };

        Self {
            registers: vec![SlimeRegister::default(); hidden_size],
            registers_tmp: vec![SlimeRegister::default(); hidden_size],
            mai_bytes: vec![0u8; hidden_size],
            kv_cache: kv_f32,
            v_cache: v_f32,
            hca_kv_cache: hca_k_f32,
            hca_v_cache: hca_v_f32,
            kv_dtype: kv_dt,
            kv_cache_f16: kv_f16,
            v_cache_f16: v_f16,
            hca_kv_f16: hca_k_f16,
            hca_v_f16,
            kv_row_scratch: vec![0.0f32; head_dim.max(1)],
            hca_window: policy.hca_window,
            hca_compression_ratio: policy.hca_ratio,
            dense_kv_cap: policy.dense_cap,
            hca_slots: policy.hca_slots,
            jepa_mu: vec![0.0f32; 2 * num_layers],
            jepa_inv_sigma: vec![0.0f32; 2 * num_layers],
            jepa_var_ema: vec![0.0f32; 2 * num_layers],
            jepa_z: vec![0.0f32; 2 * num_layers * hidden_size],
            mhc_radius: 1.5 * (hidden_size as f32).sqrt(),
            jepa_integral: 0.0,
            gemv_accum: vec![0.0f32; ffn_sz.max(hidden_size)],

            norm_i8: vec![0i8; hidden_size],
            q_f32: vec![0.0f32; hidden_size],
            k_f32: vec![0.0f32; n_kv_heads * head_dim],
            v_f32: vec![0.0f32; n_kv_heads * head_dim],
            scores: vec![0.0f32; policy.scores_len()],
            o_act_f32: vec![0.0f32; ffn_sz],
            o_act_i8: vec![0i8; ffn_sz],
            ffn_up_f32: vec![0.0f32; ffn_mid],
            ffn_gate_f32: vec![0.0f32; ffn_mid],
            ffn_mid_f32: vec![0.0f32; ffn_mid],
            ffn_out_f32: vec![0.0f32; hidden_size],

            max_pos: policy.logical_max_pos,
            n_heads,
            n_kv_heads,
            head_dim,
            hidden_size,
            ffn_mid,
            num_layers,
        }
    }

    /// L-13: absolute position → dense ring index.
    #[inline(always)]
    pub fn dense_slot(&self, pos: usize) -> usize {
        pos % self.dense_kv_cap.max(1)
    }

    #[inline(always)]
    pub fn clear_registers(&mut self) {
        self.registers.fill(SlimeRegister::default());
        self.registers_tmp.fill(SlimeRegister::default());
        self.jepa_integral = 0.0;
    }

    /// Clear all KV / HCA storage (f32 or f16 packs).
    pub fn clear_kv_all(&mut self) {
        self.kv_cache.fill(0.0);
        self.v_cache.fill(0.0);
        self.hca_kv_cache.fill(0.0);
        self.hca_v_cache.fill(0.0);
        self.kv_cache_f16.fill(0);
        self.v_cache_f16.fill(0);
        self.hca_kv_f16.fill(0);
        self.hca_v_f16.fill(0);
    }

    /// Store one head row into dense K cache at element `base`.
    #[inline]
    pub fn store_dense_k(&mut self, base: usize, row: &[f32]) {
        use crate::mud::kv_dtype::{store_row_f16, KvDtype};
        match self.kv_dtype {
            KvDtype::F32 => {
                let n = row.len();
                self.kv_cache[base..base + n].copy_from_slice(row);
            }
            KvDtype::F16 => store_row_f16(&mut self.kv_cache_f16, base, row),
        }
    }

    #[inline]
    pub fn store_dense_v(&mut self, base: usize, row: &[f32]) {
        use crate::mud::kv_dtype::{store_row_f16, KvDtype};
        match self.kv_dtype {
            KvDtype::F32 => {
                let n = row.len();
                self.v_cache[base..base + n].copy_from_slice(row);
            }
            KvDtype::F16 => store_row_f16(&mut self.v_cache_f16, base, row),
        }
    }

    #[inline]
    pub fn store_hca_k(&mut self, base: usize, row: &[f32]) {
        use crate::mud::kv_dtype::{store_row_f16, KvDtype};
        match self.kv_dtype {
            KvDtype::F32 => {
                let n = row.len();
                self.hca_kv_cache[base..base + n].copy_from_slice(row);
            }
            KvDtype::F16 => store_row_f16(&mut self.hca_kv_f16, base, row),
        }
    }

    #[inline]
    pub fn store_hca_v(&mut self, base: usize, row: &[f32]) {
        use crate::mud::kv_dtype::{store_row_f16, KvDtype};
        match self.kv_dtype {
            KvDtype::F32 => {
                let n = row.len();
                self.hca_v_cache[base..base + n].copy_from_slice(row);
            }
            KvDtype::F16 => store_row_f16(&mut self.hca_v_f16, base, row),
        }
    }

    /// Load dense K row into `self.kv_row_scratch` (len = head_dim); returns slice.
    pub fn load_dense_k(&mut self, base: usize) -> &[f32] {
        use crate::mud::kv_dtype::{load_row_f16, KvDtype};
        let n = self.head_dim;
        match self.kv_dtype {
            KvDtype::F32 => {
                self.kv_row_scratch[..n].copy_from_slice(&self.kv_cache[base..base + n]);
            }
            KvDtype::F16 => load_row_f16(&self.kv_cache_f16, base, &mut self.kv_row_scratch[..n]),
        }
        &self.kv_row_scratch[..n]
    }

    pub fn load_dense_v(&mut self, base: usize) -> &[f32] {
        use crate::mud::kv_dtype::{load_row_f16, KvDtype};
        let n = self.head_dim;
        match self.kv_dtype {
            KvDtype::F32 => {
                self.kv_row_scratch[..n].copy_from_slice(&self.v_cache[base..base + n]);
            }
            KvDtype::F16 => load_row_f16(&self.v_cache_f16, base, &mut self.kv_row_scratch[..n]),
        }
        &self.kv_row_scratch[..n]
    }

    pub fn load_hca_k(&mut self, base: usize) -> &[f32] {
        use crate::mud::kv_dtype::{load_row_f16, KvDtype};
        let n = self.head_dim;
        match self.kv_dtype {
            KvDtype::F32 => {
                self.kv_row_scratch[..n].copy_from_slice(&self.hca_kv_cache[base..base + n]);
            }
            KvDtype::F16 => load_row_f16(&self.hca_kv_f16, base, &mut self.kv_row_scratch[..n]),
        }
        &self.kv_row_scratch[..n]
    }

    pub fn load_hca_v(&mut self, base: usize) -> &[f32] {
        use crate::mud::kv_dtype::{load_row_f16, KvDtype};
        let n = self.head_dim;
        match self.kv_dtype {
            KvDtype::F32 => {
                self.kv_row_scratch[..n].copy_from_slice(&self.hca_v_cache[base..base + n]);
            }
            KvDtype::F16 => load_row_f16(&self.hca_v_f16, base, &mut self.kv_row_scratch[..n]),
        }
        &self.kv_row_scratch[..n]
    }

    /// Element count of dense KV (logical), independent of dtype packing.
    pub fn dense_kv_elems(&self) -> usize {
        if self.kv_dtype.is_f16() {
            self.kv_cache_f16.len() / 2
        } else {
            self.kv_cache.len()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slime_register_layout() {
        // v2: repr(C, align(8)) = 8 bytes
        assert_eq!(core::mem::size_of::<SlimeRegister>(), 8);
    }

    #[test]
    fn test_register_ternary_roundtrip() {
        let mut reg = SlimeRegister::default();
        let values = [0.0f32, 1.0, -1.0, 100.0, -200.5, 0.001, 65000.0];
        for &v in &values {
            reg.write_accum(v);
            let back = reg.read_accum();
            // f32 is exact
            assert_eq!(back, v);
        }
    }

    #[test]
    fn test_register_integral_roundtrip() {
        let mut reg = SlimeRegister::default();
        // Integral starts at 0 (neutral gate = 0.5)
        assert_eq!(reg.read_integral(), 0.0);
        assert!((reg.gate() - 0.5).abs() < 0.001);

        let target = 1.0;
        reg.update_integral(target);
        let i = reg.read_integral();
        assert_eq!(i, target);
        // Gate should be > 0.5
        assert!(reg.gate() > 0.5);
    }

    #[test]
    fn test_register_integral_independence_from_accum() {
        let mut reg = SlimeRegister::default();
        reg.write_accum(42.0);
        reg.write_integral(1.0);
        // Writing integral should not corrupt ternary f32
        assert_eq!(reg.read_accum(), 42.0);
        reg.update_integral(1.0);

        assert_eq!(reg.read_accum(), 42.0);
        assert_eq!(reg.read_integral(), 1.0);
    }

    #[test]
    fn test_integral_equilibrium() {
        // I-controller equilibrium: if v_jepa = c constantly, I → c (not 0)
        let mut reg = SlimeRegister::default();
        let target = 1.0f32;
        reg.update_integral(target);
        let i = reg.read_integral();
        assert_eq!(i, target);
    }

    #[test]
    fn test_slime_workspace_allocation() {
        // Isolate from env overrides
        unsafe {
            std::env::remove_var("MUD_MAX_POS");
            std::env::remove_var("MUD_HCA_WINDOW");
            std::env::remove_var("MUD_HCA_RATIO");
        }
        let ws = SlimeWorkspace::new(2048, 1024, 16, 4, 128, 2048, 30, 128.0);
        assert_eq!(ws.registers.len(), 2048);
        assert_eq!(ws.mai_bytes.len(), 2048);
        // L-13: dense ring, not full 1024 positions
        let policy = crate::mud::kv_context::KvContextPolicy::from_parts(1024, 256, 10);
        assert_eq!(ws.max_pos, policy.logical_max_pos);
        assert_eq!(ws.dense_kv_cap, policy.dense_cap);
        assert_eq!(ws.kv_cache.len(), 30 * 4 * policy.dense_cap * 128);
        assert_eq!(ws.norm_i8.len(), 2048);
        assert_eq!(ws.q_f32.len(), 2048);
        assert_eq!(ws.k_f32.len(), 4 * 128);
        assert_eq!(ws.v_f32.len(), 4 * 128);
        assert_eq!(ws.scores.len(), policy.scores_len());
        assert_eq!(ws.ffn_up_f32.len(), 2048);
        assert_eq!(ws.ffn_mid_f32.len(), 2048);
        assert_eq!(ws.ffn_out_f32.len(), 2048);
    }

    #[test]
    fn test_l13_32k_workspace_fits() {
        unsafe {
            std::env::remove_var("MUD_MAX_POS");
            std::env::remove_var("MUD_HCA_WINDOW");
            std::env::remove_var("MUD_HCA_RATIO");
        }
        // Smollm-ish: 30L, 3 kv heads, head 64, request 32k
        let ws = SlimeWorkspace::new(576, 32_768, 9, 3, 64, 1536, 30, 1.0);
        assert_eq!(ws.max_pos, 32_768);
        assert!(ws.dense_kv_cap < 512, "dense ring must stay small");
        let dense_mb = if ws.kv_dtype.is_f16() {
            (ws.kv_cache_f16.len() + ws.v_cache_f16.len()) / (1024 * 1024)
        } else {
            (ws.kv_cache.len() + ws.v_cache.len()) * 4 / (1024 * 1024)
        };
        let hca_mb = if ws.kv_dtype.is_f16() {
            (ws.hca_kv_f16.len() + ws.hca_v_f16.len()) / (1024 * 1024)
        } else {
            (ws.hca_kv_cache.len() + ws.hca_v_cache.len()) * 4 / (1024 * 1024)
        };
        assert!(
            dense_mb + hca_mb < 50,
            "32k KV footprint too large: dense={dense_mb}MB hca={hca_mb}MB"
        );
        assert!(ws.hca_slots <= crate::mud::kv_context::MAX_HCA_SLOTS);
    }
}
