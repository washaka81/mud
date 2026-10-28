/// PackedRegister v3: 4-byte dual-component register.
///
/// Bit layout (u32):
///   bits  0-15: re — ternary GEMV accumulator as IEEE binary16 (real)
///   bits 16-31: im — JEPA integral derivative as IEEE binary16 (imaginary)
///
/// In CMUD thinker mode the pair is treated as Complex<f16>:
///   real = bits 0-15, imag = bits 16-31
///   → feeds directly into phase-coherent attention and Hermitian ball clamp.
///
/// Hardware: `vcvtps2ph` (F16C) converts f32→f16 in 1 cycle on i7-1260P.
/// `vcvtph2ps` converts f16→f32 in 1 cycle. Both are inline via `half::f16`.
use half::f16;

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct SlimeRegister {
    raw: u32,
}

impl Default for SlimeRegister {
    #[inline(always)]
    fn default() -> Self {
        Self { raw: 0 }
    }
}

impl SlimeRegister {
    // ── Pack / unpack helpers ───────────────────────────────

    /// Pack real (ternary accum) + imag (JEPA/CMUD) into 32-bit register.
    #[inline(always)]
    pub fn pack(re: f32, im: f32) -> Self {
        Self {
            raw: f16::from_f32(re).to_bits() as u32
                | (f16::from_f32(im).to_bits() as u32) << 16,
        }
    }

    /// Extract real component (ternary accumulator) as f32.
    #[inline(always)]
    pub fn real(&self) -> f32 {
        f16::from_bits(self.raw as u16).to_f32()
    }

    /// Extract imaginary component (JEPA integral / CMUD) as f32.
    #[inline(always)]
    pub fn imag(&self) -> f32 {
        f16::from_bits((self.raw >> 16) as u16).to_f32()
    }

    /// Return the raw u32 (for CMUD complex access).
    #[inline(always)]
    pub fn raw_bits(&self) -> u32 {
        self.raw
    }

    /// Build from raw bits (for CMUD thinker writeback).
    #[inline(always)]
    pub fn from_raw(raw: u32) -> Self {
        Self { raw }
    }

    // ── Ternary accumulation ────────────────────────────────

    /// Read ternary accumulation. Converts f16→f32 (1 cycle via vcvtph2ps).
    #[inline(always)]
    pub fn read_accum(&self) -> f32 {
        self.real()
    }

    /// Write ternary accumulation. Converts f32→f16 (1 cycle via vcvtps2ph).
    #[inline(always)]
    pub fn write_accum(&mut self, val: f32) {
        let im = self.imag();
        self.raw = f16::from_f32(val).to_bits() as u32
            | (f16::from_f32(im).to_bits() as u32) << 16;
    }

    // ── JEPA ────────────────────────────────────────────────

    /// Read the running JEPA integral (imaginary component).
    #[inline(always)]
    pub fn read_integral(&self) -> f32 {
        self.imag()
    }

    /// Write the JEPA integral, preserving the real component.
    #[inline(always)]
    pub fn write_integral(&mut self, val: f32) {
        let re = self.real();
        self.raw = f16::from_f32(re).to_bits() as u32
            | (f16::from_f32(val).to_bits() as u32) << 16;
    }

    /// Sigmoid gate derived from the imaginary component: `σ(im)`.
    /// Returns 0.5 when im=0 (neutral gate at equilibrium).
    #[inline(always)]
    pub fn gate(&self) -> f32 {
        1.0 / (1.0 + (-self.imag()).exp())
    }

    /// Update integral in-place: stores `v_jepa` as imaginary component.
    #[inline(always)]
    pub fn update_integral(&mut self, v_jepa: f32) {
        let re = self.real();
        let clamped = v_jepa.clamp(-50000.0, 50000.0);
        self.raw = f16::from_f32(re).to_bits() as u32
            | (f16::from_f32(clamped).to_bits() as u32) << 16;
    }

    /// Initialize from an embedding value.
    /// Sets real = emb_val, imag = 0.0 (gate = 0.5, neutral).
    /// Seeds jepa_z trackers with |emb_val| (Lexical Resonance).
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
        reg.raw = f16::from_f32(emb_val).to_bits() as u32;
        if is_first_token {
            let abs_val = emb_val.abs().min(5.0);
            for head in 0..(2 * num_layers) {
                jepa_z[head * hidden + idx] = abs_val;
            }
        }
    }

    // ── CMUD intersection (complex thinker) ─────────────────

    /// Return real+imag as a (re, im) f32 pair for CMUD phase-coherent attention.
    /// The thinker treats the packed register as Complex<f16> where:
    ///   re = bits 0-15 (ternary GEMV accumulator)
    ///   im = bits 16-31 (JEPA integral derivative)
    #[inline(always)]
    pub fn as_complex_f32(&self) -> (f32, f32) {
        (self.real(), self.imag())
    }

    /// Write both real and imag components from CMUD thinker output
    /// (wave-function collapse result).
    #[inline(always)]
    pub fn write_complex(&mut self, re: f32, im: f32) {
        self.raw = f16::from_f32(re).to_bits() as u32
            | (f16::from_f32(im).to_bits() as u32) << 16;
    }

    /// Convert a register slice to a flat f32 array for the LM head.
    /// Uses AVX2 batch kernel (8 dims per call) for ~4× speedup.
    ///
    /// **Deprecated (P-08, 2026-07-27):** This convenience wrapper allocates
    /// two `Vec<f32>` (`out` + `discard`). The hot path MUST call
    /// [`registers_to_f32_into`](Self::registers_to_f32_into) with pre-allocated
    /// buffers (P-01 zero-alloc). Kept `#[cfg(test)]` so tests can use the
    /// ergonomic allocating form; dead-code-eliminated from release binaries.
    #[cfg(test)]
    #[deprecated(note = "Allocates 2× Vec<f32>; use registers_to_f32_into in hot path (P-01)")]
    pub fn registers_to_f32(regs: &[SlimeRegister]) -> Vec<f32> {
        let n = regs.len();
        let mut out = vec![0.0f32; n];
        let mut discard = vec![0.0f32; n];
        Self::registers_to_f32_into(regs, &mut out, &mut discard);
        out
    }

    /// Convert register slice into a pre-allocated f32 buffer (zero allocation).
    pub fn registers_to_f32_into(regs: &[SlimeRegister], out_real: &mut [f32], scratch_imag: &mut [f32]) {
        let n = regs.len().min(out_real.len()).min(scratch_imag.len());
        // SAFETY: `n` is clamped to the minimum of all three lengths, and
        // `SlimeRegister` is `#[repr(C)] { raw: u32 }`, so `regs.as_ptr() as
        // *const u32` is valid and aligned for n u32 reads/writes; the two
        // output buffers are distinct `&mut` slices (Rust aliasing) and each
        // has at least `n` initialized elements.
        unsafe {
            let rem = crate::asm::registers_to_f32_batch_into(
                regs.as_ptr() as *const u32,
                out_real.as_mut_ptr(),
                scratch_imag.as_mut_ptr(),
                n,
            );
            for i in n - rem..n {
                out_real[i] = regs[i].real();
            }
        }
    }

}

use crate::mud::workspace::AlignedArray;

#[derive(Clone)]
pub struct SlimeWorkspace {
    pub registers: AlignedArray<SlimeRegister>,
    pub mai_bytes: AlignedArray<u8>,
    pub kv_cache: AlignedArray<f32>,
    pub v_cache: AlignedArray<f32>,
    pub hca_kv_cache: AlignedArray<f32>, // Priority 51 / L-13: HCA compressed history
    pub hca_v_cache: AlignedArray<f32>,
    /// Stream I: when F16, dense/HCA live in packs (f32 buffers empty).
    pub kv_dtype: crate::mud::kv_dtype::KvDtype,
    pub kv_cache_f16: AlignedArray<u8>,
    pub v_cache_f16: AlignedArray<u8>,
    pub hca_kv_f16: AlignedArray<u8>,
    pub hca_v_f16: AlignedArray<u8>,
    /// Scratch for one KV head row load (f16 → f32).
    pub kv_row_scratch: AlignedArray<f32>,
    pub hca_window: usize,
    pub hca_compression_ratio: usize,
    /// L-13: physical dense ring length (not necessarily == max_pos).
    pub dense_kv_cap: usize,
    /// L-13: number of HCA compressed slots.
    pub hca_slots: usize,
    pub jepa_mu: AlignedArray<f32>,
    pub jepa_inv_sigma: AlignedArray<f32>,
    pub jepa_var_ema: AlignedArray<f32>,
    pub jepa_z: AlignedArray<f32>,
    pub mhc_radius: f32,
    pub jepa_integral: f32, // Workspace-level integral accumulator (telemetry)
    pub gemv_accum: AlignedArray<f32>, // Temporary f32 scratch for GEMV

    pub norm_i8: AlignedArray<i8>,
    pub q_f32: AlignedArray<f32>,
    pub k_f32: AlignedArray<f32>,
    pub v_f32: AlignedArray<f32>,
    pub scores: AlignedArray<f32>,
    pub o_act_f32: AlignedArray<f32>,
    pub o_act_i8: AlignedArray<i8>,
    pub ffn_up_f32: AlignedArray<f32>,
    pub ffn_gate_f32: AlignedArray<f32>,
    pub ffn_mid_f32: AlignedArray<f32>,
    pub ffn_out_f32: AlignedArray<f32>,

    /// SX-01/SX-02: SlimeX transport cell buffer (u32, 8:8:8:8 layout).
    /// Pre-allocated once (P-01); populated by `snapshot_to_slimex()` when `MUD_SLIMEX=1`.
    /// Length = hidden_size. Zero-size when SlimeX is disabled.
    pub slimex_cells: AlignedArray<u32>,

    /// SPRINT-2 / SX: Pre-allocated scratch buffer to avoid `Vec::to_vec` in the
    /// `snapshot_to_slimex` call site (P-01). Sized to `hidden_size`. The forward
    /// pass copies the per-layer jepa_z slice into this scratch before calling
    /// `snapshot_to_slimex(&self.slimex_jepa_scratch)`.
    pub slimex_jepa_scratch: AlignedArray<f32>,

    /// SPRINT-5 (5.1): Pre-allocated scratch for the CSA lightning top-k index
    /// (used in `evaluate_slime_block_moe`). Sized to `hca_slots` once at
    /// workspace creation (P-01). Filled with the per-HCA-block coarse scores
    /// before `csa_indexer::select_top_k_indices` is called. Zero-cost when
    /// CSA is disabled (path is not entered).
    pub csa_index_scratch: AlignedArray<f32>,

    /// SPRINT-5 (5.1): Pre-allocated scratch for the CSA top-k block indices
    /// (max length `top_k + tail + 8` per `CsaPolicy`). Holds the union of
    /// the lightning top-k and the forced tail blocks for the current head.
    /// Reused across heads and layers (P-01).
    pub csa_blocks_scratch: AlignedArray<usize>,

    pub max_pos: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub ffn_mid: usize,
    pub num_layers: usize,

    /// Whether `MUD_SLIMEX=1` was set at workspace creation time.
    pub slimex_enabled: bool,

    /// Stream I: Lossless KV cache compression (Delta + LZ4).
    /// Zero-cost when `MUD_KV_COMPRESS` is not set (P-01).
    pub kv_compressed: crate::mud::kv_compressed::CompressedKvCache,

    /// L-14: CellV5 buffer for C-MUD complex state (u32 8:8:8:8 packing).
    /// 4× compression for ComplexF32 (8 bytes → 4 bytes). Zero-size when
    /// `MUD_CMUD_CELLV5` is not set (P-01). P-02 safe — opt-in only.
    pub cmud_cells: AlignedArray<u32>,
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
                    AlignedArray::new(0, 64),
                    AlignedArray::new(kv_size * 2, 64),
                    AlignedArray::new(0, 64),
                    AlignedArray::new(kv_size * 2, 64),
                    AlignedArray::new(0, 64),
                    AlignedArray::new(hca_kv_size * 2, 64),
                    AlignedArray::new(0, 64),
                    AlignedArray::new(hca_kv_size * 2, 64),
                )
            } else {
                (
                    AlignedArray::new(kv_size, 64),
                    AlignedArray::new(0, 64),
                    AlignedArray::new(kv_size, 64),
                    AlignedArray::new(0, 64),
                    AlignedArray::new(hca_kv_size, 64),
                    AlignedArray::new(0, 64),
                    AlignedArray::new(hca_kv_size, 64),
                    AlignedArray::new(0, 64),
                )
            };

        // SX-01: SlimeX cell buffer — pre-alloc hidden_size u32 slots when MUD_SLIMEX=1 (P-01).
        let slimex_enabled = std::env::var("MUD_SLIMEX").as_deref() == Ok("1");
        let slimex_cells = if slimex_enabled {
            AlignedArray::new(hidden_size, 64)
        } else {
            AlignedArray::new(0, 64)
        };
        // SPRINT-2: scratch buffer for jepa_z copy inside `snapshot_to_slimex`. Always
        // allocated (zero-cost when slimex is disabled — caller skips the snapshot).
        let slimex_jepa_scratch = AlignedArray::new(hidden_size, 64);

        // SPRINT-5 (5.1): CSA scratch — sized once to the max number of HCA
        // compressed blocks and to the worst-case top-k+tail selection.
        // Both buffers live in the workspace for the whole session; the
        // forward pass `clear`s them per head as needed.
        let csa_index_scratch = AlignedArray::new(policy.hca_slots.max(1), 64);
        // 64 covers the default top_k (CSA top-k default = 64) + tail (4) + 8 slack.
        // 256 is a safe upper bound for the small-vocabulary / short-context case.
        let csa_blocks_scratch = AlignedArray::new(256.max(policy.hca_slots.min(64) + 16), 64);

        Self {
            registers: AlignedArray::from_elem(SlimeRegister::default(), hidden_size),
            mai_bytes: AlignedArray::new(hidden_size, 64),
            kv_cache: kv_f32,
            v_cache: v_f32,
            hca_kv_cache: hca_k_f32,
            hca_v_cache: hca_v_f32,
            kv_dtype: kv_dt,
            kv_cache_f16: kv_f16,
            v_cache_f16: v_f16,
            hca_kv_f16: hca_k_f16,
            hca_v_f16,
            kv_row_scratch: AlignedArray::new(head_dim.max(1), 64),
            hca_window: policy.hca_window,
            hca_compression_ratio: policy.hca_ratio,
            dense_kv_cap: policy.dense_cap,
            hca_slots: policy.hca_slots,
            jepa_mu: AlignedArray::new(2 * num_layers, 64),
            jepa_inv_sigma: AlignedArray::new(2 * num_layers, 64),
            jepa_var_ema: AlignedArray::new(2 * num_layers, 64),
            jepa_z: AlignedArray::new(2 * num_layers * hidden_size, 64),
            mhc_radius: 1.0 * (hidden_size as f32).sqrt(),
            jepa_integral: 0.0,
            gemv_accum: AlignedArray::new(ffn_sz.max(hidden_size), 64),

            norm_i8: AlignedArray::new(hidden_size, 64),
            q_f32: AlignedArray::new(hidden_size, 64),
            k_f32: AlignedArray::new(n_kv_heads * head_dim, 64),
            v_f32: AlignedArray::new(n_kv_heads * head_dim, 64),
            scores: AlignedArray::new(policy.scores_len(), 64),
            o_act_f32: AlignedArray::new(ffn_sz, 64),
            o_act_i8: AlignedArray::new(ffn_sz, 64),
            ffn_up_f32: AlignedArray::new(ffn_mid, 64),
            ffn_gate_f32: AlignedArray::new(ffn_mid, 64),
            ffn_mid_f32: AlignedArray::new(ffn_mid, 64),
            ffn_out_f32: AlignedArray::new(hidden_size, 64),

            slimex_cells,
            slimex_jepa_scratch,
            csa_index_scratch,
            csa_blocks_scratch,
            slimex_enabled,

            kv_compressed: crate::mud::kv_compressed::CompressedKvCache::new(
                crate::mud::kv_compressed::KvCompressConfig::resolve(),
                head_dim.max(1),
                n_kv_heads.max(1),
            ),

            // L-14: CellV5 buffer for C-MUD state (4× compression, opt-in).
            cmud_cells: {
                let cmud_cellv5 = std::env::var("MUD_CMUD_CELLV5").as_deref() == Ok("1");
                if cmud_cellv5 {
                    AlignedArray::new(hidden_size, 64)
                } else {
                    AlignedArray::new(0, 64)
                }
            },

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
        self.jepa_integral = 0.0;
    }

    /// SX-01: Snapshot all registers into `slimex_cells` as `SlimeXCellV5` packed u32.
    /// Uses the AVX2 unpack kernel in reverse: each register's real (f32→fp8 approx) +
    /// JEPA imag component mapped to jepa_velocity i8 + address_tag from register index.
    /// **No-op when `slimex_enabled` is false** — zero cost on the default path.
    ///
    /// Call after every block's forward pass to maintain the SlimeX cell buffer.
    #[inline]
    pub fn snapshot_to_slimex(&mut self, jepa_z_slice: &[f32]) {
        if !self.slimex_enabled {
            return;
        }
        let n = self.hidden_size.min(self.slimex_cells.len());
        let cells_ptr = self.slimex_cells.ptr;
        // SAFETY: cells_ptr points to `n` valid u32 slots pre-allocated in AlignedArray.
        // registers slice is of length hidden_size >= n. jepa_z_slice may be shorter.
        unsafe {
            for i in 0..n {
                let reg = self.registers.get_unchecked(i);
                // Map real f32 → pseudo-FP8: clamp to [0,255] representing magnitude.
                let real_abs = reg.read_accum().abs().min(255.0);
                let accum_fp8 = real_abs as u8;
                // address_tag: lower 8 bits of index → 256-slot memory address.
                let address_tag = (i & 0xFF) as u8;
                // jepa_velocity: JEPA integral mapped to i8 [-128, 127].
                let jepa_val = jepa_z_slice.get(i).copied().unwrap_or(0.0);
                let jepa_velocity = jepa_val.clamp(-128.0, 127.0) as i8;
                // ste_parity: sign bit of real component (encodes ternary direction).
                let ste_parity = if reg.read_accum() >= 0.0 { 0u8 } else { 1u8 };
                // Pack as SlimeXCellV5: [accum_fp8 | address_tag<<8 | jepa_vel<<16 | ste_parity<<24]
                let cell = (accum_fp8 as u32)
                    | ((address_tag as u32) << 8)
                    | (((jepa_velocity as u8) as u32) << 16)
                    | ((ste_parity as u32) << 24);
                *cells_ptr.add(i) = cell;
            }
        }
    }

    /// SX-01: Read back a single SlimeXCellV5 from the cell buffer (zero-alloc, by index).
    /// Returns 0 when slimex is disabled or index out of range.
    #[inline(always)]
    pub fn read_slimex_cell(&self, i: usize) -> u32 {
        if !self.slimex_enabled || i >= self.slimex_cells.len() {
            return 0;
        }
        // SAFETY: i is bounds-checked above; slimex_cells is valid AlignedArray.
        unsafe { *self.slimex_cells.ptr.add(i) }
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
        use crate::mud::kv_dtype::{store_row_bf16, store_row_f16, KvDtype};
        match self.kv_dtype {
            KvDtype::F32 => {
                let n = row.len();
                self.kv_cache[base..base + n].copy_from_slice(row);
            }
            KvDtype::F16 => store_row_f16(&mut self.kv_cache_f16, base, row),
            KvDtype::BF16 => store_row_bf16(&mut self.kv_cache_f16, base, row),
        }
    }

    #[inline]
    pub fn store_dense_v(&mut self, base: usize, row: &[f32]) {
        use crate::mud::kv_dtype::{store_row_bf16, store_row_f16, KvDtype};
        match self.kv_dtype {
            KvDtype::F32 => {
                let n = row.len();
                self.v_cache[base..base + n].copy_from_slice(row);
            }
            KvDtype::F16 => store_row_f16(&mut self.v_cache_f16, base, row),
            KvDtype::BF16 => store_row_bf16(&mut self.v_cache_f16, base, row),
        }
    }

    #[inline]
    pub fn store_hca_k(&mut self, base: usize, row: &[f32]) {
        use crate::mud::kv_dtype::{store_row_bf16, store_row_f16, KvDtype};
        match self.kv_dtype {
            KvDtype::F32 => {
                let n = row.len();
                self.hca_kv_cache[base..base + n].copy_from_slice(row);
            }
            KvDtype::F16 => store_row_f16(&mut self.hca_kv_f16, base, row),
            KvDtype::BF16 => store_row_bf16(&mut self.hca_kv_f16, base, row),
        }
    }

    #[inline]
    pub fn store_hca_v(&mut self, base: usize, row: &[f32]) {
        use crate::mud::kv_dtype::{store_row_bf16, store_row_f16, KvDtype};
        match self.kv_dtype {
            KvDtype::F32 => {
                let n = row.len();
                self.hca_v_cache[base..base + n].copy_from_slice(row);
            }
            KvDtype::F16 => store_row_f16(&mut self.hca_v_f16, base, row),
            KvDtype::BF16 => store_row_bf16(&mut self.hca_v_f16, base, row),
        }
    }

    /// Load dense K row into `self.kv_row_scratch` (len = head_dim); returns slice.
    pub fn load_dense_k(&mut self, base: usize) -> &[f32] {
        use crate::mud::kv_dtype::{load_row_bf16, load_row_f16, KvDtype};
        let n = self.head_dim;
        match self.kv_dtype {
            KvDtype::F32 => {
                self.kv_row_scratch[..n].copy_from_slice(&self.kv_cache[base..base + n]);
            }
            KvDtype::F16 => load_row_f16(&self.kv_cache_f16, base, &mut self.kv_row_scratch[..n]),
            KvDtype::BF16 => load_row_bf16(&self.kv_cache_f16, base, &mut self.kv_row_scratch[..n]),
        }
        &self.kv_row_scratch[..n]
    }

    pub fn load_dense_v(&mut self, base: usize) -> &[f32] {
        use crate::mud::kv_dtype::{load_row_bf16, load_row_f16, KvDtype};
        let n = self.head_dim;
        match self.kv_dtype {
            KvDtype::F32 => {
                self.kv_row_scratch[..n].copy_from_slice(&self.v_cache[base..base + n]);
            }
            KvDtype::F16 => load_row_f16(&self.v_cache_f16, base, &mut self.kv_row_scratch[..n]),
            KvDtype::BF16 => load_row_bf16(&self.v_cache_f16, base, &mut self.kv_row_scratch[..n]),
        }
        &self.kv_row_scratch[..n]
    }

    pub fn load_hca_k(&mut self, base: usize) -> &[f32] {
        use crate::mud::kv_dtype::{load_row_bf16, load_row_f16, KvDtype};
        let n = self.head_dim;
        match self.kv_dtype {
            KvDtype::F32 => {
                self.kv_row_scratch[..n].copy_from_slice(&self.hca_kv_cache[base..base + n]);
            }
            KvDtype::F16 => load_row_f16(&self.hca_kv_f16, base, &mut self.kv_row_scratch[..n]),
            KvDtype::BF16 => load_row_bf16(&self.hca_kv_f16, base, &mut self.kv_row_scratch[..n]),
        }
        &self.kv_row_scratch[..n]
    }

    pub fn load_hca_v(&mut self, base: usize) -> &[f32] {
        use crate::mud::kv_dtype::{load_row_bf16, load_row_f16, KvDtype};
        let n = self.head_dim;
        match self.kv_dtype {
            KvDtype::F32 => {
                self.kv_row_scratch[..n].copy_from_slice(&self.hca_v_cache[base..base + n]);
            }
            KvDtype::F16 => load_row_f16(&self.hca_v_f16, base, &mut self.kv_row_scratch[..n]),
            KvDtype::BF16 => load_row_bf16(&self.hca_v_f16, base, &mut self.kv_row_scratch[..n]),
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
        // v3: 4 bytes (u32: 16-bit real + 16-bit imag)
        assert_eq!(core::mem::size_of::<SlimeRegister>(), 4);
    }

    #[test]
    fn test_register_ternary_roundtrip() {
        let mut reg = SlimeRegister::default();
        let values = [0.0f32, 1.0, -1.0, 100.0, -200.5, 0.001, 42.0];
        for &v in &values {
            reg.write_accum(v);
            let back = reg.read_accum();
            // f16 roundtrip: allow ~0.1% relative error
            if v == 0.0 {
                assert_eq!(back, 0.0);
            } else {
                let rel_err = (back - v).abs() / v.abs().max(1e-8);
                assert!(rel_err < 0.001, "v={} back={} rel_err={}", v, back, rel_err);
            }
        }
    }

    #[test]
    fn test_register_integral_roundtrip() {
        let mut reg = SlimeRegister::default();
        assert_eq!(reg.read_integral(), 0.0);
        assert!((reg.gate() - 0.5).abs() < 0.001);

        reg.update_integral(1.0);
        let i = reg.read_integral();
        assert!((i - 1.0).abs() < 0.001, "i={}", i);
        assert!(reg.gate() > 0.5);
    }

    #[test]
    fn test_register_integral_independence_from_accum() {
        let mut reg = SlimeRegister::default();
        reg.write_accum(42.0);
        reg.write_integral(1.0);
        let accum = reg.read_accum();
        assert!((accum - 42.0).abs() < 0.01, "accum={}", accum);
        reg.update_integral(1.0);
        let accum2 = reg.read_accum();
        assert!((accum2 - 42.0).abs() < 0.01, "accum2={}", accum2);
        let integral = reg.read_integral();
        assert!((integral - 1.0).abs() < 0.001, "integral={}", integral);
    }

    #[test]
    fn test_integral_equilibrium() {
        let mut reg = SlimeRegister::default();
        reg.update_integral(1.0);
        let i = reg.read_integral();
        assert!((i - 1.0).abs() < 0.001, "i={}", i);
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

    /// SX-01/SX-07: When `MUD_SLIMEX` is not set, `slimex_cells` must be zero-size (no wasted RAM).
    #[test]
    fn test_slimex_cells_disabled_by_default() {
        unsafe {
            std::env::remove_var("MUD_SLIMEX");
            std::env::remove_var("MUD_MAX_POS");
            std::env::remove_var("MUD_HCA_WINDOW");
            std::env::remove_var("MUD_HCA_RATIO");
        }
        let ws = SlimeWorkspace::new(128, 256, 4, 4, 32, 512, 4, 1.0);
        assert!(!ws.slimex_enabled, "slimex_enabled must be false without MUD_SLIMEX=1");
        assert_eq!(ws.slimex_cells.len(), 0, "slimex_cells must be zero-size when disabled");
    }

    /// SX-01: `snapshot_to_slimex` correctly packs all registers into the cell buffer.
    /// Verifies: accum_fp8 (bits 0-7), address_tag (bits 8-15), ste_parity (bits 24-31).
    /// NOTE: Builds the workspace directly (no env-var dependency to avoid test parallelism races).
    #[test]
    fn test_slimex_snapshot_populates_cells() {
        use crate::mud::workspace::AlignedArray;
        unsafe {
            std::env::remove_var("MUD_MAX_POS");
            std::env::remove_var("MUD_HCA_WINDOW");
            std::env::remove_var("MUD_HCA_RATIO");
            std::env::remove_var("MUD_SLIMEX");
        }
        // Build workspace without slimex, then patch it directly — avoids env-var race.
        let mut ws = SlimeWorkspace::new(16, 256, 4, 4, 32, 64, 2, 1.0);
        // Manually enable slimex (avoids env-var race with parallel tests).
        ws.slimex_cells = AlignedArray::new(16, 64);
        ws.slimex_enabled = true;

        assert!(ws.slimex_enabled, "slimex_enabled must be true");
        assert_eq!(ws.slimex_cells.len(), 16, "slimex_cells must be hidden_size");

        // Write known values into registers.
        ws.registers[0].write_accum(42.0);
        ws.registers[1].write_accum(-7.0);
        ws.registers[2].write_accum(0.0);

        let jepa_z = vec![1.0f32; 16]; // uniform JEPA velocity
        ws.snapshot_to_slimex(&jepa_z);

        // Slot 0: accum=42 → fp8≈42, address_tag=0, ste_parity=0 (positive)
        let cell0 = ws.read_slimex_cell(0);
        assert_eq!((cell0 & 0xFF) as u8, 42u8, "slot0 accum_fp8");
        assert_eq!(((cell0 >> 8) & 0xFF) as u8, 0u8, "slot0 address_tag");
        assert_eq!((cell0 >> 24) as u8, 0u8, "slot0 ste_parity=0 (positive)");

        // Slot 1: accum=-7 → fp8=7 (abs), ste_parity=1 (negative)
        let cell1 = ws.read_slimex_cell(1);
        assert_eq!((cell1 & 0xFF) as u8, 7u8, "slot1 accum_fp8");
        assert_eq!((cell1 >> 24) as u8, 1u8, "slot1 ste_parity=1 (negative)");

        // Slot 2: accum=0 → fp8=0, jepa_velocity=1 (from jepa_z[2]=1.0) → bits 16-23 = 1
        let cell2 = ws.read_slimex_cell(2);
        let jepa_bits = ((cell2 >> 16) & 0xFF) as i8;
        assert_eq!(jepa_bits, 1i8, "slot2 jepa_velocity from jepa_z[2]=1.0");
    }
}
