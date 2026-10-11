//! # L-15: Gradient / activation checkpointing
//!
//! Classic trade-off: store less activation tape during the forward pass and
//! **recompute** layer segments on the reverse pass.
//!
//! | Mode | Memory (activations) | Compute |
//! |------|----------------------|---------|
//! | `Full` (default) | O(L · tape) | 1× forward |
//! | `Segmented` | O(S · tape + (L/S)·hidden) | ~2× forward in segments |
//!
//! Env:
//! - `MUD_GRAD_CKPT=1` → segmented mode
//! - `MUD_GRAD_CKPT_SEG=N` → **layers per segment** (default 4, min 1).
//!   Count of segments = `ceil(num_layers / SEG)`. Smaller `SEG` = more
//!   segments = less memory but more recompute; e.g. with `num_layers=64`:
//!   `SEG=4` → 16 segments (saves ~4× activation memory); `SEG=1` → 64
//!   segments (max savings, max recompute). To target a specific segment
//!   count `K`, set `SEG = max(1, num_layers.div_ceil(K))`.
//!
//! The `Segmented` mode keeps one residual + JEPA snapshot per segment
//! boundary; computation rewinds to the snapshot at the start of each
//! backward segment and recomputes through that segment's layers.

use crate::mud::slime::SlimeWorkspace;
use crate::mud::slime_backward::SlimeLayerTape;
use crate::mud::slime_forward::{evaluate_slime_block, layer_is_valid, SlimeLayer};

/// Default layers per recompute segment.
pub const DEFAULT_SEGMENT: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckpointMode {
    /// Keep full tapes for all layers (historical default).
    Full,
    /// Drop tapes after forward; recompute per segment on backward.
    Segmented,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointPolicy {
    pub mode: CheckpointMode,
    pub segment_size: usize,
}

impl CheckpointPolicy {
    /// Pure policy constructor from already-parsed env flags.
    ///
    /// Tests use this directly to avoid racing on the *global* `MUD_GRAD_CKPT`
    /// / `MUD_GRAD_CKPT_SEG` env vars under parallel test execution (the old
    /// `env::set_var` / `remove_var` tests produced a non-deterministic
    /// `test_policy_default_full` failure: a sibling test could re-set
    /// `MUD_GRAD_CKPT=1` between this test's `remove_var` and its `resolve`).
    pub fn from_env(on: bool, segment_size: usize) -> Self {
        if !on {
            return Self {
                mode: CheckpointMode::Full,
                segment_size: DEFAULT_SEGMENT,
            };
        }
        Self {
            mode: CheckpointMode::Segmented,
            segment_size: segment_size.max(1),
        }
    }

    pub fn resolve() -> Self {
        let on = std::env::var("MUD_GRAD_CKPT")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let segment_size = std::env::var("MUD_GRAD_CKPT_SEG")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_SEGMENT);
        Self::from_env(on, segment_size)
    }

    pub fn is_segmented(self) -> bool {
        self.mode == CheckpointMode::Segmented
    }

    /// Number of residual snapshots needed (one per segment start + final).
    pub fn num_residual_slots(self, n_layers: usize) -> usize {
        if !self.is_segmented() || n_layers == 0 {
            return 0;
        }
        n_layers.div_ceil(self.segment_size) + 1
    }

    /// Segment index for layer `l`.
    #[inline]
    pub fn segment_of(self, layer_idx: usize) -> usize {
        layer_idx / self.segment_size.max(1)
    }

    /// First layer index of the segment containing `layer_idx`.
    #[inline]
    pub fn segment_start(self, layer_idx: usize) -> usize {
        self.segment_of(layer_idx) * self.segment_size.max(1)
    }

    /// Exclusive end layer of the segment (clamped to `n_layers`).
    #[inline]
    pub fn segment_end(self, layer_idx: usize, n_layers: usize) -> usize {
        (self.segment_start(layer_idx) + self.segment_size).min(n_layers)
    }

    /// Approximate bytes for one full tape (order-of-magnitude accounting).
    pub fn approx_tape_bytes(
        hidden: usize,
        ffn_mid: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scores_len: usize,
    ) -> usize {
        let kv = n_kv_heads * head_dim;
        // i8 norms + f32 q/k/v/o/ffn/jepa + scores
        let i8s = hidden * 2;
        let f32s = hidden * 2 // q + o
            + kv * 2 // k v
            + scores_len
            + ffn_mid * 3
            + hidden * 2; // jepa
        i8s + f32s * 4 + 64
    }

    /// Peak activation estimate: Full vs Segmented.
    pub fn peak_activation_bytes(
        self,
        n_layers: usize,
        hidden: usize,
        ffn_mid: usize,
        n_kv_heads: usize,
        head_dim: usize,
        scores_len: usize,
    ) -> (usize, usize) {
        let tape = Self::approx_tape_bytes(hidden, ffn_mid, n_kv_heads, head_dim, scores_len);
        let full = n_layers.saturating_mul(tape);
        if !self.is_segmented() {
            return (full, full);
        }
        let s = self.segment_size.max(1);
        let seg_tapes = s.saturating_mul(tape);
        let residuals = self.num_residual_slots(n_layers).saturating_mul(hidden * 4);
        let segmented = seg_tapes.saturating_add(residuals);
        (full, segmented)
    }
}

/// Bank of residual-stream + JEPA snapshots at segment boundaries (stream H).
pub struct ResidualBank {
    /// Flat: slot * hidden + dim — residual matmul_accum
    data: Vec<f32>,
    /// Flat: slot * jepa_len — full `jepa_z` workspace snapshot
    jepa: Vec<f32>,
    /// Flat: slot * hidden — per-register jepa_energy (integral)
    integral: Vec<f32>,
    pub hidden: usize,
    pub jepa_len: usize,
    pub n_slots: usize,
}

impl ResidualBank {
    pub fn new(n_slots: usize, hidden: usize) -> Self {
        Self {
            data: vec![0.0; n_slots.saturating_mul(hidden)],
            jepa: Vec::new(),
            integral: vec![0.0; n_slots.saturating_mul(hidden)],
            hidden,
            jepa_len: 0,
            n_slots,
        }
    }

    /// Prefer this constructor so JEPA state is snapshotted with residuals.
    pub fn with_workspace(n_slots: usize, ws: &SlimeWorkspace) -> Self {
        let jepa_len = ws.jepa_z.len();
        Self {
            data: vec![0.0; n_slots.saturating_mul(ws.hidden_size)],
            jepa: vec![0.0; n_slots.saturating_mul(jepa_len)],
            integral: vec![0.0; n_slots.saturating_mul(ws.hidden_size)],
            hidden: ws.hidden_size,
            jepa_len,
            n_slots,
        }
    }

    pub fn save_from_workspace(&mut self, slot: usize, ws: &SlimeWorkspace) {
        if slot >= self.n_slots {
            return;
        }
        let base = slot * self.hidden;
        for i in 0..self.hidden {
            self.data[base + i] = ws.registers[i].read_accum();
            self.integral[base + i] = ws.registers[i].read_integral();
        }
        if self.jepa_len > 0 && self.jepa.len() >= (slot + 1) * self.jepa_len {
            let jb = slot * self.jepa_len;
            let n = self.jepa_len.min(ws.jepa_z.len());
            self.jepa[jb..jb + n].copy_from_slice(&ws.jepa_z[..n]);
        }
    }

    pub fn restore_to_workspace(&self, slot: usize, ws: &mut SlimeWorkspace) {
        if slot >= self.n_slots {
            return;
        }
        let base = slot * self.hidden;
        for i in 0..self.hidden.min(ws.registers.len()) {
            ws.registers[i].write_accum(self.data[base + i]);
            ws.registers[i].write_integral(self.integral[base + i]);
        }
        if self.jepa_len > 0 && self.jepa.len() >= (slot + 1) * self.jepa_len {
            let jb = slot * self.jepa_len;
            let n = self.jepa_len.min(ws.jepa_z.len());
            ws.jepa_z[..n].copy_from_slice(&self.jepa[jb..jb + n]);
        }
    }
}

/// Recompute forward for layers `[start, end)` into `tapes`, starting from current
/// workspace residual (caller must restore residual for `start` first).
pub fn recompute_segment(
    layers: &[SlimeLayer],
    start: usize,
    end: usize,
    workspace: &mut SlimeWorkspace,
    tapes: &mut [SlimeLayerTape],
    eps: f32,
    pos: usize,
) {
    let end = end.min(layers.len()).min(tapes.len());
    for l in start..end {
        if !layer_is_valid(&layers[l]) {
            continue;
        }
        tapes[l].reset();
        tapes[l].valid = true;
        tapes[l].pos = pos;
        evaluate_slime_block(&layers[l], l, workspace, pos, eps, Some(&mut tapes[l]), None);
    }
}

/// Re-seed workspace from quantized embedding values and recompute layers `0..through_end`.
/// Preferred for L-15 correctness (JEPA state rebuilt from token 0).
pub fn recompute_from_embedding(
    emb: &[f32],
    layers: &[SlimeLayer],
    through_end: usize,
    workspace: &mut SlimeWorkspace,
    tapes: &mut [SlimeLayerTape],
    eps: f32,
    pos: usize,
) {
    let hidden = workspace.hidden_size.min(emb.len());
    let n_layers = layers.len();
    workspace.clear_registers();
    for (i, &emb_i) in emb.iter().enumerate().take(hidden) {
        crate::mud::slime::SlimeRegister::init_from_embed(
            &mut workspace.registers[i],
            &mut workspace.jepa_z,
            i,
            hidden,
            n_layers,
            emb_i,
            true,
        );
    }
    recompute_segment(layers, 0, through_end, workspace, tapes, eps, pos);
}

/// Stream H: recompute only segment containing `layer_idx` after restoring residual
/// from [`ResidualBank`] at segment start (avoids full emb→prefix recompute).
///
/// Caller must have saved residuals at each segment boundary during forward.
#[allow(clippy::too_many_arguments)]
pub fn recompute_from_residual_bank(
    bank: &ResidualBank,
    layers: &[SlimeLayer],
    layer_idx: usize,
    workspace: &mut SlimeWorkspace,
    tapes: &mut [SlimeLayerTape],
    policy: CheckpointPolicy,
    eps: f32,
    pos: usize,
) {
    let n = layers.len();
    let start = policy.segment_start(layer_idx);
    let end = policy.segment_end(layer_idx, n);
    let slot = policy.segment_of(layer_idx);
    bank.restore_to_workspace(slot, workspace);
    recompute_segment(layers, start, end, workspace, tapes, eps, pos);
}

/// Whether residual-bank path is requested (`MUD_GRAD_CKPT_RESIDUAL=1`).
pub fn residual_bank_recompute_enabled() -> bool {
    std::env::var("MUD_GRAD_CKPT_RESIDUAL")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Free activation storage for a layer after its backward is done.
pub fn discard_tape(tape: &mut SlimeLayerTape) {
    tape.reset();
    tape.valid = false;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policy_default_full() {
        // Pure path: no env mutation, deterministic under parallel test execution.
        let p = CheckpointPolicy::from_env(false, DEFAULT_SEGMENT);
        assert_eq!(p.mode, CheckpointMode::Full);
        assert!(!p.is_segmented());
    }

    #[test]
    fn test_policy_segmented_env() {
        // Pure path: on=true, segment_size=3 (parsed value already resolved).
        let p = CheckpointPolicy::from_env(true, 3);
        assert!(p.is_segmented());
        assert_eq!(p.segment_size, 3);
        assert_eq!(p.segment_start(7), 6);
        assert_eq!(p.segment_end(7, 30), 9);
    }

    /// Round-trip: `resolve()` (env-backed) must agree with `from_env` (pure)
    /// for the two canonical inputs.
    #[test]
    fn test_resolve_matches_from_env() {
        // env-default (unset) == from_env(false, _)
        unsafe {
            std::env::remove_var("MUD_GRAD_CKPT");
            std::env::remove_var("MUD_GRAD_CKPT_SEG");
        }
        let resolved = CheckpointPolicy::resolve();
        let pure = CheckpointPolicy::from_env(false, DEFAULT_SEGMENT);
        assert_eq!(resolved.mode, pure.mode);
        assert_eq!(resolved.segment_size, pure.segment_size);
    }

    #[test]
    fn test_segmented_saves_memory() {
        let p = CheckpointPolicy {
            mode: CheckpointMode::Segmented,
            segment_size: 4,
        };
        let (full, seg) = p.peak_activation_bytes(30, 576, 1536, 3, 64, 512);
        assert!(seg < full, "segmented {seg} should be < full {full}");
        // Expect roughly segment_size/n_layers fraction (+ residuals)
        assert!(seg * 3 < full);
    }

    #[test]
    fn test_residual_bank_roundtrip() {
        let mut ws = SlimeWorkspace::new(32, 64, 4, 2, 8, 32, 2, 1.0);
        for i in 0..32 {
            ws.registers[i].write_accum(i as f32 * 0.1);
            ws.registers[i].write_integral(i as f32 * 0.01);
        }
        if !ws.jepa_z.is_empty() {
            ws.jepa_z[0] = std::f32::consts::PI;
        }
        let mut bank = ResidualBank::with_workspace(2, &ws);
        bank.save_from_workspace(0, &ws);
        for i in 0..32 {
            ws.registers[i].write_accum(0.0);
            ws.registers[i].write_integral(0.0);
        }
        if !ws.jepa_z.is_empty() {
            ws.jepa_z[0] = 0.0;
        }
        bank.restore_to_workspace(0, &mut ws);
        for i in 0..32 {
            assert!((ws.registers[i].read_accum() - i as f32 * 0.1).abs() < 0.001,
                "accum[{}] = {}", i, ws.registers[i].read_accum());
            assert!((ws.registers[i].read_integral() - i as f32 * 0.01).abs() < 0.001,
                "integral[{}] = {}", i, ws.registers[i].read_integral());
        }
        if !ws.jepa_z.is_empty() {
            assert!((ws.jepa_z[0] - std::f32::consts::PI).abs() < 1e-5);
        }
    }

    #[test]
    fn test_residual_bank_env_flag() {
        unsafe {
            std::env::remove_var("MUD_GRAD_CKPT_RESIDUAL");
            assert!(!residual_bank_recompute_enabled());
            std::env::set_var("MUD_GRAD_CKPT_RESIDUAL", "1");
            assert!(residual_bank_recompute_enabled());
            std::env::remove_var("MUD_GRAD_CKPT_RESIDUAL");
        }
    }
}
