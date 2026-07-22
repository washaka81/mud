//! # L-11: ExpertBus — hot mount/unmount Mini MoE
//!
//! - Up to [`MAX_EXPERT_SLOTS`] experts (`u16` slots)
//! - Router: optional ternary GEMV + [`MudRouter`] top-k / Gumbel / hash
//! - **C7 dense fallback:** if no router or a single mounted expert → that expert @ weight 1.0
//! - Forward is zero-alloc when using a prebuilt [`ExpertScratch`]

use crate::mud::routing::MudRouter;
use crate::mud::slime_expert::SlimeExpert;
use crate::mud::slime_forward::{ternary_gemv_rowwise, SlimeLayer};

/// Addressable expert slots (PLAN_MAESTRO Mini MoE).
pub const MAX_EXPERT_SLOTS: usize = 64;

/// Preallocated scratch for bus forward (P-01).
pub struct ExpertScratch {
    pub logits: Vec<f32>,
    pub indexed: Vec<(usize, f32)>,
    pub route: Vec<(usize, f32)>,
    pub up: Vec<f32>,
    pub gate: Vec<f32>,
    pub mid: Vec<f32>,
    pub expert_out: Vec<f32>,
    pub accum: Vec<f32>,
}

impl ExpertScratch {
    pub fn new(hidden: usize, ffn_mid: usize, n_slots: usize) -> Self {
        Self {
            logits: vec![0.0; n_slots.max(1)],
            indexed: Vec::with_capacity(n_slots.max(1)),
            route: Vec::with_capacity(8),
            up: vec![0.0; ffn_mid.max(1)],
            gate: vec![0.0; ffn_mid.max(1)],
            mid: vec![0.0; ffn_mid.max(1)],
            expert_out: vec![0.0; hidden.max(1)],
            accum: vec![0.0; hidden.max(1)],
        }
    }

    pub fn ensure(&mut self, hidden: usize, ffn_mid: usize, n_slots: usize) {
        if self.logits.len() < n_slots {
            self.logits.resize(n_slots, 0.0);
        }
        if self.up.len() < ffn_mid {
            self.up.resize(ffn_mid, 0.0);
            self.gate.resize(ffn_mid, 0.0);
            self.mid.resize(ffn_mid, 0.0);
        }
        if self.expert_out.len() < hidden {
            self.expert_out.resize(hidden, 0.0);
            self.accum.resize(hidden, 0.0);
        }
    }
}

/// How the bus selects experts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RouterMode {
    /// Softmax top-k on router logits (optional ternary projection).
    Softmax,
    /// Gumbel noise on logits (exploration).
    Gumbel,
    /// Parameter-free hash routing on `x`.
    Hash,
}

/// Hot-swappable expert bus.
pub struct ExpertBus {
    slots: Vec<Option<SlimeExpert>>,
    pub router: MudRouter,
    /// Ternary router weights: rows = `slots.len()`, cols = hidden (ELUT).
    router_w: *const u8,
    router_scales: *const f32,
    pub mode: RouterMode,
    pub gumbel_temperature: f32,
    /// Hidden size expected by mounted experts (0 = unset).
    pub hidden: usize,
    pub ffn_mid: usize,
}

// SAFETY: experts hold shared immutable weight pointers; bus is not Sync for mount mutations.
unsafe impl Send for ExpertBus {}

impl ExpertBus {
    pub fn with_capacity(n_slots: usize, top_k: usize) -> Self {
        let n = n_slots.clamp(1, MAX_EXPERT_SLOTS);
        Self {
            slots: (0..n).map(|_| None).collect(),
            router: MudRouter::new(n, top_k.max(1)),
            router_w: std::ptr::null(),
            router_scales: std::ptr::null(),
            mode: RouterMode::Softmax,
            gumbel_temperature: 0.5,
            hidden: 0,
            ffn_mid: 0,
        }
    }

    /// C7: dense FFN as sole expert 0 — behaves as legacy single-FFN path.
    /// If layer FFN pointers are null, returns an empty 1-slot bus with dims set.
    pub fn from_dense_layer(layer: &SlimeLayer, hidden: usize, top_k: usize) -> Self {
        let mut bus = Self::with_capacity(1, top_k.max(1));
        bus.hidden = hidden;
        bus.ffn_mid = layer.ffn_mid;
        let expert = SlimeExpert::from_dense_layer(0, layer, hidden);
        if expert.is_valid() {
            let _ = bus.mount(0, expert);
        }
        bus
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    pub fn mounted_count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    /// Dense path: single expert, or multi-expert Softmax/Gumbel without router weights.
    /// [`RouterMode::Hash`] never requires router GEMV weights.
    pub fn is_dense_mode(&self) -> bool {
        if self.mounted_count() <= 1 {
            return true;
        }
        match self.mode {
            RouterMode::Hash => false,
            RouterMode::Softmax | RouterMode::Gumbel => self.router_w.is_null(),
        }
    }

    pub fn has_router(&self) -> bool {
        !self.router_w.is_null() && !self.router_scales.is_null()
    }

    /// Attach ternary router projection (`n_slots × hidden` ELUT rows).
    pub fn set_router(&mut self, w: *const u8, scales: *const f32) {
        self.router_w = w;
        self.router_scales = scales;
    }

    pub fn clear_router(&mut self) {
        self.router_w = std::ptr::null();
        self.router_scales = std::ptr::null();
    }

    /// Hot-mount expert into `slot`. Returns previous occupant if any.
    pub fn mount(&mut self, slot: u16, expert: SlimeExpert) -> anyhow::Result<Option<SlimeExpert>> {
        let i = slot as usize;
        if i >= self.slots.len() {
            anyhow::bail!(
                "ExpertBus::mount slot {slot} out of range (capacity {})",
                self.slots.len()
            );
        }
        if !expert.is_valid() {
            anyhow::bail!("ExpertBus::mount slot {slot}: expert pointers invalid");
        }
        if self.hidden == 0 {
            self.hidden = expert.hidden;
            self.ffn_mid = expert.ffn_mid;
        } else if expert.hidden != self.hidden || expert.ffn_mid != self.ffn_mid {
            anyhow::bail!(
                "ExpertBus::mount dim mismatch: bus {}×{} vs expert {}×{}",
                self.hidden,
                self.ffn_mid,
                expert.hidden,
                expert.ffn_mid
            );
        }
        Ok(self.slots[i].replace(expert))
    }

    /// Hot-unmount; returns expert if present.
    pub fn unmount(&mut self, slot: u16) -> Option<SlimeExpert> {
        let i = slot as usize;
        if i >= self.slots.len() {
            return None;
        }
        self.slots[i].take()
    }

    pub fn get(&self, slot: u16) -> Option<&SlimeExpert> {
        self.slots.get(slot as usize).and_then(|s| s.as_ref())
    }

    /// List mounted (slot, id) pairs.
    pub fn mounted_slots(&self) -> Vec<(u16, u16)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.as_ref().map(|e| (i as u16, e.id)))
            .collect()
    }

    /// MoE / dense FFN forward: `out = Σ_k w_k · expert_k(x)`.
    ///
    /// # Safety
    /// Mounted weight pointers and optional router pointers must remain valid.
    /// `x` / `out` length ≥ `self.hidden`.
    pub unsafe fn forward(
        &self,
        x: &[f32],
        out: &mut [f32],
        scratch: &mut ExpertScratch,
        seed: u32,
    ) -> anyhow::Result<()> {
        if self.hidden == 0 || self.mounted_count() == 0 {
            anyhow::bail!("ExpertBus::forward: no experts mounted");
        }
        let h = self.hidden;
        let m = self.ffn_mid;
        if x.len() < h || out.len() < h {
            anyhow::bail!("ExpertBus::forward: buffer size mismatch");
        }
        scratch.ensure(h, m, self.slots.len());

        // --- Route ---
        self.route_into(x, scratch, seed)?;

        if scratch.route.is_empty() {
            out[..h].fill(0.0);
            return Ok(());
        }

        // --- Weighted expert sum ---
        scratch.accum[..h].fill(0.0);
        for &(slot, weight) in scratch.route.iter() {
            if weight == 0.0 || !weight.is_finite() {
                continue;
            }
            let Some(expert) = self.slots.get(slot).and_then(|s| s.as_ref()) else {
                continue;
            };
            expert.forward_swiglu(
                x,
                &mut scratch.up,
                &mut scratch.gate,
                &mut scratch.mid,
                &mut scratch.expert_out,
            );
            // accum += weight * expert_out
            let acc = scratch.accum.as_mut_ptr();
            let eo = scratch.expert_out.as_ptr();
            for i in 0..h {
                *acc.add(i) += weight * *eo.add(i);
            }
        }
        out[..h].copy_from_slice(&scratch.accum[..h]);
        Ok(())
    }

    unsafe fn route_into(
        &self,
        x: &[f32],
        scratch: &mut ExpertScratch,
        seed: u32,
    ) -> anyhow::Result<()> {
        scratch.route.clear();
        let n = self.slots.len();
        let h = self.hidden;

        // Dense fallback: single expert or missing router
        if self.is_dense_mode() {
            let slot = self
                .slots
                .iter()
                .enumerate()
                .find_map(|(i, s)| s.as_ref().map(|_| i))
                .ok_or_else(|| anyhow::anyhow!("no mounted expert for dense fallback"))?;
            scratch.route.push((slot, 1.0));
            return Ok(());
        }

        // Hash mode: no GEMV
        if self.mode == RouterMode::Hash {
            self.router.route_by_hash(&x[..h], &mut scratch.route);
            // Map logical indices → only keep mounted; re-normalize
            scratch
                .route
                .retain(|&(i, _)| i < n && self.slots[i].is_some());
            if scratch.route.is_empty() {
                // Fallback first mounted
                if let Some(slot) = self.slots.iter().position(|s| s.is_some()) {
                    scratch.route.push((slot, 1.0));
                }
            } else {
                let sum: f32 = scratch.route.iter().map(|(_, w)| *w).sum();
                if sum > 0.0 {
                    for r in scratch.route.iter_mut() {
                        r.1 /= sum;
                    }
                }
            }
            return Ok(());
        }

        // Ternary router GEMV → logits
        if self.router_w.is_null() || self.router_scales.is_null() {
            anyhow::bail!("router weights required for Softmax/Gumbel mode");
        }
        ternary_gemv_rowwise(
            &x[..h],
            self.router_w,
            &mut scratch.logits[..n],
            self.router_scales,
            n,
            h,
        );
        // Mask empty slots
        for i in 0..n {
            if self.slots[i].is_none() {
                scratch.logits[i] = f32::NEG_INFINITY;
            }
        }

        let z = match self.mode {
            RouterMode::Gumbel => self.router.route_by_q_head(
                &scratch.logits[..n],
                self.gumbel_temperature,
                seed,
                &mut scratch.indexed,
                &mut scratch.route,
            ),
            RouterMode::Softmax | RouterMode::Hash => self.router.route_in_place(
                &scratch.logits[..n],
                &mut scratch.indexed,
                &mut scratch.route,
                None,
            ),
        };
        let _ = z; // z-loss available for future aux loss wiring

        // Drop any route to empty slot (safety)
        scratch
            .route
            .retain(|&(i, _)| i < n && self.slots[i].is_some());
        if scratch.route.is_empty() {
            if let Some(slot) = self.slots.iter().position(|s| s.is_some()) {
                scratch.route.push((slot, 1.0));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny identity-ish expert with all-zero ELUT (outputs ~0) for bus plumbing tests.
    fn zero_expert(id: u16, hidden: usize, ffn_mid: usize) -> (SlimeExpert, Vec<u8>, Vec<f32>) {
        // packed bytes: (rows * cols).div_ceil(2) for ELUT nibble packing used by gemv (8/u32)
        let up_rows = ffn_mid;
        let down_rows = hidden;
        let up_u32s = up_rows * (hidden / 8).max(1);
        let down_u32s = down_rows * (ffn_mid / 8).max(1);
        let packed_up = vec![0u8; up_u32s * 4];
        let packed_gate = packed_up.clone();
        let packed_down = vec![0u8; down_u32s * 4];
        let scales_up = vec![1.0f32; up_rows];
        let scales_gate = scales_up.clone();
        let scales_down = vec![1.0f32; down_rows];
        // Leak storage for pointer stability in test (test-only)
        let up_w = Box::leak(packed_up.into_boxed_slice()).as_ptr();
        let gate_w = Box::leak(packed_gate.into_boxed_slice()).as_ptr();
        let down_w = Box::leak(packed_down.into_boxed_slice()).as_ptr();
        let up_s = Box::leak(scales_up.into_boxed_slice()).as_ptr();
        let gate_s = Box::leak(scales_gate.into_boxed_slice()).as_ptr();
        let down_s = Box::leak(scales_down.into_boxed_slice()).as_ptr();
        let expert = SlimeExpert::from_ptrs(
            id, hidden, ffn_mid, up_w, gate_w, down_w, up_s, gate_s, down_s,
        );
        (expert, Vec::new(), Vec::new())
    }

    #[test]
    fn test_mount_unmount_hot_swap() {
        let mut bus = ExpertBus::with_capacity(4, 2);
        let h = 8usize;
        let m = 16usize;
        // hidden/ffn_mid must be multiples of 8 for ELUT gemv path
        let (e0, _, _) = zero_expert(0, h, m);
        let (e1, _, _) = zero_expert(1, h, m);
        assert!(bus.mount(0, e0).unwrap().is_none());
        assert!(bus.mount(1, e1).unwrap().is_none());
        assert_eq!(bus.mounted_count(), 2);
        let prev = bus.unmount(1);
        assert!(prev.is_some());
        assert_eq!(bus.mounted_count(), 1);
        // remount
        let (e1b, _, _) = zero_expert(7, h, m);
        bus.mount(1, e1b).unwrap();
        assert_eq!(bus.get(1).unwrap().id, 7);
    }

    #[test]
    fn test_dense_fallback_forward() {
        let h = 8usize;
        let m = 16usize;
        let mut bus = ExpertBus::with_capacity(2, 2);
        let (e0, _, _) = zero_expert(0, h, m);
        bus.mount(0, e0).unwrap();
        assert!(bus.is_dense_mode());

        let x = vec![0.5f32; h];
        let mut out = vec![1.0f32; h]; // should be overwritten
        let mut scratch = ExpertScratch::new(h, m, 2);
        unsafe {
            bus.forward(&x, &mut out, &mut scratch, 0).unwrap();
        }
        // zero weights → ~0 output
        for &v in &out {
            assert!(v.abs() < 1e-3, "expected near-zero, got {v}");
        }
        assert_eq!(scratch.route.len(), 1);
        assert_eq!(scratch.route[0].0, 0);
        assert!((scratch.route[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mount_rejects_dim_mismatch() {
        let mut bus = ExpertBus::with_capacity(2, 1);
        let (e0, _, _) = zero_expert(0, 8, 16);
        bus.mount(0, e0).unwrap();
        let (e1, _, _) = zero_expert(1, 16, 32);
        assert!(bus.mount(1, e1).is_err());
    }

    #[test]
    fn test_slot_out_of_range() {
        let mut bus = ExpertBus::with_capacity(2, 1);
        let (e0, _, _) = zero_expert(0, 8, 16);
        assert!(bus.mount(5, e0).is_err());
        assert!(bus.unmount(99).is_none());
    }

    #[test]
    fn test_from_dense_layer_nulls_empty() {
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
            ffn_mid: 16,
            rope_theta: 0.0,
        };
        let bus = ExpertBus::from_dense_layer(&layer, 8, 1);
        assert_eq!(bus.capacity(), 1);
        assert_eq!(bus.mounted_count(), 0); // null FFN → not mounted
        assert_eq!(bus.hidden, 8);
        assert_eq!(bus.ffn_mid, 16);
    }

    #[test]
    fn test_hash_route_two_experts() {
        let h = 8usize;
        let m = 16usize;
        let mut bus = ExpertBus::with_capacity(4, 2);
        bus.mode = RouterMode::Hash;
        let (e0, _, _) = zero_expert(0, h, m);
        let (e1, _, _) = zero_expert(1, h, m);
        bus.mount(0, e0).unwrap();
        bus.mount(1, e1).unwrap();
        assert!(!bus.is_dense_mode());
        let x = vec![0.25f32; h];
        let mut out = vec![0.0f32; h];
        let mut scratch = ExpertScratch::new(h, m, 4);
        unsafe {
            bus.forward(&x, &mut out, &mut scratch, 42).unwrap();
        }
        assert!(!scratch.route.is_empty());
        for &(slot, w) in &scratch.route {
            assert!(bus.get(slot as u16).is_some());
            assert!(w > 0.0 && w.is_finite());
        }
        let wsum: f32 = scratch.route.iter().map(|(_, w)| *w).sum();
        assert!((wsum - 1.0).abs() < 1e-5, "route weights sum={wsum}");
    }
}
