use crate::mud::adam_state::AdamState;
use crate::mud::slime_backward::{OptimizerStrategy, select_optimizer};
use crate::mud::MudTensor;
use std::collections::HashMap;

/// SlimeX Context for a single expert. 
/// Pre-allocated buffers to respect P-01 (zero allocation on hot path).
pub struct SlimeXSlot {
    pub id: Option<u16>,
    pub ffn_up_w: Vec<f32>,
    pub ffn_gate_w: Vec<f32>,
    pub ffn_down_w: Vec<f32>,

    pub ffn_up_opt: OptimizerStrategy,
    pub ffn_gate_opt: OptimizerStrategy,
    pub ffn_down_opt: OptimizerStrategy,

    pub ffn_up_adam: Option<AdamState>,
    pub ffn_gate_adam: Option<AdamState>,
    pub ffn_down_adam: Option<AdamState>,
}

impl SlimeXSlot {
    pub fn new(hidden: usize, ffn_mid: usize) -> Self {
        let ffn_up_opt = select_optimizer(ffn_mid, hidden);
        let ffn_gate_opt = select_optimizer(ffn_mid, hidden);
        let ffn_down_opt = select_optimizer(hidden, ffn_mid);
        Self {
            id: None,
            ffn_up_w: vec![0.0; ffn_mid * hidden],
            ffn_gate_w: vec![0.0; ffn_mid * hidden],
            ffn_down_w: vec![0.0; hidden * ffn_mid],
            ffn_up_opt,
            ffn_gate_opt,
            ffn_down_opt,
            ffn_up_adam: AdamState::for_strategy(ffn_mid * hidden, ffn_up_opt),
            ffn_gate_adam: AdamState::for_strategy(ffn_mid * hidden, ffn_gate_opt),
            ffn_down_adam: AdamState::for_strategy(hidden * ffn_mid, ffn_down_opt),
        }
    }
}

/// ShadowExpertBus (SlimeX) — Dynamic Stack of Expert Shadows for Multi-expert Weighted STE
pub struct ShadowExpertBus {
    pub slots: Vec<SlimeXSlot>,
    pub hidden: usize,
    pub ffn_mid: usize,
}

impl ShadowExpertBus {
    /// Creates a pre-allocated bus with `top_k` slots. 
    /// Ensures we never allocate during the hot loop (P-01).
    pub fn new(hidden: usize, ffn_mid: usize, top_k: usize) -> Self {
        let mut slots = Vec::with_capacity(top_k);
        for _ in 0..top_k {
            slots.push(SlimeXSlot::new(hidden, ffn_mid));
        }
        Self { slots, hidden, ffn_mid }
    }

    /// Mounts an expert into an available slot or reuses if already mounted.
    /// In a real implementation, this inflates from the 4-bit ELUT to f32.
    /// Returns the index of the slot.
    pub fn mount(&mut self, expert_id: u16, _tensors: &HashMap<String, MudTensor>, _blk: usize) -> Option<usize> {
        // 1. Check if already mounted
        if let Some(idx) = self.slots.iter().position(|s| s.id == Some(expert_id)) {
            return Some(idx);
        }
        // 2. Find empty slot
        if let Some(idx) = self.slots.iter().position(|s| s.id.is_none()) {
            self.slots[idx].id = Some(expert_id);
            // TODO: Unpack ternary ELUT weights to `self.slots[idx].ffn_*_w`
            // using `unpack_ternary2bit_to_f32`
            return Some(idx);
        }
        // 3. No empty slots (requires cache eviction policy if we route more than top_k unique experts across steps)
        None
    }

    /// Unmounts an expert, freeing its slot.
    /// If using an optimizer with momentum (e.g. ChunkedAdam), state should ideally be preserved.
    pub fn unmount(&mut self, expert_id: u16) {
        if let Some(idx) = self.slots.iter().position(|s| s.id == Some(expert_id)) {
            // TODO: Save AdamState to disk/backing store if necessary
            self.slots[idx].id = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slimex_dynamic_stack() {
        let mut bus = ShadowExpertBus::new(128, 512, 2);
        assert_eq!(bus.slots.len(), 2);
        
        let empty_tensors = HashMap::new();
        
        // Mount first expert
        let idx1 = bus.mount(0, &empty_tensors, 0);
        assert_eq!(idx1, Some(0));
        assert_eq!(bus.slots[0].id, Some(0));

        // Mount second expert
        let idx2 = bus.mount(1, &empty_tensors, 0);
        assert_eq!(idx2, Some(1));
        assert_eq!(bus.slots[1].id, Some(1));

        // Try mounting a third expert when full
        let idx3 = bus.mount(2, &empty_tensors, 0);
        assert_eq!(idx3, None); // Should fail because top_k = 2

        // Unmount first expert
        bus.unmount(0);
        assert_eq!(bus.slots[0].id, None);

        // Remount third expert
        let idx4 = bus.mount(2, &empty_tensors, 0);
        assert_eq!(idx4, Some(0)); // Should take the now-empty first slot
        assert_eq!(bus.slots[0].id, Some(2));
    }
}
