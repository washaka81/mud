//! # MoE load from `.mud` (gap P0-B / L-11 product path)
//!
//! Tensor naming (universal_converter):
//! ```text
//! blk.{L}.expert.{E}.w1.weight     # gate
//! blk.{L}.expert.{E}.w3.weight     # up
//! blk.{L}.expert.{E}.w2.weight     # down
//! blk.{L}.expert.{E}.w*.prq_scale
//! # alternate:
//! blk.{L}.expert.{E}.gate.weight / up.weight
//! # router (optional):
//! blk.{L}.gate.weight  |  blk.{L}.moe_router.weight  (+ .prq_scale)
//! ```
//!
//! Dense models only ship `expert.0` → single-slot bus (C7 dense fallback).
//! Multi-expert models mount all discovered slots; Hash routing if no router weights.

use crate::mud::expert_bus::{ExpertBus, RouterMode, MAX_EXPERT_SLOTS};
use crate::mud::slime_expert::SlimeExpert;
use crate::mud::MudFile;
use crate::mud::{MudTensor, MudTensorType};
use std::collections::HashMap;

/// Canonical FFN name triple for one expert: (up, gate, down) **without** `.weight` suffix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertFfnNames {
    pub up: String,
    pub gate: String,
    pub down: String,
}

/// Resolve up/gate/down base names for `expert_id` given which keys exist in `tensors`.
pub fn resolve_expert_ffn_names(
    tensors: &HashMap<String, MudTensor>,
    layer_idx: usize,
    expert_id: u16,
) -> Option<ExpertFfnNames> {
    let p = format!("blk.{}.", layer_idx);
    let e = expert_id;
    // Prefer explicit up/gate naming
    let up_alt = format!("{p}expert.{e}.up.weight");
    let gate_alt = format!("{p}expert.{e}.gate.weight");
    let down_w2 = format!("{p}expert.{e}.w2.weight");
    let down_down = format!("{p}expert.{e}.down.weight");

    if tensors.contains_key(&up_alt) && tensors.contains_key(&gate_alt) {
        let down = if tensors.contains_key(&down_w2) {
            format!("expert.{e}.w2")
        } else if tensors.contains_key(&down_down) {
            format!("expert.{e}.down")
        } else {
            return None;
        };
        return Some(ExpertFfnNames {
            up: format!("expert.{e}.up"),
            gate: format!("expert.{e}.gate"),
            down,
        });
    }

    // BitNet / converter: w3=up, w1=gate, w2=down
    let w1 = format!("{p}expert.{e}.w1.weight");
    let w2 = format!("{p}expert.{e}.w2.weight");
    let w3 = format!("{p}expert.{e}.w3.weight");
    if tensors.contains_key(&w1) && tensors.contains_key(&w2) && tensors.contains_key(&w3) {
        return Some(ExpertFfnNames {
            up: format!("expert.{e}.w3"),
            gate: format!("expert.{e}.w1"),
            down: format!("expert.{e}.w2"),
        });
    }
    None
}

/// Discover expert ids present for a layer (sorted).
pub fn discover_expert_ids(tensors: &HashMap<String, MudTensor>, layer_idx: usize) -> Vec<u16> {
    let prefix = format!("blk.{layer_idx}.expert.");
    let mut ids = Vec::new();
    for name in tensors.keys() {
        if let Some(rest) = name.strip_prefix(&prefix) {
            // rest: "{id}.w1.weight" or "{id}.up.weight"
            if let Some((id_str, _)) = rest.split_once('.') {
                if let Ok(id) = id_str.parse::<u16>() {
                    if !ids.contains(&id) {
                        ids.push(id);
                    }
                }
            }
        }
    }
    ids.sort_unstable();
    ids
}

/// Count experts across all layers (max id+1 style capacity).
pub fn model_expert_stats(tensors: &HashMap<String, MudTensor>, n_layers: usize) -> (usize, usize) {
    let mut max_per_layer = 0usize;
    let mut multi_layers = 0usize;
    for l in 0..n_layers {
        let n = discover_expert_ids(tensors, l).len();
        if n > 1 {
            multi_layers += 1;
        }
        max_per_layer = max_per_layer.max(n);
    }
    (max_per_layer, multi_layers)
}

fn tensor_ptr_u8(tensors: &HashMap<String, MudTensor>, key: &str) -> *const u8 {
    tensors
        .get(key)
        .filter(|t| t.t_type == MudTensorType::Ternary2Bit || !t.data_ptr.is_null())
        .map(|t| t.data_ptr)
        .unwrap_or(std::ptr::null())
}

fn tensor_ptr_f32(tensors: &HashMap<String, MudTensor>, key: &str) -> *const f32 {
    tensors
        .get(key)
        .map(|t| t.data_ptr as *const f32)
        .unwrap_or(std::ptr::null())
}

fn weight_key(prefix: &str, base: &str) -> String {
    format!("{prefix}{base}.weight")
}
fn scale_key(prefix: &str, base: &str) -> String {
    format!("{prefix}{base}.prq_scale")
}

/// Build one expert from mud tensors (non-owning pointers).
pub fn load_expert(
    tensors: &HashMap<String, MudTensor>,
    layer_idx: usize,
    expert_id: u16,
    hidden: usize,
    ffn_mid: usize,
) -> Option<SlimeExpert> {
    let names = resolve_expert_ffn_names(tensors, layer_idx, expert_id)?;
    let p = format!("blk.{layer_idx}.");
    let up_w = tensor_ptr_u8(tensors, &weight_key(&p, &names.up));
    let gate_w = tensor_ptr_u8(tensors, &weight_key(&p, &names.gate));
    let down_w = tensor_ptr_u8(tensors, &weight_key(&p, &names.down));
    let up_s = tensor_ptr_f32(tensors, &scale_key(&p, &names.up));
    let gate_s = tensor_ptr_f32(tensors, &scale_key(&p, &names.gate));
    let down_s = tensor_ptr_f32(tensors, &scale_key(&p, &names.down));
    let e = SlimeExpert::from_ptrs(
        expert_id, hidden, ffn_mid, up_w, gate_w, down_w, up_s, gate_s, down_s,
    );
    if e.is_valid() {
        Some(e)
    } else {
        None
    }
}

/// Optional router weights for a layer.
pub fn load_router_ptrs(
    tensors: &HashMap<String, MudTensor>,
    layer_idx: usize,
) -> Option<(*const u8, *const f32)> {
    let p = format!("blk.{layer_idx}.");
    let candidates = [
        (
            format!("{p}moe_router.weight"),
            format!("{p}moe_router.prq_scale"),
        ),
        (format!("{p}gate.weight"), format!("{p}gate.prq_scale")),
    ];
    for (wk, sk) in candidates {
        let w = tensor_ptr_u8(tensors, &wk);
        let s = tensor_ptr_f32(tensors, &sk);
        if !w.is_null() && !s.is_null() {
            return Some((w, s));
        }
        // Dense f32 router without scale
        if let Some(t) = tensors.get(&wk) {
            if !t.data_ptr.is_null() {
                // Use dummy scales of ones? Skip — need ternary for current GEMV
                let _ = t;
            }
        }
    }
    None
}

/// Load ExpertBus for one layer. Returns None if no expert tensors found.
pub fn load_layer_bus(
    tensors: &HashMap<String, MudTensor>,
    layer_idx: usize,
    hidden: usize,
    ffn_mid: usize,
    top_k: usize,
) -> Option<ExpertBus> {
    let mut ids = discover_expert_ids(tensors, layer_idx);
    if ids.is_empty() {
        return None;
    }
    // Env: clone expert.0 into N slots for multi-expert experiments on dense models
    if let Some(n) = std::env::var("MUD_MOE_CLONE")
        .ok()
        .and_then(|s| s.parse::<u16>().ok())
    {
        if n > 1 && ids == [0] {
            ids = (0..n.min(MAX_EXPERT_SLOTS as u16)).collect();
        }
    }

    let cap = (*ids.iter().max().unwrap_or(&0) as usize + 1).clamp(1, MAX_EXPERT_SLOTS);
    let mut bus = ExpertBus::with_capacity(cap, top_k.max(1));
    bus.hidden = hidden;
    bus.ffn_mid = ffn_mid;

    for &id in &ids {
        // Clone path: all slots point at expert.0 weights
        let src_id = if std::env::var("MUD_MOE_CLONE").is_ok()
            && discover_expert_ids(tensors, layer_idx) == [0]
        {
            0
        } else {
            id
        };
        if let Some(expert) = load_expert(tensors, layer_idx, src_id, hidden, ffn_mid) {
            let mut e = expert;
            e.id = id;
            let _ = bus.mount(id, e);
        }
    }

    if bus.mounted_count() == 0 {
        return None;
    }

    if let Some((rw, rs)) = load_router_ptrs(tensors, layer_idx) {
        bus.set_router(rw, rs);
        bus.mode = RouterMode::Softmax;
    } else if bus.mounted_count() > 1 {
        bus.mode = RouterMode::Hash; // parameter-free multi-expert
    }

    Some(bus)
}

/// Load one bus per transformer layer (None if layer has no experts).
pub fn load_model_buses(
    mud: &MudFile,
    n_layers: usize,
    hidden: usize,
    ffn_mid: usize,
    top_k: usize,
) -> Vec<Option<ExpertBus>> {
    let Some(core) = mud.skills.get("core") else {
        return (0..n_layers).map(|_| None).collect();
    };
    (0..n_layers)
        .map(|l| load_layer_bus(&core.tensors, l, hidden, ffn_mid, top_k))
        .collect()
}

/// Env helpers for train-expert.
pub fn train_expert_id() -> Option<u16> {
    std::env::var("MUD_TRAIN_EXPERT")
        .ok()
        .and_then(|s| s.parse().ok())
}

/// Top-k from env (default 2).
pub fn default_top_k() -> usize {
    std::env::var("MUD_MOE_TOP_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2)
        .clamp(1, 8)
}

/// Whether multi-expert MoE forward should be used (more than one mounted expert any layer).
pub fn model_has_multi_expert(buses: &[Option<ExpertBus>]) -> bool {
    buses
        .iter()
        .filter_map(|b| b.as_ref())
        .any(|b| b.mounted_count() > 1)
}

/// FFN base names for dense SlimeLayer (expert 0, `MUD_TRAIN_EXPERT`, or stream G step expert).
pub fn dense_ffn_names_for_train(
    tensors: &HashMap<String, MudTensor>,
    layer_idx: usize,
) -> ExpertFfnNames {
    let p = format!("blk.{layer_idx}.");
    // Dense Llama/Qwen/Bonsai: blk.N.ffn_{up,gate,down}.weight (no expert.* path)
    let dense_up = format!("{p}ffn_up.weight");
    let dense_gate = format!("{p}ffn_gate.weight");
    let dense_down = format!("{p}ffn_down.weight");
    if tensors.contains_key(&dense_up)
        && tensors.contains_key(&dense_gate)
        && tensors.contains_key(&dense_down)
    {
        return ExpertFfnNames {
            up: "ffn_up".into(),
            gate: "ffn_gate".into(),
            down: "ffn_down".into(),
        };
    }
    // Stream G: multi-expert round-robin fixes expert for the whole step via begin_step.
    let eid = train_expert_id().unwrap_or_else(crate::mud::moe_train::current_step_expert);
    resolve_expert_ffn_names(tensors, layer_idx, eid).unwrap_or(ExpertFfnNames {
        up: format!("expert.{eid}.w3"),
        gate: format!("expert.{eid}.w1"),
        down: format!("expert.{eid}.w2"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mud::MudTensorType;

    fn fake_tensor(name: &str) -> (String, MudTensor) {
        (
            name.to_string(),
            MudTensor {
                name: name.to_string(),
                t_type: MudTensorType::Ternary2Bit,
                shape: vec![8, 8],
                data_ptr: std::ptr::null(),
                offset: 0,
                data_base: 0,
                mmap: None,
                owned_data: None,
            },
        )
    }

    fn map_with(names: &[&str]) -> HashMap<String, MudTensor> {
        names.iter().map(|n| fake_tensor(n)).collect()
    }

    #[test]
    fn test_discover_single_expert() {
        let m = map_with(&[
            "blk.0.expert.0.w1.weight",
            "blk.0.expert.0.w2.weight",
            "blk.0.expert.0.w3.weight",
        ]);
        assert_eq!(discover_expert_ids(&m, 0), vec![0]);
    }

    #[test]
    fn test_discover_multi_expert() {
        let m = map_with(&[
            "blk.2.expert.0.w1.weight",
            "blk.2.expert.0.w2.weight",
            "blk.2.expert.0.w3.weight",
            "blk.2.expert.3.w1.weight",
            "blk.2.expert.3.w2.weight",
            "blk.2.expert.3.w3.weight",
            "blk.2.expert.1.up.weight",
        ]);
        assert_eq!(discover_expert_ids(&m, 2), vec![0, 1, 3]);
    }

    #[test]
    fn test_resolve_w1w2w3() {
        let m = map_with(&[
            "blk.0.expert.0.w1.weight",
            "blk.0.expert.0.w2.weight",
            "blk.0.expert.0.w3.weight",
        ]);
        let n = resolve_expert_ffn_names(&m, 0, 0).unwrap();
        assert_eq!(n.gate, "expert.0.w1");
        assert_eq!(n.up, "expert.0.w3");
        assert_eq!(n.down, "expert.0.w2");
    }

    #[test]
    fn test_resolve_up_gate() {
        let m = map_with(&[
            "blk.1.expert.2.up.weight",
            "blk.1.expert.2.gate.weight",
            "blk.1.expert.2.w2.weight",
        ]);
        let n = resolve_expert_ffn_names(&m, 1, 2).unwrap();
        assert_eq!(n.up, "expert.2.up");
        assert_eq!(n.gate, "expert.2.gate");
        assert_eq!(n.down, "expert.2.w2");
    }

    #[test]
    fn test_stats() {
        let m = map_with(&[
            "blk.0.expert.0.w1.weight",
            "blk.0.expert.1.w1.weight",
            "blk.1.expert.0.w1.weight",
        ]);
        let (max_e, multi) = model_expert_stats(&m, 2);
        assert_eq!(max_e, 2);
        assert_eq!(multi, 1);
    }

    #[test]
    fn test_dense_ffn_names_default_w3_up() {
        let m = map_with(&[
            "blk.0.expert.0.w1.weight",
            "blk.0.expert.0.w2.weight",
            "blk.0.expert.0.w3.weight",
        ]);
        let n = dense_ffn_names_for_train(&m, 0);
        // Historical bug: up=w1 gate=w3. Canonical: up=w3 gate=w1.
        assert_eq!(n.up, "expert.0.w3");
        assert_eq!(n.gate, "expert.0.w1");
        assert_eq!(n.down, "expert.0.w2");
    }

    #[test]
    fn test_dense_ffn_names_qwen_bonsai() {
        let m = map_with(&[
            "blk.0.ffn_up.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_down.weight",
        ]);
        let n = dense_ffn_names_for_train(&m, 0);
        assert_eq!(n.up, "ffn_up");
        assert_eq!(n.gate, "ffn_gate");
        assert_eq!(n.down, "ffn_down");
    }

    #[test]
    fn test_dense_ffn_names_train_expert_env() {
        let m = map_with(&[
            "blk.0.expert.0.w1.weight",
            "blk.0.expert.0.w2.weight",
            "blk.0.expert.0.w3.weight",
            "blk.0.expert.2.w1.weight",
            "blk.0.expert.2.w2.weight",
            "blk.0.expert.2.w3.weight",
        ]);
        // SAFETY: test-only env mutation; serial test threads recommended.
        std::env::set_var("MUD_TRAIN_EXPERT", "2");
        let n = dense_ffn_names_for_train(&m, 0);
        std::env::remove_var("MUD_TRAIN_EXPERT");
        assert_eq!(n.up, "expert.2.w3");
        assert_eq!(n.gate, "expert.2.w1");
        assert_eq!(n.down, "expert.2.w2");
    }
}
