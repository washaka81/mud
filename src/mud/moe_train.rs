//! # Stream G: Multi-expert STE train policy
//!
//! Dense train (stream B) remaps FFN to a single `MUD_TRAIN_EXPERT`.
//! This module adds **multi-expert** product training:
//!
//! | Mode (`MUD_MOE_TRAIN`) | Behaviour |
//! |------------------------|-----------|
//! | unset / `0` | Classic single-expert dense (B) |
//! | `1` / `round_robin` | Cycle expert id each train step across discovered experts |
//! | `all` | Same as round_robin but forces clone discovery via stats |
//!
//! Joint router STE (true sparse MoE backward) remains a follow-up; this ships
//! multi-expert **weight updates** + utilization telemetry usable with
//! `MUD_MOE_CLONE` multi-slot buses.

use crate::mud::moe_load::{discover_expert_ids, train_expert_id};
use crate::mud::MudTensor;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use std::cell::Cell;

thread_local! {
    /// Expert id fixed for all layers within one train step.
    static STEP_EXPERT: Cell<Option<u16>> = const { Cell::new(None) };
}

static STEP_COUNTER: AtomicU64 = AtomicU64::new(0);
static HIT_COUNTS: [AtomicU64; 16] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeTrainMode {
    Off,
    /// Cycle expert 0..N-1 each call to [`next_train_expert`].
    RoundRobin,
    /// G+: hash-route on activation → train top-1 expert (router-free STE target).
    Hash,
}

/// Parse `MUD_MOE_TRAIN`.
pub fn moe_train_mode() -> MoeTrainMode {
    match std::env::var("MUD_MOE_TRAIN") {
        Err(_) => MoeTrainMode::Off,
        Ok(v) => {
            let t = v.trim().to_ascii_lowercase();
            match t.as_str() {
                "" | "0" | "false" | "off" | "no" => MoeTrainMode::Off,
                "hash" | "route" | "joint" | "2" => MoeTrainMode::Hash,
                "1" | "true" | "on" | "yes" | "round_robin" | "rr" | "all" => {
                    MoeTrainMode::RoundRobin
                }
                _ => MoeTrainMode::RoundRobin,
            }
        }
    }
}

pub fn moe_train_enabled() -> bool {
    !matches!(moe_train_mode(), MoeTrainMode::Off)
}

/// Expert ids present on layer 0 (or union across layers if empty).
pub fn discover_train_expert_pool(
    tensors: &HashMap<String, MudTensor>,
    n_layers: usize,
) -> Vec<u16> {
    let mut ids = discover_expert_ids(tensors, 0);
    if ids.is_empty() && n_layers > 1 {
        for l in 1..n_layers {
            for id in discover_expert_ids(tensors, l) {
                if !ids.contains(&id) {
                    ids.push(id);
                }
            }
        }
        ids.sort_unstable();
    }
    if ids.is_empty() {
        ids.push(0);
    }
    ids
}

/// Next expert id for dense FFN remapping under multi-expert train.
///
/// Priority:
/// 1. Explicit `MUD_TRAIN_EXPERT` always wins (B path).
/// 2. Else if `MUD_MOE_TRAIN` round-robin → cycle pool.
/// 3. Else expert 0.
pub fn next_train_expert(pool: &[u16]) -> u16 {
    if let Some(e) = train_expert_id() {
        record_hit(e);
        return e;
    }
    if !moe_train_enabled() || pool.is_empty() {
        record_hit(0);
        return 0;
    }
    let step = STEP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let e = pool[(step as usize) % pool.len()];
    record_hit(e);
    e
}

/// Begin a train step: pick one expert for **all** layers until [`end_step`].
pub fn begin_step(pool: &[u16]) -> u16 {
    let e = next_train_expert(pool);
    STEP_EXPERT.with(|c| c.set(Some(e)));
    e
}

/// G+: begin step with **hash routing** on `x` (token activation / STE emb).
/// Maps router slot indices into `pool` ids; trains top-1.
///
/// Returns `(primary_expert, route: [(expert_id, weight), ...])`.
pub fn begin_step_hash(pool: &[u16], x: &[f32], top_k: usize) -> (u16, Vec<(u16, f32)>) {
    if let Some(e) = train_expert_id() {
        record_hit(e);
        STEP_EXPERT.with(|c| c.set(Some(e)));
        return (e, vec![(e, 1.0)]);
    }
    if pool.is_empty() {
        record_hit(0);
        STEP_EXPERT.with(|c| c.set(Some(0)));
        return (0, vec![(0, 1.0)]);
    }
    let n = pool.len();
    let router = crate::mud::routing::MudRouter::new(n, top_k.max(1).min(n));
    let mut results = Vec::with_capacity(top_k.max(1));
    router.route_by_hash(x, &mut results);
    let mut route: Vec<(u16, f32)> = results
        .into_iter()
        .filter_map(|(slot, w)| {
            if slot < n {
                Some((pool[slot], w))
            } else {
                None
            }
        })
        .collect();
    if route.is_empty() {
        route.push((pool[0], 1.0));
    }
    // Normalize weights
    let sum: f32 = route.iter().map(|(_, w)| *w).sum();
    if sum > 0.0 {
        for r in route.iter_mut() {
            r.1 /= sum;
        }
    }
    let primary = route[0].0;
    record_hit(primary);
    // Also credit secondary hits for util telemetry (fractional not stored — full hit)
    for &(eid, _) in route.iter().skip(1) {
        record_hit(eid);
    }
    STEP_EXPERT.with(|c| c.set(Some(primary)));
    let _ = STEP_COUNTER.fetch_add(1, Ordering::Relaxed);
    (primary, route)
}

/// Expert fixed for current train step (or 0).
pub fn current_step_expert() -> u16 {
    STEP_EXPERT.with(|c| c.get().unwrap_or(0))
}

/// Clear step-local expert (end of pair/window).
pub fn end_step() {
    STEP_EXPERT.with(|c| c.set(None));
}

fn record_hit(eid: u16) {
    let i = eid as usize;
    if i < HIT_COUNTS.len() {
        HIT_COUNTS[i].fetch_add(1, Ordering::Relaxed);
    }
}

/// Snapshot utilization counts (expert_id → hits) for logging.
pub fn utilization_snapshot() -> Vec<(u16, u64)> {
    HIT_COUNTS
        .iter()
        .enumerate()
        .filter_map(|(i, a)| {
            let c = a.load(Ordering::Relaxed);
            if c > 0 {
                Some((i as u16, c))
            } else {
                None
            }
        })
        .collect()
}

/// Reset counters (tests / session start).
pub fn reset_utilization() {
    STEP_COUNTER.store(0, Ordering::Relaxed);
    for a in HIT_COUNTS.iter() {
        a.store(0, Ordering::Relaxed);
    }
}

/// One-line summary for trainer panel / health.
pub fn summary_line(pool: &[u16]) -> String {
    let util = utilization_snapshot();
    match moe_train_mode() {
        MoeTrainMode::Off => "MoE-train=off (single dense expert)".into(),
        MoeTrainMode::RoundRobin => {
            format!("MoE-train=round_robin pool={pool:?} util={util:?}")
        }
        MoeTrainMode::Hash => {
            format!("MoE-train=hash pool={pool:?} util={util:?}")
        }
    }
}

/// Dense FFN names for the expert selected by multi-train policy.
pub fn dense_ffn_names_for_moe_train(
    tensors: &HashMap<String, MudTensor>,
    layer_idx: usize,
    pool: &[u16],
) -> crate::mud::moe_load::ExpertFfnNames {
    let eid = next_train_expert(pool);
    crate::mud::moe_load::resolve_expert_ffn_names(tensors, layer_idx, eid).unwrap_or(
        crate::mud::moe_load::ExpertFfnNames {
            up: format!("expert.{eid}.w3"),
            gate: format!("expert.{eid}.w1"),
            down: format!("expert.{eid}.w2"),
        },
    )
}

/// F+ orbit #1 (research): **multi-expert weighted STE** building block.
///
/// `G+` currently trains a single top-1 expert per step (round-robin or hash).
/// True MoE trains the **top-k** experts jointly, weighting each expert's STE
/// update by its routing weight. This primitive computes, for `k` experts with
/// per-parameter gradients `expert_grads[j]` (all same length) and routing
/// weights `w` (need not be normalized), each expert's parameter delta:
///
/// ```text
/// delta_j[p] = lr * w_j * expert_grads[j][p]
/// ```
///
/// The trainer would route to top-k via [`begin_step_hash`] (`MUD_MOE_TOP_K>1`),
/// gather each expert's grad, and call this to fuse the weighted updates without
/// a single shared backward graph. Kept separate from the live top-1 path so the
/// proven trainer is untouched.
pub fn weighted_expert_deltas(
    expert_grads: &[Vec<f32>],
    weights: &[f32],
    lr: f32,
) -> Vec<Vec<f32>> {
    assert!(
        !expert_grads.is_empty(),
        "weighted_expert_deltas: need at least one expert grad"
    );
    let n = expert_grads[0].len();
    assert!(
        expert_grads.iter().all(|g| g.len() == n),
        "weighted_expert_deltas: all expert grads must share length"
    );
    assert_eq!(
        expert_grads.len(),
        weights.len(),
        "weighted_expert_deltas: grads/weights length mismatch"
    );
    expert_grads
        .iter()
        .zip(weights.iter())
        .map(|(g, &w)| g.iter().map(|&x| lr * w * x).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mud::{MudTensor, MudTensorType};
    use std::sync::Mutex;

    // Serialize tests that mutate the process-global expert counter / hit counts
    // (STEP_COUNTER, HIT_COUNTS). They otherwise race under cargo's parallel
    // test runner and produce flaky results.
    static EXPERT_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn fake(name: &str) -> (String, MudTensor) {
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

    #[test]
    fn test_round_robin_cycles() {
        let _g = EXPERT_TEST_LOCK.lock().unwrap();
        reset_utilization();
        unsafe {
            std::env::set_var("MUD_MOE_TRAIN", "1");
            std::env::remove_var("MUD_TRAIN_EXPERT");
        }
        let pool = vec![0u16, 1, 2];
        let a = next_train_expert(&pool);
        let b = next_train_expert(&pool);
        let c = next_train_expert(&pool);
        let d = next_train_expert(&pool);
        unsafe {
            std::env::remove_var("MUD_MOE_TRAIN");
        }
        assert_eq!([a, b, c, d], [0, 1, 2, 0]);
        let util = utilization_snapshot();
        assert!(util.iter().any(|(e, c)| *e == 0 && *c >= 2));
    }

    #[test]
    fn test_explicit_train_expert_wins() {
        let _g = EXPERT_TEST_LOCK.lock().unwrap();
        reset_utilization();
        unsafe {
            std::env::set_var("MUD_MOE_TRAIN", "1");
            std::env::set_var("MUD_TRAIN_EXPERT", "3");
        }
        let pool = vec![0u16, 1, 2, 3];
        assert_eq!(next_train_expert(&pool), 3);
        assert_eq!(next_train_expert(&pool), 3);
        unsafe {
            std::env::remove_var("MUD_MOE_TRAIN");
            std::env::remove_var("MUD_TRAIN_EXPERT");
        }
    }

    #[test]
    fn test_discover_pool() {
        let m: HashMap<_, _> = [
            fake("blk.0.expert.0.w1.weight"),
            fake("blk.0.expert.2.w1.weight"),
            fake("blk.0.expert.1.w1.weight"),
        ]
        .into_iter()
        .collect();
        assert_eq!(discover_train_expert_pool(&m, 1), vec![0, 1, 2]);
    }

    #[test]
    fn test_weighted_expert_deltas() {
        // F+ orbit #1: top-k experts weighted STE.
        let g0 = vec![1.0_f32, 2.0_f32];
        let g1 = vec![3.0_f32, 4.0_f32];
        let deltas = weighted_expert_deltas(&[g0, g1], &[0.7, 0.3], 0.1);
        // delta_j = lr * w_j * grad_j (approx, f32)
        assert!((deltas[0][0] - 0.07).abs() < 1e-6);
        assert!((deltas[0][1] - 0.14).abs() < 1e-6);
        assert!((deltas[1][0] - 0.09).abs() < 1e-6);
        assert!((deltas[1][1] - 0.12).abs() < 1e-6);
        // Sum of weighted deltas = lr * weighted sum of grads (shared-param view).
        let combined: Vec<f32> = deltas[0]
            .iter()
            .zip(deltas[1].iter())
            .map(|(a, b)| a + b)
            .collect();
        assert!((combined[0] - 0.16).abs() < 1e-6); // 0.1*(0.7*1+0.3*3)
        assert!((combined[1] - 0.26).abs() < 1e-6); // 0.1*(0.7*2+0.3*4)
    }

    #[test]
    fn test_hash_route_picks_pool_member() {
        let _g = EXPERT_TEST_LOCK.lock().unwrap();
        reset_utilization();
        unsafe {
            std::env::set_var("MUD_MOE_TRAIN", "hash");
            std::env::remove_var("MUD_TRAIN_EXPERT");
        }
        let pool = vec![0u16, 1, 2, 3];
        let x = vec![0.1f32; 64];
        let (primary, route) = begin_step_hash(&pool, &x, 2);
        unsafe {
            std::env::remove_var("MUD_MOE_TRAIN");
        }
        assert!(pool.contains(&primary));
        assert!(!route.is_empty());
        assert!(route.iter().all(|(e, w)| pool.contains(e) && *w > 0.0));
        assert_eq!(current_step_expert(), primary);
        end_step();
    }
}
