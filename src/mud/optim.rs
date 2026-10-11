//! SPRINT-4: Optimizer strategy module (leaf — no deps on `corpus_trainer` or `slime_backward`).
//!
//! This module **breaks the dependency cycle** between `corpus_trainer` and
//! `slime_backward` by collecting:
//!   - `OptimizerStrategy` enum
//!   - `select_optimizer` + `optimizer_policy_override`
//!   - `mhc_scale_sgd_step` (SGD for the mHC scale vectors)
//!   - `apply_optimizer_cpu_step_and_pack` (full optimizer dispatcher + STE pack)
//!   - `update_prq_scales_only` + `apply_sgd_shadow_update` (private helpers)
//!
//! Both `slime_backward` and `corpus_trainer` now depend on this module, but this
//! module depends on neither. The graph becomes:
//!
//! ```text
//!   slime_backward  ─┐
//!                     ├─►  optim  (no outgoing edges to trainer/backward)
//!   corpus_trainer  ─┘
//! ```
//!
//! Also: a [`Optimizer`] trait is exposed as the OCP-friendly extension path. The
//! enum remains for back-compat with `SlimeLayerShadowF32` field types; future
//! extension can wrap `Box<dyn Optimizer>` inside the variants without breaking
//! the public API.

#![allow(clippy::too_many_arguments)]

/// SGD weight clamp maximum (prevents explosion on unstable gradients).
/// Replaces the magic number `1e6` previously inline in `SgdOptimizer::step`.
pub const SGD_CLAMP_MAX: f32 = 1e6;

use crate::mud::adam_state::AdamState;
use crate::mud::muon;

// ── OptimizerStrategy enum (data-only, back-compat) ─────────────────────────

/// Strategy for the per-layer optimizer dispatcher. The variants are still
/// data-only; the runtime behavior is implemented by matching on this enum in
/// `apply_optimizer_cpu_step_and_pack`.
///
/// Sprint-4 / 4.2: future extension path is to convert variants to wrap
/// `Box<dyn Optimizer>` so adding a new optimizer (Lion, Shampoo) becomes 1
/// struct + 1 enum variant + 1 dispatch arm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OptimizerStrategy {
    /// Plain SGD (no NS / GaLore) — default for quick smoke trains.
    Sgd,
    Muon { ns_iters: usize },
    GaLore { rank: usize, update_freq: usize },
    ChunkedAdam { chunk_cols: usize },
    SparseAdam { only_active_rows: bool },
    Adam,
}

impl OptimizerStrategy {
    /// Stable string identifier for telemetry and env round-tripping.
    pub fn name(&self) -> &'static str {
        match self {
            OptimizerStrategy::Sgd => "sgd",
            OptimizerStrategy::Muon { .. } => "muon",
            OptimizerStrategy::GaLore { .. } => "galore",
            OptimizerStrategy::ChunkedAdam { .. } => "chunked_adam",
            OptimizerStrategy::SparseAdam { .. } => "sparse_adam",
            OptimizerStrategy::Adam => "adam",
        }
    }
}

// ── Optimizer trait (OCP-friendly extension path) ───────────────────────────

/// Strategy interface. Implementors are stateless w.r.t. the weight matrix
/// (state lives in `AdamState` keyed by matrix id in the caller). `step`
/// is the only required method; the default `needs_moments` returns false
/// for pure-SGD-style optimizers.
pub trait Optimizer: Send + Sync {
    fn name(&self) -> &'static str;
    fn step(
        &self,
        w: &mut [f32],
        g: &mut [f32],
        lr: f32,
        weight_decay: f32,
    ) -> anyhow::Result<()>;
    fn needs_moments(&self) -> bool {
        false
    }
}

/// Plain SGD: `w -= lr * (g + wd * w)`.
pub struct SgdOptimizer;
impl Optimizer for SgdOptimizer {
    fn name(&self) -> &'static str {
        "sgd"
    }
    fn step(&self, w: &mut [f32], g: &mut [f32], lr: f32, wd: f32) -> anyhow::Result<()> {
        for (wi, gi) in w.iter_mut().zip(g.iter()) {
            let upd = lr * (*gi + wd * *wi);
            *wi = (*wi - upd).clamp(-SGD_CLAMP_MAX, SGD_CLAMP_MAX);
        }
        Ok(())
    }
}

// M-3 PURGE (2026-08-03): `AdamWOptimizer` removed — compiled, exported via the
// `Optimizer` trait, but never instantiated from any production call site
// (the live optimizer path is `OptimizerStrategy::Adam` → `adam_step` in
// adam_state.rs). `parking_lot::Mutex` usage removed with it (P-08).

// ── Policy: env var override + shape dispatch ───────────────────────────────

/// Global optimizer policy from env (applied before shape dispatch).
///
/// `MUD_OPT` / `MUD_TRAIN_OPT`:
/// - `sgd` — always SGD
/// - `adam` — always Adam
/// - `muon` — always Muon (ns from [`crate::mud::muon::muon_ns_iters`])
/// - `galore` — always GaLore
/// - unset — shape dispatch; if `MUD_TRAIN_MAX_CHUNKS` set → SGD (smoke default)
pub fn optimizer_policy_override() -> Option<OptimizerStrategy> {
    let raw = std::env::var("MUD_OPT")
        .or_else(|_| std::env::var("MUD_TRAIN_OPT"))
        .unwrap_or_default();
    let key = raw.trim().to_ascii_lowercase();
    match key.as_str() {
        "sgd" | "plain" => Some(OptimizerStrategy::Sgd),
        "adam" => Some(OptimizerStrategy::Adam),
        "muon" => Some(OptimizerStrategy::Muon {
            ns_iters: muon::muon_ns_iters(),
        }),
        "galore" => Some(OptimizerStrategy::GaLore {
            rank: 64,
            update_freq: 100,
        }),
        "" => {
            // Quick sessions: skip expensive Muon NS unless user forced MUD_OPT
            if std::env::var("MUD_TRAIN_MAX_CHUNKS")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .filter(|&m| m > 0)
                .is_some()
            {
                Some(OptimizerStrategy::Sgd)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Shape-based optimizer dispatch (used by `SlimeLayerShadowF32::new`).
pub fn select_optimizer(rows: usize, cols: usize) -> OptimizerStrategy {
    if let Some(over) = optimizer_policy_override() {
        return over;
    }
    let ratio = rows as f32 / cols as f32;
    let total = rows * cols;
    let ns = muon::muon_ns_iters();

    if total < 100_000 && (0.5..=2.0).contains(&ratio) {
        OptimizerStrategy::Muon { ns_iters: ns }
    } else if (0.5..=2.0).contains(&ratio) {
        // BitNet-2B size square matrices also use Muon
        OptimizerStrategy::Muon { ns_iters: ns }
    } else if ratio > 2.5 {
        let rank = (cols / 4).max(8);
        OptimizerStrategy::GaLore {
            rank,
            update_freq: 100,
        }
    } else if ratio < 0.4 {
        let chunk = 512;
        OptimizerStrategy::ChunkedAdam { chunk_cols: chunk }
    } else if total > 10_000_000 {
        OptimizerStrategy::SparseAdam {
            only_active_rows: true,
        }
    } else {
        OptimizerStrategy::Adam
    }
}

// ── mHC scale SGD step ──────────────────────────────────────────────────────

/// Phase 1 (mHC trainable): in-place dense SGD step on a Float32 hyper-connection scale
/// vector (`alpha` or `beta`), with token-mean grad scaling and a hard clamp to keep the
/// residual geometry bounded (HC paper uses tanh; a clamp is simpler + STE-friendly).
///
/// # Safety
/// `w` must point to `grad.len()` valid, writable `f32` (the model's owned mHC tensor data).
pub unsafe fn mhc_scale_sgd_step(w: *mut f32, grad: &[f32], lr: f32, inv_tok: f32) {
    if w.is_null() {
        return;
    }
    for (i, &g) in grad.iter().enumerate() {
        let g = if g.is_finite() { g * inv_tok } else { 0.0 };
        let p = unsafe { w.add(i) };
        let updated = (*p - lr * g).clamp(0.0, 4.0);
        *p = updated;
    }
}

// ── Private helpers (used by `apply_optimizer_cpu_step_and_pack`) ────────────

/// SGD-style shadow update (shared by all strategies after optional grad transform).
/// L-09: AVX2 when available, else EZOP scalar raw pointers.
fn apply_sgd_shadow_update(
    shadow_w: &mut [f32],
    grad_w: &[f32],
    lr: f32,
    weight_decay: f32,
    num_tokens: f32,
) {
    let n = shadow_w.len().min(grad_w.len());
    if n == 0 {
        return;
    }
    if core_arch_x86_64_has_avx2() {
        unsafe {
            forge_autograd::avx_math::sgd_step_avx2(
                &mut shadow_w[..n],
                &grad_w[..n],
                lr,
                weight_decay,
                num_tokens,
            );
        }
    } else {
        // SAFETY: n elements of each slice.
        unsafe {
            crate::mud::ezop::sgd_step(
                shadow_w.as_mut_ptr(),
                grad_w.as_ptr(),
                n,
                lr,
                weight_decay,
                num_tokens,
            );
        }
    }
}

fn core_arch_x86_64_has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Update PRQ scales only: keep existing ELUT codes, fit s per row to shadow.
///
/// For each row: `s = Σ(W·T) / Σ(T²)` with T ∈ {-1,0,+1} from packed ELUT.
/// Then project shadow `W ← s·T` so next STE step stays on the ternary manifold.
///
/// # Safety
/// `packed_ptr` / `scales_ptr` must cover `rows` of ELUT (cols/8 u32s per row) and `rows` f32.
unsafe fn update_prq_scales_only(
    shadow_w: &mut [f32],
    packed_ptr: *mut u8,
    scales_ptr: *mut f32,
    rows: usize,
    cols: usize,
) {
    if rows == 0 || cols == 0 || shadow_w.len() < rows * cols {
        return;
    }
    let u32s_per_row = cols.div_ceil(8);
    let packed_u32 = packed_ptr as *const u32;
    for r in 0..rows {
        let row_off = r * cols;
        let mut dot = 0.0f32;
        let mut t2 = 0.0f32;
        for c in 0..cols {
            let word = *packed_u32.add(r * u32s_per_row + c / 8);
            let nibble = (word >> ((c % 8) * 4)) & 0xF;
            let t = if nibble == 0x1 {
                1.0f32
            } else if nibble == 0xF {
                -1.0f32
            } else {
                0.0f32
            };
            let w = shadow_w[row_off + c];
            dot += w * t;
            t2 += t * t;
        }
        let s = if t2 > crate::mud::constants::EPSILON_FLOOR {
            (dot / t2).abs().max(crate::mud::constants::EPSILON_FLOOR)
        } else {
            // Fallback absmean of shadow if row is all zeros in codes
            let mut abs_sum = 0.0f32;
            for c in 0..cols {
                abs_sum += shadow_w[row_off + c].abs();
            }
            (abs_sum / cols as f32).max(crate::mud::constants::EPSILON_FLOOR)
        };
        // Preserve sign of least-squares scale if anti-correlated (rare):
        // W≈s·T with T∈{-1,0,+1}, so a negative `dot` means the shadow is
        // anti-correlated with the frozen codes and the signed scale must stay
        // negative (the forward applies `out = w * scale`, scale carries sign).
        let s_signed = if t2 > crate::mud::constants::EPSILON_FLOOR && dot < 0.0 {
            -s
        } else {
            s
        };
        // Clamp magnitude into a sane PRQ range; keep sign.
        let mag = s_signed
            .abs()
            .clamp(crate::mud::constants::EPSILON_FLOOR, 1.0f32);
        *scales_ptr.add(r) = if s_signed < 0.0 { -mag } else { mag };
        let s_use = *scales_ptr.add(r);
        // Project shadow onto frozen codes so magnitude tracks scales for next step
        for c in 0..cols {
            let word = *packed_u32.add(r * u32s_per_row + c / 8);
            let nibble = (word >> ((c % 8) * 4)) & 0xF;
            let t = if nibble == 0x1 {
                1.0f32
            } else if nibble == 0xF {
                -1.0f32
            } else {
                0.0f32
            };
            shadow_w[row_off + c] = t * s_use;
        }
    }
}

// ── Dispatcher: apply strategy + STE pack to ELUT 4-bit ──────────────────────

/// Apply optimizer according to [`OptimizerStrategy`], then STE-pack to ELUT 4-bit.
///
/// **L-01 + P0:** Muon/GaLore/Chunked preprocess → SGD; **Adam / SparseAdam** use real
/// moments via [`AdamState`] (`adam_step_avx2` when available).
/// **L-09:** EZOP TLS grad scratch.
///
/// # Safety
///
/// `packed_ptr` must point to a writable buffer of at least
/// `shadow_w.len()` ELUT 4-bit nibbles (`(shadow_w.len() + 1) / 2` bytes);
/// the caller guarantees it is valid for the duration of the call and the
/// `grad_w`/`shadow_w` slices are non-overlapping and properly aligned.
#[allow(clippy::too_many_arguments)]
pub unsafe fn apply_optimizer_cpu_step_and_pack(
    shadow_w: &mut [f32],
    grad_w: &[f32],
    packed_ptr: *mut u8,
    scales_ptr: *mut f32,
    lr: f32,
    weight_decay: f32,
    num_tokens: f32,
    cols: usize,
    pool: &crate::mud::pcore_pool::PCorePool,
    strategy: OptimizerStrategy,
    adam: Option<&mut AdamState>,
) {
    let cols = cols.max(1);
    let n = shadow_w.len();
    let rows = n / cols;
    if rows == 0 || n != rows * cols || grad_w.len() < n {
        return;
    }

    // L-09: reuse TLS scratch instead of grad_w.to_vec() every call (P-01).
    crate::mud::ezop::with_grad_scratch(n, |grad| {
        // SAFETY: grad len n; grad_w at least n.
        unsafe {
            crate::mud::ezop::copy_f32(grad.as_mut_ptr(), grad_w.as_ptr(), n);
            crate::mud::ezop::sanitize_f32(grad.as_mut_ptr(), n);
        }
        crate::mud::adam_state::scale_grad_by_tokens(grad, num_tokens);

        match strategy {
            OptimizerStrategy::Sgd => {
                apply_sgd_shadow_update(shadow_w, grad, lr, weight_decay, 1.0);
            }
            OptimizerStrategy::Muon { ns_iters } => {
                muon::newton_schulz_orthogonalize(grad, rows, cols, ns_iters);
                apply_sgd_shadow_update(shadow_w, grad, lr, weight_decay, 1.0);
            }
            OptimizerStrategy::GaLore {
                rank,
                update_freq: _,
            } => {
                crate::mud::galore::galore_step(grad, rows, cols, rank.max(1));
                apply_sgd_shadow_update(shadow_w, grad, lr, weight_decay, 1.0);
            }
            OptimizerStrategy::ChunkedAdam { chunk_cols } => {
                let chunk = chunk_cols.max(1);
                let ns = muon::muon_ns_iters();
                if cols > chunk {
                    muon::chunked_muon_step(grad, rows, cols, chunk, ns);
                } else {
                    muon::newton_schulz_orthogonalize(grad, rows, cols, ns);
                }
                apply_sgd_shadow_update(shadow_w, grad, lr, weight_decay, 1.0);
            }
            OptimizerStrategy::Adam => {
                if let Some(st) = adam {
                    crate::mud::adam_state::adam_step(shadow_w, grad, st, lr, weight_decay, 1.0);
                    st.store_shadow_bf16(shadow_w);
                } else {
                    // No state allocated — safe SGD fallback
                    apply_sgd_shadow_update(shadow_w, grad, lr, weight_decay, 1.0);
                }
            }
            OptimizerStrategy::SparseAdam { only_active_rows } => {
                if let Some(st) = adam {
                    crate::mud::adam_state::sparse_adam_step(
                        shadow_w,
                        grad,
                        st,
                        rows,
                        cols,
                        lr,
                        weight_decay,
                        1.0,
                        only_active_rows,
                    );
                    st.store_shadow_bf16(shadow_w);
                } else {
                    apply_sgd_shadow_update(shadow_w, grad, lr, weight_decay, 1.0);
                }
            }
        }

        // D2: Store bf16 shadow snapshot of updated weights (memory reduction).
        // Integrated into each Adam-family arm because `adam` is moved by `if let`.
        // Only active when MUD_BF16_SHADOW=1; reuses buffer on subsequent calls.

        // ── Shadow magnitude guard (P1 fix, 2026-07-20) ───────────────────────
        // Prevent PRQ-scale inflation: an unbounded shadow (high LR / weak decay)
        // makes per-row absmean → large `s`, so W = s·T explodes (observed 27× on
        // last layers → logit domination → vocabulary collapse). Clamp each element
        // to ±K·row_absmax before pack. K default 8 (env MUD_TRAIN_WCLAMP_K, 0=off).
        {
            let k = std::env::var("MUD_TRAIN_WCLAMP_K")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(8.0);
            if k > 0.0 {
                // FIX (2026-08-03): Vectorized WCLAMP — replaces scalar per-element
                // sum + clamp with AVX2 sum_abs (8 f32/iter) + clamp_finite (8/iter).
                // For a typical 128x128 matrix: 128K scalar ops → 16K vector ops (~8×).
                let p = shadow_w.as_mut_ptr();
                let floor = crate::mud::constants::EPSILON_FLOOR;
                for r in 0..rows {
                    let base = (r * cols) as isize;
                    // AVX2 sum-of-abs (8 f32/iter, 4-accumulator unroll)
                    let abs_sum = if cols >= 8 && std::is_x86_feature_detected!("avx2") {
                        unsafe { crate::mud::ezop::sum_abs_avx2(p.offset(base), cols) }
                    } else {
                        let mut s = 0.0f32;
                        for c in 0..cols {
                            s += unsafe { (*p.offset(base + c as isize)).abs() };
                        }
                        s
                    };
                    let bound = (abs_sum / cols as f32 * k).max(floor);
                    // AVX2 clamp (8 f32/iter, NaN→0, clamp to [-bound, bound])
                    if cols >= 8 && std::is_x86_feature_detected!("avx2") {
                        unsafe {
                            crate::mud::ezop::clamp_finite_avx2(
                                p.offset(base),
                                -bound,
                                bound,
                                0.0,
                                cols,
                            );
                        }
                    } else {
                        for c in 0..cols {
                            let off = base + c as isize;
                            let v = unsafe { *p.offset(off) };
                            unsafe { *p.offset(off) = v.clamp(-bound, bound) };
                        }
                    }
                }
            }
        }

        // ── Scales-only path (Bonsai / native ternary seating) ─────────────────
        // Freeze ELUT trit codes; update PRQ scales so W≈s·T matches post-SGD shadow.
        // Avoids re-thresholding {-1,0,+1} that destroys native low-bit structure.
        let scales_only = std::env::var("MUD_TRAIN_SCALES_ONLY")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if scales_only {
            unsafe {
                update_prq_scales_only(shadow_w, packed_ptr, scales_ptr, rows, cols);
            }
            return;
        }

        // STE pack: serial `pack_elut_prq` is faster than PCorePool for post-step pack
        // (avoids 30L×7 task-queue roundtrips; pool stays free for next GEMV).
        // Optional parallel pack: MUD_TRAIN_PACK_POOL=1
        let use_pack_pool = std::env::var("MUD_TRAIN_PACK_POOL")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if use_pack_pool && rows >= 64 && pool.num_threads() > 1 {
            let n_tasks = pool.num_threads().max(1);
            let rows_per_task = (rows / n_tasks).max(1);
            let shadow_ptr = shadow_w.as_ptr() as usize;
            let packed_p = packed_ptr as usize;
            let scales_p = scales_ptr as usize;
            for i in 0..n_tasks {
                let start_row = i * rows_per_task;
                let end_row = if i + 1 == n_tasks {
                    rows
                } else {
                    start_row + rows_per_task
                };
                if start_row >= end_row {
                    break;
                }
                let nrows = end_row - start_row;
                pool.execute(move || {
                    let sw = (shadow_ptr as *const f32).wrapping_add(start_row * cols);
                    let sc = (scales_p as *mut f32).wrapping_add(start_row);
                    let pk = packed_p as *mut u8;
                    unsafe {
                        // Pack row range into global buffer (absolute nibble indices)
                        for r in 0..nrows {
                            let abs_row = start_row + r;
                            let row_start = abs_row * cols;
                            let mut abs_sum = 0.0f32;
                            for c in 0..cols {
                                abs_sum += (*sw.add(r * cols + c)).abs();
                            }
                            let s = ((abs_sum / cols as f32) * std::f32::consts::FRAC_1_SQRT_2)
                                .max(crate::mud::constants::EPSILON_FLOOR);
                            *sc.add(r) = s;
                            let threshold = s * crate::mud::constants::SPARSITY_THRESHOLD_RATIO;
                            for c in 0..cols {
                                let idx = row_start + c;
                                let v = *sw.add(r * cols + c);
                                let bit = if v > threshold {
                                    0x1u8
                                } else if v < -threshold {
                                    0xFu8
                                } else {
                                    0x0u8
                                };
                                let byte_idx = idx / 2;
                                let nibble_pos = (idx % 2) * 4;
                                let current = *pk.add(byte_idx);
                                let mask = !(0xFu8 << nibble_pos);
                                *pk.add(byte_idx) = (current & mask) | (bit << nibble_pos);
                            }
                        }
                    }
                });
            }
            pool.wait_all();
        } else {
            // SAFETY: shadow/scales/packed valid for full matrix after optimizer step.
            unsafe {
                crate::mud::ezop::pack_elut_prq(
                    shadow_w.as_ptr(),
                    rows,
                    cols,
                    scales_ptr,
                    packed_ptr,
                );
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_strategy_name() {
        assert_eq!(OptimizerStrategy::Sgd.name(), "sgd");
        assert_eq!(OptimizerStrategy::Adam.name(), "adam");
        assert_eq!(
            OptimizerStrategy::Muon { ns_iters: 5 }.name(),
            "muon"
        );
    }

    #[test]
    fn test_select_optimizer_square_small_picks_muon() {
        let opt = select_optimizer(128, 128);
        assert!(matches!(opt, OptimizerStrategy::Muon { .. }));
    }

    #[test]
    fn test_select_optimizer_tall_picks_galore() {
        let opt = select_optimizer(8192, 128);
        assert!(matches!(opt, OptimizerStrategy::GaLore { .. }));
    }

    #[test]
    fn test_select_optimizer_wide_picks_chunked_adam() {
        let opt = select_optimizer(128, 8192);
        assert!(matches!(opt, OptimizerStrategy::ChunkedAdam { .. }));
    }

    #[test]
    fn test_sgd_optimizer_step() {
        let mut w = vec![1.0f32, 2.0, 3.0];
        let mut g = vec![0.1f32, -0.2, 0.3];
        let sgd = SgdOptimizer;
        sgd.step(&mut w, &mut g, 0.1, 0.0).unwrap();
        // w[i] -= 0.1 * g[i]
        assert!((w[0] - 0.99).abs() < 1e-6);
        assert!((w[1] - 2.02).abs() < 1e-6);
        assert!((w[2] - 2.97).abs() < 1e-6);
    }

    #[test]
    fn test_mhc_scale_sgd_step_clamps() {
        let mut w = vec![0.0f32; 4];
        let grad = vec![100.0f32, -100.0, 0.0, 5.0];
        // lr=0.1, inv_tok=1.0  →  w[0] = 0 - 10 = -10 → clamped to 0
        //                        w[1] = 0 + 10 = 10  → clamped to 4
        //                        w[2] = 0 (grad=0)
        //                        w[3] = 0 - 0.5 = -0.5 → clamped to 0
        unsafe { mhc_scale_sgd_step(w.as_mut_ptr(), &grad, 0.1, 1.0) };
        assert_eq!(w[0], 0.0);
        assert_eq!(w[1], 4.0);
        assert_eq!(w[2], 0.0);
        assert_eq!(w[3], 0.0);
    }
}
