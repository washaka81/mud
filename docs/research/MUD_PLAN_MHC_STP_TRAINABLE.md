# MUD Plan: Make JEPA + mHC Contribute to Learning (Trainable mHC + STP loss)

**Date:** 2026-07-17
**Author:** agent (research-backed)
**Status:** Phase 1 DONE (mHC α/β trainable, verified). Phase 2 DONE
(`src/mud/stp_loss.rs` + `MUD_TRAIN_STP` hook, verified). Phase 3 optional/deferred.

**Default-on (2026-07-17):** `./mud.sh train` now sets `MUD_TRAIN_STP=1`,
`MUD_TRAIN_STEPS_PER_CHUNK=64`, `MUD_TRAIN_NUM_NEG=255`, `MUD_OPT=adam`, and
assembles a project-adapted corpus (`build_project_corpus`) — see `mud.sh`.
**Supersedes:** the informal "fix JEPA v_jepa" idea from the prior session.

> Goal: turn the currently *passive* JEPA + mHC compound (verified inert in
> `MUD_TRAINER_TERNARY_JEPA_MHC.md`) into an *active learning* contributor, with
> **zero inference cost**, respecting P-00/P-01/P-06/P-07/P-13/P-27 and the
> low-resource target (i7-1260P, ~15 GiB, Iris Xe).

---

## 0. Why (research grounding)

| Mechanism in MUD | Equivalent in literature | Measured benefit | Inference cost |
|---|---|---|---|
| `mhc_residual` = `alpha*h + (1-gate)*beta*f` | **Hyper-Connections** (ByteDance, ICLR 2025, arXiv:2409.19606) | 1.8x faster convergence, -0.027..-0.034 loss, ~50% fewer tokens, +6 ARC-C, **no loss spikes**, negligible cost | ~0 |
| JEPA OU stabilizer (currently inert) | **JEPA aux loss** (LLM-JEPA arXiv:2509.14252) / **STP** (arXiv:2602.22617) | LLM-JEPA: faster convergence, LoRA r512 ~ full-FT. STP: **match baseline with 16x less data**, negligible overhead | 0 (train-only) |
| Ternary ELUT + STE + RMSNorm | BitNet b1.58 / "Extra RMSNorm" (ACL 2025) | matches FP16 from 3B; stability from RMSNorm+STE+lambda schedule, low/zero weight-decay on shadow | 0 |

Key facts that shape this plan:
1. **HC with `n=1` does NOT beat baseline** (paper Fig.5). The big 1.8x needs `n>=2`.
   -> Phase 1 (`n=1` trainable) buys *stability + anti-collapse + zero-cost*, not the 1.8x.
   The 1.8x is Phase 3 (`n=2`), which costs inference memory — optional.
2. **`L_LLM` does NOT implicitly minimize `L_JEPA`** (LLM-JEPA Fig.3/4). A passive JEPA
   controller (our `v_jepa~=0`) therefore *cannot* learn. It needs its own gradient term.
3. **LLM-JEPA costs 2-3x compute** (extra forward passes) -> too expensive for 15 GiB.
   **STP costs ~0** (reuses `h_s,h_r,h_t` already in the forward) -> correct choice for MUD,
   and it is literally a manifold-trajectory regularizer = the project's "MUD" thesis.
4. Ternary substrate: **do not add exotic regularizers**; keep RMSNorm+STE. Consider
   **low/zero weight-decay on shadow weights** and **sum (not mean) loss reduction**.

---

## 1. Scope & Non-Goals

**In scope**
- Phase 1: `mhc_alpha` / `mhc_beta` become trainable via STE (dense f32 params).
- Phase 2: replace inert JEPA-OU gradient coupling with an **STP trajectory loss**.
- Phase 3 (optional, gated): mHC state expansion `n=2` for the 1.8x regime.

**Non-goals (this plan)**
- No `mhc_radius` training in Phase 1/2 (interacts with norm projection; instability risk).
- No LLM-JEPA two-view double-forward (violates low-resource budget).
- No change to ELUT wire format, GEMV kernels, or Vulkan path.
- No `n>=2` unless Phase 1+2 are green AND user opts in (inference memory cost).

---

## 2. Current truth (verified in code)

- `SlimeLayerGradients` (`slime_backward.rs:5`) has NO `mhc_*` fields.
- `mhc_alpha_w/beta_w/radius_w` are `*const f32` (`slime_forward.rs:62`), read-only.
- Forward mHC: `val = alpha[i]*h_in[i] + (1-gate[i])*beta[i]*f_h[i]` (`slime_forward.rs:564`).
  - post-attn call: `h_in=registers`, `f_h=o_act_f32`, out=`registers_tmp` (line 1185).
  - post-ffn call:  `h_in=registers_tmp`, `f_h=ffn_out_f32`, out=`registers` (line 1277).
- `gate[i] = registers[i].gate() = sigmoid(v_jepa)`; with `v_jepa~=0` -> `gate~=0.5`.
- Tape already stores `attn_v_jepa` and `ffn_v_jepa` (`slime_forward.rs:1164,1255`).
- Backward JEPA coupling: `kinetic_grad = -2*0.005*v_jepa ~= 0` (`slime_backward.rs:513`).

---

## 3. Phase 1 — Trainable mHC alpha/beta (zero inference cost)

**Rationale:** tensors `blk.N.mhc_alpha/beta` already exist physically (converter creates
them). We only need to (a) compute their gradient, (b) feed the optimizer, (c) pack back.

### 3.1 Math
Forward per element `i`:
```
val[i] = alpha[i]*h_in[i] + (1-gate[i])*beta[i]*f_h[i]
```
Given `grad_val[i]` (grad wrt mHC output, before norm-projection — see 3.4):
```
dL/dalpha[i] = grad_val[i] * h_in[i]
dL/dbeta[i]  = grad_val[i] * (1-gate[i]) * f_h[i]
```
`alpha,beta` are **dense f32** params (NOT ternary) — no ELUT pack, plain Adam.

### 3.2 Changes
1. `SlimeLayerGradients`: add `mhc_alpha_grad: Vec<f32>`, `mhc_beta_grad: Vec<f32>`
   (len `hidden`), init `vec![0.0; hidden]` in `new()`, reset in the zero/clear path.
2. Tape: ensure `h_in`, `f_h`, `gate` for BOTH mHC calls are available in backward.
   - `f_h`: post-attn = `o_act_f32`, post-ffn = FFN out. Add tape buffers if not present.
   - `gate`: store `1-gate[i]` (or `v_jepa`, recompute sigmoid) per mHC site.
   - `h_in`: reconstructable from existing register tape; add buffer if cheaper.
   (P-01: all tape buffers preallocated in `SlimeLayerTape::new`, zero alloc in hot loop.)
3. `backward_slime_block`: after computing `grad_val` for each mHC site, accumulate
   `mhc_alpha_grad`/`mhc_beta_grad` with raw-pointer loop (P-00), AVX2 optional later.
4. Optimizer wiring in `apply_optimizer_cpu_step_and_pack` (corpus_trainer.rs:3693):
   treat `mhc_alpha/beta` as a dense f32 tensor -> Adam moment update -> write back to the
   in-memory `.mud` tensor bytes directly (they are stored as f32, not ELUT).
5. Guard: if `mhc_alpha_w.is_null()` (base model) -> skip grad + skip step (no invention, P-13).

### 3.3 Init / stability
- If tensors exist but are all `1.0` (converter default), that's the residual identity — safe start.
- Clamp `alpha,beta` to e.g. `[0, 4]` after each step (HC paper uses `tanh` bounding;
  a hard clamp is simpler and STE-friendly). Prevents residual blow-up under ternary noise.
- Keep **weight-decay = 0** on alpha/beta (they are scales, not features).

### 3.4 Norm-projection caveat
`mhc_residual` applies a post-hoc norm clamp (`if norm>radius: scale`). For a correct
gradient we either (a) backprop through the clamp (piecewise: identity when `norm<=radius`,
scaled when `>`), or (b) approximate by treating the clamp as a constant scale for the step
(cheaper, biased but stable). **Decision:** start with (b) constant-scale approximation;
revisit if alpha/beta grads look wrong. Document the approximation inline.

### 3.5 Acceptance (Phase 1)
- `cargo clippy --all-targets` 0/0 (P-06); `cargo test --lib` green.
- New unit test: analytic grad of mHC vs finite-difference on random `h_in,f_h,gate` < 1e-3.
- 32-step Adam run on `models/smollm2.mud`:
  - `cmp` on `blk.N.mhc_alpha` bytes before/after: **changed**.
  - `alpha/beta` diverge from `1.0` (log min/max/mean per few layers).
  - loss **<=** baseline trajectory (no regression), no NaN, sigma% ~50% stable.

---

## 4. Phase 2 — Replace inert JEPA-OU with STP trajectory loss (zero inference cost)

**Rationale:** JEPA is currently a controller with `v_jepa~=0` and a `-2*lambda*v_jepa`
spring that contributes ~0 gradient. STP gives JEPA a *real* objective at ~0 cost, using
hidden states already computed in the full-seq forward.

### 4.1 STP objective (Semantic Tube Prediction, arXiv:2602.22617)
For three positions `s < r < t` in the sequence, with hidden states `h_s,h_r,h_t`:
```
d1 = h_r - h_s
d2 = h_t - h_s
L_STP = 1 - cos( d1, d2 )                       # keep h_r on the geodesic s->t
L_total = L_softmax + lambda_stp * L_STP        # lambda_stp ~ 0.05 (tune)
```
**Formula correction (impl):** the original `1 - cos(proj_parallel(d1,d2), d2)` is
degenerate — `proj_parallel(d1,d2)` is by construction parallel to `d2`, so its
cosine with `d2` is `sign(<d1,d2>)` (±1) with no usable gradient. The correct,
non-degenerate objective is `1 - cos(d1, d2)` (step aligned with chord); it is
zero exactly when `h_r` is on the segment `h_s -> h_t`. See `src/mud/stp_loss.rs`.

**Impl notes (as shipped):**
- `stp_loss_and_grad` returns dL/dh_{s,r,t}; TLS scratch, AVX2 dot, zero alloc.
- Hook: per-window ring of the top-of-stack residual (`pre_norm_x`, i.e.
  `matmul_accum` pre-output-norm). At each step (position = `t`) a random past
  pair `s<r` is sampled and `λ·dL/dh_t` is added into the backward seed `grad_in`
  (which is exactly `dL/dpre_norm_x`). `h_s,h_r` are frozen history that step —
  a stochastic estimator; over a window every position gets STP signal as `t`.
- Gated behind `MUD_TRAIN_STP=1` (default OFF); `MUD_TRAIN_STP_LAMBDA` (default 0.05).
- Verified smollm2: NTP loss unchanged (step1 identical off/on), tok/s 2.9 both
  (overhead ~0), L_STP logged+bounded, no NaN, `varj` loosens slightly.
- Identity predictor (local-linearity) => no predictor network, no extra params.
- `s,r,t` sampled randomly per step from the current full-seq window
  (`MUD_TRAIN_FULL_SEQ` path already gives us the sequence of `h`).
- Grad flows into the same last-N layers already trained; reuses tape.

### 4.2 Changes
1. New module `src/mud/stp_loss.rs`: `stp_loss_and_grad(h_s,h_r,h_t) -> (f32, grads)`,
   raw-pointer AVX2 dot/axpy from `forge_autograd::avx_math` (P-00/P-01, TLS scratch).
2. Hook in `train_on_sequence` (corpus_trainer.rs): after collecting per-position final
   hidden states, sample `k` triples, add `lambda_stp * grad` into the backward seed for
   those positions. Gate behind `MUD_TRAIN_STP=1` (default OFF first, then flip to default ON
   once verified — mirrors AWAKE-01 discipline).
3. **Repurpose JEPA's gate coupling:** feed `L_STP`-derived signal into `v_jepa` so the
   existing `gate=sigmoid(v_jepa)` in mHC becomes meaningful (JEPA integral tracks the
   trajectory-error, not a self-referential z-score). Alternatively keep JEPA-OU for gating
   and let STP act purely as loss — **Decision:** first ship STP as pure aux loss
   (decoupled from gate), measure, then optionally wire STP -> v_jepa in a follow-up.
4. Keep JEPA-OU stabilizer as-is for now (it still bounds magnitude); do NOT delete until
   STP is proven (P-08 applies only to confirmed-dead code).

### 4.3 Acceptance (Phase 2)
- Unit test: `L_STP=0` when `h_r` lies exactly on segment `h_s->h_t`; `>0` otherwise;
  analytic grad vs finite-diff < 1e-3.
- 64-step Adam run smollm2 with `MUD_TRAIN_STP=1` vs OFF:
  - softmax loss trajectory **no worse** (STP must not trade away NTP — paper confirms it
    doesn't); ideally faster descent or lower plateau.
  - `L_STP` decreases over steps (log it).
  - tok/s within ~5% of baseline (overhead must stay negligible; if not, cache triples).
- No NaN, clippy 0/0, tests green.

---

## 5. Phase 3 — mHC expansion n=2 (OPTIONAL, has inference cost)

Only if Phase 1+2 green AND user opts in. This is where the paper's 1.8x lives.
- Expand residual stream to `n=2` copies with a learnable `(n+1)x(n+1)` mix matrix per layer.
- **Inference cost:** +1x hidden-state memory + small matmul per layer. On 15 GiB / 1.7B
  this is significant -> must be measured (`gemv_auto_bench` style) before adoption.
- Deferred; separate plan doc when/if requested.

---

## 6. Order, risk, rollback

| Step | Deliverable | Risk | Inference cost | Rollback |
|------|-------------|------|----------------|----------|
| 1 | Trainable mHC alpha/beta | Low | 0 | grads unused if flag off / null tensors |
| 2 | STP aux loss (`MUD_TRAIN_STP`) | Med | 0 | env flag default OFF |
| 3 | mHC n=2 | High | +mem | separate build path |

- Each phase = its own commit with tests. Do NOT proceed to next until acceptance met.
- All new behavior behind env flags first (AWAKE-01 discipline), flip default only after proof.

## 7. What we expect to gain (summary for the user)

- **Training:** faster, spike-free convergence; fewer tokens for the same loss (decisive at
  1.9-3.2 tok/s on this hardware); reduced layer representation-collapse (`varj` unfreezes).
- **Inference:** **no cost** for Phase 1+2 (mHC uses existing `alpha,beta`; STP is train-only).
- **Alignment with MUD thesis:** STP = manifold trajectory regularizer = "Modular
  Understanding Dynamics" made literal.

## 8. Files to touch

| Phase | Files |
|-------|-------|
| 1 | `src/mud/slime_backward.rs` (grads+bwd), `src/mud/slime_forward.rs` (tape), `src/mud/corpus_trainer.rs` (optimizer wiring), tests |
| 2 | `src/mud/stp_loss.rs` (new), `src/mud/corpus_trainer.rs` (hook+flag), `forge_autograd/src/avx_math.rs` (reuse), tests |
| 3 | separate |

## 9. Open decisions to confirm before coding

1. Phase 1 norm-projection gradient: (b) constant-scale approx first — OK?
2. Phase 2: ship STP as **pure aux loss** first (decoupled from `gate`) — OK?
   (vs immediately wiring STP -> v_jepa -> gate.)
3. Keep JEPA-OU stabilizer alive during Phase 2 (delete only after STP proven) — OK?
4. Start with **Phase 1 only**, verify, then decide on Phase 2 — OK?
