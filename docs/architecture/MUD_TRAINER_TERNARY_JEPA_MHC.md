# MUD Trainer: Ternary + JEPA + mHC — Anatomy & Verification

**Date:** 2026-07-17
**Scope:** `src/mud/corpus_trainer.rs`, `src/mud/slime_jepa.rs`, `src/mud/slime_forward.rs`,
`src/mud/slime_backward.rs`, `src/mud/ezop.rs`, `forge_autograd/src/avx_math.rs`.
**Verified by:** live short-run telemetry (`mud_train_metrics.log`) on `models/smollm2.mud`
(SmolLM2, 147M params) + `weights/checkpoints/model_latest_checkpoint.mud` (Bonsai 1.7B).

> This document describes the **actual implemented** compound, not aspirational design.
> Claims marked OK were confirmed in code AND in a running training session.

---

## 1. The Compound in One Picture

```
token ->[ELUT emb]-> h0
                    |
   for each blk l (full forward; BWD only last-N):
     |- QKV GEMV (ELUT 4-bit x PRQ scale) -> Attn -> O -> o_act
     |- JEPA stabilizer  --> z_attn, v_jepa_attn, gate_attn   -- (no weights)
     |- mHC residual     --> alpha.h + (1-gate).beta.f_h, norm<=radius -- (alpha,beta,radius FIXED)
     |- FFN RMSNorm -> SwiGLU (ELUT) -> ffn_out
     |- JEPA stabilizer  --> z_ffn, v_jepa_ffn, gate_ffn      -- (no weights)
     `- mHC residual     --> alpha.h + (1-gate).beta.f_h, norm<=radius -- (alpha,beta,radius FIXED)
                    |
   output_norm -> final_x -> Sampled Softmax(1 target + N rand neg) -> CE loss
                    |
   BWD: EZOP zero-alloc -> shape-dispatch optimizer (Muon/GaLore/Chunked/Adam/SGD)
                    |
   STE pack: shadow f32 -> pack_elut_prq -> ELUT 4-bit + PRQ scale (re-quantize)
```

Four stacked mechanisms:

| # | Mechanism | Role | Trainable weights? |
|---|-----------|------|--------------------|
| 1 | **Ternary (ELUT 4-bit)** | physical weight storage: codes {-1,0,+1}, 2/nibble, PRQ f32 scale/row | OK shadow f32 updated by optimizer, re-packed each step |
| 2 | **JEPA** (OU stabilizer) | activation manifold stabilization; produces gate=sigma(v_jepa) | NO controlador, no params. Backward term ~= 0 when v_jepa~=0 |
| 3 | **mHC** (hyper-connections) | dynamic residual blend alpha.h+(1-gate).beta.f_h, norm-bounded by radius | NO alpha,beta,radius are *const f32 read-only; never in grads |
| 4 | **Sampled Softmax** | loss: 1 target + N random negatives (rejection sampling, no hard mining) | n/a (loss fn) |

---

## 2. Ternary (ELUT 4-bit) — the substrate

- Weights stored as nibbles: 0x1=+1, 0x0=0, 0xF=-1. 2 weights/byte.
- Per-row f32 PRQ scale (absmean x 1/sqrt2, floor EPSILON_FLOOR).
- Train path: unpack_ternary2bit_to_f32 -> shadow Vec<f32> -> optimizer step ->
  ezop::pack_elut_prq (raw pointers, threshold scale*0.7) -> re-pack + update scale.
- matmul_accum is f32 (P-02). GEMV: AVX2x8 (dot_product_avx2) on CPU, Vulkan optional.

## 3. JEPA — Ornstein-Uhlenbeck stabilizer (slime_jepa.rs:148)

Per block, twice (post-attn, post-ffn):

```
y_norm  = zscore(block_out)              # mean 0, RMS 1
z_next  = (1-alpha).z + alpha.y_norm + eps  # OU, alpha=0.01 (+ tiny jitter, anti-collapse)
v_jepa  = (z - mu_ctx) . inv_sigma_ctx     # centered gate signal
I[t]    = 0.99.I[t-1] + 0.01.v_jepa        # integral low-pass
gate    = sigma(v_jepa)                    # written to each register
```

mu_ctx, inv_sigma_ctx are EMA-tracked (0.9/0.1, 0.99/0.01). inv_sigma_ctx is tanh-bounded
by sqrt(hidden) (slime_jepa.rs:214).

Backward coupling (slime_backward.rs:511):
```
spring_force = v_jepa
kinetic_grad = -2.lambda.spring_force     # lambda = kinetic_lambda = 0.005
gb[i]        = g + kinetic_grad           # branch gradient
go[i]        = g                           # residual gradient UNCHANGED
```
JEPA is not a layer with weights; it is a controller. Its only gradient influence is a
kinetic spring term scaled by lambda=0.005 x v_jepa.

## 4. mHC — Manifold-Constrained Hyper-Connections (slime_forward.rs:538)

```
val = alpha.h_in + (1 - gate).beta.f_h
if ||h|| > radius: scale h down to radius   # radius = sqrt(hidden) (or per-layer tensor)
```
- gate=sigma(v_jepa) from JEPA. If gate~=0 -> plain residual alpha.h+beta.f. If gate~=1 ->
  f_h suppressed, h_in dominates.
- alpha, beta, radius come from model tensors (blk.l.mhc_alpha/beta/radius). Read-only in
  forward; absent from SlimeLayerGradients -> never updated by optimizer.

---

## 5. Verification Against Live Telemetry

Short runs (2-10 chunks, SmolLM2 147M; 4 chunks Bonsai 1.7B). Summary of observed signals:

### 5.1 Ternary substrate — OK healthy
| Metric | Observed | Verdict |
|--------|----------|---------|
| sigma (ternary distribution %) | 48-51% | OK no collapse (BUG-6 avoided) |
| weight byte-diff before/after | changed | OK assimilation works (STE pack writes) |
| varh (activation std) | 1.70-1.82 | OK activations alive, no DEAD_ACT |

### 5.2 JEPA — WARN passive (by design, not bug)
| Metric | Observed | Verdict |
|--------|----------|---------|
| varj (var of jepa_z) | 0.07-0.08 (flat) | JEPA latent barely moves |
| jepa integral | +-0.03 around 0 | v_jepa~=0 -> gate~=0.5 constant |
| kinetic_grad in BWD | -2*0.005*~0 ~= 0 | JEPA contributes ~0 gradient |

Conclusion: JEPA sits at v_jepa~=0 -> gate~=0.5 -> mHC reduces to
0.85.h + 0.5*0.15.f (static attenuated residual). The OU tracker needs many more steps
than a 10-chunk smoke test to leave v_jepa~=0 (alpha=0.01 is deliberately slow). JEPA therefore
acts as a structural regularizer (keeps ||h|| bounded, prevents magnitude blow-up),
NOT as a learning engine in short runs.

### 5.3 mHC — OK bounded, WARN not adaptive in short run
- radius=sqrt(hidden) binding keeps activations finite (consistent with stable varh).
- Because gate~=0.5 and alpha/beta are fixed, mHC does not adapt; it is a fixed residual
  geometry. No gradient flows to alpha/beta/radius.

### 5.4 Loss & convergence — depends on optimizer budget
| Config | Loss trajectory | tok/s (AVX2x8, Iris Xe) |
|--------|-----------------|--------------------------|
| SGD, 16 steps/chunk, 31 neg | 6.3->5.5->5.5->3.6->4.7->3.0->6.1->5.7->3.0->5.8 (noise) | ~3.2 |
| Adam, 32 steps/chunk, 63 neg | 9.99->4.30->4.25->3.93->5.30->3.69->3.77->3.68 (down) | ~1.9 |
| Bonsai 1.7B, auto, 16 steps | 3.8->6.1->6.8->6.2 (up, <1 tok/s) | 0.4-1.1 |

Conclusion: Real learning is carried by the optimizer on last-N layers, not by the
JEPA/mHC composite. With enough steps/chunk (>=32) + Adam, loss falls. With SGD + 16 steps,
gradient noise dominates and loss oscillates. On 1.7B the throughput (<1 tok/s) makes a full
epoch impractical on 15 GiB hosts.

---

## 6. Bugs Found & Fixed During This Review

1. tok/s telemetry spike (16000 on first chunk). toks_per_sec = chunks_per_sec x steps
   used the session-average elapsed, giving a spurious peak on chunk 1.
   Fix: measure per-chunk wall-clock (chunk_t0/chunk_dt) in corpus_trainer.rs;
   fall back to average only if chunk was instantaneous. OK verified: now ~3.2 tok/s stable.
2. AWAKE-01 default ON (SGD on synthetic noise, all layers) - wasted CPU/RAM on
   low-resource hosts. Fix: default OFF; opt-in MUD_TRAIN_AWAKE=1. OK
3. distill_workflow stub - loaded full emb, then return Ok(()) without training.
   Fix: removed; --distill falls back to run_alignment_session. OK
4. new_train.rs - dead duplicate train_on_sequence. Fix: deleted. OK
5. clippy deny errors (raw-pointer unsafe fn, PI constant) - fixed to keep P-06. OK

---

## 7. Recommendations

- Default --align to Adam + MUD_TRAIN_STEPS_PER_CHUNK>=32. SGD/16-steps produces
  non-converging noise (sec 5.4). The config that actually learns should be the default.
- Telemetry hygiene: drop dead columns (sat zent tsoft align dedt |v2| are always 0.0)
  and rename misleading ones (conf_pct = exp(-loss) is not prediction accuracy;
  cognitive = mean|h|*100 is not cognition).
- JEPA/mHC are regularizers, not learners. If the goal is for JEPA to modulate training,
  either (a) train longer so v_jepa leaves 0, or (b) expose alpha/beta/radius (or a JEPA gain)
  as optimizer parameters. As-is, they only prevent magnitude collapse.
- Throughput on 1.7B is the real blocker (<1 tok/s). For 15 GiB hosts, keep last-N small
  and consider SCALES_ONLY (updates PRQ scales, freezes ELUT codes) to cut pack cost.

---

## 8. File Map

| Concern | File |
|---------|------|
| Training loop / session | src/mud/corpus_trainer.rs (run_alignment_session, train_on_sequence, apply_optimizer_cpu_step_and_pack) |
| JEPA OU stabilizer | src/mud/slime_jepa.rs (jepa_stabilizer) |
| mHC residual blend | src/mud/slime_forward.rs (mhc_residual) |
| Forward block | src/mud/slime_forward.rs (evaluate_slime_block) |
| Backward (EZOP + JEPA spring) | src/mud/slime_backward.rs (backward_slime_block) |
| ELUT pack / SGD (AVX2x8) | src/mud/ezop.rs, forge_autograd/src/avx_math.rs |
| Optimizer dispatch | src/mud/optimizer.rs (via select_optimizer in slime_backward.rs) |
| Vulkan QAT (optional) | src/mud/ash_qat_dispatcher.rs |
| STP trajectory loss (Phase 2) | src/mud/stp_loss.rs |

---

## 9. Update 2026-07-17 — mHC & JEPA made adaptive (Phase 1 + Phase 2)

Sections 5.2/5.3/7 above described the **passive** compound (α/β fixed, JEPA
`v_jepa≈0`). That is now superseded: the two recommendations in §7 ("expose
α/β as optimizer params" / "give JEPA a real objective") are **implemented and
verified**. Plan: `docs/research/MUD_PLAN_MHC_STP_TRAINABLE.md`.

### 9.1 Phase 1 — trainable mHC α/β (zero inference cost)
- `mhc_alpha.weight` / `mhc_beta.weight` are dense f32 params trained by SGD.
- Grad (constant-scale approx through the norm clamp):
  `dL/dα[i] = g·h_in[i]`, `dL/dβ[i] = g·(1-gate[i])·f_h[i]`.
- Wired in **both** the CPU pack path and the Vulkan (`ash=on`) path
  (α/β stay CPU-resident; ash only owns ternary GEMV weights). Clamp `[0,4]`, WD=0.
- Entry points: `slime_backward.rs` (`SlimeLayerGradients.mhc_{alpha,beta}_grad`,
  grads in `backward_slime_block`), `slime_forward.rs` (tape capture of `h_in`/`f_h`),
  `corpus_trainer.rs::mhc_scale_sgd_step`.

### 9.2 Phase 2 — STP trajectory loss (train-only, `MUD_TRAIN_STP=1`, default OFF)
- Semantic Tube Prediction (arXiv:2602.22617). For `s<r<t`:
  `L_STP = 1 − cos(h_r−h_s, h_t−h_s)` — 0 iff `h_r` is on the geodesic `s→t`.
  (Plan's `proj_parallel` form was degenerate; corrected in `stp_loss.rs`.)
- Hook: per-window ring of the top-of-stack residual (`pre_norm_x`). At each step
  (position `t`) a random past pair `s<r` is sampled and `λ·∂L_STP/∂h_t` is added
  into the backward seed `grad_in` (= `∂L/∂pre_norm_x`). Stochastic estimator;
  every position gets STP signal as it becomes `t`. `λ` via `MUD_TRAIN_STP_LAMBDA`
  (default 0.05). Zero alloc / AVX2 dot / TLS scratch (P-00/P-01).

### 9.3 Validation (2026-07-17, SmolLM2 147M, i7-1260P, `ash=on`)
| Check | Result |
|-------|--------|
| `cargo test --lib` | **218 passed**, 0 failed (215 base + 3 STP + 2 mHC already counted) |
| `cargo clippy --all-targets` | 0 warnings in STP/mHC code (exit 0); 28 pre-existing P-06 debt untouched |
| mHC finite-diff test | `test_mhc_scale_grad_matches_finite_diff` OK (<1e-3) |
| STP finite-diff + geodesic-zero | `test_stp_{grad_matches_finite_diff,zero_on_geodesic,positive_off_geodesic}` OK |
| α/β writeback | `blk.22/29.mhc_{alpha,beta}.weight` diverge per-element from 0.85/0.15; `cmp` DIFFERs; persists on save |
| STP overhead | tok/s 2.9 OFF vs 2.9 ON (≈0, well under 5%) |
| NTP not traded | loss step1 identical OFF/ON (4.9779); trajectory ~unchanged |
| L_STP | logged + bounded (~0.51), mild descent at higher λ; no NaN |
| Integrated F1+F2 run | loss 4.98→3.41, α/β changed, L_STP active, σ≈49%, no NaN |

**Net:** mHC is now adaptive and JEPA's role is superseded by an explicit STP
objective — both at zero inference cost. Phase 3 (mHC `n=2`, +inference memory)
remains optional/deferred per the plan.

---

## 10. Update 2026-07-17 — Unified trainer console (UI)

The end-to-end trainer output was a patchwork of two boxes (banner in
`run_trainer.rs` + "Architecture" box in `corpus_trainer.rs`), emoji banners
(`📊🆕✅🌀🚑💾`), and mixed stderr/stdout `[RAM]/[STP]/[quick]` notes. Consolidated:

- **Single source of geometry:** `src/mud/trainer_ui.rs` — `box_top/box_title/
  box_section/box_kv/note/phase` (W=80, INNER=78).
- `run_trainer.rs` prints one slim header (Hardware + Session + Corpus); the
  full Training Configuration box lives in `corpus_trainer.rs` (single authority).
- All setup/teardown chatter uses `note(kind, msg)` → uniform indented `[ok]/
  [ram]/[stp]/[warn]/[err]` tags. `[TELEM]` stays on stderr (machine-readable).
- Progress line simplified to `Speed: N tk/s` (dropped the bogus `ops/s` label).
- Validated: `cargo clippy --all-targets` 0 new warnings in trainer/UI code;
  218 tests pass; short STP-on run renders cleanly.

### 10.1 Project-adapted `mud.sh train` default
`./mud.sh train` now (see `mud.sh` `build_project_corpus` / `compute_project_chunks`):
- Assembles `training/corpus/project_corpus.txt` from ES/EN align text + repo
  `*.md` (AGENTS/GEMINI/README/docs) + project `*.rs` (src/forge_autograd/tools),
  mtime-gated so the AOT cache is reused across runs.
- Defaults (override via env): `MUD_TRAIN_STP=1`, `MUD_TRAIN_STEPS_PER_CHUNK=64`,
  `MUD_TRAIN_NUM_NEG=255`, `MUD_OPT=adam`, `MUD_TRAIN_TEXT_ONLY=1`.
- `MUD_TRAIN_MAX_CHUNKS` auto-sized to cover the project corpus (capped at 64) so a
  bare `./mud.sh train` is a sane, bounded, project-aware session.

