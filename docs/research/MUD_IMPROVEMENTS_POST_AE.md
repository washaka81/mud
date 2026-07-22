# MUD Improvements Research — Post Streams A–E (2026-07-16)

> **Context:** Ledger L-01…L-15 closed; depth streams **A–K** + orbit **F–L** largely landed.  
> **SSOT:** `GEMINI.md` · launch: `docs/manuals/LAUNCH_COUNTDOWN.md`

---

## 1. Closed streams (do not re-plan)

| Stream | Outcome | Code / tools |
|--------|---------|--------------|
| **A** Adam moments | Real m/v; SparseAdam skips zero rows | `adam_state.rs` |
| **B** MoE product path | Load experts; train-expert dense FFN | `moe_load.rs` |
| **C** GEMV auto | Profiled break-even | `gemv_policy.rs`, `gemv_auto_bench` |
| **D** Full-seq train | Causal windows | `sequence_pack` + trainer |
| **E** CSA top-k | Lightning rank over HCA | `csa_indexer.rs` |
| **F** QKV one CB | 3 dispatches / 1 fence | `dispatch_gemv_qkv_host_sync` |
| **K** Loss cert | Trajectory gate + CI unit tests | `loss_cert.rs`, `cert-loss` |
| **G** Multi-expert STE | Round-robin expert rebind per step | `moe_train.rs`, `MUD_MOE_TRAIN=1` |
| **H** Long full-seq | Auto L-15 ckpt + residual bank API | `maybe_enable_grad_ckpt_*`, `recompute_from_residual_bank` |
| **I** KV f16 | IEEE half packs for dense+HCA | `kv_dtype.rs`, `MUD_KV_DTYPE=f16` |
| **J** CSA LSH v2 | SimHash prefilter before top-k | `MUD_CSA_LSH=1` |
| **L** Converter P-13 | Canonical aliases + auditor | `ensure_canonical_metadata_aliases` |

Validate: `./mud.sh ci` · `cargo test --lib` · `./mud.sh audit-full`.

---

## 2. Remaining depth (honest / research)

| Item | Notes |
|------|-------|
| Sparse MoE **multi-expert backward** in one graph (weighted sum of expert grads) | G+ trains **top-1** hash/round-robin expert dense STE; not simultaneous multi-expert STE |
| Learned `W_compress` CSA in `.mud` | J SimHash is fixed hyperplanes |
| True Google BF16 | I uses IEEE f16 |
| Nightly e2e `cert-loss` on smollm2 | Unit synthetic 20-step gate in CI; e2e via `MUD_CI_LOSS_CERT=1` |
| Exotic converter sources without tokenizer.json | May still lack `tokenizer.tokens` |
| **C-MUD × log-gas** (JHEP CFT ideas) | `cmud.rs` kernel done (L-14); map Dyson circular-ensemble / contour-rotation to mHC + thinking loop. Feasibility: `docs/research/CMUD_LOGGAS_FEASIBILITY.md`. **DEFERRED to future-study gate** (25-epoch train clean + circuit honors + mHC stability observed). Experiments E1–E5 only after reliability gate. P-02-safe (opt-in). |

---

## 3. Env map (product knobs)

| Area | Vars |
|------|------|
| Optimizers | `MUD_USE_VULKAN` (Muon NS GPU) |
| MoE | `MUD_TRAIN_EXPERT`, `MUD_MOE_CLONE`, `MUD_MOE_TOP_K`, **`MUD_MOE_TRAIN=1\|hash`** |
| GEMV | `MUD_GPU_GEMV=auto\|0\|1`, `MUD_GPU_GEMV_MIN` |
| Full-seq | `MUD_TRAIN_FULL_SEQ`, `MUD_TRAIN_SEQ_LEN` (try **128**), `MUD_TRAIN_CKPT_SEQ_THR` |
| Grad ckpt | `MUD_GRAD_CKPT`, `MUD_GRAD_CKPT_SEG`, **`MUD_GRAD_CKPT_RESIDUAL=1`** (residual+JEPA bank) |
| CSA | `MUD_CSA`, `MUD_CSA_TOP_K`, `MUD_CSA_INDEX_DIM`, **`MUD_CSA_LSH=1`**, `MUD_CSA_LSH_BITS`, `MUD_CSA_LSH_RADIUS` |
| KV | **`MUD_KV_DTYPE=f32\|f16\|bf16`** |
| Loss cert | `MUD_LOSS_CERT_FAST`, `MUD_CI_LOSS_CERT=1` |
| Context | `MUD_MAX_POS`, `MUD_HCA_WINDOW`, `MUD_HCA_RATIO` |

---

## 4. Suggested next work (research / scale)

```
1. Multi-expert weighted STE (train all top-k experts with route weights) — **FOUNDATION (2026-07-20):** `weighted_expert_deltas` en `moe_train.rs` + `test_weighted_expert_deltas`. Aún no cableado al trainer vivo (para no regresionar G+ top-1).
2. Learned CSA W_compress tensors in .mud
3. Nightly job: cert-loss --fast smollm2
4. Measure f16 KV vs f32 quality @ 8k–32k
5. C-MUD log-gas spike: E1 phase-repulsion + E3 contour-rotation probe (see CMUD_LOGGAS_FEASIBILITY.md)
```

---

## 5. Changelog

| Date | Note |
|------|------|
| 2026-07-16 | Initial post–A–E backlog (F–L). |
| 2026-07-16 | **F DONE** QKV one CB. |
| 2026-07-16 | **K DONE** loss_cert. |
| 2026-07-16 | **G·H·I·J·L** foundation implementations landed. |
| 2026-07-16 | **H residual-bank wired** in train (save/restore residual+JEPA). |
| 2026-07-16 | **G+ hash route** `MUD_MOE_TRAIN=hash` + util log. |
| 2026-07-20 | **G+ SlimeX PoC** `ShadowExpertBus` and `SlimeXSlot` implemented zero-alloc style in `slime_x.rs`. Ready for wiring into trainer. |
