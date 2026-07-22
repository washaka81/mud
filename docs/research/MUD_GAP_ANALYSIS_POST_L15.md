# MUD Gap Analysis — Post L-01…L-15 (2026-07-16)

## 1. What is LIVE

| Area | Status | Notes |
|------|--------|-------|
| L-01 Optimizer strategy in step | ✅ | Muon/GaLore/Chunked + **Adam moments (A)** |
| L-02 Newton-Schulz Vulkan | ✅ | Hybrid ash + CPU |
| L-03 InferenceWorkspace purge | ✅ | SlimeWorkspace only |
| L-04 ASM orphans | ✅ | 11 live kernels |
| L-05 Double-buffer QAT | ✅ | DoubleFrame + deferred readback |
| L-06 mha/rms_norm | ✅ | Dispatch + output_norm GPU threshold |
| L-07 P-13 pool/eps | ✅ | `default_pcore_threads`, EPSILON SSOT |
| L-08 NaN guards ASM | ✅ | non-finite → 0 |
| L-09 EZOP | ✅ | TLS grad scratch, pack path |
| L-10 Sequence packing | ✅ | Full-chunk pairs + **full-seq windows (D)** |
| L-11 Mini MoE ExpertBus | ✅ | + **`.mud` load / train-expert (B)** |
| L-12 CI + P-13 props | ✅ | `mud.sh ci`, GHA, `p13` |
| Phase B/B+ GEMV | ✅ | Shared-tile; **auto policy (C)** |
| L-13 HCA 32k-ready | ✅ | Dense ring + HCA + **CSA top-k (E)** |
| L-14 C-MUD kernel | ✅ | Research math; `MUD_CMUD_THINK=1` |
| L-15 Grad checkpoint | ✅ | `MUD_GRAD_CKPT=1` recompute on reverse |

**Tools:** `training_healthcheck`, `mud_full_audit`, `converter_auditor`, `gemv_auto_bench`, `./mud.sh ci|audit-full`.

**Next-wave backlog (F+):** [`MUD_IMPROVEMENTS_POST_AE.md`](./MUD_IMPROVEMENTS_POST_AE.md).

---

## 2. Gaps by priority

### P0 — Correctness / product readiness
1. ~~**Adam / SparseAdam real moments**~~ — **DONE 2026-07-16** (`adam_state`, `adam_step_avx2`, sparse rows).
2. ~~**MoE train path**~~ — **DONE 2026-07-16** (`moe_load`: discover/load buses from `.mud`, `MUD_TRAIN_EXPERT` dense FFN target, `MUD_MOE_CLONE` synthetic multi-expert, fixed w3=up/w1=gate in trainer). Remaining: full multi-expert STE for experts 1..N in one step (not expert-only dense).
3. **Checkpoint recompute cost** — L-15 recomputes prefix `0..end` (correct JEPA); segment residual-only path not yet used (future optimization).
4. ~~**Training is single-token (pos=0)**~~ — **DONE 2026-07-16** Stream D: causal windows (`MUD_TRAIN_FULL_SEQ` default on, `MUD_TRAIN_SEQ_LEN`); pos/RoPE/KV grow per step; grads accumulate then one optimizer step. True multi-token joint BPTT / full HCA stress still partial (window length capped; truncated online, not full unrolled graph).

### P1 — Performance
1. ~~**GPU GEMV default off**~~ — **DONE 2026-07-16** (`gemv_policy` auto). ~~QKV multi-matrix~~ **DONE (F)** one CB / 3 dispatches.
2. **GPU forward for full block** — only GEMV/RMSNorm/MHA pieces; not fused residual+JEPA.
3. **Shared-memory GEMV on CPU?** — already AVX2; further tile/prefetch (EZOP remaining loops).
4. **Parallel multi-expert MoE** — sequential weighted sum today.

### P2 — Scale / context
1. ~~**True CSA lightning indexer**~~ — **DONE 2026-07-16** (`csa_indexer`: coarse top-k over HCA + dense window; train tapes force full HCA). Remaining: learned W_compress, 1M-scale LSH.
2. **KV in bf16 / quantized cache** — still f32 dense ring + f32 HCA.
3. **Prefill batching** — decode-oriented workspace; multi-token prefills limited.

### P3 — Research / 2027
1. **Full C-MUD network** — complex GEMV ASM + complex weights in `.mud`.
2. **Federated / multi-node** — out of scope.
3. **Discrete GPU** — ash targets iGPU UMA; no CUDA.

### Tooling gaps
1. ~~Auto-benchmark suite comparing CPU vs `MUD_GPU_GEMV`~~ — **DONE** (`gemv_auto_bench` + auto policy).
2. ~~End-to-end “loss must decrease N steps”~~ — **DONE (K)** `loss_cert` + `cert-loss`; e2e optional `MUD_CI_LOSS_CERT=1`.
3. Converter → always emit complete P-13 key set (some models use alternate names only) — backlog **L**.

---

## 3. Recommended next workstreams

| Stream | Why | Effort |
|--------|-----|--------|
| ~~**A. Adam moments + Sparse rows**~~ | **DONE** | — |
| ~~**B. MoE .mud load + train-expert**~~ | **DONE** (`src/mud/moe_load.rs`; inference buses; trainer FFN names + `MUD_TRAIN_EXPERT`) | — |
| ~~**C. Profile GPU GEMV auto-on**~~ | **DONE** (`src/vulkan/gemv_policy.rs`; default auto; `gemv_auto_bench`) | — |
| ~~**D. Full-seq train (packed)**~~ | **DONE** (`seq_windows` + causal `pos` in `train_on_sequence`; default on) | — |
| ~~**E. CSA top-k indexer**~~ | **DONE** (`src/mud/csa_indexer.rs`; top-k∪tail over HCA; dense window full) | — |

### Stream D env knobs (full-seq train)
| Env | Effect |
|-----|--------|
| `MUD_TRAIN_FULL_SEQ` unset/`1` | Causal windows: pos grows, KV/HCA retained within window |
| `MUD_TRAIN_FULL_SEQ=0` | Classic L-10 independent pairs at pos=0 |
| `MUD_TRAIN_SEQ_LEN=N` | Window length (default 32, clamp 2..512, also ≤ dense_kv_cap) |

### Stream E env knobs (CSA)
| Env | Effect |
|-----|--------|
| `MUD_CSA` unset/`1` | Lightning top-k over HCA when `#blocks > top_k` (inference; train tapes = full) |
| `MUD_CSA=0` | Always full HCA scan |
| `MUD_CSA_TOP_K=N` | Keep N blocks (default 64) |
| `MUD_CSA_INDEX_DIM=D` | Coarse rank dim (default 16; `0` = full head) |
| `MUD_CSA_TAIL=N` | Always keep last N HCA blocks (default 4) |

### Stream B env knobs (product)
| Env | Effect |
|-----|--------|
| `MUD_TRAIN_EXPERT=N` | Dense QAT FFN targets `expert.N` (w3=up, w1=gate, w2=down or up/gate alt) |
| `MUD_MOE_CLONE=N` | Clone expert.0 into N slots for multi-expert inference tests on dense models |
| `MUD_MOE_TOP_K=K` | Top-k routing (default 2, clamp 1..8) |

### Stream C env knobs (GEMV)
| Env | Effect |
|-----|--------|
| `MUD_GPU_GEMV=auto` (default unset) | One-shot CPU vs GPU micro-bench; GPU only past break-even |
| `MUD_GPU_GEMV=1` | Force GPU when work ≥ 256² (or `MUD_GPU_GEMV_MIN`) |
| `MUD_GPU_GEMV=0` | Always AVX2 |
| `MUD_GPU_GEMV_MIN=N` | Force min work units (`n_in*n_out`) |
| `MUD_GPU_GEMV_LOG=1` | Print calibration table |
| `cargo run --release --bin gemv_auto_bench` | Standalone profile |

---

## 4. Validation commands

```bash
./mud.sh ci                    # tests + clippy + health
./mud.sh audit-full [model]    # ledger structural audit
./mud.sh health [model]        # training preflight
cargo test --lib -- --test-threads=2
```

Optional stress:
```bash
export MUD_GRAD_CKPT=1 MUD_GRAD_CKPT_SEG=4
export MUD_GPU_GEMV=auto MUD_USE_VULKAN=1   # or =1 force / =0 CPU
export MUD_MAX_POS=32768
cargo run --release --bin gemv_auto_bench    # print break-even on this silicon
```

---

## 5. Conclusion

The **Phase A–E ledger items L-01…L-15** have foundational implementations in-tree, and post-ledger streams **A–E** (Adam · MoE load · GEMV auto · full-seq train · CSA indexer) are **DONE**.

**Living next backlog** (ordered research + product):  
→ **[`MUD_IMPROVEMENTS_POST_AE.md`](./MUD_IMPROVEMENTS_POST_AE.md)** — F QKV one-CB · K loss cert · G multi-expert STE · H joint BPTT · I KV quant · J CSA v2.

This file keeps **residual P0–P3** and **env knobs** for A–E; prefer F+ doc for new session planning.
