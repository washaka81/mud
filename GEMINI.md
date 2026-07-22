# Forge LLM (MUD) — Canonical Project Instructions

> **Document hierarchy (2026-07-16)**
>
> | Document | Role | Authority |
> |----------|------|-----------|
> | **`GEMINI.md`** (this file) | Policies (P-#), architecture truth, priority ledger, compliance | **SSOT** for agents & humans |
> | **`AGENTS.md`** | Session context, recent fixes, commands for AI agents | Derived from this file — must not contradict |
> | **`VISION_ROADMAP.md`** | Product vision + Q3–Q4 2026 phases | SSOT for *direction* |
> | **`PLAN_MAESTRO.md`** | Deep architecture narrative + MoE design | Expansion of vision; metrics table |
>
> If two docs disagree, **this file wins for policies/status**, and **`VISION_ROADMAP.md` wins for product direction**.

---

## 0. Canonical Status Snapshot (2026-07-16, post streams A–E)

### What works
- Forward + backward (STE QAT) on ternary ELUT 4-bit weights
- `SlimeRegister` with **native `f32` `matmul_accum`** + JEPA state in workspace (`jepa_z`)
- Forward GEMV: **AVX2 ELUT × `PCorePool(8)`** (Rayon OFF), plus **auto GPU** via `gemv_policy` (`MUD_GPU_GEMV=auto|0|1`); default `auto` in `mud.sh`. **Auto** = one-shot CPU-vs-GPU(ash) micro-bench; GPU only past break-even. On **i7-1260P + Iris Xe (ADL GT2)** the hot-path real picks **AVX2** (per-GEMV dispatch on UMA doesn't amortize for 147M; synthetic contiguous bench shows 2.5–14× GPU win). Force `0`=CPU / `1`=GPU. Vulkan backend = **ash 0.38** (`src/vulkan/ash_backend.rs`, `probe_gpu()`), also used in RMSNorm output-norm / Muon NS / heartbeat. **⚠️ Vulkan SIGSEGV on Intel Iris Xe (2026-07-18):** el driver Intel (`libvulkan_intel.so`, ADL GT2 UMA) hace SIGSEGV en `submit_and_wait` al dispatchar QKV GEMV (determinista en block 11/64). Fix: `mud.sh train` fuerza `MUD_USE_VULKAN=0`/`MUD_GPU_GEMV=0` (override `MUD_TRAIN_FORCE_VULKAN=1`). Un fallo de driver NO es capturable como `Result`. Detalle: `docs/sessions/MUD_SESSION_REPORT_2026-07-18.md` §2.
- SiLU + attention dots + **LM head logits** on ASM (`silu_vectorial`, `dot_product`, `lm_head_logits_avx2`)
- JEPA OU tracker + mHC residual (spring force / repulsion **removed**)
- Sampled Softmax training objective (NCE-8 abandoned)
- **Full-seq train** (causal windows, `MUD_TRAIN_FULL_SEQ` / `MUD_TRAIN_SEQ_LEN`) + L-10 packing
- **Mini MoE load** from `.mud` + dense train-expert (`moe_load`, `MUD_TRAIN_EXPERT`)
- **CSA lightning top-k** over HCA (`csa_indexer`; dense window always full; train tapes = full HCA)
- L-13 HCA dense ring + 32k-ready policy; L-14 C-MUD math; L-15 grad checkpoint
- Rayon removed from runtime crate; AOT corpus cache; converter `--check` / auditors
- Compute stack: `docs/architecture/MUD_COMPUTE_STACK.md`
- ~186 lib tests; clippy-clean target (P-06); `./mud.sh ci|audit-full`

### 0.1 Status addendum (2026-07-22)

- **Word fusion tokenizer fix:** Fixed AOT corpus & C-MUD data loaders (`corpus_trainer.rs`, `cmud_train.rs`) to tokenize continuous text blocks, preserving `Ġ` space prefixes across line/chunk boundaries. Added `Tokenizer::has_space_prefix` helper and unit test `test_has_space_prefix_subwords`.
- **C-MUD Manifold & Cognition Validator (`cmud_manifold_validator` / `./mud.sh cmud-manifold`):** New standalone tool validating 5 cognitive dimensions (Léxico, Pensamiento, Coherencia, Resolución de Problemas, Resultados) with side-by-side baseline vs C-MUD comparative table and 100% verified certification.
- **Project-wide Table Framing & Alignment:** All CLI tools and UI banners reviewed and formatted using `unicode-width` (`UnicodeWidthStr`) for pixel-perfect box borders across UTF-8 characters (`Ġ`, `Δ`, `µ`, `τ`, etc.). 257 lib tests passing, `cargo clippy --all-targets` 0 errors / 0 warnings.

### 0.2 Historical Status addendum (2026-07-20)

- **Model base saned:** `models/smollm2.mud` reconverted healthy (FIX D + `scale_audit`); prior-training vocabulary-collapse (PRQ scales inflated ~14–28× on last layers) fixed via `MUD_TRAIN_WCLAMP_K` shadow clamp (default 8) + `MUD_TRAIN_RESET_EPOCH=1`. See `TRAIN_TELEMETRY_FORENSIC_2026-07-20.md` §7.
- **Live training telemetry TUI (`train_telemetry`):** parses `[TELEM]`/`[DW]` from `mud_train_metrics.log` (trainer now writes to **stderr AND file**); panels: PosLoss / VarH / VarJ / JEPA-integral / Cognitive + new **Weight Δ (bytes moved / sync)**. fixes the empty-panel bug where `[TELEM]` was stderr-only. Verified via `tmux capture-pane`.
- **Pointer-optimized hot loops (P-00/P-01):** `apply_optimizer_cpu_step_and_pack` clamp, `dequantize_ternary_row` (LUT branchless), `unpack_ternary2bit_to_f32` (raw ptr, 8/u32), `pack_ternary_into` (8/u32 word), `pack_elut_prq` (2/byte) — zero-alloc.
- **Debate/circuit writeback (`run_debate_session`):** now compares `hash_trained_weights` in vs out and prints ✓/⚠ NO-OP; reuses `sync_shadow_to_mud` (PRQ-scale-aware). fixes C1/C2 collapsed-writeback class (forensic §8).
- **STE deadzone caveat:** on a converged base at default `QAT_LEARNING_RATE=0.0005` (threshold `s*0.7`) ΔW≈0 is expected, not a bug; use higher LR (e.g. `MUD_QAT_LR=0.03`) for visible weight movement.
- **Open debt:** trainer console banner hardcodes `lr_init=3e-4` (corpus_trainer.rs:1703) while real LR=0.0005 — display-only, no functional impact.

### Depth streams A–E (post-ledger, 2026-07-16) — **DONE**
| Stream | Item |
|--------|------|
| **A** | Adam / SparseAdam real moments (`adam_state`, sparse zero-row skip) |
| **B** | MoE `.mud` load + train-expert; FFN names w3=up w1=gate |
| **C** | GPU GEMV auto-threshold (`gemv_policy`, `gemv_auto_bench`) |
| **D** | Full-seq packed train (causal pos/KV) |
| **E** | CSA top-k indexer over HCA |

### Orbit F–L (2026-07-16)
| Item | Reality |
|------|---------|
| **F QKV one CB** | **DONE** `dispatch_gemv_qkv_host_sync` |
| **G Multi-expert STE** | **DONE** round-robin + **hash route** (`MUD_MOE_TRAIN=1\|hash`) |
| **H Long full-seq** | **DONE** auto L-15 + **residual bank wired** in train (`MUD_GRAD_CKPT_RESIDUAL=1`) |
| **I KV f16** | **DONE** `MUD_KV_DTYPE=f16` IEEE half packs |
| **J CSA LSH** | **DONE** `MUD_CSA_LSH=1` SimHash prefilter |
| **K Loss cert** | **DONE** `loss_cert` + `cert-loss` |
| **L Converter P-13** | **DONE** canonical aliases + auditor |

### Active optimizer
**L-01 LIVE:** `select_optimizer()` → **Muon / GaLore / Chunked** preprocess then STE SGD + ELUT pack; **Adam / SparseAdam** use real moments (`adam_state` + `adam_step_avx2` when available).

---

## 1. Architectural Mandates

- **Pointer Mastery (P-00):** "El que domina los punteros domina el núcleo de la programación". Hot paths use `*mut T` / `*const T` and pre-allocated arenas. High-level slices are acceptable only outside the compute kernel.
- **Zero-Allocation Policy (P-01):** Inference and QAT hot loops MUST NOT allocate. Use `SlimeWorkspace` (and trainer flat buffers) pre-allocated before the loop; reuse via `copy_from_slice` / `fill`.
- **SlimeRegister (P-02 — CURRENT):** Fundamental activation cell. **Runtime truth (post 2026-06-23):** `matmul_accum: f32` (native magnitude; i16/f16 accum **deprecated** after saturation crises). JEPA orbital state lives in `SlimeWorkspace.jepa_z` (flat `f32`), not as a second packed f16 that drives the math path. Dual-f16 packing remains a historical design note, not the active accumulation contract.
- **ELUT Wire Format (P-03):** Ternary weights on the hot path use **4-bit nibble ELUT** packing (8 weights / `u32`) + **Per-Row Quantization (PRQ)** scales.
- **JEPA Deterministic Gate (P-05):** JEPA is deterministic (no learnable JEPA weights). Per block: z-score `y` → OU update `z` → `v_jepa` → sigmoid gate into **mHC** residual blend. Diagnostics MUST expose VarH, VarJ, gate stats.
- **Equilibrium Mandate:** Statistical ternary path and JEPA gate should keep healthy VarH (> ~0.1 target band) and VarJ ~ O(1) after z-score. Collapse is a **critical bug**, not a tuning footnote.
- **Anti-Hardcoding (P-13):** Network dimensions MUST come from `.mud` metadata (or explicit panic/error). No silent magic fallbacks for hidden/vocab/layers/heads.
- **Rust-Only Hardware Target (P-07 / P-12):** No Python/PyTorch in runtime, training, conversion, or validation. Hot compute: handwritten AVX2 ASM (P-cores) and/or Vulkan compute on Iris Xe. No discrete-GPU requirement.
- **Rayon (P-27 — CLARIFIED):** **Forbidden in the `forge_llm` runtime and tools hot path.** Custom `PCorePool` only. Exception: `forge_autograd` may still list Rayon only if unused on the training critical path; prefer removal. Do not reintroduce Rayon into `src/`.
- **Jamba / Mamba:** Historical research track (`mamba.s` may exist). **Not** the current product spine. Current spine = dense ternary Transformer (+ future Mini MoE per vision). Do not treat Mamba as mandatory until re-prioritized in the ledger.

---

## 2. Technical Standards

- **SIMD:** Critical GEMV/elementwise in AVX2 ASM (`src/asm/*.s`) where profiled. Scalar only for edges/tests.
- **0-Error, 0-Warning (P-06):** `cargo clippy --all-targets` clean.
- **Tests (P-09):** New modules ship `#[cfg(test)]` with public-fn coverage + edge case.
- **Benchmarks (P-10):** New compute modules ship a `tools/*_bench.rs` `[[bin]]` + `./mud.sh bench` entry when applicable.
- **Dead Code (P-08):** Delete unused modules/bins/shaders call-sites; do not comment-out or `allow(dead_code)` as storage.
- **Memory Safety (P-16):** Every `unsafe` has `// SAFETY:`.
- **Fail-fast (P-17):** Null / missing metadata → log + skip or hard error; never silent corruption.
- **Justified constants (active):**
  - `DEPTH_DAMPENING_FACTOR = 0.7071` (1/√2)
  - `SPARSITY_THRESHOLD_RATIO = 0.7` (~26% sparsity)
  - `NEURAL_KICK_JITTER = 1e-5` (or current equivalent micro-jitter on `z`)
  - `EPSILON_FLOOR = 1e-8` — **single definition** in `constants.rs` only
  - `JEPA_ATTRACTOR_LR` / OU rates: document actual rates used in `slime_jepa.rs` (currently EMA ~0.1 on `z`)
- **Deprecated constants (do not reintroduce as mandates):**
  - `SLIME_RESEAT_STRIDE` / i16 mid-row reseat (P-04) — obsolete after f32 accum
- **Bilingual corpus (P-23):** ES + EN.
- **Docs layout (P-18 — CLARIFIED):** Long-form docs under `docs/{audits,sessions,architecture,research,manuals,dumps}/`. **Allowed at repo root:** `GEMINI.md`, `AGENTS.md`, `VISION_ROADMAP.md`, `PLAN_MAESTRO.md`, `README.md`, `TREE.md`, `LICENSE`. Other root markdown is debt (move or delete).

---

## 3. Calibration & Restoration (UCP)

Every new model conversion SHOULD follow:

1. **Convert** — `universal_converter` (PRQ + ELUT 4-bit)
2. **Check** — `--check` / `audit-conv` (metadata + topology)
3. **Bound** — ternary grid + scale floors
4. **Health** — shapes / optimizer selection report
5. **Restore IQ** — STE QAT via corpus trainer / `./mud.sh` pipeline
6. **Validate** — iteration / composite gates when tools are available

> **Converter health caveat (FIX D, 2026-07-18):** `MudFile::save` previously
> corrupted mmap-backed tensors on the post-conversion ECC rewrite (read
> `tensor.offset` as absolute, but it is relative to the data region). Output
> norms/scales decoded as `~±1e38`. Fixed via `MudTensor::data_base` +
> `data_base + offset`. `--check` only validates metadata — always
> `diagnose_model` the converted `.mud` to confirm norms ≈ 1.0 and scales
> in a sane band (not 1e-8 nor 1e38).

Production gate targets (when validators run): SQNR ≥ 10.5 dB (conversion); composite score ≥ 96% (restore). Treat misses as blockers, not warnings.

---

## 4. Development Workflow

1. Read **this file** + `VISION_ROADMAP.md` Phase A before architectural changes.
2. New module → tests → (bench if compute) → `mud.sh` entry → doc under `docs/`.
3. After sessions: `cargo clippy --all-targets` (and `-D dead_code` when purging).
4. Prefer `./mud.sh <cmd>` over raw `cargo run` for operator workflows (P-19).
5. Do not mark a Priority **DONE** unless the **call site** is live (not just a module or shader file).

---

## 5. Product Mandate (aligned with vision)

**North star:** A ~2B ternary LLM (~400MB) that trains and runs on a commodity Intel laptop (e.g. i7-1260P), offline, private, with near-zero marginal cost.

**Success is not** "beat GPT-4 on MMLU". **Success is** accessible, trainable, local intelligence (code assistant, personal fine-tunes, air-gapped devices).

**Engineering pillars:** *Fast* (AVX2 + optional Vulkan), *Efficient* (ternary + zero-alloc hot path), *Stable* (JEPA + mHC + sanitized grads). Intelligence gains come from QAT + data + architecture — not from marketing wording.

Full narrative: `VISION_ROADMAP.md`, `PLAN_MAESTRO.md`.

---

## 6. Priority Ledger (monotonic — SSOT)

IDs below are the **only** active tracking IDs. Old "Priority N" numbers in session reports are historical; they must not be reused.

### 6.1 Historical achievements (collapsed — details in `AGENTS.md` / `docs/sessions/`)

Completed capability areas (not exhaustive): SlimeForward/Backward STE, ELUT 4-bit, JEPA/mHC recovery, ash migration, PCorePool HT saturation, AOT corpus cache, Sampled Softmax, embedding dequant fixes, EZOP benches (+8% certified), converter auditor, telemetry TUI, speculative drafter zero-alloc, Vulkan teardown UAF fix, many VarH/VarJ/aphasia recoveries.

### 6.2 Open ledger (do these next)

| ID | Title | Status | Notes |
|----|-------|--------|-------|
| **L-01** | Wire Muon/GaLore/Chunked/Sparse to real optimizer step | **DONE** | Muon/GaLore/Chunked + **Adam/Sparse moments (stream A)** |
| **L-02** | Allocate + dispatch Newton-Schulz Vulkan path when strategy=Muon | **DONE** | `AshContext::dispatch_newton_schulz_sync`; hybrid in `muon.rs` |
| **L-03** | Delete `InferenceWorkspace` + unused ASM/tool orphans | **DONE** | workspace.rs 426→~180 LOC; keep UnifiedBuffer |
| **L-04** | FFI/call sites for high-value ASM or delete orphans | **DONE** | Purged elut/pext/lut/slime_rmsnorm/mamba/ternary_backward; 11 live `.s` |
| **L-05** | True double-buffer CPU/GPU overlap in QAT/inference | **DONE** | `DoubleFrame`; `step_async_deferred` + flush after next backward |
| **L-06** | Dispatch `mha.comp` + `rms_norm.comp` where profitable | **DONE** | Shared-mem shaders; output_norm GPU; `try_gpu_dense_mha` |
| **L-07** | P-13 cleanup: no dim fallbacks; single `EPSILON_FLOOR`; pool size from policy/metadata | **DONE** | `constants::default_pcore_threads`, EPSILON SSOT, fail-fast mHC meta |
| **L-08** | NaN guards / finite checks on remaining ASM edges | **DONE** | dot/sum_sq/rmsnorm/silu/lm_head/gemv/batch4 sanitize non-finite → 0 |
| **L-09** | EZOP raw-pointer pass on remaining core loops | **DONE** | `mud::ezop`; TLS grad scratch; pack/dequant; zero-alloc backward branch |
| **L-10** | Sequence packing (no pad) | **DONE** | `sequence_pack`; full-chunk pairs; tail kept; no EOS cross |
| **L-11** | Mini MoE bus (ExpertBus mount/unmount) | **DONE** | + **stream B** `moe_load` / train-expert product path |
| **L-12** | P-13 property tests + CI health battery | **DONE** | `mud::p13`; `./mud.sh ci`; `.github/workflows/ci.yml` |
| **L-13** | CSA/HCA KV / 32k context | **DONE** | Dense ring + HCA; **stream E** top-k lightning indexer |
| **L-14** | C-MUD complex registers | **DONE** | `mud::cmud` math kernel; opt-in `MUD_CMUD_THINK=1` |
| **L-15** | Gradient checkpointing | **DONE** | `MUD_GRAD_CKPT=1` recompute-on-reverse; segment policy |
| **F1** | Trainable mHC α/β (dense f32 SGD) | **DONE** | `mhc_scale_sgd_step`; CPU+ash paths; clamp [0,4]; finite-diff test |
| **F2** | STP trajectory loss (JEPA→geodesic aux) | **DONE** | `stp_loss`; `MUD_TRAIN_STP=1` (default ON in `./mud.sh train`); `λ`=`MUD_TRAIN_STP_LAMBDA` |
| **UI** | Unified trainer console format | **DONE** | `src/mud/trainer_ui.rs`; single box, `note()` tags, no emoji/stderr noise |
| **F3** | RLVR debate: juez + reward/penalty + aprendizaje | **DONE** | `arena_judge.rs` (Verifiable/Rust/Text/Professor, no-API); `run_game` infinito hasta basta; `TextJudge` local cosine; `MUD_DEBATE_LEARN` default OFF |
| **F3+** | Seed-driven Training Circuit (`--circuit`) | **DONE** | `corpus_trainer::run_training_circuit`; rota baterías por semilla (align/debate/games/professor) con LCG sin RNG; time-box por fase; telemetría unificada + `logs/circuit.log` (ver con `tail -f logs/circuit.log`); guarda al `quit`. **`./mud.sh circuit` lanza TUI `circuit_telemetry`** (Juez + J\|A/B, +Profesor/Alumno, event-log; Ctrl-C/q stop&save). **Honors-mode eval:** `circuit_eval_integrity` (structural gate: tensors present/non-null + **norm-weights no-cero** post-`materialize_writable`; NO per-weight nibble-skew flag) + `circuit_benchmark_games` win-rate vs baseline; rollback a `.bak_circuit` si falla. CPU caveat: benchmark puede dar 0 matches → solo integridad guarda. **Health-check previo:** modelo colapsado (`attn_norm.weight`=0) es rechazado con error claro (no panic en worker thread de PCorePool). **⚠️ REQUISITO:** usar `.mud` sano. `models/smollm2.mud` fue **reconvertido sano** (2026-07-18, FIX D) desde `models/smollm2/` y ya NO está colapsado; `weights/checkpoints/model_latest_checkpoint.mud` es el checkpoint vivo del trainer (25 épocas CPU en curso, 0 crashes). `models/ternary_bonsai_1.7b.mud` sigue siendo alternativa sana. Ver `docs/research/MUD_PLAN_CIRCUIT_ALGORITHMS.md`. |
| **TLM** | Live training telemetry TUI (key-parse + ΔW panel + `[TELEM]`→log) | **DONE** | `train_telemetry.rs` `kv_f64`; `[DW]` every sync; trainer writes stderr+file; verified via tmux capture |
| **G+** | Multi-expert STE joint (SlimeX Dynamic Stack / ShadowExpertBus) | **WIP** | `slime_x.rs` PoC with zero-alloc `SlimeXSlot` + `ShadowExpertBus` created (2026-07-20). Ready for forward/backward wire into `corpus_trainer.rs` for joint training of top-k experts. Future vision: Hybrid GPU/CPU hot-mounting. |

### 6.3 Phase mapping (see also `VISION_ROADMAP.md`)

| Phase | When | Ledger focus |
|-------|------|----------------|
| **A — Close debt** | Jul 2026 | L-01 … L-08 |
| **B — Extreme perf** | Aug 2026 | ASM/GPU tiling, further prefetch (lm_head logits already live) |
| **C — Modular** | Sep 2026 | L-09–L-11 (EZOP, packing, Mini MoE) |
| **D — Scale train** | Oct 2026 | 4-pillar efficiency, bf16 shadows |
| **E — Maturity** | Nov–Dec 2026 | L-12–L-13, CI benches, docs |

### 6.4 Next session handoff

**Ledger L-01…L-15 + streams A–E + F1/F2/UI/F3/F3+ + TLM CLOSED. Launch countdown T-0 = GO (2026-07-16).**  

**2026-07-20 addendum:** model base reconverted sane; live telemetry TUI fixed (`[TELEM]`→log + key parser + Weight Δ panel); hot loops pointer-optimized (P-00/P-01). Open: trainer banner LR hardcoded `3e-4` (corpus_trainer.rs:1703). STE deadzone: converged base at default LR → ΔW≈0 is expected.
See `docs/manuals/LAUNCH_COUNTDOWN.md`.

**F1/F2/UI verification (2026-07-17):** `docs/architecture/MUD_TRAINER_TERNARY_JEPA_MHC.md` §9–§10; plan `docs/research/MUD_PLAN_MHC_STP_TRAINABLE.md`. Trainer default now project-adapted corpus + STP on + 64 steps/chunk (`mud.sh train`).

| Order | Action |
|-------|--------|
| 1 | **Orbit F–L foundations DONE** — see `MUD_IMPROVEMENTS_POST_AE.md` remaining depth (router BPTT, residual-bank wire, nightly cert) |
| 2 | Residuals — `docs/research/MUD_GAP_ANALYSIS_POST_L15.md` |
| 3 | Keep green: `./mud.sh ci` · `audit-full` · `gemv_auto_bench` |

**Read first:** this §0 · `LAUNCH_COUNTDOWN.md` · `MUD_IMPROVEMENTS_POST_AE.md`.

---

## 7. Documentation Mandate (P-18)

| Path | Content |
|------|---------|
| `docs/audits/` | `MUD_AUDIT_REPORT_VXX*.md`, `MUD_AUDIT_LATEST.md` |
| `docs/sessions/` | `MUD_SESSION_REPORT_YYYY-MM-DD.md` |
| `docs/architecture/` | Specs, manifests |
| `docs/research/` | Papers, gap analysis, **MUD_IMPROVEMENTS_POST_AE.md** (F+) |
| `docs/manuals/` | User/protocol guides + **LAUNCH_COUNTDOWN.md** |
| `docs/STATUS_REPORT.md` | Logros vs deuda (moved from root P-18) |
| `docs/dumps/` | Temporary dumps |

Root exceptions listed in §2. Update **this file's §0 and §6** when status changes — do not invent parallel status sections in other root files.

---

## 8. Orchestration Mandate (`mud.sh`)

- Prefer `./mud.sh <command>` as operator entry (P-19).
- Group commands by domain; no orphan menu entries without `[[bin]]`.
- Destructive ops require confirmation.
- Auto-select trained checkpoints when present for chat/train.

---

## 9. Complete Policy Index

| ID | Policy | Severity | Notes |
|----|--------|----------|-------|
| **P-00** | Raw pointer mastery in hot path | CRITICAL | |
| **P-01** | Zero-allocation hot-loop (`SlimeWorkspace`) | CRITICAL | |
| **P-02** | `SlimeRegister` / activation contract | CRITICAL | **f32 matmul_accum** is current truth |
| **P-03** | ELUT 4-bit nibble packing | CRITICAL | |
| **P-04** | i16 partial reseat ≤256 | **DEPRECATED** | |
| **P-05** | JEPA gate + mHC at block boundaries | CRITICAL | |
| **P-06** | 0-error 0-warning clippy | CRITICAL | |
| **P-07** | Rust-only — no Python/PyTorch | CRITICAL | |
| **P-08** | Dead code deleted | CRITICAL | L-03 InferenceWorkspace purged |
| **P-09** | Inline unit tests per module | MANDATORY | |
| **P-10** | Benchmark binary for compute modules | MANDATORY | |
| **P-11** | Every `[[bin]]` has `./mud.sh` entry | MANDATORY | |
| **P-12** | AVX2 P-cores + optional Vulkan iGPU | MANDATORY | |
| **P-13** | Anti-hardcoding dimensions | MANDATORY | Open cleanup L-07 |
| **P-14** | Gradient finite + clamp | MANDATORY | |
| **P-15** | Hot PRQ clamp to ternary grid | MANDATORY | sparsity 0.7·s |
| **P-16** | `// SAFETY:` on every unsafe | MANDATORY | |
| **P-17** | Fail-fast over silent corruption | MANDATORY | |
| **P-18** | Docs under `docs/` + root exceptions | MANDATORY | §2 |
| **P-19** | `mud.sh` preferred entry | MANDATORY | |
| **P-20** | UCP ordering for restore pipeline | MANDATORY | |
| **P-21** | Universal agnosticism (multi-arch metadata) | MANDATORY | |
| **P-22** | Dynamic context from `max_position_embeddings` | MANDATORY | |
| **P-23** | Bilingual ES+EN corpus | STANDARD | |
| **P-24** | PRQ on all ternary tensors | CRITICAL | |
| **P-25** | SQNR ≥ 10.5 dB conversion | CRITICAL | |
| **P-26** | iteration_validator ≥ 96% prod cert | CRITICAL | |
| **P-27** | No Rayon in runtime hot path | CRITICAL | §1 clarification |

---

## 10. Architecture Facts (engine)

- **Format:** `.mud` (`MUD\x01`) Ternary2Bit ELUT + `prq_scale` + norms; optional expert tensors
- **Forward (`evaluate_slime_block` / `_moe`):** RMSNorm → QKV GEMV → SDPA/GQA (HCA+CSA) → O → Residual+JEPA/mHC → FFN/ExpertBus → Residual+JEPA/mHC
- **Training:** STE QAT; full-seq windows (default); Sampled Softmax (~512); Adam/Muon/GaLore/Chunked LIVE
- **Threading:** `PCorePool` via `default_pcore_threads()` / `MUD_PCORE_THREADS`
- **Device:** GEMV auto (`gemv_policy`); NS/MHA/RMSNorm GPU helpers when profitable

```
fn select_optimizer(rows, cols) -> Strategy  // LIVE on step (L-01 + stream A)
```

**Next improvements:** `docs/research/MUD_IMPROVEMENTS_POST_AE.md`.

---

*MUD: Bare-metal ternary engine. Policies live here. Vision lives in VISION_ROADMAP.md. Session detail lives in AGENTS.md.*
