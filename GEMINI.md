# Forge LLM (MUD) — Canonical Project Instructions

> **Document hierarchy (2026-10-11)**
>
> | Document | Role | Authority |
> |----------|------|-----------|
> | **`docs/planning/MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md`** | **PRIORIDAD 0 — backlog activo (`R-##`)**. Correcciones, limpieza y mejoras con criterios de aceptación | **SSOT de ejecución** — manda sobre cualquier otro backlog |
> | **`GEMINI.md`** (this file) | Policies (P-#), architecture truth, priority ledger, compliance | **SSOT** for agents & humans |
> | **`AGENTS.md`** | Session context, recent fixes, commands for AI agents | Derived from this file — must not contradict |
> | **`ROADMAP.md`** | Product vision + phases + architectural narrative (consolidated 2026-07-26) | **SSOT** for *direction* |
> | **`docs/architecture/MUD_COMPUTE_STACK.md`** | Compute stack LIVE detail | Technical reference |
> | **`docs/audits/MUD_AUDIT_2026-10-11_LOW_COMPUTE.md`** | Auditoría vigente (evidencia `file:line` + mediciones) | Fuente de los hallazgos del plan |
>
> **Orden de prioridad:** (1) plan `R-##` — empezar por la **Ola 0**; (2) el resto de backlogs (`MUD_IMPROVEMENTS_POST_AE.md`, `MUD_GAP_ANALYSIS_POST_L15.md`) queda **congelado** hasta cerrar la Ola 1 y sólo se reanuda si no contradice la auditoría 2026-10-11.
>
> If two docs disagree, **this file wins for policies/status**, and **`ROADMAP.md` wins for product direction**.

---

## 0. Canonical Status Snapshot

> ### ⚠️ 0.3 AUDITORÍA 2026-10-11 — estado real verificado (tiene prioridad sobre lo que sigue)
>
> Auditoría completa: [`docs/audits/MUD_AUDIT_2026-10-11_LOW_COMPUTE.md`](docs/audits/MUD_AUDIT_2026-10-11_LOW_COMPUTE.md) · Backlog activo: [`docs/planning/MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md`](docs/planning/MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md) (`R-##`).
>
> **Hechos verificados en vivo (comando ejecutado, no inspección):**
>
> | Hecho | Evidencia |
> |---|---|
> | `cargo test --lib --release -- --test-threads=2` **PASA** (468 pass / 0 fail / 2 ignored); ub firmwick purgado (R-03, 2026-10-11) | `cargo test --lib --release -- --test-threads=2` |
> | CI workflow **válido**: `on:` movido a raíz (`push`, `pull_request`, `schedule`, `workflow_dispatch`); slug `washaka81/mud` (R-01) | `actionlint .github/workflows/ci.yml` exit 0 |
> | `./mud.sh ci` usa **release + all-targets + all-features** (`-D warnings`) | `mud.sh` `ci)` case |
> | Repo con **git init** + `.gitignore` limpio (R-04) | `git status` limpio tras commit |
> | ✅ **R-10**: el trainer SÍ entrena la cabeza evaluada — `sync_shadow_to_mud` espeja `token_embd.weight`→`output.weight` (+ `output.prq_scale`) vía `sync_tied_output_weight`; inferencia (`inference.rs:395`) y trainer (`train_step.rs:593`) coinciden. No-op para modelos *tied*. | `checkpoint.rs` + tests; `cargo test --lib --release` 462/0 |
> | ✅ **R-11**: FFN backward SiLU — `slime_backward.rs` reemplazó ReLU² (`g²·up`) por `silu(gate)·up` vía `ezop::silu_backward_avx2` (AVX2 8-f32/iter + escalar, σ via exp2 ≈ forward `silu_vectorial`). `test_silu_backward_matches_reference` ✅; `cargo test --lib --release` 463/0. |
> | ✅ **R-12**: mHC activation backward — `grad_out *= alpha[i]`, `grad_branch *= (1-gate)·beta[i]` vía `ezop::mhc_activation_backward` (AVX2 + escalar, gate=sigmoid(v_jepa) recomputado, `kinetic` preservado). Ambas rutas (ternaria + F32). `test_mhc_activation_backward_matches_reference` ✅ (alpha=0.85/beta=1.2/gate≠0.5). |
> | ✅ **R-13**: deadzone SSOT `TERNARY_DEADZONE=0.7` unificada conversor↔trainer↔checkpoint (`quantizer.rs:125`, `alignment.rs:249`, `checkpoint.rs` emb+layer pack) | `cargo test --lib --release` 466/0 | test |
> | ✅ **R-14..R-20** (cerrada Ola 1): escala PRQ punto fijo; packing determinista; `scale_up` backward; clip global `MUD_GRAD_CLIP`; AdamW decoupled; sampled-softmax score (mismo STE target+negs); chunked-Muon stride + scales_only √2 | `cargo test --lib --release -- --test-threads=2` 480/0/2 (2026-10-12) |
> | ✅ **R-30/R-31** (2026-10-12): `validate_model_shapes` metadata↔shape en `load`+forward; loader con `checked_*` + `MAX_TENSOR_DIM=1<<28` | tests R-30 ×3 + R-31 ×2; 480/0/2 |
> | ✅ **R-32** (2026-10-13): parser GGUF con cotas (`GgufCursor` bounds-checked, caps SEC-H1, arrays ≤4M/depth ≤8, `alignment` validado, offsets checked + byte-size exacto por tipo) | tests R-32 ×12 + proptest ×2; 500/0/2 |
> | ✅ **R-33/R-34/R-35** (2026-10-13): ECC valida paridad contra extent base (`Result`); `worm::get_embedding_row` exige Float32 rank-2 + dims del shape; GEMV wrappers exactos para `n%8∈1..7` (`ternary_gemv_tail`) + strides `div_ceil(8)` | tests R-33 ×2, R-34 ×4, R-35 (falla-antes ✅); 507/0/2 |
> | ✅ **R-36** (2026-10-13): `./mud.sh unsafe-audit` (awk en `mud.sh`, gate en `ci` + GitHub workflow) → **0 `unsafe` sin justificar** sobre 532 productivos; contratos `//! SAFETY:` de módulo + per-site; solo comentarios | `./mud.sh unsafe-audit` OK; clippy 0; 507/0/2 |
> | ✅ **R-14**: PRQ scale punto fijo — `stable_pack_ternary_rows` (helper compartido emb+layer) preserva `s_old` cuando los códigos no cambian (save→load→save bit-estable) y recomputa `s_new=absmean/√2` SOLO cuando `ΔW` real mueve los códigos (Phase-1 pack con s_old → byte-diff → Phase-2 repack); band-clamp `[½,2]·s_old` opcional omitido para no bloquear ajustes reales. Deriva ×0.52/sesión eliminada. Tests `test_prq_scale_stable_roundtrip` (5 ciclos, ±1e-6) + `test_prq_scale_recomputes_on_movement`. `cargo test --lib --release -- --test-threads=2` **468/0/2 ✅**; clippy 0. | `checkpoint.rs` `stable_pack_ternary_rows`; `cargo test --lib --release -- --test-threads=2 prq_scale` 2/0 |
> | Wire format ternario = **4 bits/peso** (2.52× la entropía de 1.585 bits) — pendiente R-42; ✅ **R-40** (2026-10-13): LM head ternario batcheado (`ternary_head_logits`, 4/8 filas + PCorePool + PRQ), sin materializar f32 → **~159 tok/s** default / 400 tok/s greedy en `models/ale.mud` (A/B: 2.4× default, 4.6× cómputo; RSS 201→127 MB; logits idénticos) | `gemv_gpu.rs`, `main.rs`; medición A/B 2026-10-13 |
> | ✅ **R-41** (2026-10-18): embedding ya no se materializa como `vocab×hidden` f32 (~155 MB). Packed ternary / mmap F32 slice se conservan; `dequant_emb_row` de-cuantiza solo la fila necesaria en `emb_row_buf` (pre-alloc). `emb_rms` y `computed_max_emb` se calculan iterando filas. `scale_up` se aplica per-row. Tests 508/0/2, clippy 0, build release OK | `main.rs` |
> | ✅ **R-43** (2026-10-18): round-trips i8 eliminados. RMSNorm almacena f32 directamente en `ws.ffn_out_f32` (copy de `gemv_accum`). Tape almacena `norm_f32_attn`/`norm_f32_ffn` (f32) en lugar de `norm_i8_*` + scale. Backward usa `copy_from_slice` directo (elimina `scale_i8_to_f32_avx2` en 4 sitios). `SlimeLayerTape` actualizado. Tests 508/0/2, clippy 0 | `slime_forward.rs`, `slime_backward.rs` |
> | 🟠 **R-42** (parcial, 2026-10-18): `MudTensorType::TernaryTQ = 5` + `data_size`/`checked_data_size` + parser + 7 match arms. `unpack_ternary_tq_to_f32` decodificador base-3. `ternary_tq_gemv` scalar + `ternary_tq_gemv_avx2` AVX2 (LUT 256×5 + multiply-accumulate). Tests roundtrip + GEMV scalar + GEMV AVX2 (2 rows, 8 cols). Pendiente: conversor, trainer, migración, inferencia wiring. Tests 511/0/2, clippy 0 | `mod.rs`, `slime_backward.rs` |
> | ⏸ **Ola 3 diferidos** (R-42 kernel/converter, R-44 fusionar pases, R-45 residual f32, R-46 KV sub-8-bit, R-47 LUT mpGEMM): sesiones dedicadas con rationale en plan maestro. Ningún TODO abierto en Ola 3. |
> | 🟠 **R-44** (parcial, 2026-10-18): slots cache-line-padded (`#[repr(align(64))]`) + `PCORE_THREADS_CAP` 8→16 (P+E cores). `MUD_PCORE_THREADS` soporta 1..64. Pendiente: fusionar pases norm/attn/JEPA + tabla tok/s. Tests 508/0/2, clippy 0 | `pcore_pool.rs`, `constants.rs` |
> | 🟠 **R-45** (parcial, 2026-10-18): `f16::from_f32` en `write_accum` ya clampa a ±65504. mHC projection + finite guard. Env `MUD_RESIDUAL_F32` para promoción a f32 diferido (requiere refactor P-02). | `slime.rs`, `slime_forward.rs` |
> | ⏸ **R-42/R-46/R-47** diferidos: formato 5 trits/byte (4-6d), KV sub-8-bit (2-3d), LUT mpGEMM (3d). Sesiones dedicadas.
> | Inferencia medida (post-R-40): **~159 tok/s** default (`MUD_INFER_MAX_TOKENS=256`); cuello restante = sampling nucleus (~3.8 ms/token) | medición 2026-10-13 |
> | ✅ **R-30/R-31** (2026-10-12): validación `metadata↔shape` (`validate_model_shapes` en `load`+forward) + aritmética checked en el loader (`MAX_TENSOR_DIM`); ✅ **R-32** (2026-10-13): parser GGUF con cotas; ✅ **R-36** (2026-10-13): `./mud.sh unsafe-audit` = 0 `unsafe` sin justificar (532 productivos; gate en CI); ✅ **R-37/R-38** (2026-10-13): RCE feature-gated documentado estructuralmente (+ gate tests), `data_analysis` confinado a `AgentWorkspace`; symlink escape cerrado (`symlink_metadata`+`O_NOFOLLOW`, 4 tests) → **Ola 2 completa** | tests R-30 ×3/R-31 ×2/R-32 ×12+proptest/R-38 ×4; `./mud.sh unsafe-audit` OK; `cargo test --lib --release --features proptest` 513/0/2 |
> | ✅ **R-05**: env vars SSOT (117 `R` + 7 shell + 6 retiradas en `MUD_ENV_REFERENCE.md`); `./mud.sh env-audit` → `OK — 0 divergencias (R-05)`. `rayon` sigue en `Cargo.toml` (R-66) | `bash mud.sh env-audit` exit 0 |
>
> **Claims retirados** (aparecían como ✅ en §0/§6 y en `AGENTS.md`): "460 tests ✅" (sólo debug; ahora 468 en release tras purgar batch4/batch8 UB, R-03), "0 OOB" (falso, ver R-30/R-31), "Rayon removed from runtime crate" (`rayon` sigue en `[dependencies]`, R-66), "`MUD_GPU_GEMV` default `auto` en mud.sh" (es `0`), "Fenwick batch prefix sum COMPLETADO (~4×)" (UB + sin llamadores ⇒ purgado, R-03), "wormholes peer-to-peer −50 % host compute" (mismo trabajo con un multiplicador local, R-67).
>
> **Regla nueva (R-73):** toda fila de estado exige `(comando, salida, fecha)`. Prohibido `COMPLETADO` sin ejecución.

### 0.0 Snapshot 2026-07-26 (post Sprint 5 + Sprints 6-7 + Batch A/B/C/D-partial session) — *parcialmente obsoleto, ver 0.3*

### What works
- Forward + backward (STE QAT) on ternary ELUT 4-bit weights
- `SlimeRegister` with **4-byte packed dual-component register** (f16 re + f16 im) + JEPA state in workspace (`jepa_z`)
- Forward GEMV: **AVX2 ELUT × `PCorePool(8)`** (Rayon OFF), plus **auto GPU** via `gemv_policy` (`MUD_GPU_GEMV=auto|0|1`); `mud.sh` **fuerza `0`** (`mud.sh:48`) — el default `auto` sólo aplica si la env está sin definir (corregido 2026-10-11, ver §0.3). **Auto** = one-shot CPU-vs-GPU(ash) micro-bench; GPU only past break-even. On **i7-1260P + Iris Xe (ADL GT2)** the hot-path real picks **AVX2** (per-GEMV dispatch on UMA doesn't amortize for 147M; synthetic contiguous bench shows 2.5–14× GPU win). Force `0`=CPU / `1`=GPU. Vulkan backend = **ash 0.38** (`src/vulkan/ash_backend.rs`, `probe_gpu()`), also used in RMSNorm output-norm / Muon NS / heartbeat. **⚠️ Vulkan SIGSEGV on Intel Iris Xe (2026-07-18):** el driver Intel (`libvulkan_intel.so`, ADL GT2 UMA) hace SIGSEGV en `submit_and_wait` al dispatchar QKV GEMV (determinista en block 11/64). Fix: `mud.sh train` fuerza `MUD_USE_VULKAN=0`/`MUD_GPU_GEMV=0` (override `MUD_TRAIN_FORCE_VULKAN=1`). Un fallo de driver NO es capturable como `Result`. Detalle: `docs/sessions/MUD_SESSION_REPORT_2026-07-18.md` §2.
- SiLU + attention dots + **LM head logits** on ASM (`silu_vectorial`, `dot_product`, `lm_head_logits_avx2`)
- JEPA OU tracker + mHC residual (spring force / repulsion **removed**)
- Sampled Softmax training objective (NCE-8 abandoned)
- **Full-seq train** (causal windows, `MUD_TRAIN_FULL_SEQ` / `MUD_TRAIN_SEQ_LEN`) + L-10 packing
- **Mini MoE load** from `.mud` + dense train-expert (`moe_load`, `MUD_TRAIN_EXPERT`)
- **CSA lightning top-k** over HCA (`csa_indexer`; dense window always full; train tapes = full HCA)
- L-13 HCA dense ring + 32k-ready policy; L-14 C-MUD math; L-15 grad checkpoint
- Rayon **no se usa** en `src/`, pero sigue declarado en `Cargo.toml:44` (deuda R-66); AOT corpus cache; converter `--check` / auditors
- Compute stack: `docs/architecture/MUD_COMPUTE_STACK.md`
- Register bit packing: `docs/architecture/MUD_REGISTER_BIT_PACKING.md`
- **459 lib tests en debug / 459 en release / 0 fallos** (R-03: purgados `prefix_sum_batch4_avx2` + `prefix_sum_batch8_avx2` por UB + 0 llamadores); clippy-clean target (P-06); `./mud.sh ci|audit-full`
- **F4 Fenwick trees** (`src/mud/fenwick.rs`) — `FenwickF32`/`FenwickU32`/`FenwickI32`/`FenwickLogF32` + AVX2/BMI1/BMI2 ops (`find_kth_avx2` LIVE en `debate_trainer.rs:557`; `find_kth_walk` test-only, sin UB)
- **Sprint 4b:** `prefix_sum_batch4_avx2` **PURGADO** (R-03, 2026-10-11): UB `_mm256_storeu_ps` sobre binding inmutable + 0 llamadores de producción → eliminado bajo P-08. `find_kth_walk` (BMI1/BMI2) sigue vivo (test-only).
- **Sprints 6-7:** Security documentation (`docs/seguridad/THREAT_MODEL.md`, `HARDENING.md`, `INCIDENT_RESPONSE.md`)
- **ASM kernel benchmarks:** 16 kernels (8.8× geometric mean)
- **Fenwick benchmarks:** 10 benchmarks (19.5× geometric mean)
- **mud.sh v2.1:** Reorganizado, comandos agrupados, `--json`/`--quiet` flags
- **Small-matrix GEMV threshold** (2026-08-03): Inline AVX2 path when `n_out < 256`, avoiding PCorePool dispatch overhead on small models (pico/nano Q/K/V). Speed: 6.8 → 11.3 tk/s.
- **Parallel F32 GEMV** (2026-08-03): `f32_gemv_submit` (no-wait) overlaps FFN up+gate and Q/K/V projections. Single `wait_all` instead of 3.
- **Telemetry fix** (2026-08-03): `push_grad` Welford double-division bug fixed (`metrics.rs`). Entropy display was 64× under-reported (8.7 → 0.136). Added `test_push_grad_no_double_division`.
- **Gradient fix** (2026-08-03): Class logit gradient uses `* logit_scale` (not `* temp_scale`), recovering 60× gradient magnitude.

### 0.1 Status addendum (2026-07-22)

- **Word fusion tokenizer fix:** Fixed AOT corpus & C-MUD data loaders (`corpus_trainer.rs`, `cmud_train.rs`) to tokenize continuous text blocks, preserving `Ġ` space prefixes across line/chunk boundaries. Added `Tokenizer::has_space_prefix` helper and unit test `test_has_space_prefix_subwords`.
- **C-MUD Manifold & Cognition Validator (`cmud_manifold_validator` / `./mud.sh cmud-manifold`):** New standalone tool validating 5 cognitive dimensions (Léxico, Pensamiento, Coherencia, Resolución de Problemas, Resultados) with side-by-side baseline vs C-MUD comparative table and 100% verified certification.
- **Project-wide Table Framing & Alignment:** All CLI tools and UI banners reviewed and formatted using `unicode-width` (`UnicodeWidthStr`) for pixel-perfect box borders across UTF-8 characters (`Ġ`, `Δ`, `µ`, `τ`, etc.). 257 lib tests passing, `cargo clippy --all-targets` 0 errors / 0 warnings.
- **Sprint 5 hot-path optimizations (2026-07-24):** 5.1 zero-alloc `evaluate_slime_block_moe` (workspace-backed `Vec` views); 5.2 `Mutex`→`AtomicU8` GPU dispatch fast path; 5.3 `ternary_gemv_backward` LUT; 5.4 branchless `pack_elut_prq`. 312 tests, clippy 0 errors. Ver `MUD_SESSION_REPORT_2026-07-24_SPRINT_5.md`.

### 0.2 Historical Status addendum (2026-07-20)

- **Model base saned:** `models/smollm2.mud` reconverted healthy (FIX D + `scale_audit`); prior-training vocabulary-collapse (PRQ scales inflated ~14–28× on last layers) fixed via `MUD_TRAIN_WCLAMP_K` shadow clamp (default 8) + `MUD_TRAIN_RESET_EPOCH=1`. See `TRAIN_TELEMETRY_FORENSIC_2026-07-20.md` §7.
- **Live training telemetry TUI (`train_telemetry`):** parses `[TELEM]`/`[DW]` from `mud_train_metrics.log` (trainer now writes to **stderr AND file**); panels: PosLoss / VarH / VarJ / JEPA-integral / Cognitive + new **Weight Δ (bytes moved / sync)**. fixes the empty-panel bug where `[TELEM]` was stderr-only. Verified via `tmux capture-pane`.
- **Pointer-optimized hot loops (P-00/P-01):** `apply_optimizer_cpu_step_and_pack` clamp, `dequantize_ternary_row` (LUT branchless), `unpack_ternary2bit_to_f32` (raw ptr, 8/u32), `pack_ternary_into` (8/u32 word), `pack_elut_prq` (2/byte) — zero-alloc.
- **Debate/circuit writeback (`run_debate_session`):** now compares `hash_trained_weights` in vs out and prints ✓/⚠ NO-OP; reuses `sync_shadow_to_mud` (PRQ-scale-aware). fixes C1/C2 collapsed-writeback class (forensic §8).
- **STE deadzone caveat:** on a converged base at default `QAT_LEARNING_RATE=0.0005` (threshold `s*0.7`) ΔW≈0 is expected, not a bug; use higher LR (e.g. `MUD_QAT_LR=0.03`) for visible weight movement.
- **Fixed:** trainer console banner now uses `qat_learning_rate()` (was hardcoded `3e-4`).

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
| **MUD-WORM** | **PARCIAL** `src/mud/worm.rs` — WormPropagator + JEPARouter + MhCoupler + WormChain + WormholeRouter/Network (30 tests). Auto-conexión de modelos `.mud` como modelo único ✅. ⚠️ **"wormholes peer-to-peer (50% reducción host compute)" es FALSO** (2026-10-11): `WormholeNetwork::forward` ejecuta el mismo trabajo en ambas ramas con un multiplicador local `strength`; no hay red ni peers (R-67). Ver `docs/research/MUD_WORM_IMPLEMENTATION.md` |
| **K Loss cert** | **DONE** `loss_cert` + `cert-loss` |
| **L Converter P-13** | **DONE** canonical aliases + auditor |

### Active optimizer
**L-01 LIVE:** `select_optimizer()` → **Muon / GaLore / Chunked** preprocess then STE SGD + ELUT pack; **Adam / SparseAdam** use real moments (`adam_state` + `adam_step_avx2` when available).

---

## 1. Architectural Mandates

- **Pointer Mastery (P-00):** "El que domina los punteros domina el núcleo de la programación". Hot paths use `*mut T` / `*const T` and pre-allocated arenas. High-level slices are acceptable only outside the compute kernel.
- **Zero-Allocation Policy (P-01):** Inference and QAT hot loops MUST NOT allocate. Use `SlimeWorkspace` (and trainer flat buffers) pre-allocated before the loop; reuse via `copy_from_slice` / `fill`.
- **SlimeRegister (P-02 — CURRENT):** Fundamental 4-byte packed dual-component register. **Runtime truth (post 2026-07-22, commit `9bc665d1f`):** `raw: u32` packs `(re: f16, im: f16)` where **re** is the ternary GEMV accumulator (IEEE binary16) and **im** is the JEPA integral derivative. `read_accum()` returns `f32` (1-cycle `vcvtph2ps` on F16C hardware) — the accumulator **executes in f32 precision** but is **stored as f16** (range ±65504, ~3-4 significant digits). The f16 store is intentional: it packs 2× density vs f32 and enables C-MUD thinker complex mode (`as_complex_f32`). `SlimeWorkspace.gemv_accum` (`AlignedArray<f32>`) is the transient f32 scratch during GEMV before being committed to a register via `write_accum()` (f32→f16, 1-cycle `vcvtps2ph`). Old i16 accum algorithms are **deprecated and purged** (L-04); the f16 path inherits the saturation safety without the i16 dynamic-range cliff. JEPA orbital `jepa_z` lives separately in `SlimeWorkspace.jepa_z` (flat `f32`) — the `im` component inside the register is the per-token integral derivative, not a duplicated workspace buffer.
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
  - `SLIME_RESEAT_STRIDE` / i16 mid-row reseat (P-04) — obsolete after f16-packing accum
- **Bilingual corpus (P-23):** ES + EN.
- **Docs layout (P-18 — CLARIFIED):** Long-form docs under `docs/{audits,sessions,architecture,research,manuals,dumps}/`. **Allowed at repo root:** `GEMINI.md`, `AGENTS.md`, `ROADMAP.md`, `README.md`, `TREE.md`, `LICENSE`. Other root markdown is debt (move or delete).

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

1. Read **this file** + `ROADMAP.md` Phase A before architectural changes.
2. New module → tests → (bench if compute) → `mud.sh` entry → doc under `docs/`.
3. After sessions: `cargo clippy --all-targets` (and `-D dead_code` when purging).
4. Prefer `./mud.sh <cmd>` over raw `cargo run` for operator workflows (P-19).
5. Do not mark a Priority **DONE** unless the **call site** is live (not just a module or shader file).

---

## 5. Product Mandate (aligned with vision)

**North star:** A ~2B ternary LLM (~400MB) that trains and runs on a commodity Intel laptop (e.g. i7-1260P), offline, private, with near-zero marginal cost.

**Success is not** "beat GPT-4 on MMLU". **Success is** accessible, trainable, local intelligence (code assistant, personal fine-tunes, air-gapped devices).

**Engineering pillars:** *Fast* (AVX2 + optional Vulkan), *Efficient* (ternary + zero-alloc hot path), *Stable* (JEPA + mHC + sanitized grads). Intelligence gains come from QAT + data + architecture — not from marketing wording.

Full narrative: `ROADMAP.md`.

---

## 6. Priority Ledger (monotonic — SSOT)

IDs below are the **only** active tracking IDs. Old "Priority N" numbers in session reports are historical; they must not be reused.

### 6.1 Historical achievements (collapsed — details in `AGENTS.md` / `docs/sessions/`)

Completed capability areas (not exhaustive): SlimeForward/Backward STE, ELUT 4-bit, JEPA/mHC recovery, ash migration, PCorePool HT saturation, AOT corpus cache, Sampled Softmax, embedding dequant fixes, EZOP benches (+8% certified), converter auditor, telemetry TUI, speculative drafter zero-alloc, Vulkan teardown UAF fix, many VarH/VarJ/aphasia recoveries.

### 6.2 Open ledger (do these next)

> **⚡ BACKLOG ACTIVO (2026-10-11): `R-##`.** Plan maestro: [`docs/planning/MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md`](docs/planning/MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md) — 55 ítems en 7 olas. Las filas `L-##`/`F#`/`Sprint #` de abajo son **histórico cerrado**; cualquier fila que contradiga la auditoría (§0.3) se considera retirada.
>
> | Ola | Foco | IDs | Estado |
> |---|---|---|---|
> | **0** | Verdad y CI (workflow válido, release tests, git, env SSOT) | R-01…R-06 | ✅ cerrada |
> | **1** | Correctitud de entrenamiento (cabeza, SiLU, mHC, escala, deadzone) | R-10…R-20 | ✅ cerrada |
> | **2** | Seguridad de entrada (metadata↔shape, GGUF, SAFETY) | R-30…R-38 | ✅ cerrada (R-30…R-36 ✅; R-37 RCE estructural + confinamiento; R-38 symlink escape ✅) |
> | **3** | Eficiencia de inferencia (head ternario, 1.6 bpp, i8, hilos) | R-40…R-48 | 🟠 en curso (R-40 head ternario ✅, R-48 doc GPU_GEMV ✅) |
> | **4** | Calidad de conversión (paridad, rotación, emb 8-bit) | R-50…R-55 | pendiente |
> | **5** | Limpieza (skills/, bins, tests, P-18, rayon) | R-60…R-68 | pendiente |
> | **6** | Verificación y honestidad (gates, energía, bench-infer) | R-70…R-74 | pendiente |

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
| **G+** | Multi-expert STE joint (SlimeX Dynamic Stack / ShadowExpertBus) | **DONE** (2026-07-23) | `slime_x.rs` `SlimeXSlot` + `ShadowExpertBus`; `moe_train.rs` `apply_joint_multi_expert_step`; `MUD_MOE_TRAIN=1\|hash`. Forward/backward wireado en `corpus_trainer.rs`. |
| **Sprint 5** | Hot-path perf: zero-alloc + zero-lock + LUT + branchless | **DONE** | 5.1 `csa_*_scratch` workspace views; 5.2 `AtomicU8` GPU dispatch; 5.3 `TERNARY_NIBBLE_TO_F32` LUT; 5.4 branchless `pack_elut_prq`. 312 tests, clippy 0. Ver `MUD_SESSION_REPORT_2026-07-24_SPRINT_5.md`. |

### 6.3 Phase mapping (see also `ROADMAP.md`)

| Phase | When | Ledger focus |
|-------|------|----------------|
| **A — Close debt** | Jul 2026 | L-01 … L-08 |
| **B — Extreme perf** | Aug 2026 | ASM/GPU tiling, further prefetch (lm_head logits already live) |
| **C — Modular** | Sep 2026 | L-09–L-11 (EZOP, packing, Mini MoE) |
| **D — Scale train** | Oct 2026 | 4-pillar efficiency (3/4 DONE), bf16 shadows (KV DONE, training pending), grad ckpt (DONE), pipeline overlap (double-buffer DONE), **Vulkan/ash compute delegation (2026-07-26)**: STP loss + mHC SGD + JEPA dynamics → `dispatch_optimizer_batch_async` (safe on Iris Xe, no `submit_and_wait`); NS/RMSNorm/MHA/GEMV-regular → probar shaders no-QKV en Iris Xe (SIGSEGV es específico del shader `qkv_pipe`, no de `submit_and_wait` general). Ver `docs/research/MUD_VULKAN_DELEGATION_ANALYSIS_2026-07-26.md`. **D5 (2026-07-26): GPU signal keep-alive plan** — batch backward pass (14M ops) + mHC (49K) + STP (2K) + JEPA (25K) en un solo `dispatch_optimizer_batch_async` para mantener Iris Xe en boost clock. Prioridad: 1) conectar `ternary_backward.comp` (existe, no conectado) → `dispatch_ternary_backward_host_sync`; 2) shader genérico para STP+JEPA (element-wise + reducción); 3) batch + DoubleFrame overlap. Ver `docs/research/MUD_GPU_SIGNAL_KEEPALIVE_PLAN_2026-07-26.md`. |
| **E — Maturity** | Nov–Dec 2026 | L-12–L-13, CI benches, docs |
| **Sprint 5** | Jul 24 2026 | Hot-path perf (zero-alloc/lock/LUT/branchless) — **DONE** |
| **F-K streams** | Jul 23 2026 | QKV one-CB, loss cert, multi-expert STE, long full-seq, KV bf16, CSA LSH, converter P-13 — **ALL DONE** |

### 6.4 Next session handoff

**Ledger L-01…L-15 + streams A–E + F1/F2/UI/F3/F3+ + TLM CLOSED. Sprint 5 CLOSED (2026-07-24). Sprints 6-7 CLOSED (2026-07-24).**

**2026-07-20 addendum:** model base reconverted sane; live telemetry TUI fixed (`[TELEM]`→log + key parser + Weight Δ panel); hot loops pointer-optimized (P-00/P-01). Banner LR fixed (now uses `qat_learning_rate()`). STE deadzone: converged base at default LR → ΔW≈0 is expected.

**2026-07-24 Sprint 5 addendum:** Hot path zero-alloc + zero-lock:
- **5.1** `evaluate_slime_block_moe` — 2 `Vec` allocs/llamada → workspace pre-alloc (`csa_index_scratch` + `csa_blocks_scratch` en `SlimeWorkspace`; `Vec::from_raw_parts` views + `mem::forget`)
- **5.2** GPU dispatch `Mutex<GemvGpuCache>` → `AtomicU8` tri-state (`GPU_GEMV_DISPATCH`: unknown/enabled/disabled); `mark_gpu_gemv_disabled()` desde `publish_no_device` paths
- **5.3** `ternary_gemv_backward` inner `match` → `TERNARY_NIBBLE_TO_F32[16]` LUT (branchless)
- **5.4** `pack_elut_prq` 3-way if/else → branchless `(over & pos) * 0x1 | (over & !pos) * 0xF`
- **Validación:** 312 tests (↑2 nuevos), clippy 0 errors, build `--all-targets` verde
- Ver `docs/sessions/MUD_SESSION_REPORT_2026-07-24_SPRINT_5.md`

**2026-07-24 Sprint 4b + Sprints 6-7 addendum:**
- **Sprint 4b:** `prefix_sum_batch4_avx2` (AVX2 batch 4-queries) + `find_kth_walk` (BMI1/BMI2 `_pdep_u32`+`_tzcnt_u32`) en `src/mud/fenwick.rs`. 2 tests nuevos.
- **Sprints 6-7:** `docs/seguridad/THREAT_MODEL.md` (threat model + DREAD), `HARDENING.md` (build/runtime hardening + audit trail), `INCIDENT_RESPONSE.md` (playbook + runbooks). `SECURITY.md` actualizado.
- **Herramientas:** `asm_kernel_bench` (16 kernels, `--quiet`, JSON válido, geometric mean), `fenwick_bench` (10 benchmarks, `--json`/`--quiet`, `FenwickLogF32`, correctness column). `mud.sh` v2.1 reorganizado.
- **Validación:** 344 tests (↑2 nuevos), clippy 0 warnings, `./mud.sh ci` pass.

**2026-07-26 Pendientes session addendum (Batch A/B/C/D-partial):**
- **F1 Fenwick nucleus** wired into `debate_trainer.rs` (eliminates per-token `vec![0;vocab_size]` + sort O(n log n) → `FenwickF32` binary search + `find_kth`). Match `main.rs:438` pattern. 4 per-doppel pre-allocated buffers (`logits_buf`, `reg_f32_buf`, `scratch_imag`, `fenwick_logits`) — P-01 eliminated.
- **F5b percentile loss gate** added to `loss_cert.rs`: `CertMode::Percentile` (`CertMode::Endpoints/WindowMeans/Percentile`) + `pub fn percentile_loss(...)` via FenwickU32 histogram, midpoint interp. Noise-robust median-of-window cert + `head_pct`/`tail_pct` knobs + env (`MUD_LOSS_CERT_MODE=percentile`). 6 tests.
- **8.1 / 8.3 (Muon/Newton-Schulz):** in-place NS iter + lifted `newton_schulz_inner_cpu` accept scratch + chunked reuse — ~4 allocs/NS-call → 0/call (the `grad.to_vec()` copy + per-NUM-iter scratch both eliminated).
- **8.2 (GaLore):** `compute_projection_matrix` now takes `&mut [f32] p_matrix_out` (no Vec return); `galore_step` pre-allocs p_matrix + g_low once.
- **8.5 doc:** `MUD_GRAD_CKPT_SEG` explained as layers-per-segment (count = `ceil(L/SEG)`).
- **8.9 (run_trainer):** checkpoint + corpus dirs now absolute via `current_dir().join(...)`; env override `MUD_TRAIN_CKPT_DIR` / `MUD_TRAIN_CORPUS_DIR`.
- **SEC-H1 caps** in `MudFile::load` (`MAX_META_ENTRIES=4096`, `MAX_TENSOR_ENTRIES=65536`, `MAX_STRING_LEN`, `MAX_SHAPE_DIMS`). 3 tests.
- **SEC-H8** `MUD_CIRCUIT_MAX_CYCLES` cap on infinite `run_training_circuit` loop.
- **SEC-H7** `COLLECT_FILES_MAX_DEPTH=32` recursion cap in `corpus::collect_files_inner`. 1 test (deep chain returns empty).
- **PERF-7** `forward_last_hidden_and_head` tied-output path: `emb_f32.clone()` (≈295 MB on smollm2) → `std::mem::take` after forward loop (zero copy, NLL allows the deferred move).
- **MUD_KV_COMPRESS_CELLV5** env var added with sync-warning in `kv_compressed.rs` resolver (Fase-3 reserved surface).
- **407 tests (histórico) | clippy 0 ✅** — *superado; el conteo vigente y su estado release están en §0.3*

**Deferred with rationale (next-session scope):**
- 5.1 (CellV5 Fase 2 ASM kernels `cmud_cell_v5_avx2.s`) — weight-heavy rewrite with bit-exact SlimeXCellV5 + FFI + AVX2-disabled fallback. Dedicated session.
- 5.2 (CellV5 Fase 3 KV pre-compression pipeline F32→CellV5→Delta→LZ4) — must preserve Stream I lossless invariant; design with validators. Dedicated session.

**2026-07-27 BATCH E COMPLETADO:** `step_elementwise_sync` (elementwise.comp) integrado en `jepa_stabilizer` (modos 4/5: normalize + ema, alpha fijo 0.1) y `stp_loss_and_grad` (modo 6: axpy para gradient injection). `evaluate_slime_block`/`evaluate_slime_block_moe` aceptan `Option<&mut AshQatDispatcher>`; `train_on_sequence` pasa `vk_qat.as_deref_mut()`. Fallback CPU automático cuando `gpu` es `None` (Iris Xe blacklisteado por `probe_gpu()`). 406 tests ✅ | clippy 0 ✅. Ver `docs/sessions/MUD_SESSION_REPORT_2026-07-27_BATCH_E_GPU_INTEGRATION.md`.

**2026-07-27 D5 signal keep-alive fase 2 COMPLETADA:** `dispatch_combined_batch_async` en `ash_backend.rs` ahora despacha optimizer updates (no-mHC, 4 buffers) y mHC updates (2 buffers) en un **único `vkQueueSubmit`**. Nueva función `step_combined_async_deferred` en `ash_qat_dispatcher.rs` combina optimizer step + mHC scale SGD. `train_step.rs` usa `step_combined_async_deferred` cuando `MUD_PIPELINE_OVERLAP=0` (default). Pipeline overlap path mantiene `step_pipeline` + `step_mhc_async` separados. 406 tests ✅ | clippy 0 ✅.

**2026-07-24 Test fixes + debt cleanup addendum:**
- `test_dense_ffn_names_default_w3_up` — stale `MUD_TRAIN_EXPERT=2` env leak. Fix: `remove_var` at test start.
- `test_slime_layer_shadow_opts_match_select` — missing `OPT_TEST_LOCK`. Fix: acquire lock + `clear_opt_env()`.
- Clippy: removed unnecessary `let _ =` on unit-returning `remove_var`.
- **P-08 dead code:** removed stale `#[allow(dead_code)]` from `DwReport` (used in 5 places); gated `ComplexSlimeRegister` + `complex_gemv_gauss_ref` with `#[cfg(test)]`; removed dead Vulkan QAT stub in `alignment.rs`.
- **Stale docs:** removed 4 stale banner-LR debt references from `AGENTS.md` + `GEMINI.md` (banner already uses `qat_learning_rate()`).
- **344 tests, clippy 0 warnings, release build OK.** Ver `docs/sessions/MUD_SESSION_REPORT_2026-07-24_TEST_FIXES.md`.

See `docs/manuals/LAUNCH_COUNTDOWN.md`.

**2026-08-03 Session addendum — Telemetry phantom collapse + parallel GEMV:**
- **Telemetry double-division bug fix** (`metrics.rs`): `push_grad()` was called once per chunk with already-averaged values but applied Welford's `÷ n` formula again (n=64), causing entropy to display as 8.56/64 = 0.136 instead of 8.56. Fix: direct assignment. Added `test_push_grad_no_double_division`.
- **Gradient `logit_scale` fix** (`train_step.rs:690`): `d_loss/d_logit = (prob - target) * logit_scale` (was `* temp_scale`, missing `scale_up` factor). 60× gradient magnitude recovery (0.0012 → 0.080).
- **Small-matrix GEMV threshold** (`gemv_gpu.rs` + `slime_forward.rs`): Inline AVX2 path when `n_out < 256`, avoiding PCorePool dispatch overhead on small model projections. Speed: 6.8 → 11.3 tk/s.
- **Parallel F32 GEMV** (`slime_forward.rs`): Added `f32_gemv_submit` (no-wait variant) — overlaps FFN up+gate and Q/K/V projections as 3 parallel tasks with a single `wait_all`.
- **P-01 fix**: `try_gpu_rms_output_norm` per-call `vec![]` allocations → workspace buffer reuse (`ws.q_f32`/`ws.k_f32`).
- **Telemetry header** (`session.rs:944`): `<pad0> <pad0> <pad0>` → `grad_norm stp_loss entropy`.
- **Consolidated `logit_scale`** (`train_step.rs:592`): Merged `scale_up` + `temp_scale` into single expression.
- **GRASSHOPPER FIX (2026-08-03):** Added SGDR (cosine restarts) to `apply_lr_schedule()`. The old single-cosine schedule caused coincidental LR match at step 2 and step 1059 (both ~1.07e-4, 21% of base) due to cosine symmetry — the LR monotonically decayed, leaving the model at suboptimal LR for 73% of training. SGDR creates periodic LR bounces (step 1059 LR: 1.07e-4 → 4.81e-4, 4.5× higher). Control: `MUD_LR_RESTART=0` (disable), `MUD_LR_CYCLES=N` (set count). 4 new tests. Constants in `constants.rs`.
- **460 tests (debug) | clippy 0 ✅** (was 456 + 4 new tests) — ⚠️ **release falla 1/461 (UB fenwick), ver §0.3**

**F1/F2/UI verification (2026-07-17):** `docs/architecture/MUD_TRAINER_TERNARY_JEPA_MHC.md` §9–§10; plan `docs/research/MUD_PLAN_MHC_STP_TRAINABLE.md`. Trainer default now project-adapted corpus + STP on + 64 steps/chunk (`mud.sh train`).

| Order | Action |
|-------|--------|
| **0** | **Backlog activo = `R-##`** → [`docs/planning/MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md`](docs/planning/MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md). Empezar por la **Ola 0** (R-01 CI válido, R-03 UB fenwick, R-04 git) |
| 1 | Órdenes de ejecución: Ola 1 (entrenamiento) y Ola 2 (seguridad) en paralelo; Ola 3/4 después de Ola 1 |
| 2 | Histórico (sólo si no contradice §0.3): `MUD_IMPROVEMENTS_POST_AE.md`, `MUD_GAP_ANALYSIS_POST_L15.md` |
| 3 | Medir, no afirmar: `./mud.sh ci` (release) · `bench-infer` · `cert-loss` |

**Read first:** §0.3 · plan maestro `R-##` · `LAUNCH_COUNTDOWN.md`.

---

## 7. Documentation Mandate (P-18)

| Path | Content |
|------|---------|
| `docs/audits/` | `MUD_AUDIT_REPORT_VXX*.md`, **`MUD_AUDIT_2026-10-11_LOW_COMPUTE.md`** (vigente), `MUD_AUDIT_LATEST.md` |
| `docs/planning/` | **`MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md`** (backlog activo `R-##`), `PHASE_D_TRAINING_SCALE_PLAN.md` |
| `docs/seguridad/` | `THREAT_MODEL.md`, `HARDENING.md`, `INCIDENT_RESPONSE.md` |
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
- `./mud.sh plan` is the **PRIORIDAD 0** entry (prints the `R-##` backlog; documentation-only, wraps no bin).
- Destructive ops require confirmation.
- Auto-select trained checkpoints when present for chat/train.

---

## 9. Complete Policy Index

| ID | Policy | Severity | Notes |
|----|--------|----------|-------|
| **P-00** | Raw pointer mastery in hot path | CRITICAL | |
| **P-01** | Zero-allocation hot-loop (`SlimeWorkspace`) | CRITICAL | |
| **P-02** | `SlimeRegister` / activation contract | CRITICAL | **f32 compute, f16 storage** — packed dual `(re:f16, im:f16)` register, `read_accum()` returns f32 via F16C |
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
| **P-15** | Hot PRQ clamp to ternary grid | MANDATORY | sparsity **0.7·s** — ⚠️ **violado en `checkpoint.rs:211,683` (usa 0.5·s)**; alinear en R-13 |
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

- **Format:** `.mud` (`MUD\x01`) Ternary2Bit ELUT + `prq_scale` + norms; optional expert tensors. **Nota (2026-10-11):** "Ternary2Bit" empaqueta **4 bits/peso** (8 pesos/`u32`) ⇒ 2.52× la entropía ternaria de 1.585 bits; el formato denso (5 trits/byte ≈ 1.6 bpp) es el objetivo de R-42
- **Forward (`evaluate_slime_block` / `_moe`):** RMSNorm → QKV GEMV → SDPA/GQA (HCA+CSA) → O → Residual+JEPA/mHC → FFN/ExpertBus → Residual+JEPA/mHC
- **Training:** STE QAT; full-seq windows (default); Sampled Softmax (~512); Adam/Muon/GaLore/Chunked LIVE
- **Threading:** `PCorePool` via `default_pcore_threads()` / `MUD_PCORE_THREADS`
- **Device:** GEMV auto (`gemv_policy`); NS/MHA/RMSNorm GPU helpers when profitable

```
fn select_optimizer(rows, cols) -> Strategy  // LIVE on step (L-01 + stream A)
```

**Next improvements:** `docs/research/MUD_IMPROVEMENTS_POST_AE.md` (F+ backlog) · `docs/research/MUD_FENWICK_TREE_OPTIMIZATION_PLAN.md` (F4, deferred) · Sprints 6-7 (docs/seguridad, backlog).

---

*MUD: Bare-metal ternary engine. Policies live here. Vision lives in ROADMAP.md. Session detail lives in AGENTS.md.*
