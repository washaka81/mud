# INFORME DE LOGROS vs DEUDA — Forge LLM

**Fecha:** 2026-10-11 (secciones §1 en adelante son **histórico 2026-07-26**; el estado vigente es §0)  
**Base:** `GEMINI.md` (SSOT) + verificación en código

---

## 0. ESTADO VIGENTE (2026-10-11) — auditoría completa

> Este informe reflejaba un estado optimista basado en inspección, no en ejecución. La auditoría del 2026-10-11 ejecutó el proyecto y encontró que **varias afirmaciones de §1–§3 son falsas o prematuras**.

| Comprobación ejecutada | Resultado |
|---|---|
| `cargo clippy --all-targets` | ✅ 0 warnings |
| `cargo test --lib` (debug) | ✅ 461 passed |
| `cargo test --lib --release` | ❌ **460 passed / 1 FAILED** — UB en `fenwick.rs:307-312` (pasa en debug, falla con `-O`) |
| CI remoto | ❌ workflow inválido (`on:` anidado dentro de un job) ⇒ **nunca se ejecuta** |
| Entrenamiento | ❌ el trainer optimiza `token_embd` pero la inferencia usa `output.weight` (nunca sincronizada); backward del FFN con derivada ReLU² sobre un forward SiLU; mHC sin factores `alpha`/`(1-gate)·beta`; deriva de escala PRQ ×0.52 por guardado; loss plano en todos los logs |
| Inferencia `models/ale.mud` | ⚠️ **~46 tok/s** (9 % del BW), salida incoherente; LM head en f32 denso (77.6 MB/token), single-thread |
| Formato de pesos | ⚠️ **4 bits/peso** (2.52× la entropía ternaria de 1.585 bits) |
| Seguridad | ⚠️ sin validación `metadata↔shape` (OOB/OOM), parser GGUF sin cotas, 19 `unsafe` sin `// SAFETY:` |
| Trazabilidad | ❌ repo **sin `.git`** |

**Deuda abierta = plan maestro:** [`planning/MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md`](planning/MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md) (`R-##`, 7 olas).
**Evidencia completa:** [`audits/MUD_AUDIT_2026-10-11_LOW_COMPUTE.md`](audits/MUD_AUDIT_2026-10-11_LOW_COMPUTE.md).

---

## 1. LOGROS RECIENTES (histórico — 2026-07-26)

### 2026-07-26 — Sprint 8 (Batch A/B/C/D-partial) — cierre de pendientes

Detalle y pendientes deferral rationale: `docs/sessions/MUD_SESSION_REPORT_2026-07-26_PENDIENTES.md`.

| Item | Estado |
|------|--------|
| **A.** F1 Fenwick nucleus en `debate_trainer.rs` (patrón `main.rs:438`, +4 Doppelganger pre-alloc buffers, P-01 per-token alloc → 0) | ✅ |
| **A.** F5b `CertMode::Percentile` + `percentile_loss` via `FenwickU32` (Stream K noise-robust; 6 tests) | ✅ |
| **B.** Muon/Newton-Schulz in-place + lifted scratch (chunked reuse) — `grad.to_vec()` eliminado, ~4 alloc/NS-call → 0 | ✅ |
| **B.** GaLore `compute_projection_matrix → &mut [f32]`; `galore_step` prealloc p_matrix + g_low (P-01) | ✅ |
| **B.** `grad_checkpoint.rs` doc clarifier SEG↔SEGS; `run_trainer.rs`绝对 paths + `MUD_TRAIN_CKPT_DIR`/`MUD_TRAIN_CORPUS_DIR` env | ✅ |
| **C.** SEC-H1 caps `MudFile::load` (`MAX_META_ENTRIES=4096`, `MAX_TENSOR_ENTRIES=65536`, `MAX_STRING_LEN`, `MAX_SHAPE_DIMS`) + 3 tests | ✅ |
| **C.** SEC-H8 `MUD_CIRCUIT_MAX_CYCLES` cap on `run_training_circuit` infinite loop | ✅ |
| **C.** SEC-H7 `COLLECT_FILES_MAX_DEPTH=32` recursion cap en `corpus::collect_files_inner` + 1 test | ✅ |
| **C.** PERF-7 tied-output `forward_last_hidden_and_head`: `emb_f32.clone()` (~295 MB smollm2) → `std::mem::take` (zero copy) | ✅ |
| **D.** `MUD_KV_COMPRESS_CELLV5` env reservado con sync warning (Fase-3 not-yet-implemented) | ✅ (plumbing) |
| **D.** 5.1 CellV5 Fase 2 ASM kernels (`cmud_cell_v5_avx2.s` con 3 batch: wave_collapse/phase_repulsion/gauss_mul) | 🟡 DEFERRED |
| **D.** 5.2 CellV5 Fase 3 KV pre-compression pipeline (F32→CellV5→Delta→LZ4; invariante lossless Stream I) | 🟡 DEFERRED |
| **E.** Vulkan D5 `step_elementwise_sync` integration en `slime_jepa`/`stp_loss` (opt-in MUD_USE_VULKAN) | 🟡 DEFERRED HW |

**Tests:** 407 (397→407; +10); **clippy `--all-targets`:** 0 warnings; **release build:** verde.

### 2026-07-16 — Depth streams A–E

| Stream | Item | Estado |
|--------|------|--------|
| A | Adam / SparseAdam moments | ✅ |
| B | MoE `.mud` load + train-expert | ✅ |
| C | GPU GEMV auto policy | ✅ |
| D | Full-seq packed train | ✅ |
| E | CSA top-k lightning indexer | ✅ |

Detalle: `docs/sessions/MUD_SESSION_REPORT_2026-07-16_STREAMS_AE.md` · backlog F+: `docs/research/MUD_IMPROVEMENTS_POST_AE.md`.

### 2026-07-23 — Q2_0→F32 conversion fix & Optimization Sprint (P0-P5 + Orbit G+/J/I)

Detalle y Cálculos Matemáticos: `docs/audits/MUD_SYSTEM_PERFORMANCE_AND_COGNITIVE_GAINS_2026-07-23.md`.

| Item | Estado |
|------|--------|
| Q2_0→ELUT+PRQ per-row scale averaging corregido | ✅ |
| `f32_gemv_rowwise` + `is_f32_weights` dispatch | ✅ |
| Backward null-scale SEGFAULT guard | ✅ |
| Dead code `q2_0_to_elut_prq`/`elut_bits`/`PlanItem::Q2` eliminado | ✅ |
| **P0:** AVX2 batch register ops (`registers_to_f32_batch`, `f32_to_registers_batch`, `gate_batch`) | ✅ |
| **P1:** Eliminar `registers_tmp` (doble buffer reusando `ffn_out_f32`) | ✅ |
| **P2:** Raw pointer workspace (`AlignedArray<T>` reemplaza `Vec<T>` en `SlimeWorkspace`) | ✅ |
| **P3:** Vectorización SIMD AVX2 (batch 8) del JEPA stabilizer | ✅ |
| **P4:** LM head batch desempaque cero-alloc (`registers_to_f32_into`) | ✅ |
| **P5:** Alineación nativa a 64 bytes para modelos nano/micro (`hidden ≤ 384`) | ✅ |
| **G+ SlimeX Unpacking + Joint Step:** Desintercalado ternario en `ShadowExpertBus::mount` + `apply_joint_multi_expert_step` | ✅ |
| **Learned CSA `W_compress`:** `lsh_signature_learned` con proyecciones AVX2 | ✅ |
| **Native Google BF16:** `KvDtype::BF16` en `kv_dtype.rs` y `slime.rs` | ✅ |
| **MUD-WORM Architecture:** Banco de memoria neuronal latente acoplable en caliente (`docs/research/MUD_LATENT_NEURAL_MEMORY_WORM_ARCHITECTURE.md`) | 📘 Especificado |
| **Tabla de Continuidad & Fusión `.mud`:** Tabla plana de punteros C-repr en RAM + fusión física en un solo binario (`docs/research/MUD_MEMORY_CONTINUITY_TABLE_AND_FILE_FUSION.md`) | 📘 Especificado |
| **MUD Master Cell v5 (FP8 + 256-Slot Memory Table):** Célula `u32` ultra-comprimida (FP8 + 256 slots de direccionamiento + JEPA + STE) (`docs/research/MUD_ULTRA_COMPRESSED_REGISTER_AND_MEMORY_CONTINUITY_MASTER.md`) | 📘 Especificado |
| **Cálculos Matemáticos de Ganancia MUD v5 & WORM:** Densidad SIMD (4.0×), Slots (16.0×), Resolución Punteros (54.87×), Hot-Plug (37,500×) (`docs/audits/MUD_MASTER_CELL_V5_AND_WORM_GAIN_CALCULATIONS_2026-07-23.md`) | 📊 Auditado |
| **AOT Engine Boot Indexer (`BootMemoryIndexer`):** Indexación automática de 256 slots `.mud` en $t_0$ con 0 ms cold-start (`docs/research/MUD_ENGINE_BOOT_INDEXER_SPECIFICATION.md`) | 📘 Especificado |
| **Ecuación Maestra C-MUD Thinker:** Ecuaciones de interferencia cuántica-like (constructiva/destructiva) sobre la célula `u32` v5 (`docs/research/CMUD_THINKER_INTEGRATION_MASTER_EQUATION.md`) | 📘 Especificado |
| **Demostración de Proporción Óptima 8:8:8:8 (u32):** Análisis de rendimiento SIMD (1 ciclo) y 100% L1 cache fit para 256 slots (`docs/audits/OPTIMAL_U32_BIT_ALLOCATION_RATIO_PROOF.md`) | 📊 Demostrado |
| **Fase 1 SlimeX LLM (ASM & Punteros P-00/P-01):** Kernel AVX2 ensamblador puro `src/asm/slimex_cell_v5_avx2.s` + FFI + tests (`docs/architecture/SLIMEX_PHASE1_ASM_POINTERS_LOG.md`) | ✅ Certificado |
| **SlimeX Trainer Integration (SX-01…SX-07):** `slimex_cells: AlignedArray<u32>` en `SlimeWorkspace` (P-01); `snapshot_to_slimex()` raw-ptr hot path; `ShadowExpertBus` inicializado en `SlimeLayerShadowF32::new()` cuando `MUD_SLIMEX=1`; entry `./mud.sh slimex-train` | ✅ LIVE |
| 271 tests + clippy 0 errors | ✅ |

### 2026-07-15/16 — Ledger y polish

| # | Item | Estado |
|---|------|--------|
| 1 | Document hierarchy GEMINI / AGENTS / VISION | ✅ |
| 2 | ASM polish + NaN guards | ✅ |
| 3 | `lm_head_logits_avx2` wired | ✅ |
| 4 | L-01…L-15 + Phase B/B+ | ✅ |
| 5 | Compute stack doc | ✅ |

---

## 2. RUNTIME TRUTH (do not contradict)

| Subsystem | Live path | Not live / next |
|-----------|-----------|-----------------|
| Forward GEMV | AVX2×8 + **auto ash** (`gemv_policy`) | QKV multi-matrix one CB (**F**) |
| Accum dtype | **f32** | dual-f16 / i8-act GEMV |
| Weight storage | ELUT 4-bit + PRQ | — |
| QAT step | Muon/GaLore/Chunked + **Adam moments** + STE pack | Multi-expert STE joint (**G**) |
| Train sequence | **Full-seq windows** (default) | Joint multi-token BPTT (**H**) |
| Attention history | HCA + dense + **CSA top-k** (infer) | CSA v2 LSH / W_compress (**J**) |
| KV storage | f32 ring + f32 HCA | bf16/quant (**I**) |
| LM head | `lm_head_logits_avx2` | — |

---

## 3. DEUDA ABIERTA

**Ledger L-01…L-15:** all **DONE** (see `GEMINI.md` §6).

**Open product backlog:** streams **F–L** in `docs/research/MUD_IMPROVEMENTS_POST_AE.md`.

---

## 4. SUPERSEDED CLAIMS (correct if you see old text)

| Old claim | Correction |
|-----------|------------|
| Adam moments still stub / missing | **LIVE** stream A |
| GPU GEMV API only / not in forward | **LIVE** auto + force-on |
| Train only pairs @ pos=0 | **Default full-seq** (D); pairs if `MUD_TRAIN_FULL_SEQ=0` |
| MoE ExpertBus only, no `.mud` load | **LIVE** `moe_load` (B) |
| CSA/HCA = mean-pool only | **+ top-k indexer** (E) |
| Muon only “planned” | L-01 LIVE |
| Next work = L-01 | **Wrong** — next = F+ improvements |

---

## 5. TOOLING

- `./mud.sh ci` · `audit-full` · `health`
- `mud_full_audit` sections 3b CSA, 6a GEMV, 6b MoE, 8 full-seq
- `gemv_auto_bench` — break-even on device
- `training_healthcheck` — LIVE policy lines

---

## 6. HANDOFF (siguiente)

1. **T-0 GO** — foundation shippable (`docs/manuals/LAUNCH_COUNTDOWN.md`)  
2. **Orbit F–L foundations DONE** · remaining: router STE BPTT, residual-bank wire, nightly e2e cert  
3. Keep green: `./mud.sh ci` · `audit-full` · `gemv_auto_bench`

*Update this file when closing F+ items; do not invent parallel ledgers.*
