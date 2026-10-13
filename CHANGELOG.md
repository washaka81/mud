# Changelog

All notable changes to Forge LLM (MUD) are documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/).

---

## [Unreleased] — 2026-10-13 — Ola 2 (F2) completa + Ola 3 iniciada (R-40, R-48)

### Added
- **`./mud.sh unsafe-audit`** (R-36): gate embebido (awk, sin Python — P-07) que verifica que todo `unsafe` de producción esté justificado (`SAFETY` adyacente, contrato `//! SAFETY:` de módulo para sitios no críticos, alto riesgo siempre adyacente). Excluye tests/`#[cfg(test)]`/`src/bin`. Añadido a `mud.sh ci` y `.github/workflows/ci.yml`.
- **`gemv_gpu::ternary_head_logits`** (R-40): GEMV ELUT por filas 4/8 + `PCorePool` + PRQ por fila para el LM head; sin materializar la matriz f32.
- **Tests de gate por build** (R-37): `test_terminal_sandbox_disabled_by_default`, `test_rlvr_critic_disabled_by_default` (default build ⇒ `PermissionDenied` / `-1.0`).
- **Tests R-32/R-33/R-34/R-35/R-38/R-40** (28 tests nuevos + 2 proptest GGUF fuzz).

### Changed
- **`src/gguf/mod.rs`** (R-32): parser con `GgufCursor` bounds-checked, caps SEC-H1, arrays ≤4M/depth ≤8, `general.alignment` validado, offsets `checked_add` + tamaño exacto por tipo.
- **`src/mud/mod.rs`** (R-33): `ecc_verify_all -> Result`; paridad validada contra el extent del tensor base (owned `len>=need`, mmap `.get()`).
- **`src/mud/worm.rs`** (R-34): `get_embedding_row` exige `Float32` rank-2 y deriva dims del shape.
- **`src/asm/mod.rs` + `gemv_gpu.rs` + `slime_backward.rs`** (R-35): `ternary_gemv_tail` hace exactos los wrappers para `n%8∈1..7`; strides `/8`→`div_ceil(8)` (layout canónico R-15).
- **`src/mud/workspace_agent.rs`** (R-38): resolución por ancestro existente más profundo (`symlink_metadata` + `canonicalize`), enlace colgante ⇒ `Err`, apertura con `O_NOFOLLOW`.
- **`src/main.rs`** (R-40): head ternario packed (tied/untied) sin `out_f32_owned`; post-escalado de logits equivalente al pre-escalado denso. `models/ale.mud`: greedy **4.6×**, default **2.4×** (3.5× vs baseline congelado), RSS **201→127 MB**, logits idénticos.
- **`SECURITY.md`** (R-37): el RCE de `unsafe-rce`/`unsafe-cli` documentado como **estructural** (feature-gated, no "0 RCE"); filas loader/GGUF/ASM/skills actualizadas; sección path-traversal con R-38.
- **`skills/data_analysis.rs`** (R-37): lecturas confinadas vía `AgentWorkspace::read_file` (inerte sin workspace).
- **`mud.sh` + `MUD_ENV_REFERENCE.md`** (R-48): decisión escrita de `MUD_GPU_GEMV` (producción CPU=`0`; `auto` opt-in).
- **Docs de estado**: `GEMINI.md §0.3`/§6.2, `AGENTS.md` (last sync), plan maestro §12, `MUD_AUDIT_LATEST.md` → Ola 2 cerrada, 28/55.

### Fixed
- **GGUF truncado/hostil** ya no puede hacer panic/OOM (R-32); **ECC diminuta** ⇒ `Err` (R-33); **embedding ternario en worm** ya no se lee como f32 (R-34); **GEMV** con `n%8∈1..7` exacto y sin over-read (R-35); **symlink colgante** ya no escapa del workspace (R-38); **LM head** ya no re-lee 77.6 MB densos por token (R-40).

### Evidencia
- `cargo test --lib --release --features proptest -- --test-threads=2` → **515 passed / 0 failed / 2 ignored** (2026-10-13).
- `cargo clippy --all-targets --all-features -- -D warnings` → 0 warnings.
- `./mud.sh unsafe-audit` → `OK - 0 sin comentario (R-36)` (532 `unsafe` productivos).
- Informe: [`docs/sessions/MUD_SESSION_REPORT_2026-10-13_OLA2_OLA3.md`](docs/sessions/MUD_SESSION_REPORT_2026-10-13_OLA2_OLA3.md).

---

## [Unreleased] — 2026-10-11 — Auditoría + Plan Maestro (**documentación + hook operativo; sin cambios en el motor**)

### Added
- **`docs/audits/MUD_AUDIT_2026-10-11_LOW_COMPUTE.md`** — auditoría completa: correctitud (entrenamiento/inferencia), seguridad de entrada, eficiencia/roofline con mediciones en vivo, deriva documentación↔código y contraste con literatura arXiv 2024-2026.
- **`docs/planning/MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md`** — plan maestro de correcciones, limpieza y mejoras (`R-##`, 55 ítems, 7 olas, criterios de aceptación, riesgos, Definition of Done). Declarado **PRIORIDAD 0**.
- **`./mud.sh plan`** — hook operativo que imprime el backlog `R-##` con su estado por ítem y el progreso (`n/55`). Sólo lectura de documentación; no envuelve ningún `[[bin]]`.

### Changed
- **`GEMINI.md`**: jerarquía de documentos con el plan como **PRIORIDAD 0**; nueva sección **§0.3** con el estado verificado; §6.2 apunta al backlog `R-##`; §7 indexa `docs/planning/` y `docs/seguridad/`; §9 nota la violación de **P-15** (deadzone 0.7·s en política vs 0.5·s en `checkpoint.rs`); §10 aclara que `Ternary2Bit` son **4 bits/peso**.
- **`AGENTS.md`**: nuevo bloque *Last sync 2026-10-11*; corregidas in situ las afirmaciones falsas (conteos de tests, "0 OOB", `MUD_GPU_GEMV` default, wormholes −50 %, Fenwick batch prefix sum); ledger marcado como histórico frente al backlog `R-##`.
- **`ROADMAP.md`**: bloque **PRIORIDAD 0** que bloquea las fases A–E hasta cerrar las Olas 0–1; estado corregido.
- **`docs/manuals/LAUNCH_COUNTDOWN.md`**: el `T-0 GO` de 2026-07-16 queda **RETIRADO** (gates G2/G4/G5/G6/G7 falsos o vacíos) y se re-gatea el despegue por las olas `R-##` (board "RE-LAUNCH").
- **`docs/audits/MUD_AUDIT_LATEST.md`** pasa a ser **puntero** a la auditoría vigente + plan maestro.
- **`README.md`**, **`docs/README.md`**, **`TREE.md`**, **`docs/STATUS_REPORT.md`**, **`SECURITY.md`** — estado vigente, índices y claims de seguridad corregidos.

### Retirado (afirmaciones, no código)
- "460 tests ✅" (release falla 1/461 por UB en `fenwick.rs:307-312`), "0 OOB", "Rayon removed from runtime crate" (`rayon` sigue en `Cargo.toml`), "`MUD_GPU_GEMV` default `auto` en `mud.sh`" (fuerza `0`), "Fenwick batch prefix sum COMPLETADO (~4×)" (UB + 0 llamadores), "wormholes peer-to-peer −50 % host compute" (mismo trabajo con un multiplicador local).
- Regla nueva **R-73**: ninguna fila de estado sin `(comando, salida, fecha)`.

---

## [Unreleased] — 2026-07-24

### Added
- **SPRINT-6 (ASM benchmarks):** `asm_kernel_bench` tool expanded from 10 to 13 benchmarks:
  added `ternary_gemv_4rows`, `ternary_gemv_8rows`, `ternary_gemm_batch4` (batch4 multi-token).
  Added `--quiet` flag for CI/scripting. JSON output fixed (valid JSON with no trailing commas).
  Geometric mean replaces arithmetic mean for speedup summary (more appropriate for ratios).
- **SPRINT-6 (Fenwick benchmarks):** `fenwick_bench` tool expanded from 9 to 10 benchmarks:
  added `log_prefix_product` (FenwickLogF32 — log-space multiplicative prefix products).
  Added `--json` and `--quiet` flags. AVX2 detection in header. Correctness column (✅/❌).
  Geometric mean replaces arithmetic mean.

### Added (Sprint 4b)
- **Fenwick `prefix_sum_batch4_avx2`:** AVX2-optimized batch prefix_sum for 4 indices
  simultaneously. Processes 4 independent Fenwick tree walks in lockstep using SIMD
  horizontal addition. 2 new tests.
- **Fenwick `find_kth_walk`:** BMI1/BMI2-optimized find_kth using `_pdep_u32` + `tzcnt`
  for hardware bit manipulation walk. 1 new test.

### Added (Sprints 6-7: Security Documentation)
- **`docs/seguridad/THREAT_MODEL.md`:** Threat model, 6 componentes, DREAD scoring
- **`docs/seguridad/HARDENING.md`:** Build/runtime hardening, input validation, audit trail
- **`docs/seguridad/INCIDENT_RESPONSE.md`:** Playbook, runbooks, PR checklist
- **`SECURITY.md` actualizado:** Enlaces a docs/seguridad/ + security test commands

### Fixed (P-13 soft debt)
- **Stray `1e-8` → `EPSILON_FLOOR`:** Replaced hardcoded `1e-8` with `EPSILON_FLOOR`
  (SSOT constant from `constants.rs`) in 6 files: `galore.rs`, `holographic_loss.rs`,
  `integral_threshold.rs`, `inference.rs`, `optim.rs`, `self_play.rs`, `train_step.rs`.
  P-13 compliance: no more magic epsilon values outside the SSOT.
- **Banner LR:** Already fixed — uses `qat_learning_rate()` (0.0005) via `constants.rs`.
  The AGENTS.md debt item (`corpus_trainer.rs:1703` `3e-4`) is resolved.
- **Stray `0.7` → `SPARSITY_THRESHOLD_RATIO`:** Replaced hardcoded `0.7` with
  `SPARSITY_THRESHOLD_RATIO` (SSOT) in `ezop.rs` (2 ocurrencias) and `optim.rs` (1 ocurrencia).
- **Debate constants centralizados:** Added `DEBATE_R_LOSE` (0.7), `DEBATE_JEPA_LAMBDA` (0.05),
  `DEBATE_WD` (0.01) to `constants.rs` SSOT. Reemplazados en `arena_judge.rs` y `debate_trainer.rs`.
- **Debate loss threshold:** Added `DEBATE_LOSS_THRESHOLD` (0.01) to `constants.rs`. Reemplazado en
  `debate_trainer.rs` (2 ocurrencias).
- **Redundant clones eliminados (P-08):** Removed 4 redundant `.clone()` calls:
  `debate.rs` (Arc<AtomicUsize> + Arc<AtomicBool>), `session.rs` (Vec<String>), `tests.rs` (Vec<u8>).
  Eliminado dead `ex_idx` variable en `debate.rs` (shadowed, unused).
- **RMSNorm epsilon centralizado:** Added `RMS_NORM_EPS` (1e-5) to `constants.rs`.
  Reemplazado en `debate_trainer.rs` (1e-6 → 1e-5) e `inference.rs` (fallback 1e-5).
- **JEPA alpha constants:** Added `JEPA_ALPHA_MIN` (0.01) + `JEPA_ALPHA_RANGE` (0.09) to `constants.rs`.
  Reemplazado en `slime_jepa.rs` (AVX2 path + scalar path).

### Fixed
- **P-06 (CI broken):** 19 clippy warnings from Sprint 5 — `ezop.rs` branchless pack
  (`identity_op` + `operator_precedence`), `corpus_trainer/mod.rs` unused imports + macro
  ordering, `train_step.rs` doc overindentation + `too_many_arguments`.
  `cargo clippy --lib -- -D warnings` now passes. `./mud.sh ci` completes.
- **P-08 (dead code):** Removed legacy shims `ternary_state_f32` / `jepa_energy_f32`
  from `slime.rs:182-189` (no callers).
- **P-18 (root files):** Moved 5 stray files from repo root to `docs/` and `training/`:
  `arreglar_trainer.md`, `mud_disassembly.txt`, `dump.txt`, `corpus.txt`. Deleted
  `test_qwen_rope.py` (P-07 violation).
- **P-17 (unwraps):** Replaced 6 `owned_data.as_ref().unwrap()` with safe `if let Some(d)`
  pattern in `alignment.rs`. Replaced 2 `tensors.get(...).unwrap()` with `match` + `continue`
  + `log::warn!` in `alignment.rs`.
- **PERF-7:** Eliminated `head_w.to_vec()` allocation in `inference.rs:455`. Non-tied
  path now moves `out_f32_owned` directly (~77 MB save for smollm2, ~1.3 GB for large models).
- **Macro `println!`:** Moved before `pub mod` declarations in `corpus_trainer/mod.rs`
  so `MUD_CIRCUIT_TUI` gating is visible to all submodules.

### Added
- **F4 Fenwick Trees:** `src/mud/fenwick.rs` with `FenwickF32`, `FenwickU32`, `FenwickI32`.
  Zero-alloc `with_capacity` + `reset_zeroed`. 14 tests. `MUD_FENWICK=1` (default on).
- **SECURITY.md:** Threat model, feature flag documentation, path traversal protection.
- **Model:** `models/smollm2.mud` created from `models/smollm2/` (SmolLM2, hidden=576,
  30 layers, vocab=128256). Health: 🟢 STRUCTURALLY HEALTHY. Scale audit: ratio 0.374.

### Changed
- `corpus_trainer/mod.rs`: Removed `#![allow(clippy::type_complexity)]` blanket allow.
- `train_step.rs`: Added `WeightMatrixRef` type alias to fix `type_complexity` warning.
  Added `#[allow(clippy::too_many_arguments)]` on `train_on_sequence` with TODO for struct refactor.

### Tests
- 326 tests (312 original + 14 new Fenwick tests), 0 failures, 2 ignored.
- `cargo clippy --all-targets`: 0 warnings.
- `cargo clippy --lib -- -D warnings`: 0 errors.

---

## [Sprint 6] — 2026-07-24

### Added
- **ASM Kernel Benchmark Suite:** `tools/asm_kernel_bench.rs` (13 kernels: ternary_gemv, dot_product, sum_squares, silu, rms_norm, lm_head, adam_step, sgemm_abt, ezop_axpy, ezop_scale, fenwick_prefix_sum, fenwick_find_kth, fenwick_build). Correctness ✅ ALL PASS.
- **Fenwick Benchmark Suite:** `tools/fenwick_bench.rs` (9 benchmarks: F4 threshold, F1 nucleus, F3 neg-sample, prefix_sum, find_kth, build_from, build_from_linear, prefix_sum_raw, find_kth_avx2).
- **AVX2 Batch Scale Application:** `src/asm/apply_scales.s` + `apply_row_scales_8_avx2` — replaces scalar loop in `ternary_gemv_rowwise_submit` for 8-row batches.
- **CMUD AVX2 Batch Complex Multiply:** `src/asm/cmud_batch.s` + `cmud_gauss_mul_batch_avx2` — 8 complex multiplies in 6 instructions.
- **CMUD Batch Wave-Collapse:** `wave_collapse_batch_avx2` using Rust AVX2 intrinsics (precision-guaranteed).
- **FenwickLogF32:** Multiplicative prefix products via log-space (`log(a·b) = log(a) + log(b)`). 4 new tests.
- **mud.sh:** Added `fenwick-bench` command.

### Fixed
- **P-08 (dead code):** Removed `haar_wavelet_decompose` from converter quantizer. Removed `repack_bitnet_to_mud`, `pack_native_ternary_f32`, `pack_ternary_from_f32`, `embedding_rowwise_ternarize` from converter (dead paths `is_native_ternary`/`is_u8_direct` were hardcoded `false`). Removed unused `HashMap` import.
- **P-13 (hardcoding):** Reviewed all source — no hardcoded model dimensions in production code. Test/benchmark fixtures use small hardcoded values (documented).
- **Housekeeping:** Reviewed SlimeX, CMUD, Universal Converter, and Trainer. SlimeX hot path is GEMV (already optimized). CMUD complex math vectorized. Converter dead code purged. Trainer already uses raw pointers + LUT + Fenwick + AVX2 scale.

### Tests
- 342 tests (335 + 7 new: 4 FenwickLogF32 + 3 CMUD batch), 0 failures, 2 ignored.
- `cargo clippy --all-targets`: 0 warnings.
- `cargo clippy --lib -- -D warnings`: 0 errors.
- `./mud.sh ci`: All suites pass (tests + clippy + health + cmud + pointer).

---

## [V37] — 2026-07-21

### Fixed
- Full CI battery (`./mud.sh ci`) — 0 clippy warnings, all tests passing.
- C-MUD reasoning validation — `forward_ok = true`, `logits_finite = true`.
- Pointer-address layout audit — 106,168,320 ternary elements decode identically.

### Tests
- 256 core tests, 11 P-13 property tests, 9 loss cert tests, 28 C-MUD tests.

---

## [Sprint 5] — 2026-07-24

### Fixed
- 5.1: Zero-alloc `evaluate_slime_block_moe` (workspace pre-alloc).
- 5.2: `Mutex` → `AtomicU8` GPU dispatch fast path.
- 5.3: `TERNARY_NIBBLE_TO_F32` LUT for `ternary_gemv_backward`.
- 5.4: Branchless `pack_elut_prq`.

### Tests
- 312 tests (310 + 2 new), clippy 0 errors.

---

## [Sprint 4] — 2026-07-24

### Fixed
- `corpus_trainer.rs` split into 7 modules (`corpus_trainer/`).
- `MudError` enum with `thiserror`.
- `trait Optimizer` in `src/mud/optim.rs`.
- Broke `corpus_trainer ↔ slime_backward` circular dependency.

### Tests
- 310 tests, clippy 0 errors.
