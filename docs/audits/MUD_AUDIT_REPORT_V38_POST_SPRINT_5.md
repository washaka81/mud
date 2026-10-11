# MUD Audit Report V38 — Post-Sprint 5 Comprehensive Audit

**Fecha:** 2026-07-24  
**Alcance:** Build, tests, clippy, security, performance, architecture, docs, model health  
**Auditor:** Sistema de auditoría automatizada  
**Build status:** `cargo check --release` ✅ | `cargo clippy --all-targets` ❌ 19 warnings | `cargo test --lib` ✅ 312/312  

---

## 0. Resumen Ejecutivo

| Categoría | Estado | Detalle |
|---|---|---|
| **Build** | ✅ PASS | `cargo check --release` — 6 warnings (solo en `corpus_trainer/mod.rs`) |
| **Tests** | ✅ PASS | 312 passed, 0 failed, 2 ignored |
| **Clippy (P-06)** | ❌ **VIOLADO** | 19 warnings → `cargo clippy --lib -- -D warnings` FALLA (CI roto) |
| **Security (SEC-1/2/3)** | ✅ FIXED | RCE detrás de feature flags; path traversal validado |
| **Performance (PERF-1…7)** | ✅ 6/7 FIXED | PERF-7 parcialmente abordado |
| **Bugs (BUG-1/2)** | ✅ FIXED | OOB clampeado; panics → `nan_skip` |
| **Architecture (ARCH-1…5)** | ✅ FIXED | Split de `corpus_trainer`; `trait Optimizer`; `MudError`; ciclo roto |
| **Docs (P-18)** | ❌ VIOLADO | 5 archivos stray en root |
| **Docs (P-06 stale)** | ❌ STALE | `MUD_AUDIT_LATEST.md` (V37) claims 0 warnings, hay 19 |

**Veredicto global:** El proyecto es funcional y 312 tests pasan, pero **el CI está roto** por 19 warnings de clippy introducidos en el Sprint 5 (2026-07-24). La documentación de auditoría más reciente (V37) es **obsoleta** y no refleja el estado actual.

---

## 1. Build & Test Results

### 1.1 `cargo check --release`
```
Finished `release` profile [optimized] target(s) in 4.06s
```
**6 warnings** — todos en `src/mud/corpus_trainer/mod.rs`:
- `unused import: MudTensorType`
- `unused imports: Duration and Instant`
- `unused import: Ordering`
- `unused macro definition: println`
- `unused import: TrainChunkMetrics` (en `session.rs`)
- `unused import: crate::mud::slime_backward::OptimizerStrategy` (en `train_step.rs`)

### 1.2 `cargo clippy --all-targets`
**19 warnings** (P-06 VIOLATION):

| Categoría | Count | Archivo | Detalle |
|---|---|---|---|
| `identity_op` | 3 | `ezop.rs:176,181,190` | `(over & pos) * 0x1u8` — el `* 0x1` es identidad |
| `operator_precedence` | 3 | `ezop.rs:176,181,190` | `|` vs `*` precedence no obvio en branchless pack |
| `unnecessary_unsafe` | 1 | `ezop.rs:349` | `unsafe` block innecesario |
| `needless_range_loop` | 1 | `ezop.rs:350` | `for r in 0..rows` indexa `scales_a` |
| `too_many_arguments` | 1 | `train_step.rs:27` | `train_on_sequence` tiene 12 args (límite 7) |
| `doc_list_item_overindented` | 6 | `train_step.rs:6-10` | Items de doc con indentación excesiva |
| `unused_imports` | 5 | `mod.rs`, `session.rs`, `train_step.rs` | Imports no usados tras el split |
| `unused_macros` | 1 | `mod.rs:21` | Macro `println!` definida pero no visible a submodules |

### 1.3 `cargo clippy --lib -- -D warnings` (usado por `./mud.sh ci`)
```
error: could not compile `forge_llm` (lib) due to 19 previous errors
```
**❌ CI BATTERY FALLA** en el paso de clippy. El comando `./mud.sh ci` ejecuta `cargo clippy --lib -- -D warnings` (línea 447) sin manejo de errores, por lo que la batería de CI se detiene aquí.

### 1.4 `cargo test --lib -- --test-threads=2`
```
test result: ok. 312 passed; 0 failed; 2 ignored; 0 measured; 0 filtered out
```
✅ Coincide con el reporte de AGENTS.md (312 tests).

### 1.5 `./mud.sh health`
```
verdict: 🔴 STRUCTURAL ISSUES
```
El checkpoint `weights/checkpoints/model_latest_checkpoint.mud` (hidden=128, 6 layers, 2 heads, vocab=151643) tiene 18 tensores de experto faltantes — **esperado** para un modelo denso (num_experts=1). El kernel C-MUD self-check pasa. ✅

---

## 2. Security Audit (V25 SEC-1/2/3)

### SEC-1 · `RlvrCritic::evaluate_rust_code` — RCE via `rustc`
- **Archivo:** `src/mud/rlvr.rs:33-34`
- **Estado:** ✅ **FIXED** — Detrás de `#[cfg(feature = "unsafe-rce")]`. El `Cargo.toml` define `default = []` y el feature `unsafe-rce` está deshabilitado por defecto. Builds de producción son seguros.

### SEC-2 · `TerminalSandbox::execute_command` — RCE via `bash -c`
- **Archivo:** `src/mud/sandbox.rs:36`
- **Estado:** ✅ **FIXED** — Detrás de `#[cfg(feature = "unsafe-cli")]`. Builds de producción son seguros.

### SEC-3 · `AgentWorkspace::read_file` / `write_file` — Path Traversal
- **Archivo:** `src/mud/workspace_agent.rs:91-114`
- **Estado:** ✅ **FIXED** — Usa `fs::canonicalize` + `starts_with(&self.root_canon)` para validar que paths no escapen del workspace root. Tests: `test_absolute_path_rejected`, `test_path_traversal_blocked`.

### SEC-Hallazgos restantes (V25 SEC-H1…H11)
- **SEC-H1** (MudFile::load sin caps): No verificado en este audit.
- **SEC-H2** (No fuzzing tests): No verificado.
- **SEC-H3** (unsafe impl Send/Sync sin miri): No verificado.
- **SEC-H4** (apply_optimizer sin bounds check): `optim.rs:373` — no verificado.
- **SEC-H6** (log/env_logger no usados): Verificado parcialmente — hay 72 `eprintln!` + 70 `println!` en el código.

---

## 3. Performance Audit (V25 PERF-1…7)

### PERF-1 · RoPE usa `powf + sin_cos` en vez de kernel AVX2
- **Estado:** ✅ **FIXED** — `compute_rope_inv_freq()` precomputa `inv_freq[i]` una vez. El hot path usa recurrencia `theta *= inv_freq[i]` (1 multiply ~1 ns) en vez de `powf` (~50 ns). El campo `rope_inv_freq` está en `SlimeLayer` (línea 235).

### PERF-2 · 6+ allocs en `evaluate_slime_block_moe` (P-01 violado)
- **Estado:** ✅ **FIXED** — Sprint 5.1: `csa_index_scratch` y `csa_blocks_scratch` movidos a `SlimeWorkspace` como `AlignedArray`. Views via `Vec::from_raw_parts` + `mem::forget`. **0 allocs** en el hot path.

### PERF-3 · JEPA SIMD "batch 8" degrada a scalar
- **Estado:** ✅ **FIXED** — Sprint 5.3: `TERNARY_NIBBLE_TO_F32[16]` LUT reemplaza el `match` por elemento. Branchless indexed load.

### PERF-4 · `csa_indexer::select_topk` aloca 3 Vec por cabeza × layer
- **Estado:** ✅ **FIXED** — Buffers pre-allocados en `SlimeWorkspace`.

### PERF-5 · `Mutex<GemvGpuCache>` en hot path GPU dispatch
- **Estado:** ✅ **FIXED** — Sprint 5.2: `AtomicU8` tri-state (`GPU_GEMV_DISPATCH`: 0=unknown, 1=enabled, 2=disabled). Hot path: 1 `Relaxed` load + early return.

### PERF-6 · `ternary_gemv_backward` con `match` por weight
- **Estado:** ✅ **FIXED** — Sprint 5.3: LUT `TERNARY_NIBBLE_TO_F32` reemplaza el `match`.

### PERF-7 · `cmud_training_forward` aloca 200 MB por step
- **Estado:** ⚠️ **PARCIALMENTE ABORDADO** — `inference.rs:455`: `let head_w: Vec<f32> = out_slice_ternary.to_vec()` **aún existe**. Comentario dice "owned for the training backward". Para modelos grandes (vocab=128k, hidden=2560), esto es ~1.3 GB/step. La sugerencia del V25 (reuse `emb_f32` si tied) no se implementó. **No es hot path de inferencia**, pero es un problema de memoria en entrenamiento.

### PERF-H1/H2/H3 · `std::env::var` + `is_x86_feature_detected!` en hot path
- **Estado:** ✅ **FIXED** — `prefer_gemv_8_rows()` y `use_vulkan_cached()` cachean en `AtomicU64`. `is_x86_feature_detected!` cacheado en `AtomicU8` (verificado en `slime_forward.rs`).

---

## 4. Bug Audit (V25 BUG-1/2)

### BUG-1 · `main.rs:437` — OOB sin clamp de vocab_size
- **Estado:** ✅ **FIXED** — `main.rs:456-461`: `let tid = (tokens[current_pos] as usize).min(vocab_len - 1)`.

### BUG-2 · 7 `panic!` en hot path del forward (SIGABRT)
- **Estado:** ✅ **FIXED** — `slime_forward.rs:5-26`: 7 sites reemplazados con `nan_skip()` que usa `AtomicU64` counters + throttling de logging (primero 4, luego cada 64k). El proceso sobrevive a `.mud` corruptos.

---

## 5. Architecture Audit (V25 ARCH-1…5)

### ARCH-1 · `corpus_trainer.rs` (5407 líneas) — God module
- **Estado:** ✅ **FIXED** — Split en 7 módulos en `src/mud/corpus_trainer/`:
  - `mod.rs` (41 LOC) — entry point, macros, re-exports
  - `alignment.rs` (21K) — `deep_local_alignment` + L-QAT
  - `circuit.rs` (22K) — `run_training_circuit`
  - `debate.rs` (22K) — `run_debate_session`
  - `emb.rs` (4.3K) — `load_emb_row_into`
  - `integrity.rs` (11K) — `model_norms_alive`, `circuit_eval_integrity`
  - `metrics.rs` (2.6K) — `TrainChunkMetrics`
  - `session.rs` (70K) — `run_alignment_session`
  - `train_step.rs` (47K) — `train_on_sequence`

  **APIs externas intactas:** `MudCorpusTrainer`, `SHOULD_TERMINATE`, `CHUNK_SIZE`, `run_alignment_session`, `run_debate_session`, `run_training_circuit`.

### ARCH-3 · Strategy pattern cerrado (viola OCP)
- **Estado:** ✅ **FIXED** — `src/mud/optim.rs:70`: `pub trait Optimizer: Send + Sync` con `name()`, `step()`, `needs_moments()`, `clone_box()`.

### ARCH-4 · Ciclo de dependencia `corpus_trainer ↔ slime_backward`
- **Estado:** ✅ **FIXED** — `mhc_scale_sgd_step` y `apply_optimizer_cpu_step_and_pack` movidos a `src/mud/optim.rs`. El test en `slime_backward.rs:867` confirma: "SPRINT-4: `mhc_scale_sgd_step` moved to `crate::mud::optim` to break [the cycle]".

### ARCH-5 · 297 `unwrap()` + 50 `expect()` + 11 `panic!` (P-17 mal aplicado)
- **Estado:** ⚠️ **PARCIALMENTE ABORDADO** — `MudError` enum existe en `src/mud/error.rs` (con `thiserror`). Los 7 `panic!` del hot path fueron reemplazados. Pero aún hay **135** `unwrap()`/`expect()`/`panic!` en `src/mud/`. P-17 sigue parcialmente violado.

---

## 6. P-06 Violation (CRITICAL — CI Roto)

### 6.1 Resumen
`cargo clippy --lib -- -D warnings` (el comando usado por `./mud.sh ci` línea 447) **falla con 19 errores**. Esto significa que **el CI está roto** y cualquier PR no puede ser aprobada.

### 6.2 Detalle de los 19 warnings

#### 6.2.1 `ezop.rs` — 8 warnings (introducidos en Sprint 5.4)
El branchless pack `(over & pos) * 0x1u8 | (over & !pos) * 0xFu8` (Sprint 5.4) genera:
- **3× `identity_op`**: `* 0x1u8` es identidad, clippy sugiere `((over & pos))`
- **3× `operator_precedence`**: `|` tiene menor precedencia que `*`, clippy sugiere paréntesis explícitos
- **1× `unnecessary_unsafe`** en línea 349
- **1× `needless_range_loop`** en línea 350

**Fix recomendado:** Reescribir como:
```rust
let b0 = ((over0 & pos0) as u8) | ((over0 & !pos0) * 0xFu8);
```
Esto elimina tanto el `identity_op` como el `operator_precedence` warning.

#### 6.2.2 `corpus_trainer/mod.rs` — 6 warnings (introducidos en Sprint 4 split)
- **5× unused imports**: `MudTensorType`, `Duration`, `Instant`, `Ordering`, `TrainChunkMetrics` (en session.rs), `OptimizerStrategy` (en train_step.rs)
- **1× unused macro**: `println!` definido en `mod.rs:21` pero **no visible a los submodules** porque está definido DESPUÉS de las declaraciones `pub mod` (líneas 4-11)

**Bug asociado:** El macro `println!` en `mod.rs` está diseñado para gatear output con `MUD_CIRCUIT_TUI`, pero como está definido después de `pub mod`, los submodules usan `std::println!` directamente. **El gateo de TUI está roto.**

**Fix recomendado:** Mover la definición del macro ANTES de las declaraciones `pub mod`, o usar `#[macro_use]` / `#[macro_export]`.

#### 6.2.3 `train_step.rs` — 7 warnings
- **6× `doc_list_item_overindented`**: Items de doc con 8 espacios en vez de 5
- **1× `too_many_arguments`**: `train_on_sequence` tiene 12 args (límite clippy: 7)

#### 6.2.4 `corpus_trainer/mod.rs` — `#![allow(clippy::type_complexity)]`
El archivo `mod.rs` tiene un `#![allow(clippy::type_complexity)]` en la línea 1, que es un blanket allow. Esto podría estar ocultando otros warnings.

### 6.3 Impacto
- `./mud.sh ci` falla en el paso de clippy
- No se puede mergear PRs sin limpiar estos warnings
- P-06 (0-error 0-warning clippy) está violado

---

## 7. P-08 Violation (Dead Code)

### `slime.rs:182-189` — Legacy shims sin callers
```rust
pub fn ternary_state_f32(&self) -> f32 { self.real() }
pub fn jepa_energy_f32(&self) -> f32 { self.imag() }
```
Estas funciones son wrappers triviales sin callers en el codebase. El V25 audit las identificó como "legacy shims sin callers". Aunque son `pub` (por lo que no disparan `dead_code` warning), son código muerto que debería eliminarse (P-08).

---

## 8. P-18 Violation (Root-level files)

| Archivo | Líneas | Problema |
|---|---|---|
| `arreglar_trainer.md` | 690 | Documento de sesión de agente en root — debería estar en `docs/sessions/` |
| `mud_disassembly.txt` | 538 | Dump de disassembler en root — debería estar en `docs/dumps/` |
| `test_qwen_rope.py` | 3 | **Archivo Python** — viola P-07 (no Python/PyTorch) |
| `dump.txt` | 2 | Dump de metadata en root — debería estar en `docs/dumps/` |
| `corpus.txt` | 40 | Corpus de entrenamiento en root — debería estar en `training/corpus/` |

**Total:** 5 archivos en root que violan P-18. El `test_qwen_rope.py` también viola P-07.

---

## 9. Documentation Audit

### 9.1 `MUD_AUDIT_LATEST.md` (V37) — STALE
- **Fecha:** 2026-07-21
- **Claim:** "0 errors, 0 warnings (`cargo clippy --all-targets -D warnings`)"
- **Realidad:** 19 warnings, CI falla
- **Problema:** El V37 fue emitido antes del Sprint 5 (2026-07-24), que introdujo los warnings en `ezop.rs` y `corpus_trainer/mod.rs`. El documento está obsoleto y engaña.

### 9.2 `BUG_LOGITS_CORRUPTION.md` — VERIFIED
- **Fix:** Q2_0→F32 conversion via `Q2ToF32` path en `bonsai_gguf_to_mud.rs:478-580`
- **Dispatch:** `is_f32_weights` flag en `SlimeLayer` (inference.rs:323) → `f32_gemv_rowwise` (slime_forward.rs:317, 855, 901, 988, 1091, 1444)
- **Estado:** ✅ Activo y documentado

### 9.3 `MUD_AUDIT_REPORT_V25_COMPREHENSIVE.md` — Historical
- **Fecha:** 2026-07-23
- **Hallazgos:** 236 (15 P0, 60 P1, 110 P2, 51 P3)
- **Estado:** La mayoría de los P0 han sido abordados en Sprint 5, pero se introdujeron nuevos warnings de clippy.

---

## 10. Model Health

### Modelos disponibles
| Modelo | Tamaño | Hidden | Layers | Heads | Vocab | Estado |
|---|---|---|---|---|---|---|
| `models/ale.mud` | 22M | 128 | 6 | 2 | 151643 | ✅ Healthy (dense, C-MUD OK) |
| `models/pico_alpha.mud` | 22M | ? | ? | ? | ? | No verificado |
| `models/pico_beta.mud` | 22M | ? | ? | ? | ? | No verificado |
| `weights/checkpoints/model_latest_checkpoint.mud` | 22M | 128 | 6 | 2 | 151643 | ✅ Healthy (training checkpoint) |
| `models/smollm2.mud` | — | — | — | — | — | ❌ **MISSING** |

### `models/smollm2.mud` — MISSING
AGENTS.md (línea 187) y GEMINI.md (línea 42) mencionan que `models/smollm2.mud` fue "reconvertido sano" en 2026-07-18 (FIX D). Sin embargo, **el archivo no existe** en el filesystem. El directorio `models/smollm2/` existe (con `config.json`, `model.safetensors`, `tokenizer.json`), pero el `.mud` convertido falta.

**Impacto:** Los comandos que dependen de `models/smollm2.mud` (como `./mud.sh circuit models/smollm2.mud`) fallarán con "No such file or directory".

### `models/ternary_bonsai_1.7b.mud` — MISSING
AGENTS.md menciona `models/ternary_bonsai_1.7b.mud` como "alternativa sana", pero el directorio `models/ternary_bonsai_1.7b/` contiene archivos GGUF y fuentes, no un `.mud`.

---

## 11. Prioridad de Remediación

### 🚨 CRÍTICO (inmediato)
1. **Limpiar los 19 warnings de clippy** — CI está roto. `./mud.sh ci` falla en `cargo clippy --lib -- -D warnings`.
   - Fix `ezop.rs`: reescribir branchless pack para eliminar `identity_op` + `operator_precedence`
   - Fix `corpus_trainer/mod.rs`: remover imports no usados, mover macro `println!` antes de `pub mod`
   - Fix `train_step.rs`: corregir indentación de doc, refactorizar `train_on_sequence` (12→7 args)
   - Fix `ezop.rs:349`: remover `unsafe` innecesario

2. **Actualizar `MUD_AUDIT_LATEST.md`** — V37 claims 0 warnings, hay 19. Documentar el estado real.

3. **Mover/eliminar archivos root stray** — P-18 violation. Especialmente `test_qwen_rope.py` (P-07).

### ⚠️ ALTO
4. **Verificar/crear `models/smollm2.mud`** — AGENTS.md lo menciona como modelo sano pero no existe.
5. **Eliminar dead code** `jepa_energy_f32` / `ternary_state_f32` (P-08).
6. **PERF-7**: Reusar `emb_f32` en vez de `to_vec()` en `inference.rs:455` para tied output.

### 📝 MEDIO
7. **Refactorizar `train_on_sequence`** (47K LOC, 12 args) — ARCH-MAINT.
8. **Auditar `unsafe impl Send/Sync`** con miri (SEC-H3).
9. **Eliminar `#![allow(clippy::type_complexity)]`** blanket en `corpus_trainer/mod.rs`.

---

## 12. Compliance Summary

| Política | Estado | Notas |
|---|---|---|
| **P-00** (raw pointers) | ✅ PARCIAL | Hot loops OK; algunos `match` reemplazados por LUT |
| **P-01** (zero alloc) | ✅ FIXED | Sprint 5.1 eliminó allocs en `evaluate_slime_block_moe` |
| **P-02** (matmul_accum f32) | ✅ CUMPLE | |
| **P-06** (clippy 0) | ❌ **VIOLADO** | 19 warnings, CI falla |
| **P-07** (no Python) | ❌ **VIOLADO** | `test_qwen_rope.py` en root |
| **P-08** (no dead code) | ⚠️ PARCIAL | `jepa_energy_f32`/`ternary_state_f32` legacy shims |
| **P-13** (no hardcode dims) | ✅ CUMPLE | |
| **P-17** (fail-fast) | ⚠️ PARCIAL | 135 unwrap/expect/panic en `src/mud/` |
| **P-18** (docs layout) | ❌ **VIOLADO** | 5 archivos stray en root |
| **P-27** (no Rayon hot path) | ✅ CUMPLE | PCorePool usado |

---

## 13. Conclusión

Forge LLM es un motor maduro con 312 tests pasando y la mayoría de los hallazgos críticos del V25 audit resueltos (security, performance, bugs, architecture). Sin embargo, **el Sprint 5 (2026-07-24) introdujo 19 warnings de clippy que rompen el CI**, violando P-06. La documentación de auditoría (V37) es obsoleta y no refleja este estado.

**Recomendación inmediata:** Limpiar los 19 warnings de clippy antes de cualquier merge. Luego actualizar `MUD_AUDIT_LATEST.md` y mover los archivos root stray.
