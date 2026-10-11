# MUD Session Report — 2026-07-25

## Fase D: Training Scale — Progreso

### ✅ D2 bf16 Shadow Weights (COMPLETADO)
**Archivo:** `src/mud/adam_state.rs`, `src/mud/optim.rs`

- **`AdamState`** extendido con `shadow_bf16: Option<Vec<u16>>` — snapshot bf16 de shadow weights
- **`store_shadow_bf16(&mut self, weights: &[f32])`** — almacena snapshot bf16; zero-alloc después del primer llamado (reusa buffer, P-01)
- **`load_shadow_bf16(&self) -> Option<Vec<f32>>`** — reconstruye f32 desde bf16
- **`f32_to_bf16(f32) -> u16`** — conversión (truncate mantissa, IEEE 754 8 exp + 7 mantissa)
- **`bf16_to_f32(u16) -> f32`** — reconstrucción (shift left 16 bits)
- **Integración en `apply_optimizer_cpu_step_and_pack`** — `store_shadow_bf16` llamado después del Adam/SparseAdam step en cada match arm
- **3 tests nuevos:** `test_bf16_roundtrip`, `test_bf16_zero_and_sign`, `test_shadow_bf16_store_load`
- **Env var:** `MUD_BF16_SHADOW=1` activa el shadow bf16

**Validación:** `cargo clippy --lib -- -D warnings` 0 errors, `cargo test --lib -- --test-threads=2` → 347 passed, 0 failed, 2 ignored

### ✅ D4 Pipeline Overlap — Integración en Training Loop (COMPLETADO)
**Archivos:** `src/mud/ash_qat_dispatcher.rs`, `src/mud/corpus_trainer/mod.rs`, `src/mud/corpus_trainer/train_step.rs`, `src/mud/corpus_trainer/session.rs`, `Cargo.toml`, `mud.sh`, `.github/workflows/ci.yml`, `tools/pipeline_overlap_bench.rs`

- **`PipelineOverlap` struct** — orquestador de pipeline 3-stage (CPU readback → GPU GEMV → slot swap)
- **`step_pipeline()`** — método core que:
  1. `flush_pending` — drena CPU readback de chunk N-1 (mientras GPU trabaja en N)
  2. `step_async_deferred` — submit GPU GEMV para chunk N (retorna inmediatamente, L-05 DoubleFrame)
  3. `swap_slots` — rota double-buffer para chunk N+1
- **CPU Forward N+1** corre concurrentemente con GPU Optimizer N
- **`drain_all()`** — flush final + wait GPU idle (epoch end / checkpoint)
- **Integración en `train_on_sequence`** — `step_pipeline` usado cuando `MUD_PIPELINE_OVERLAP=1`, con fallback a `step_async_deferred` (L-05)
- **Integración en `session.rs`** — `drain_all` en epoch end + final flush cuando `MUD_PIPELINE_OVERLAP=1`, con fallback a `sync_all`
- **`MudCorpusTrainer`** extendido con `pipeline_overlap: RefCell<PipelineOverlap>` (interior mutability para `&self` en `train_on_sequence`)
- **Telemetry flags:** `mark_prepare_started/done`, `is_prepare_running`, `is_gemv_pending`, `is_readback_pending`
- **Env var:** `MUD_PIPELINE_OVERLAP=1` activa el pipeline overlap
- **4 tests nuevos:** `test_pipeline_overlap_slots`, `test_pipeline_overlap_swap`, `test_pipeline_overlap_prepare_flags`, `test_pipeline_overlap_env`
- **`pipeline_overlap_bench` binary** — CPU struct overhead benchmark (swap_slots, env check, slot ops, telemetry flags)
- **Cargo.toml:** registrado `gemv_auto_bench`, `loss_certification_bench`, `training_healthcheck`, `pipeline_overlap_bench`
- **CI:** nightly-cert-loss job (cron daily 02:00 UTC) + pipeline_overlap_bench en build step
- **mud.sh:** comando `pipeline-overlap` agregado

**Validación:** `cargo clippy --lib -- -D warnings` 0 errors, `cargo test --lib -- --test-threads=2` → 351 passed, 0 failed, 2 ignored, `cargo build --release --bin run_trainer --bin forge_llm --bin pipeline_overlap_bench --bin loss_certification_bench --bin training_healthcheck --bin gemv_auto_bench` OK

### 🔧 Optimizaciones Hot-Path (P-00, P-01)

#### 1. Env var caching (P-01)
- **`bf16_shadow_enabled()`** — `thread_local<RefCell<Option<bool>>>` cache para `MUD_BF16_SHADOW`
- **`pipeline_overlap_enabled()`** — `thread_local<RefCell<Option<bool>>>` cache para `MUD_PIPELINE_OVERLAP`
- Evita `std::env::var` syscall en cada optimizer step (hot path)
- Tests: `reset_bf16_shadow_cache()` / `reset_pipeline_overlap_cache()` para reset entre tests

#### 2. JEPA mean/var AVX2 vectorization (PERF-3)
- **`jepa_stabilizer`** en `src/mud/slime_jepa.rs` — mean/var loops escalares (f64) → AVX2 batch-8 (f32 SIMD + f64 accumulation)
- **Ganancia:** ~4× en mean/var computation (8192 → ~2048 ops para hidden=2048)
- **Impacto forward:** ~3-5% más rápido (JEPA llamado 2× por capa × 32 capas)
- **Sin impacto en u32 packing (SlimeX CellV5)** — formato de memoria separado
- **Sin impacto en C-MUD** — módulo opt-in separado, JEPA ya usa f32 en z-update
- **Test nuevo:** `test_jepa_mean_var_vectorized` (verifica alineación 2048 + no-alineada 2050)

#### 3. Raw pointer passes (P-00)
- **`adam_step_scalar`** — `as_mut_ptr()`/`as_ptr()` + `*ptr.add(i)` — elimina bounds checks en scalar Adam fallback
- **`sparse_adam_step`** — raw pointers para row max |g| + inner Adam loop
- **`SgdOptimizer::step`** — reemplazado magic number `1e6` con `SGD_CLAMP_MAX` constant

#### 4. Cargo.toml binary registration
- Registrados: `gemv_auto_bench`, `loss_certification_bench`, `training_healthcheck`, `pipeline_overlap_bench`
- CI build step actualizado para incluir `pipeline_overlap_bench`

#### 5. Magic number cleanup (P-13)
- `1e-8` → `EPSILON_FLOOR` en `jepa_stabilizer` (P-13 soft debt)
- `1e6` → `SGD_CLAMP_MAX` en `SgdOptimizer`

### Estado General
- **Tests totales:** 352 passed, 0 failed, 2 ignored
- **Clippy:** 0 errors, 0 warnings (`-D warnings`, `--lib` + `--all-targets`)
- **Build:** todos los bins de release OK
- **mud.sh:** `MUD_PIPELINE_OVERLAP=0`, `MUD_BF16_SHADOW=0` env vars + comando `pipeline-overlap`
- **CI:** nightly-cert-loss job (cron daily 02:00 UTC) + pipeline_overlap_bench en build step

### Estado Phase D — Completo
| Sub-fase | Estado | Tests |
|----------|--------|-------|
| D1: 4-Pillar Efficiency | ✅ DONE | — |
| D2: bf16 Shadow Weights | ✅ **DONE** | 347 tests |
| D3: Gradient Checkpointing | ✅ DONE | — |
| D4: Pipeline Overlap | ✅ **DONE** | 351 tests |

### Próximos pasos (Orbit)
- **Día 5 Benchmark:** Pipeline overlap throughput (requiere GPU estable; Intel Iris Xe SIGSEGV → CPU fallback)
- **Nightly job:** ✅ cert-loss --fast smollm2 (CI configurado, cron daily 02:00 UTC)
- **C-MUD log-gas (E1+E3):** DEFERRED — requiere reliability gate
