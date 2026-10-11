# Sprint 4 — Refactor Estructural P0 + Fenwick Tree Pivot

> **Fecha:** 2026-07-24 (reformulado tras freeze, **split culminado**)
> **Estado base:** Sprints 1-3 cerrados. **280 tests**, clippy 0, PPL C-MUD validado, SlimeX wired, RoPE precomputed.
> **Estado Sprint 4 (cerrado):** `MudError` enum ✅ + `optim.rs` leaf module ✅ (cycle broken) + `checkpoint.rs` / `corpus.rs` extraídos ✅ + **`corpus_trainer.rs` split ✅** (4534 → 9 archivos, 7 módulos cohesivos). 310 tests pasando, clippy 0 errors.
> **Freeze (2026-07-23→24):** el split masivo (4.1, 10 archivos nuevos) absorbió el sprint sin entregar valor end-user. **Pivotamos a Fenwick trees** como nuevo eje (ver `docs/research/MUD_FENWICK_TREE_OPTIMIZATION_PLAN.md`).
> **Split re-ejecutado (2026-07-24):** enfoque incremental — 8 mini-PRs con `cargo check` después de cada extracción. Ver `MUD_SESSION_REPORT_2026-07-24.md`.
> **Foco reformulado:** 4.FW — Fenwick tree core (`fenwick.rs`) + F1 nucleus sample + F4 histogram threshold. 4.1 ✅ cerrado; ahora backlog = Fenwick trees.

---

## 0. Estado de entrada

| Problema | Origen | Impacto |
|----------|--------|---------|
| Ciclo `corpus_trainer ↔ slime_backward` | `corpus_trainer.rs:4845` define `mhc_scale_sgd_step`; `slime_backward.rs:942` lo importa (en `#[cfg(test)]`); `corpus_trainer.rs:4883` importa `OptimizerStrategy` de `slime_backward` | Cargo no puede deduplicar; tests aislados se rompen si el path de carga cambia |
| `OptimizerStrategy` enum cerrado | `slime_backward.rs:84-172` | Añadir un optimizador (Lion, Shampoo) requiere modificar 9 sitios |
| 297 `unwrap()` + 50 `expect()` + 11 `panic!` | dispersos en `src/mud/` | P-17 mal aplicado: errores no se propagan al caller |
| `corpus_trainer.rs` 5415 líneas | god module | sin split, imposible de revisar en code review |

---

## 1. Tareas

### 4.4 — Romper ciclo `corpus_trainer ↔ slime_backward`

**Estrategia:** crear `src/mud/optim.rs` como **módulo hoja** sin dependencias a `corpus_trainer` ni a `slime_backward`. Mover:
- `mhc_scale_sgd_step` (de `corpus_trainer.rs:4845-4868`)
- `apply_optimizer_cpu_step_and_pack` (de `corpus_trainer.rs:4870-5072`)
- `OptimizerStrategy` enum (de `slime_backward.rs:84-101`)
- `select_optimizer` (de `slime_backward.rs:142-172`)
- `optimizer_policy_override` (de `slime_backward.rs:110-140`)

**Resultado:** grafo de dependencias:
- `slime_backward → optim` (para `select_optimizer` al construir `SlimeLayerShadowF32`)
- `corpus_trainer → optim` (para `mhc_scale_sgd_step` y `apply_optimizer_cpu_step_and_pack`)
- `optim → ∅` (no imports de los dos)
- `slime_backward ↛ corpus_trainer` (ciclo roto)

### 4.2 — `trait Optimizer` con dispatch vía enum

**Estrategia pragmática:** mantener `OptimizerStrategy` como enum (back-compat con `SlimeLayerShadowF32`) pero las **variantes** ahora contienen `Box<dyn Optimizer>` por variante. Esto preserva la API actual (no rompe `slime_backward.rs:241 SlimeLayerShadowF32::new` que matchea sobre el enum) y habilita OCP: añadir un optimizador = 1 struct + 1 variante enum.

```rust
// src/mud/optim.rs
pub trait Optimizer: Send + Sync {
    fn name(&self) -> &'static str;
    fn step(&self, w: &mut [f32], g: &mut [f32], lr: f32, weight_decay: f32) -> anyhow::Result<()>;
}

pub struct Sgd;
pub struct Muon { pub ns_iters: usize }
pub struct AdamW { pub m: Vec<f32>, pub v: Vec<f32>, pub step: u64 }

pub enum OptimizerStrategy {
    Sgd(Box<Sgd>),
    Muon(Box<Muon>),
    AdamW(Box<AdamW>),
    GaLore { rank: usize, ... },   // keep data-only for now
    ChunkedAdam { chunk_cols: usize },
    SparseAdam { only_active_rows: bool },
}
```

### 4.3 — `MudError` enum (subset)

**Estrategia:** crear `src/mud/error.rs` con `thiserror::Error` y convertir los 11 `expect("Missing ...")` críticos en `Result<T, MudError>`. Mantener `expect` solo en:
- Init del workspace (donde cualquier fallo es fatal).
- Tests.
- Paths de baja latencia (LM head) donde el error se loguea en vez de propagarse.

**Subset a convertir en este sprint:**
- `main.rs:30-59` — 6 `expect("Missing hidden_size/num_heads/...")` → `MudError::MissingMetadata`.
- `inference.rs:154-184` — los mismos 6.
- `corpus_trainer.rs:719-748, 2013-2042` — 6 más.
- `inference.rs:485, 868` — 2 `unwrap` en MoE bus → `MudError::BusEmpty`.

### 4.1 — Split `corpus_trainer.rs` (documentado, no ejecutado en Sprint 4)

El split de 5415 líneas en `trainer/` con 10 archivos (alignment, session, circuit, debate, integrity, checkpoint, mhc_optimizer, telemetry, env) es un refactor de 2-3 semanas por sí solo. **Se documenta el plan en `docs/research/CORPUS_TRAINER_SPLIT_PLAN.md`** pero no se ejecuta en este sprint.

---

## 2. Validación

### 2.1 — Validación Sprint 4 cerrado (parcial)

| Check | Comando | Criterio | Estado |
|-------|---------|----------|--------|
| Build | `cargo build --release` | 0 errors | ✅ |
| Clippy | `cargo clippy --all-targets --release` | 0 warnings | ✅ |
| Tests | `cargo test --lib --release` | 280+ passed | ✅ |
| Cycle broken | `cargo tree \| grep corpus_trainer.*slime_backward` | sin match (no cycle) | ✅ |
| E2E train | `./mud.sh circuit models/pico_beta.mud` | entrena sin error, peso cambia | ✅ |
| E2E fuse | `./mud.sh slimex-fuse models /tmp/m.mud` | OK | ✅ |

### 2.2 — Validación Sprint 4 FW (esta mitad)

| Check | Comando | Criterio |
|-------|---------|----------|
| Build | `cargo build --release` | 0 errors |
| Clippy | `cargo clippy --all-targets --release` | 0 warnings |
| Tests | `cargo test --lib --release` | 280+18 (FW) = 298+ passed |
| Fenwick distribution KS-test | `cargo test --lib fenwick_distribution` | p > 0.05 sobre 10 000 samples |
| Fenwick integral_threshold regression | `cargo test --lib integral_threshold_fenwick` | threshold diff < 1e-5 vs naive |
| Fallback (MUD_FENWICK=0) | `MUD_FENWICK=0 cargo test --lib fenwick` | código lineal usado; paridad bit-exacta |
| Bench (i7-1260P) | `cargo run --release --bin fenwick_bench` | prefix_sum n=32768 ≤ 60 ns; lower_bound ≤ 75 ns |
| E2E debate | `./mud.sh debate models/pico_beta.mud` | tokens sampleados con MUD_FENWICK=1; distribución ~uniforme |

---

## 3. Entregables

### 3.1 — Cerrados en la primera mitad de Sprint 4 (pre-freeze)

1. ✅ `src/mud/optim.rs` (módulo hoja con `mhc_scale_sgd_step`, `apply_optimizer_cpu_step_and_pack`, `OptimizerStrategy`, `select_optimizer`, `optimizer_policy_override`).
2. ✅ `src/mud/error.rs` (`MudError` enum con `thiserror`).
3. ✅ `src/mud/optim.rs::Optimizer` trait + structs `SgdOptimizer` + `AdamWOptimizer`.
4. ✅ Reemplazo de 10-15 `expect("Missing ...")` críticos por `Result<T, MudError>`.
5. ✅ Tests pasando, ciclo roto, E2E funcionando.
6. ✅ `src/mud/checkpoint.rs` y `src/mud/corpus.rs` extraídos como módulos `pub(crate)` (parte del split 4.1 — pero sin culminar).
7. ✅ Documentación: este archivo + `docs/research/CORPUS_TRAINER_SPLIT_PLAN.md` (aún no escrito; bloqueado por pivot).

### 3.2 — Cerrado: split de `corpus_trainer.rs` (2026-07-24)

> **Reporte completo:** `MUD_SESSION_REPORT_2026-07-24.md`. Resumen:

| ID | Entregable | LOC | Estado |
|----|------------|-----|--------|
| 4.1.1 | `metrics.rs` (TrainChunkMetrics + ActStatsAccum) | 87 | ✅ |
| 4.1.2 | `emb.rs` (load_emb_row_into + slice_stddev) | 131 | ✅ |
| 4.1.3 | `integrity.rs` (model_norms_alive + circuit_eval_integrity + benchmark) | 270 | ✅ |
| 4.1.4 | `alignment.rs` (validate_metadata + deep_local_alignment + audit_tokenization) | 507 | ✅ |
| 4.1.5 | `debate.rs` (run_debate_session) | 516 | ✅ |
| 4.1.6 | `circuit.rs` (run_training_circuit) | 318 | ✅ |
| 4.1.7 | `session.rs` (run_alignment_session) | 1556 | ✅ |
| 4.1.8 | `train_step.rs` (train_on_sequence) | 1045 | ✅ |
| 4.1.9 | `mod.rs` thin entry (struct, macros, shims, re-exports) | 380 | ✅ |
| 4.1.10 | Tests nuevos (+30) | — | ✅ 280 → 310 |
| 4.1.11 | Compilación + clippy + tests | — | ✅ verde |

**Total split:** 4534 → 4810 LOC (delta +276 de headers/imports), **0 regresiones**, **APIs externas sin cambios** (4 herramientas siguen usando `forge_llm::mud::corpus_trainer::*`).

### 3.3 — Pendiente en esta mitad reformulada (Sprint 4 FW)

> **Plan completo en** `docs/research/MUD_FENWICK_TREE_OPTIMIZATION_PLAN.md`. Resumen:

| ID | Entregable | LOC | Esfuerzo |
|----|------------|-----|----------|
| 4.FW.1 | `src/mud/fenwick.rs` core (F32 + U32 + I32 variants) | 250 | 3 h |
| 4.FW.2 | Tests unitarios del core (15+ tests) | 200 | 2 h |
| 4.FW.3 | `pub mod fenwick;` en `src/mud/mod.rs:64` | 1 | 1 min |
| 4.FW.4 | Wrap F1 nucleus sample (`debate_trainer.rs:485-505`) | 20 | 1 h |
| 4.FW.5 | Wrap F4 histogram (`integral_threshold.rs:18-60`) | 35 | 1 h |
| 4.FW.6 | Test integración debate_trainer (KS-test sobre 1 000 samples) | 30 | 1 h |
| 4.FW.7 | Test regresión integral_threshold (vs naive 200-pass) | 25 | 30 min |
| 4.FW.8 | `MUD_FENWICK=0` fallback test (paridad con baseline) | 15 | 30 min |
| 4.FW.9 | `cargo test --lib` + `cargo clippy --all-targets` 0 | — | 40 min |
| 4.FW.10 | `tools/fenwick_bench.rs` con n ∈ {128, 1024, 32768, 131072} | 80 | 1 h |

**Total Sprint 4 FW:** ~575 LOC Rust + 80 LOC bench + 18 tests = **~9 horas**, default-on con MUD_FENWICK env flag, fallback lineal si está en 0 (CI matrix).

### 3.3 — Backlog (Sprint 5+)

- **Sprint 4b (opcional):** ASM AVX2 `prefix_sum_batch4` + `find_kth_walk` para F1 batched (~8 h).
- **Sprint 5:** retomar 4.1 (split completo de `corpus_trainer.rs`) ahora con `fenwick.rs` marcando el patrón de módulo cohesivo.
- **F2** (first-fit pack) y **F3** (CSA top-k) si los hot loops lo justifican (> 4096 HCA blocks o bins > 32).
- **F5b** (loss percentile gate) — feature nuevo, no optimización.
- **Lion optimizer** ya en `optim.rs` backlog.

### 3.4 — Por qué pivotamos (lección del freeze)

El Sprint 4 original era 80% split estructural (4.1). Empezar el split de 4534 LOC en 10 archivos sin drivers concretos es:
- **Alto riesgo:** cada extracción cambia el grafo de imports; un ciclo de imports no detectado paraliza el sprint.
- **Bajo valor end-user:** el split no acelera el motor, solo lo hace más mantenible.
- **Largo:** estimado 2-3 semanas, real 3-4 con re-trabajo.

Pivotar a **Fenwick trees** da:
- ✅ Valor medible (F1: 3.9× nucleus sample; F4: 198× integral threshold).
- ✅ Patrón arquitectónico (módulo cohesivo `pub(crate)`) que **prepara** el split futuro.
- ✅ Bajo riesgo (feature-flag default on, fallback lineal en CI).
- ✅ Corto (1 semana vs 3-4).

**Convención:** cuando un refactor estructural no tenga driver concreto, se difiere; cuando haya un hotspot medible, se pivotará a una optimización basada en el refactor (no al revés).
