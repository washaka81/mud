# Implementation Plan V2 — Forge LLM Post SlimeX/CMUD Sprint

> **Fecha:** 2026-07-23 (tras SlimeX u32 + CMUD + Memory Continuity sprint)
> **Estado base:** P0–P5 + Q2_0→F32 fix + SX-01…SX-07 + ST-01 + CMUD-01 = COMPLETE
> **Próximas prioridades:** basadas en `MUD_AUDIT_REPORT_V25_COMPREHENSIVE.md` + extensiones lógicas de SlimeX/CMUD

---

## 0. Contexto

El motor está **production-ready** para el path f32 (P-02 SSOT), con dos líneas opt-in:
- **SlimeX** (`MUD_SLIMEX=1`): u32 8:8:8:8 transport + ASM AVX2 unpacker + workspace + bus + table + boot.
- **CMUD** (`MUD_CMUD_THINK=1`): complex manifold con trainable sidecar.

**Siguiente horizonte:** completar la integración end-to-end de ambas líneas + cerrar los P0 de auditoría (seguridad + perf).

---

## Sprint 1 — Quick wins P0 (seguridad + perf) — 1 semana

> Los hallazgos críticos de la auditoría (V25) requieren acción inmediata. Ver `docs/audits/MUD_AUDIT_REPORT_V25_COMPREHENSIVE.md`.

### 1.1 SEG — Eliminar/behind-feature-flag RCE triviales

**P0 — Crítico de seguridad:**

- **SEC-1**: Mover `RlvrCritic::evaluate_rust_code` (`src/mud/rlvr.rs:20-54`) detrás de `#[cfg(feature = "unsafe-rce")]`. Default = off.
- **SEC-2**: Mover `TerminalSandbox::execute_command` (`src/mud/sandbox.rs:22-66`) detrás de `#[cfg(feature = "unsafe-cli")]`. Default = off.
- **SEC-3**: Añadir path validation en `AgentWorkspace::read_file` / `write_file` (`src/mud/workspace_agent.rs:58-70`):
  ```rust
  let full = self.root_dir.join(relative_path).canonicalize()?;
  if !full.starts_with(&self.root_dir.canonicalize()?) { bail!("path escape"); }
  ```
- **SEC-4**: Validar `model_path` en `tools/iteration_validator.rs:378-389` antes de invocar el binario.
- **SEC-5**: Whitelist dir + extensión en `DataAnalysisSkill` (`src/mud/skills/data_analysis.rs:34`).

**Estimación:** 2-3 horas. **Impacto:** 0 RCE triviales.

### 1.2 BUG — Corregir OOB en `main.rs:437`

**P0 — Crítico de correctness:**
- Clamp `tid` con el patrón de `inference.rs:462`:
  ```rust
  let vocab_len = vocab_size.max(1);
  let tid = (tokens[current_pos] as usize).min(vocab_len - 1);
  ```
- **Estimación:** 1 línea. **Impacto:** 0 OOB en inferencia.

### 1.3 BUG — Convertir 7 `panic!` en hot path a telemetría + skip

**P0 — Crítico de robustness:**
- `slime_forward.rs:839, 862, 1173, 1190, 1211, 1234, 1308` — todos abortan con SIGABRT (release). Convertir a `eprintln! + NaN counter + skip`.
- **Estimación:** 30 min. **Impacto:** el proceso sobrevive inputs corruptos.

### 1.4 PERF — RoPE precomputed cos/sin table + `apply_rope_asm` (kernel ya existe)

**P0 — Speedup 10-50×:**
- Precomputar `inv_freq[i]` en `SlimeLayer` (loader).
- Precomputar `cos[pos*half_d + i]` y `sin[pos*half_d + i]` por posición.
- Reemplazar el loop de `powf + sin_cos` libm en `slime_forward.rs:935-959` con llamada a `apply_rope_asm` (que ya existe en `src/asm/mod.rs:73`).
- **Estimación:** 2-3 horas. **Impacto:** 1-2 s/chunk ahorrados.

### 1.5 PERF — Cachear `env::var` y `is_x86_feature_detected!` en `AtomicBool`

**P1 — Bajo esfuerzo, alto impacto:**
- `MUD_GEMV_ROWS` (215k lookups/chunk en `slime_forward.rs:318`)
- `MUD_USE_VULKAN` (30k lookups/chunk en `slime_forward.rs:1556, 1613`)
- `is_x86_feature_detected!` en `slime_jepa.rs:178, 278`
- **Estimación:** 1 hora. **Impacto:** ~5-10 ms/chunk.

**Total Sprint 1:** 1 semana · ~5 quick wins · 0 vulnerabilidades críticas · 2-3 s/chunk speedup.

---

## Sprint 2 — SlimeX end-to-end wiring — 1-2 semanas

> Conectar `snapshot_to_slimex()` y `read_slimex_cell()` al flujo real del forward + Continuity Table lookup.

### 2.1 Wire `snapshot_to_slimex()` post-block

- En `evaluate_slime_block_moe` (`src/mud/slime_forward.rs:799-1469`), después del residual final, llamar `ws.snapshot_to_slimex(&ws.jepa_z[..])`.
- Hacer que la llamada sea **zero-cost cuando `MUD_SLIMEX ≠ 1`** (early return).
- **Test:** `test_slimex_snapshot_in_forward_path` (integration test).

### 2.2 Conectar `read_slimex_cell()` ↔ `MemoryContinuityTable::get_entry()`

```rust
// En el GEMV de FFN cuando se monta un experto:
let slot = ws.read_slimex_cell(dim_idx) as u8;  // bits 8..15
if let Some(entry) = continuity_table.get_entry(slot) {
    // Usar entry.weight_ptr + entry.scale_ptr en lugar de buscar en HashMap
}
```

- **Impacto:** 54.87× lookup speedup vs HashMap (ya calculado en `MUD_MASTER_CELL_V5_AND_WORM_GAIN_CALCULATIONS_2026-07-23.md`).

### 2.3 Profiling benchmark del kernel AVX2

- Crear `tools/slimex_asm_bench.rs` que mide unpack de N cells (N=1k, 10k, 100k) y compara con versión escalar.
- **Target:** demostrar ~10× speedup en i7-1260P.

### 2.4 Integrar `MudFileMerger` en el flujo de save

- En `save_checkpoint` (`corpus_trainer.rs:2989-3000`), si hay submódulos mounted, fusionar antes de escribir.
- Default: off. Opt-in via `MUD_MERGE_ON_SAVE=1`.

**Total Sprint 2:** 1-2 semanas · SlimeX pasa de "especificado" a "wired".

---

## Sprint 3 — CMUD end-to-end + trainable sidecar — 1-2 semanas

### 3.1 Validar `cmud_spectral_health` end-to-end

- `inference.rs:650` ya testea `MUD_CMUD_THINK=1`; falta validar que **mejora perplexity** vs baseline.
- Crear `tools/cmud_perplexity_bench.rs` que mide PPL en `wikitext-2` subset con y sin C-MUD.
- **Target:** PPL no-regresión (< +2%) + Latency no-regresión (< +10%).

### 3.2 Wire `cmud_backward` al trainer

- Conectar el backward de CMUD al loop de training cuando `MUD_CMUD_THINK=1`.
- Backward debe propagar a las dimensiones de `phase_bias`, `alpha_per_dim`, `gate_temp` en `CmudLayerParams`.

### 3.3 Sidecar versioning

- Sidecar actual (`{model}.cmud.json`) no tiene version. Añadir `"version": 1` y check al load.
- Si la versión cambia, fallar con error claro (P-17).

### 3.4 `tools/cmud_train.rs` robusto

- Auto-tokenización (ya implementado con `Tokenizer::from_mud_metadata`).
- Validar que el corpus no esté vacío.
- Limitar el número de iteraciones (`MUD_CMUD_TRAIN_MAX_ITERS`).

**Total Sprint 3:** 1-2 semanas · CMUD pasa de "research" a "production-trainable".

---

## Sprint 4 — Refactor estructural P0 (V25 audit) — 1-2 semanas

### 4.1 Split `corpus_trainer.rs` (5407 líneas) en submódulo

```
src/mud/trainer/
├── mod.rs               (MudCorpusTrainer struct, ~100 LOC)
├── alignment.rs         (deep_local_alignment + L-QAT, ~400 LOC)
├── session.rs           (run_alignment_session, ~1500 LOC)
├── circuit.rs           (run_training_circuit, ~300 LOC)
├── debate.rs            (run_debate_session wrapper, ~100 LOC)
├── integrity.rs         (model_norms_alive, circuit_eval_integrity, ~250 LOC)
├── checkpoint.rs        (save_checkpoint + hash_trained_weights, ~100 LOC)
├── mhc_optimizer.rs     (mhc_scale_sgd_step, apply_sgd_shadow_update, ~100 LOC)
├── telemetry.rs         (telemetría, ~150 LOC)
└── env.rs               (MUD_TRAIN_* env var parsing, ~200 LOC)
```

### 4.2 `trait Optimizer` en lugar de enum cerrado

```rust
pub trait Optimizer: Send + Sync {
    fn name(&self) -> &'static str;
    fn step(&mut self, w: &mut [f32], g: &mut [f32], lr: f32, weight_decay: f32) -> anyhow::Result<()>;
    fn needs_moments(&self) -> bool { false }
    fn clone_box(&self) -> Box<dyn Optimizer>;
}
```

**Beneficio:** añadir Lion, Shampoo, etc. es 1 archivo + 1 línea en `select_optimizer`. Antes eran 9 sitios.

### 4.3 `MudError` enum + propagación de errores

- Convertir 297 `unwrap()` + 11 `panic!` en producción a `Result<T, MudError>`.
- Solo en init / tests / donde cualquier fallo es fatal, mantener `unwrap`.

### 4.4 Romper ciclo `corpus_trainer ↔ slime_backward`

- Mover `mhc_scale_sgd_step` y `apply_optimizer_cpu_step_and_pack` a `src/mud/optim/`.

**Total Sprint 4:** 1-2 semanas · code health +50%.

---

## Sprint 5 — Perf P1 (V25 audit) — 2-3 semanas

### 5.1 Eliminar allocs en `evaluate_slime_block_moe`

- Mover a `SlimeWorkspace`:
  - `csa_blocks_scratch: AlignedArray<usize>` (capacity hca_max_pos/2)
  - `csa_index_scratch: AlignedArray<f32>` (capacity hca_max_pos)
  - `csa_pairs: AlignedArray<(f32, usize)>`
  - `csa_forced: SmallVec<[usize; 32]>` (stack-first)

### 5.2 Mutex → AtomicBool en GPU dispatch

```rust
static GPU_DISABLED: AtomicBool = AtomicBool::new(false);
// Init: if !probe_gpu().0 { GPU_DISABLED.store(true, Ordering::Relaxed); }
// Hot path: if GPU_DISABLED.load(Ordering::Relaxed) { return false; }
```

### 5.3 Vectorizar `ternary_gemv_backward` con AVX2

- 8 weights por u32 → 8 lanes FMA.
- Reemplazar `match val { 0x1 => 1.0, 0xF => -1.0, _ => 0.0 }` con `TERNARY_LUT[bits] * scale_gy`.

### 5.4 Pack path vectorizado

- Hoy `pack_elut_prq` en `corpus_trainer.rs:5019-5046` tiene un branchy inner loop. Vectorizar.

**Total Sprint 5:** 2-3 semanas · +5-10 ms/chunk adicionales + 30k allocs/chunk eliminados.

---

## Sprint 6 — Documentación, CI, seguridad — 1-2 semanas

### 6.1 SECURITY.md + threat model

- Documentar:
  - `.mud` se trata como blob no confiable (validador con caps de counts).
  - `rlvr.rs` y `sandbox.rs` están detrás de feature flags.
  - No hay telemetría a internet.

### 6.2 Fuzzing harness

- `cargo-fuzz` targets:
  - `fuzz_mud_load` — MUD file parser
  - `fuzz_gguf_load` — GGUF file parser
  - `fuzz_safetensors_load` — safetensors wrapper

### 6.3 Caps en `MudFile::load`

```rust
const MAX_META_COUNT: u32 = 1_000_000;
const MAX_TENSOR_COUNT: u32 = 1_000_000;
if meta_count > MAX_META_COUNT || tensor_count > MAX_TENSOR_COUNT {
    bail!("absurd count, possible DoS");
}
```

### 6.4 Cap recursion en `collect_files`

```rust
const MAX_DEPTH: usize = 5;
const MAX_FILES: usize = 100_000;
```

### 6.5 CHANGELOG.md

- Consolidar todos los fixes (1-41) + P0-P5 + SX-01..07 + ST-01 + CMUD-01 en un CHANGELOG formal.

**Total Sprint 6:** 1-2 semanas · hardening + docs.

---

## Sprint 7 — Mejoras nice-to-have — backlog

### 7.1 Doxygen de kernels ASM

- 15 archivos `.s` con comentarios mínimos → añadir header con descripción del algoritmo, references, why-this-layout.

### 7.2 Referencias académicas

- `slime_jepa.rs` → paper OU process
- `mhc_residual` → paper DeepSeek-V4 (realmente mHC)
- `cmud.rs` → paper ComplexFormer
- `rlvr.rs` → paper Verifiable Rewards

### 7.3 Magic numbers → `constants.rs`

- `127.0` (i8 sat), `0x9E37_79B9` (hash mult), `1000.0` (mHC radius default), Adam betas, etc.

### 7.4 Eliminar macro `println!` redefinido

- `corpus_trainer.rs:10-29` redefine `println!` con check `MUD_CIRCUIT_TUI` — anti-pattern severo. Reemplazar con `tui_safe_println!` con nombre explícito.

### 7.5 Long functions refactor

- `train_on_sequence` (1018 LOC) → split en `forward_pass`, `compute_gradients`, `apply_optimizer`, `log_metrics`.
- `run_alignment_session` (653 LOC) → split.
- `evaluate_slime_block_moe` (670 LOC) → split.

---

## Resumen consolidado

| Sprint | Esfuerzo | Output |
|--------|----------|--------|
| 1 | 1 sem | 0 RCE + 0 OOB + 2-3 s/chunk |
| 2 | 1-2 sem | SlimeX end-to-end wired + benchmark |
| 3 | 1-2 sem | CMUD perplexity validado + sidecar versionado |
| 4 | 1-2 sem | corpus_trainer split + trait Optimizer + cycle broken |
| 5 | 2-3 sem | 5-10 ms/chunk adicionales + 30k allocs/chunk eliminados |
| 6 | 1-2 sem | SECURITY.md + fuzzing + caps + CHANGELOG |
| 7 | backlog | Doxygen + papers + constants + macro cleanup |
| **TOTAL** | **8-12 sem** | **Motor production-hardened + 2 nuevas features live** |

---

## Métricas de éxito (post todos los sprints)

| Métrica | Hoy | Target |
|---------|-----|--------|
| **Vulnerabilidades P0** | 5 RCE + 1 OOB + 7 panic | **0** |
| **Speedup inferencia** | baseline (post P0-P5) | +30-50% (P1 RoPE + zero-alloc + 5-10 ms overhead removed) |
| **Tests** | 271 | 350+ (incluye fuzz corpus) |
| **Clippy warnings** | 0 | 0 |
| **`corpus_trainer.rs` LOC** | 5407 | <500 (subdividido) |
| **`unwrap()` en producción** | 297 | <50 (todo lo demás en `Result`) |
| **SlimeX live** | workspace + bus, no wired | wired + benchmark + usado en producción |
| **CMUD live** | research code, opt-in | sidecar versionado + perplexity validado + trainable |

---

## Riesgos y mitigaciones

| Riesgo | Mitigación |
|--------|-----------|
| SlimeX wiring introduce regresión en path default | Feature flag `MUD_SLIMEX=1`; tests separados; CI matrix con/sin |
| CMUD perplexity regresión | Guard con `circuit_eval_integrity` que rechaza si $P(\text{token}_0) > 0.15$ |
| Refactor `corpus_trainer.rs` rompe algo | Hacer en PRs pequeños con tests; `git bisect` friendly |
| RoPE precomputed tables inflan RAM | Solo precomputar para `max_pos` (no ilimitado) |
| Fuzzing encuentra bug real | Triage + fix + regression test |
