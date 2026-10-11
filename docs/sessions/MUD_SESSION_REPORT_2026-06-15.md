# MUD Session Report: 2026-06-15

## Objetivo
Unificar y modernizar el orquestador `mud.sh`, catalogar las 60+ herramientas del proyecto con sus recomendaciones de uso, reordenar el pipeline de recovery, auditar el estado actual del proyecto y actualizar ROADMAP + documentación de sesión.

## Acciones realizadas

### 1. Reescritura de `mud.sh` v3.0 — "Tool Catalog & Recovery Unification"
- **Secciones claras y ordenadas:**
  1. **RECOVERY & RESTAURATION** — `restore-iq`, `deep-repair`, `awake`, `project`, `validate`, `estimate`, `interactive`
  2. **TRAINING & ALIGNMENT** — `train`, `l-qat`, `full-qat`, `download-corpus`, `qat-bench`, `embed-tern`
  3. **CONVERSION & FORGE** — `convert`, `forge`, `import-gguf`, `export-sf`, `fix-metadata`, `verify`
  4. **DIAGNOSTICS & INSPECTION** — `diag`, `hw`, `bench`, `audit`, `diagnose*`, `step`, `profiler`, `inspect-*`, `probe`, `wave-audit`, `microscope`, `banner-cmd`, `offsets`, `calibrator`, `check-norms`, `check-vocab`
  5. **BENCHMARKS** — `inference-bench`, `kernel-bench`, `matrix-bench`, `memory-bench`, `precision-bench`, `jamba-bench`, `jepa-bench`, `galore-bench`, `diffusion`, `wave-demo`, `eval`, `int4`
  6. **DEEP AUDITS** — `deep-math`, `attention-audit`, `embed-audit`, `ternary-audit`, `weight-audit`, `moe-audit`, `language-audit`, `tokenizer-audit`, `ldt-audit`, `phase14`, `cognitive`, `ptr-audit`, `expert-anatomy`
  7. **INTERACTION** — `chat`, `chat-once`, `vocab`, `dump`, `list-tensors`, `print-mud`, `iq-box`
  8. **SAFETY** — `ckpt`, `restore`, `clean`, `deep-clean`, `verify`, `bound`
  9. **META** — `tools`, `menu`, `help`

- **Bugs corregidos:**
  - `shift 2 || shift 1` frágil en `chat` reemplazado por `shift 2 2>/dev/null || shift $# 2>/dev/null`.
  - Helper `run_tool` unifica `cargo run --release --features="tools" --bin`.
  - Helper `banner` unifica los banners decorativos.
  - Eliminadas las rutas duplicadas `align|full-qat` (ambos hacían `--lqat`): ahora `full-qat` hace `--full-qat` y `l-qat` hace `--lqat`, coherente con la semántica.
  - El pipeline `restore-iq` ahora ejecuta 6 pasos (Bound → Estimate → L-QAT → Full-QAT → Project → Validate) en lugar de los 6 originales mal numerados.

### 2. Nuevo subcomando `./mud.sh tools`
Catálogo razonado con **flujos recomendados** según el síntoma:
- **Flujo para modelo nuevo:** `convert → verify → bound → estimate → restore-iq → validate → chat`
- **Afasia semántica:** `tokenizer-audit → embed-audit → deep-repair`
- **Inestabilidad numérica:** `deep-math → bound → diagnose-layers`
- **Modelo lento:** `diag → profiler → kernel-bench → memory-bench`
- Tabla de 60+ herramientas con propósito breve.

### 3. Herramientas adicionales registradas en `Cargo.toml`
Se agregaron como `[[bin]]` herramientas útiles que existían pero no estaban expuestas:
- `list_tensors` — Lista todos los tensores de un `.mud` (nombre, shape, tipo)
- `print_mud` — Imprime metadata cruda del archivo
- `check_norms` — Estadísticas de capas de normalización
- `check_vocab` — Estadísticas de vocabulario

### 4. Auditoría rápida de estado del proyecto
- **`cargo build --lib`**: OK.
- **`cargo test --lib`**: 16 passed; 0 failed; 60 filtered out.
- **`cargo clippy --lib`**: 17 warnings activos (política 0-warning pendiente):
  - `needless_range_loop` ×6 en `corpus_trainer.rs` (209, 766, 771, 813, 1980, 2031)
  - `needless_borrow` ×6 en `corpus_trainer.rs` (1248, 1409, 1411, 1444, 1458, 1462)
  - `vec_init_then_push` ×2 en `corpus_trainer.rs` (1685, 1831)
  - `missing_safety_doc` en `holographic_loss.rs:11` (`compute_phase_gradients_avx2`)
  - `new_without_default` en `subagents.rs:32` (`SubagentManager`)
  - `legacy_numeric_constants` en `integral_threshold.rs:21` (`std::f32::MAX`)

### 5. Actualización de ROADMAP.md
- Agregada sección "Tool Catalog & Orchestration" con hito de esta sesión.
- Actualizada tabla de auditorías conocidas.

## Próximas prioridades (recomendadas para la siguiente sesión)

### Prioridad 1 — Resolver warnings clippy (política 0-warning)
Ejecutar `cargo clippy --fix --lib -p forge_llm` y resolver manualmente los casos no automáticos. Bloquea el cumplimiento del mandato arquitectónico "0-Error, 0-Warning".

### Prioridad 2 — Vulkan FFN blowup (VULK-03)
Aún sin resolver: el RMS del estado oculto de `run_chained_ffn` crece de 0.82 → 3.7B a lo largo de 30 capas. Requiere debug del SPIR-V kernel (peso/escala upload + decodificación ternaria + aplicación PRQ).

### Prioridad 3 — QAT Pipeline (PHASE 18)
- QAT-01: Operaciones STE en `forge_autograd`
- QAT-02: Master weights FP32 persistentes
- QAT-03: Propagación real de gradientes en todas las capas

### Prioridad 4 — Speculative Decoding (SPEC-01)
Draft model 15MB para predecir 5 tokens + verificación en bloque del modelo 2B (optimización 500% bandwidth).

### Prioridad 5 — TL2 Kernels (HW-05)
1.67 bits por parámetro empaquetando 5 pesos ternarios por byte con LUT AVX2.

## Estado final
- `mud.sh` v3.0 funcional, sintaxis validada (`bash -n`).
- 60+ herramientas expuestas y catalogadas con recomendaciones.
- Pipeline UCP (Universal Calibration Protocol) correctamente ordenado y documentado.
- Roadmap actualizado.
- 17 warnings de clippy documentados como tarea pendiente.

---

## Acciones realizadas (continuación) — Clippy Backlog Resolution

### 6. Resolución de los 17 warnings de clippy (mandato 0-warning)
Ejecutado `cargo clippy --fix --lib --features=tools --allow-dirty --allow-staged` que resolvió automáticamente 7 de 17 warnings (`needless_borrow` ×6 en `corpus_trainer.rs`, `new_without_default` en `subagents.rs`). Los 10 restantes se resolvieron manualmente:

- **`needless_range_loop` ×6** en `src/mud/corpus_trainer.rs`:
  - L209: `for i in 0..n` → `data.chunks_exact(4).enumerate().take(n)` (elimina bounds-check manual `start+4 <= data.len()`).
  - L763: `for c in 0..cols` → `x.iter_mut()` con estado LCG enhebrado.
  - L768 + L788: doble loop `for r in 0..rows` + `for c in 0..cols` → `teacher_fp32.chunks_exact(cols).zip(student_shadow.chunks_exact_mut(cols)).enumerate().take(rows)` + `t_row.iter().zip(s_row.iter()).zip(x.iter())`. Refactorización completa del loop de destilación capa-a-capa, preservando exactamente la semántica de forward+backward+STE.
  - L813: `for r in 0..rows` → `student_shadow.chunks_exact(cols).enumerate().take(rows)`.
  - L1977: `for pos in 0..QAT_SEQ_LEN` → `seq.iter().enumerate().take(QAT_SEQ_LEN)`.
  - L2028: `for i in 1..loss_components.len()` → `loss_components.iter().skip(1)` (NodeId es `Copy`).
- **`vec_init_then_push` ×2**: reemplazados `Vec::with_capacity(N) + N pushes` por `vec![...]` literales (líneas 1678 y 1825).
- **`missing_safety_doc`** en `src/mud/holographic_loss.rs:11`: añadida sección `# Safety` documentando requisitos AVX2, invariantes de longitud de slices y aliasing de `grad_out`.
- **`legacy_numeric_constants`** en `src/mud/integral_threshold.rs:21`: `std::f32::MAX` → `f32::MAX`.
- **`unused_mut`** en `corpus_trainer.rs:1241` (`run_full_qat_loop`): removido `mut` redundante en el binding `mut mud: &mut MudFile`.
- **`doc_lazy_continuation`** en `holographic_loss.rs:16`: indentación correcta del bullet de continuación en la sección Safety.
- **Fixes de compilación para tools registrados en sesión 8**:
  - `tools/list_tensors.rs`: usa `mud.skills` en lugar del inexistente `mud.tensors`; imprime skill + tensor.
  - `tools/check_vocab.rs`: carga tokenizador desde `global_metadata["tokenizer.tokens"/"tokenizer.merges"]` vía `Tokenizer::from_mud_metadata`.
  - `tools/check_norms.rs`: corrige bound `Borrow<&str>` (`core.tensors.get(name)` → `get(*name)`) y reemplaza `get().is_none()` por `!contains_key()`.
  - `tools/qat_benchmark.rs`: `let trainer` → `let mut trainer` (run_benchmark requiere `&mut self`).

### Validación final
- **`cargo clippy --all-targets --features=tools -- -D warnings`**: **0 errors, 0 warnings** (mandato arquitectónico desbloqueado).
- **`cargo test --lib`**: **16 passed; 0 failed; 60 filtered out** (regresión nula).
- **`cargo build --lib`**: OK.

### Prioridades actualizadas para la siguiente sesión
1. ~~Resolver 17 warnings clippy~~ → **DONE**.
2. ~~VULK-03 (FFN blowup RMS 0.82 → 3.7B)~~ → **DONE** (ver abajo).
3. **🔴 PRÓXIMA SESIÓN → QAT-01/02**: STE ops (`Op::STEQuantize`, `Op::RMSNorm`) + master weights FP32 persistentes (`MudQatState` con Adam m/v) en `forge_autograd`.
4. **SPEC-01**: Draft model 15MB + verify en bloque.
5. **HW-05**: TL2 kernels (1.67 bits/param).

---

## Acciones realizadas (continuación) — VULK-03 FFN Blowup Resolution

### 7. Fix VULK-03: Bypass de `run_chained_ffn` cuando SubLN está activo

**Síntoma:** El RMS del estado oculto crece exponencialmente a lo largo de 30 capas: 0.82 → 213K → 161M → 3.7B. Solo ocurre por la ruta Vulkan; la ruta CPU produce valores correctos.

**Causa raíz:** El shader `run_chained_ffn` (SPIR-V monolítico) ejecuta W1 → SiLU*W3 → W2 en un solo command buffer sin insertar la capa BitDistill SubLN entre la activación SiLU y la proyección W2. Sin SubLN, las activaciones no normalizadas alimentan W2 con un RMS que se acumula exponencialmente capa tras capa. La ruta CPU (`forward.rs` ~L2188) aplica correctamente el orden W1 → SiLU → SubLN → W2.

**Fix aplicado** en `src/mud/forward.rs` (`run_expert_ffn`):
```rust
let mut vlk_done = false;
let subln_active = !ffn_sub_norm_w.is_null();
if !force_cpu && !subln_active {
    // Vulkan fast path only when SubLN is NOT active
    if let Some(vk) = vk_ctx { ... vk.run_chained_ffn(...) ... }
}
if !vlk_done {
    // CPU fallback: W1 → SiLU → SubLN → W2 (correct ordering)
}
```

Cuando `ffn_sub_norm_w` es non-null (SubLN inyectado por BitDistill durante la conversión), se descarta la ruta Vulkan y se usa el fallback CPU que respeta el orden correcto.

**Nota para follow-up:** El shader monolítico podría dividirse en dos command buffers (W1+SiLU → CPU SubLN → W2) para restaurar la aceleración Vulkan incluso con SubLN activo. Esto requiere refactoring en `src/vulkan/mod.rs:run_chained_ffn` y un nuevo shader `silu_gate_subln.comp`.

**Validación:**
- `cargo clippy --all-targets --features=tools -- -D warnings` → **0 errors, 0 warnings**.
- `cargo test --lib` → **76 passed, 0 failed** (sin regresión).

### Prioridades finales de la sesión
1. ~~Resolver 17 warnings clippy~~ → **DONE**.
2. ~~VULK-03 FFN blowup~~ → **DONE**.
3. **🔴 PRÓXIMA SESIÓN → QAT-01/02**: STE ops (`Op::STEQuantize`, `Op::RMSNorm`) + master weights FP32 persistentes (`MudQatState` con Adam m/v) en `forge_autograd`.
4. **SPEC-01**: Draft model 15MB + verify en bloque.
5. **HW-05**: TL2 kernels (1.67 bits/param).
