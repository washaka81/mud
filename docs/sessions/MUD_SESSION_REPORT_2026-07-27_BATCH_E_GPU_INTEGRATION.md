# MUD Session Report — 2026-07-27 — BATCH E: Vulkan D5 Elementwise Integration

## Resumen

Integración del `step_elementwise_sync` (elementwise.comp) en `jepa_stabilizer` y
`stp_loss_and_grad`, completando el **BATCH E** deferido del reporte de sesión
2026-07-26. Los shaders y dispatch functions ya existían; faltaba el *wiring* en
el trainer hot path.

## Cambios

### 1. `src/mud/slime_jepa.rs` — JEPA GPU path (modos 4/5)

- **`jepa_stabilizer`**: agregado parámetro `mut gpu: Option<&mut AshQatDispatcher>`.
- **GPU path**: cuando `gpu` es `Some`, los steps 1-2 (z-score normalize + OU update)
  se offloden a `step_elementwise_sync`:
  - Mode 5 (normalize): `y_norm[i] = (block_out[i] - mean) / rms`
  - Mode 4 (ema): `z_next[i] = 0.1 * y_norm[i] + 0.9 * z[i]` (alpha fijo = 0.1, OU puro)
- **Fallback CPU**: si el dispatch falla o `gpu` es `None`, usa el path escalar
  existente (adaptive alpha + jitter).
- Thread-local `JEPA_GPU_SCRATCH` para temporales `[y_norm_tmp, z_next_tmp]`.
- Constante `JEPA_GPU_ALPHA = 0.1` (matches AGENTS.md §5).
- Fix: `reg.jepa_energy` → `reg.read_integral()` (campo no existía en `SlimeRegister`).

### 2. `src/mud/stp_loss.rs` — STP GPU path (modo 6)

- **`stp_loss_and_grad`**: agregado parámetro `mut gpu: Option<&mut AshQatDispatcher>`.
- **GPU path**: cuando `gpu` es `Some`, los gradientes `g_d1`/`g_d2` se computan
  en CPU (dependen de escalares), luego se usa `step_elementwise_sync` (mode 6: axpy)
  para la inyección de gradientes:
  - `grad_h_r += g_d1` (axpy: `out = grad_h_r + 1.0 * g_d1`)
  - `grad_h_t += g_d2`
  - `grad_h_s += -(g_d1 + g_d2)`
- Thread-local `STP_AXPY_SCRATCH` para `[neg_sum, axpy_out]`.
- CPU path: loop escalar existente (sin cambios).

### 3. `src/mud/slime_forward.rs` — Threading del parámetro GPU

- **`evaluate_slime_block`**: agregado parámetro `gpu`, pasado a `evaluate_slime_block_moe`.
- **`evaluate_slime_block_moe`**: agregado parámetro `mut gpu`, pasado a ambas llamadas
  de `jepa_stabilizer` (atención + FFN) via `gpu.as_deref_mut()` (reborrow sin move).

### 4. `src/mud/corpus_trainer/train_step.rs` — Trainer wiring

- `train_on_sequence`: `vk_qat` ahora es `mut` (para `as_deref_mut()`).
- Forward pass: `evaluate_slime_block` recibe `vk_qat.as_deref_mut()`.
- STP: `stp_loss_and_grad` recibe `vk_qat.as_deref_mut()`.

### 5. Call sites actualizados (pasando `None`)

- `debate_trainer.rs`, `grad_checkpoint.rs`, `inference.rs`, `self_play.rs`,
  `speculative.rs`, `main.rs`, `slime_forward.rs` (test) — todos pasan `None`
  para el parámetro `gpu` (no tienen `AshQatDispatcher`).

## Diseño

- **Opt-in**: el path GPU solo se activa cuando `AshQatDispatcher` es `Some`.
  En Iris Xe (blacklisteado por `probe_gpu()`), el path CPU se usa automáticamente.
- **Safe fallback**: si `step_elementwise_sync` retorna error, se cae al path CPU.
- **Alpha fijo**: el path GPU usa alpha=0.1 (OU puro). El adaptive alpha + jitter
  es CPU-only (el shader genérico no los soporta).
- **Batching futuro**: las operaciones individuales se pueden batchar via
  `dispatch_combined_batch_async` (ya implementado) para D5 signal keep-alive.

## Validación

| Métrica | Estado |
|---------|--------|
| `cargo check --release` | ✅ Verde |
| `cargo test --lib` | ✅ 406 passed / 0 failed / 2 ignored |
| `cargo clippy --all-targets` | ✅ 0 errors / 0 warnings |
| `cargo build --bin run_trainer` | ✅ Verde |

## D5 Signal Keep-Alive (segunda fase — COMPLETADA)

Además del BATCH E, se completó la segunda fase del D5 signal keep-alive:

### Cambios

| Archivo | Cambio |
|---------|--------|
| `ash_backend.rs` | `dispatch_combined_batch_async` ahora despacha **ambos** updates mHC y no-mHC en la sección de optimizador (antes saltaba los no-mHC con `if !u.is_mhc { continue; }`) |
| `ash_qat_dispatcher.rs` | Nueva función `step_combined_async_deferred` — combina optimizer step + mHC scale SGD en un **único `vkQueueSubmit`** (via `dispatch_combined_batch_async`) |
| `train_step.rs` | Trainer usa `step_combined_async_deferred` cuando `MUD_PIPELINE_OVERLAP=0` (default), combinando optimizer + mHC en un solo submit. Pipeline overlap path mantiene `step_pipeline` + `step_mhc_async` separados. |

### Cómo funciona

1. Trainer colecta optimizer updates (`all_steps`) y mHC updates (`mhc_updates`)
2. `step_combined_async_deferred` uploadea ambos tipos de gradientes
3. Construye un único `Vec<AshOptimizerUpdate>` con `is_mhc=false` (optimizer) y `is_mhc=true` (mHC)
4. Llama `dispatch_combined_batch_async` → **un solo `vkQueueSubmit`** con todos los updates
5. Iris Xe mantiene boost clock (sin power-gate entre optimizer y mHC)

### Validación

| Métrica | Estado |
|---------|--------|
| `cargo check --release` | ✅ Verde |
| `cargo test --lib` | ✅ 406 passed / 0 failed / 2 ignored |
| `cargo clippy --all-targets` | ✅ 0 errors / 0 warnings |

## Próximos pasos

- **CellV5 Fase 2 ASM**: kernels AVX2 para wave_collapse/phase_repulsion/gauss_mul.
- **CellV5 Fase 3 KV pre-compresión**: F32→CellV5→Delta→LZ4.
