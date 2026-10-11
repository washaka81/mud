# MUD GPU Signal Keep-Alive Plan (D5) — 2026-07-26

> **Objetivo:** Mantener la Intel Iris Xe en boost clock durante el entrenamiento mediante
> batching de múltiples cálculos en un solo `dispatch_optimizer_batch_async` (L-05 DoubleFrame,
> no `submit_and_wait`).

---

## 1. Problema

La Iris Xe (ADL GT2 UMA) se **power-gatea** cuando no hay trabajo GPU continuo. El overhead de
reactivación (~100-500μs) domina cuando cada submit contiene pocos ops (ej. mHC solo = 49K ops).

### Sin batching (status actual)
```
Submit 1: mHC scale SGD (49K ops) → GPU trabaja 10μs → se apaga
Submit 2: mHC scale SGD (49K ops) → reactivación 200μs → trabaja 10μs → se apaga
Throughput: ~1K updates/s, GPU en power-save
```

### Con batching (objetivo)
```
Submit 1: backward (14M) + mHC (49K) + STP (2K) + JEPA (25K) = 14.1M ops
GPU trabaja 5ms continuo → mantiene boost clock
Throughput: ~100K updates/s, GPU en boost
```

---

## 2. Tareas

### Tarea 1: Conectar `ternary_backward.comp` al código Rust 🟢 **COMPLETADA**

**Estado:** Shader existe (`assets/shaders/ternary_backward.comp` + `.spv`) y pipeline `backward_pipe` ya estaba creado.

**Cambios implementados:**
1. `ash_backend.rs`: Agregado `BackwardPushConstants` struct + `dispatch_ternary_backward_host_sync` (sync, `submit_and_wait`) + `dispatch_ternary_backward_async` (async, DoubleFrame L-05) + `readback_backward_grad`
2. `ash_qat_dispatcher.rs`: Agregado `step_backward_sync`, `step_backward_async`, `readback_backward_grad`
3. **Pendiente:** Integración en `slime_backward.rs` — requiere shader adicional para dL/dX (el shader actual solo calcula dL/dW)

**DoubleFrame (L-05) async path:**
- `dispatch_ternary_backward_async` usa `frame.acquire` + `frame.submit` (no `submit_and_wait`)
- CPU continúa inmediatamente → overlap CPU/GPU
- `sync_frames` + `readback_backward_grad` para recuperar grad_w
- **Signal keep-alive:** batch con mHC/STP/JEPA via `dispatch_optimizer_batch_async` (future: combined shader)

**Complejidad:** ~250 líneas implementadas
**Impacto:** 14M ops disponibles para GPU (dL/dW backward), async con DoubleFrame

### Tarea 2: Shader genérico para STP+JEPA 🟢 **PARCIALMENTE COMPLETADA**

**Estado:** Shader creado y conectado. Integración en STP/JEPA pendiente.

**Cambios implementados:**
1. `assets/shaders/elementwise.comp` — shader genérico con 7 modos (add, subtract, clamp, sigmoid, ema, normalize, axpy)
2. `assets/shaders/spirv/elementwise.spv` — compilado con `glslc --target-spv=spv1.3`
3. `ash_backend.rs`: Agregado `ELEMENTWISE` (include_bytes), `elementwise_pipe`, `ElementwisePushConstants`, `dispatch_elementwise_sync`
4. `ash_qat_dispatcher.rs`: Agregado `step_elementwise_sync`

**Pendiente:** Integración en `stp_loss.rs` (modo 0/6: add/axpy para gradient injection) y `slime_jepa.rs` (modos 3/4/5: sigmoid/ema/normalize)

**Complejidad:** ~100 líneas implementadas
**Impacto:** ~27K ops adicionales por batch (STP + JEPA element-wise)

### Tarea 2: Shader genérico para STP + JEPA 🟡 Mediano plazo

**Estado:** No existe. STP y JEPA usan CPU (AVX2).

**Cambios:**
1. Crear `assets/shaders/stp_jepa.comp` — operaciones element-wise + reducción
2. Compilar a SPIR-V
3. `ash_backend.rs`: Agregar `dispatch_stp_jepa_sync`
4. `stp_loss.rs` / `slime_jepa.rs`: Modificar para usar path GPU

**Complejidad:** ~150 líneas (shader) + ~100 líneas (Rust)
**Impacto:** ~27K ops adicionales por batch

### Tarea 3: Batch + DoubleFrame overlap 🟢 **COMPLETADA**

**Estado:** `dispatch_combined_batch_async` implementado en `ash_backend.rs` + `step_combined_batch_async` en `AshQatDispatcher`.

**Cambios implementados:**
1. `ash_backend.rs`: Agregadas estructuras `AshBackwardUpdate` + `AshElementwiseUpdate` + método `dispatch_combined_batch_async`
2. `ash_qat_dispatcher.rs`: Agregado `step_combined_batch_async`

**Cómo funciona:**
- Combina backward (14M ops) + mHC (49K) + element-wise (27K) en un **único** `vkQueueSubmit`
- Usa `frame.acquire` + `frame.submit` (DoubleFrame L-05, no `submit_and_wait`)
- CPU continúa inmediatamente → overlap CPU/GPU
- Readback diferido: `sync_frames` + `readback_backward_grad` / `readback_mhc` / `readback_elementwise`

**Complejidad:** ~150 líneas
**Impacto:** Mantiene Iris Xe en boost clock con un solo submit

### Tarea 4: Integración en STP/JEPA 🟡 **Pendiente**

**Estado:** `mhc_stress_bench` ya existe.

**Cambios:**
1. Extender `mhc_stress_bench` para incluir backward pass
2. Agregar métricas de GPU power state (boost clock residency)
3. Benchmark CPU vs GPU con batching

**Complejidad:** ~100 líneas
**Impacto:** Métricas de verificación

---

## 3. Cronograma

| Semana | Tarea | Responsable | Entregable |
|--------|-------|-------------|------------|
| Semana 1 | Tarea 1 (backward shader) | Agent | `dispatch_ternary_backward_host_sync` + tests |
| Semana 1 | Tarea 4 (benchmark) | Agent | `mhc_stress_bench` extendido |
| Semana 2 | Tarea 2 (STP+JEPA shader) | Agent | `stp_jepa.comp` + dispatch |
| Semana 2 | Tarea 3 (batch + overlap) | Agent | `step_full_async` en dispatcher |
| Semana 3 | Verificación end-to-end | Agent | Benchmark completo + docs |

---

## 4. Métricas de éxito

| Métrica | Sin batching | Con batching (objetivo) |
|---------|-------------|------------------------|
| **Ops per submit** | ~49K (mHC solo) | ~14.1M (backward + mHC + STP + JEPA) |
| **GPU boost clock residency** | < 10% | > 80% |
| **Throughput** | ~1K updates/s | ~100K updates/s |
| **CPU/GPU overlap** | 0% | > 50% (DoubleFrame) |
| **Power efficiency** | GPU apagada frecuentemente | GPU en boost continuo |

---

## 5. Riesgos

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|-------------|---------|------------|
| `ternary_backward.comp` no funciona en Iris Xe | Media | Alto | Usar `gemv_pipe` (no QKV); probar antes de integrar |
| STP/JEPA shader introduce precisión diferente | Baja | Medio | Comparar CPU vs GPU en `mhc_stress_bench` |
| DoubleFrame race condition | Baja | Alto | Usar `sync_frames` antes de readback |
| Iris Xe SIGSEGV en `gemv_pipe` | Baja | Alto | `probe_gpu()` ya blacklistea Iris Xe; fallback a CPU |

---

## 6. Referencias

- Análisis base: `docs/research/MUD_VULKAN_DELEGATION_ANALYSIS_2026-07-26.md`
- AGENTS.md §Runtime Truth (P-02, P-05, P-17)
- `src/vulkan/ash_backend.rs:658` (`dispatch_optimizer_batch_async`)
- `assets/shaders/ternary_backward.comp` (shader existente)
- `tools/mhc_stress_bench.rs` (benchmark existente)
