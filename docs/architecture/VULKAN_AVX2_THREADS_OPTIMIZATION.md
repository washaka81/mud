# OPTIMIZACIÓN VULKAN + AVX2 + 8 THREADS
*Pipeline de cómputo extremo combinado CPU/GPU*

---

## DIAGNÓSTICO ACTUAL

### Pipeline real (serial):
```
Forward (CPU, 8 threads) → Backward (CPU, 8 threads) → GPU Optimizer → sync_all()
                                                                          ↓
Forward (CPU, 8 threads) → Backward (CPU, 8 threads) → GPU Optimizer → sync_all()
```

### Pipeline posible (overlap):
```
Forward N (CPU)  ──────→ Backward N (CPU) ──────→ GPU Optimizer N ──→ sync ──→ Forward N+1
                         Forward N+1 (CPU) ──start early──┘ (sin esperar)
```

**Problema raíz:** `corpus_trainer.rs:2329` llama `sync_all()` al PRINCIPIO de cada iteración. `sync_and_readback_all()` se llama INMEDIATAMENTE después de `step_async()`. El `DoubleFence` existe pero es puramente ceremonial — nunca hay overlap real.

---

## FASE 1: TRUE DOUBLE-BUFFER OVERLAP (DÍA 1)
*Mayor impacto: 1.5-2× throughput sin nuevo hardware.*

### Cambios:

**1.1** Mover `sync_all()` del principio de `train_on_sequence` al final del backward, justo antes de `step_async`.

**1.2** Cambiar `sync_and_readback_all` de `device_wait_idle` a `vkWaitForFences` con timeout=0 (poll). En lugar de bloquear, retornar estado. El forward del siguiente chunk puede empezar inmediatamente.

**1.3** El forward del chunk N+1 comienza con pesos del chunk N (GPU aún optimizando). Cuando GPU termina, los pesos nuevos están en mapped_ptr (UMA = zero-copy, no necesitan copy explícita). El forward lee los pesos nuevos automáticamente porque el mapped_ptr apunta a la misma RAM.

### Flujo resultante:
```
Forward N (CPU, 8 threads) ──────────→ Backward N (CPU, 8 threads)
                                            ↓
                                      GPU Optimizer N (async)
                                            ↓
                                      sync_all() light (fence poll, no block)
                                            ↓
Forward N+1 (CPU, 8 threads) ←─── readback zero-copy (no memcpy, UMA)
```

**Archivos:** `corpus_trainer.rs:2326-2330`, `ash_qat_dispatcher.rs:247-271`, `ash_backend.rs:770`

---

## FASE 2: DESPLEGAR SHADERS EXISTENTES (DÍA 2)
*Los shaders ya existen pero nunca se despachan.*

### 2.1 `mha.comp` — Multi-Head Attention
**Estado:** Compilado (`ash_backend.rs` lo carga) pero **nunca despachado** desde `slime_forward.rs`.
**Implementación:** Añadir `AshContext::dispatch_mha()` que lance `[n_heads, seq_len, 1]` workgroups. El shader usa `__shared s_scores[64]` + `subgroupAdd` para softmax.
**Speedup:** 5-10× sobre el actual loop scalar en CPU.

### 2.2 `rms_norm.comp` — RMSNorm
**Estado:** Compilado, nunca despachado.
**Implementación:** Añadir `AshContext::dispatch_rms_norm()` para lanzar `[hidden/32, 1, 1]` workgroups. Push constants: `hidden_size`, `eps`.
**Speedup:** 3-5× sobre CPU.

### 2.3 Eliminar `newton_schulz_step1/2.comp` del pipeline
**Estado:** Shaders compilados, pipelines creados, pero `dispatch_optimizer_batch_async` NUNCA los despacha. No hay VRAM buffers para Muon.
**Acción:** Quitar de `ash_backend.rs:305-306` (creación de pipelines). Ahorra ~100ms de init time.

**Archivos:** `slime_forward.rs`, `ash_backend.rs`, `assets/shaders/mha.comp`, `assets/shaders/rms_norm.comp`

---

## FASE 3: SHARED MEMORY TILING EN GEMV SHADER (DÍA 3-4)

### 3.1 `ternary_gemv_unified.comp` — Error crítico de memoria global

**Problema:** El shader lee `x[]` desde global memory para CADA output row. Para hidden=2560 con 2560 rows: `2560 × 2560/8 × 32 threads = 26M` lecturas globales.

**Fix:** Cargar `x[]` en `__shared` una vez por workgroup (Iris Xe: 64KB SLM, 2560 f32 = 10KB, cabe sobrado):

```glsl
// Antes del loop de rows:
__shared float s_x[2560];  // cabe en SLM (10KB)
if (gl_LocalInvocationIndex < hidden) {
    s_x[gl_LocalInvocationIndex] = x[gl_LocalInvocationIndex];
}
barrier();

// En el loop de rows, leer de s_x (3-5 cycles) en vez de global (200-400 cycles):
float x_val = s_x[x_idx + lane];  // shared memory, ~20× más rápido
```

**Speedup:** 2-3× en GPU GEMV. Fundamental para que GPU sea competitiva vs AVX2.

**Archivo:** `assets/shaders/ternary_gemv_unified.comp`

---

## FASE 4: PARALLEL QKV DISPATCH (DÍA 4)

**Problema:** `slime_forward.rs:293-318` llama `ternary_gemv_rowwise` 3 veces (Q, K, V) con `pool.wait_all()` entre cada una.

**Fix A (GPU):** Añadir dimensión `matrix_id` al dispatch de `ternary_gemv_unified.comp`. Un solo dispatch computa Q, K, V simultáneamente → `[out_dim, 3, 1]` workgroups.

**Fix B (CPU):** Despachar Q, K, V a 3 grupos de threads distintos en PCorePool sin `wait_all` intermedio:

```rust
pool.execute(|| q_gemv(...));
pool.execute(|| k_gemv(...));
pool.execute(|| v_gemv(...));
pool.wait_all();  // un solo wait para los 3
```

**Speedup:** 3× para QKV section (5-8% del forward total).

**Archivo:** `slime_forward.rs:293-318`

---

## FASE 5: UMA READBACK ELIMINATION (DÍA 5)

**Problema:** `sync_and_readback_all` en `ash_qat_dispatcher.rs:247-271` hace memcpy de mapped_ptr a buffers temporales. En UMA (Iris Xe), mapped_ptr ya es la misma RAM que usa la CPU.

**Fix:** Pasar los mapped pointers directamente al `MudSave` path:
```rust
// Antes: buf.read_f32(&mut dest)  — memcpy innecesario
// Después: dest = &buf.mapped_ptr  — zero-copy, mismo physical memory
```

Requiere que el fence check garantice que GPU terminó de escribir. Usar `vkWaitForFences` con timeout finito en lugar de `device_wait_idle`.

**Archivos:** `ash_qat_dispatcher.rs:247-271`, `ash_backend.rs:770`

---

## FASE 6: PCORE POOL DURING GPU OPTIMIZER (DÍA 5-6)

**Problema:** Mientras GPU ejecuta el optimizer, los 8 threads del PCorePool están idle.

**Fix:** Después de `step_async()`, usar PCorePool para:
1. Tokenizar el siguiente chunk (actualmente en E-core, mover a PCorePool)
2. Limpiar workspaces para el siguiente forward
3. Cargar embeddings

Esto es posible porque `step_async()` retorna inmediatamente — la GPU procesa en paralelo con CPU.

**Flujo resultante:**
```
GPU:  [Optimizer N] → → → → → → → → [Optimizer N+1]
CPU:  [Forward N] → [Backward N] → [step_async N] → [clean/prepare N+1] → [Forward N+1]
                                                      ^^^^ 8 threads en uso ^^^^
```

**Archivos:** `corpus_trainer.rs:2657-2753`

---

## MAPA DE TIEMPOS ESTIMADO (hidden=2560, modelo pequeño)

| Operación | CPU AVX2 8T | GPU Async | Overlap posible |
|-----------|------------|-----------|-----------------|
| RMSNorm (por capa) | 30 µs | 5 µs | ❌ Serial dentro de capa |
| Q GEMV (576×576) | 50 µs | 25 µs | ✅ Con K+V (3 paralelo) |
| K GEMV (192×576) | 18 µs | 10 µs | ✅ Con Q+V |
| V GEMV (192×576) | 18 µs | 10 µs | ✅ Con Q+K |
| Attention (20 heads) | 120 µs | 12 µs | ❌ Serial post-QKV |
| O GEMV (576×576) | 50 µs | 25 µs | ❌ Serial post-Attn |
| FFN Up (1536×576) | 75 µs | 30 µs | ✅ Con Gate |
| FFN Gate (1536×576) | 75 µs | 30 µs | ✅ Con Up |
| SiLU | 10 µs | 3 µs | ❌ Serial post-Up+Gate |
| FFN Down (576×1536) | 100 µs | 35 µs | ❌ Serial |
| **Forward total** | **~2.5 ms** | **~1 ms** | — |
| **Backward total** | **~3 ms** | — | — |
| **Optimizer (7 mats)** | ~5 ms CPU | ~2 ms GPU | ✅ Overlap con próximo forward |

**Con pipeline overlap completo: throughput estimado: 2.5× respecto a CPU-only.**

---

## ORDEN DE EJECUCIÓN

```
Día 1: Fase 1 (Double-Buffer Overlap) — mayor impacto, arquitectura pura
Día 2: Fase 2 (Desplegar shaders mha + rms_norm) — shaders ya existen, solo wiring
Día 3-4: Fase 3 (Shared memory tiling en GEMV shader) — necesario para GPU GEMV útil
Día 4: Fase 4 (Parallel QKV) — 3 líneas de Rust, bajo esfuerzo
Día 5: Fase 5 (UMA readback elimination) + Fase 6 (PCorePool + GPU overlap)
```

---

## RIESGOS

| Riesgo | Probabilidad | Mitigación |
|--------|-------------|------------|
| GPU GEMV sin shared memory es más lento que AVX2 | Alta | No usar GPU GEMV hasta Fase 3. Usar solo attention + rms_norm en GPU |
| `device_wait_idle` reemplazado por fence poll puede tener race conditions | Media | Test estricto con checksum de pesos después de cada epoch |
| UMA readback sin copia puede exponer datos a medio escribir | Media | Fence check ANTES de leer en forward. Usar `vkWaitForFences` con timeout |
| PCorePool + GPU compiten por ancho de banda de memoria | Baja | Intel Iris Xe UMA comparte RAM con CPU. En P-cores, esto puede causar contención. Monitorear con `perf stat` |
