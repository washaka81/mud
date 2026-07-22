# INFORME DE AUDITORÍA — Forge LLM
**Fecha:** 2026-07-15
**Hash:** `ae6056e4`

---

## 1. OPORTUNIDADES DE OPTIMIZACIÓN ASM (ALTO IMPACTO)

| # | Componente | Problema | Speedup Estimado |
|---|-----------|----------|-----------------|
| P1 | **`lm_head.s`** (vocab 128k) | Llama a `dot_product_avx2` **128,256 veces** por forward pass. Cada llamada tiene call overhead + stack save/restore. | **1.5-3×** inferencia |
| P2 | **`ternary_gemv.s`** | Solo 4 acumuladores YMM (ymm0,9,13,14) de 16 disponibles. Podrían ser 8 → 64 elementos por iteración vs 32. | **1.3-1.5×** GEMV |
| P3 | **`slime_rmsnorm.s`** | 3 pases separados sobre mismos datos (sum_sq → peak → quantize). Fusión a 2 pases posible. | **1.2-1.4×** RMSNorm |
| P4 | **Rust intrinsics** (`avx_math.rs`) | 23 usos de `_mm256_loadu_ps` sin verificar alineación. El `AlignedBuffer` está alineado a 64B pero se usan loads no-alineados. | 5-10% ops vectoriales |
| P5 | **6 ASM files sin prefetch** | `adam_step.s`, `silu.s`, `sgemm.s`, `q4_0_gemv.s`, `rope.s`, `mamba.s` — cero prefetch. | 5-15% en cada op |
| P6 | **Prefetch distance** en `ternary_gemv.s:42` | Pesos prefetch a solo 32B (una línea de caché). Debería ser 256-512B. | 2-5% GEMV |

---

## 2. OPORTUNIDADES EN MANEJO DE PUNTEROS

| # | Ubicación | Problema | Riesgo |
|---|-----------|----------|--------|
| R1 | `slime_backward.rs:374` | `vec![0.0; hidden]` en hot loop — violación **P-01** | Alloc por token en backward |
| R2 | `pcore_pool.rs:113` | Spin-loop `while pending > 0` quema 100% CPU esperando | Desperdicio en sincronización |
| R3 | `slime_forward.rs:86-89` | Pointer → usize → pointer round-trip en closures | UB potencial |
| R4 | `workspace.rs:267-425` | `InferenceWorkspace` — 426 líneas de struct + métodos **nunca instanciados** | P-08 violation |
| R5 | `slime_backward.rs:147-161` | `ternary_gemv_backward_avx2` con body dummy | Mantenimiento engañoso |

---

## 3. MÉTRICAS DEL PROYECTO

| Métrica | Valor |
|---------|-------|
| LOC total Rust | ~24,450 (99 archivos) |
| Archivos ASM | 17 (5 huérfanos — compilados pero sin call sites) |
| Bloques `unsafe` | 251 |
| Funciones `unsafe fn` | 42 |
| Tests totales | 89 funciones, ~15-20% de cobertura estimada |
| Binarios en Cargo.toml | **27** (en un solo crate → build lento) |
| Dependencias | 31 direct + 1 workspace |

---

## 4. VIOLACIONES P-13 (HARDCODED DIMENSIONS) CRÍTICAS

| Archivo | Línea | Violación |
|---------|-------|-----------|
| `main.rs:335` | `max_gen = 256` hardcoded | Debe ser CLI arg o metadata |
| `pcore_pool.rs:121` | `PCorePool::new(8)` | Debe ser `hardware.preferred_threads` |
| `corpus_trainer.rs:472-473` | Fallback `30` layers / `4096` hidden | Debe propagar error si metadata falta |
| `workspace.rs:163` | `max_pos.min(8192)` | Cap arbitrario |
| `constants.rs` vs `workspace.rs` vs `slime_jepa.rs` | `EPSILON_FLOOR` triplicado | Usar import único |

---

## 5. CÓDIGO MUERTO (P-08 VIOLATIONS)

**Archivos ASM huérfanos** (compilados sin call sites):
- `src/asm/qat_step.s`, `ternary_lut.s`, `ternary_pext.s`, `elut_gemv.s`, `lm_head.s`
- ~600+ líneas ASM compiladas pero nunca ejecutadas

**Scratch/artefactos de merge:**
- `src/main.rs.orig`, `src/main.rs.rej`, `src/model/tokenizer.rs.orig`, `src/model/tokenizer.rs.rej`
- `scratch.rs`, `scratch2.rs`, `scratch3.rs`, `scratch4.rs`, `scratch_telemetry.rs`, `test_affinity.rs`

---

## 6. RIESGOS DE COBERTURA DE TESTS

| Archivo | LOC | Tests | Riesgo |
|---------|-----|-------|--------|
| `corpus_trainer.rs` | 2,949 | **0** | 🔴 Crítico |
| `slime_forward.rs` | 865 | **1** | 🟠 Alto |
| `slime_backward.rs` | 739 | **2** | 🟠 Alto |
| `workspace.rs` | 426 | **0** | 🟡 Medio |
| `self_play.rs` | 170 | **0** | 🟡 Medio |
| `pcore_pool.rs` | 121 | **0** | 🟢 Bajo |

---

## 7. ACCIONES RECOMENDADAS (POR PRIORIDAD)

| Prioridad | Acción | Archivos | Impacto |
|-----------|--------|----------|---------|
| 🔴 **CRÍTICA** | Fusionar `lm_head.s` en GEMM batch (eliminar 128k call overhead) | `lm_head.s`, `main.rs` | 1.5-3× inferencia |
| 🔴 **CRÍTICA** | Eliminar/migrar `InferenceWorkspace` (426 LOC dead) + 5 ASM orphans | `workspace.rs`, `build.rs` | P-08 compliance |
| 🔴 **CRÍTICA** | Pre-allocar buffers en backward pass (P-01) | `slime_backward.rs` | Zero alloc hot path |
| 🟠 **ALTA** | Expandir accumuladores YMM en `ternary_gemv.s` (4→8) | `ternary_gemv.s` | 1.3-1.5× GEMV |
| 🟠 **ALTA** | Añadir dispatch alineado en `avx_math.rs` (`load_ps` vs `loadu_ps`) | `avx_math.rs` | 5-10% vector ops |
| 🟠 **ALTA** | Parchear P-13: `max_gen`, `PCorePool(8)`, fallbacks corpus_trainer | `main.rs`, `pcore_pool.rs`, `corpus_trainer.rs` | P-13 fix |
| 🟡 **MEDIA** | Añadir prefetch a 6 ASM files + widening en ternary_gemv | Ver P5/P6 | 5-15% cada op |
| 🟡 **MEDIA** | Fuse `slime_rmsnorm.s` 3→2 pases | `slime_rmsnorm.s` | 1.2-1.4× RMSNorm |
| 🟡 **MEDIA** | Add yield-after-spin en `pcore_pool.rs` | `pcore_pool.rs` | Menos CPU waste |
| 🟢 **BAJA** | `strip = "symbols"` + `debug = 0` en release profile | `Cargo.toml` | -20-30% bin size |
| 🟢 **BAJA** | Unificar `EPSILON_FLOOR` en `constants.rs` | `workspace.rs`, `slime_jepa.rs` | Cleanup |
| 🟢 **BAJA** | Eliminar scratch/merge artifacts | ~10 archivos | Cleanup |

---

## TOP 3 INMEDIATOS

1. **Fusionar `lm_head.s`** — es el bottleneck más grande del proyecto (128k dot products seriales → GEMM batch)
2. **8 accumulators en `ternary_gemv.s`** — dobla el unroll del kernel más llamado con cambio de 4 registros
3. **Dispatch alineado en `avx_math.rs`** — +5-10% gratis en todas las ops vectoriales, ~1 hora de trabajo
