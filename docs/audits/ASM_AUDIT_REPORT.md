# INFORME DE AUDITORÍA ASM — Forge LLM

**Fecha:** 2026-07-15
**Total archivos:** 17 `.s` (3,627 LOC) + `mod.rs` (176 LOC FFI) + `tests.rs` (550 LOC) + `build.rs` (131 LOC)

---

## 1. TABLA RESUMEN

| Archivo | LOC | Exports | YMM usados | Prefetch | vzeroupper | Unroll | Syntax | ¿Testeado? |
|---------|-----|---------|-----------|----------|------------|--------|--------|-----------|
| `ternary_gemv.s` | 283 | 1 | 16/16 | 2 | ✅ | 8×8=64 | AT&T | ✅ |
| `ternary_gemv_4rows.s` | 150 | 1 | 14/16 | **5** (best) | ✅ | 8×4=32 | AT&T | ✅ |
| `ternary_gemm_batch4.s` | 217 | 1 | 14/16 | **0** | ✅ | 16×4=64 | AT&T | ✅ |
| `ternary_backward.s` | 355 | 1 | many | 0 | ✅ | GCC-gen | AT&T | ❌ |
| `ternary_lut.s` | 102 | 1 | 6/16 | 0 | ✅ | 32 | AT&T | ❌ |
| `ternary_pext.s` | 69 | 1 | 4/16 | 0 | ✅ | N/A | AT&T | ❌ |
| `elut_gemv.s` | 127 | 1 | 8/16 | 0 | **❌** | 32 | AT&T | ❌ |
| `q4_0_gemv.s` | 102 | 1 | 8/16 | 0 | ✅ | 32 | AT&T | ✅ |
| `adam_step.s` | 162 | 1 | 8/16 | 0 | ✅ | 8 | **Intel** | ❌ |
| `silu.s` | 153 | 1 | 8/16 | 0 | ✅ | 8 | AT&T | ❌ |
| `rmsnorm.s` | 57 | 1 | 3/16 | 0 | ✅ | 8 | AT&T | ✅ |
| `slime_rmsnorm.s` | 255 | 1 | 14/16 | 0 | ✅ | 8 | AT&T | ❌ |
| `math.s` | 278 | **5** | 5/16 | **4** | ✅(×5) | 16 | AT&T | Parcial |
| `sgemm.s` | 246 | 2 | 5/16 | 0 | ✅(×2) | 16 | AT&T | Parcial |
| `mamba.s` | 193 | 2 | 12/16 | 0 | ✅(×2) | 8 | AT&T | ❌ |
| `rope.s` | 64 | 2 | 8/16 | 0 | ✅(×2) | 8 | AT&T | ❌ |
| `lm_head.s` | 89 | 1 | 4/16 | 0 | ✅ | N/A | AT&T | ❌ |
| **Missing** | — | — | — | — | — | — | — | — |
| `qat_step.s` | **NO EXISTE** | — | — | — | — | — | — | — |

---

## 2. FFI PERDIDAS — ASM COMPILADO PERO NO LLAMABLE DESDE RUST 🔴

Estas funciones ASM existen en `.s` pero **no tienen declaración `extern "C"` en `mod.rs`**, por lo que el motor las ejecuta como fallback en Rust (100× más lento):

| Función | Archivo | Impacto |
|---------|---------|---------|
| **`lm_head_avx2`** | `lm_head.s` | **CRÍTICO** — LM head sobre 128k vocab corre en Rust |
| **`adam_step_avx2`** | `adam_step.s` | **ALTO** — Optimizer Adam corre en Rust |
| **`slime_rmsnorm_i8_avx2`** | `slime_rmsnorm.s` | **ALTO** — RMSNorm del forward pass corre en Rust |
| `elut_gemv_avx2` | `elut_gemv.s` | Medio |
| `peak_abs_avx2` | `math.s` | Bajo |
| `apply_gradient_avx2` | `math.s` | Bajo |
| `hadamard_transform_avx2` | `math.s` | Bajo |
| `apply_rope_interleaved_asm` | `rope.s` | Bajo (stub) |
| `sgemm_abt_avx2` | `sgemm.s` | Bajo (fallback Rust) |
| `sgemm_avx2` | `sgemm.s` | Bajo (fallback Rust) |

---

## 3. HALLAZGOS CRÍTICOS (P0)

### P0-1: `lm_head.s` — Arquitectura Incorrecta
**Archivo:** `src/asm/lm_head.s`
**Problema:** Llama a `dot_product_avx2` **128,256 veces** por forward pass (una por token del vocabulario). Cada llamada:
- Crea stack frame (`sub $24, %rsp`)
- Salva/restaura `xmm1` en stack
- Ejecuta `call` con ~8 ciclos de overhead
- El stride `hidden*4` se recalcula cada iteración (`imul` + `shl`)
- **Cero prefetch** para un flujo de ~1.3GB de pesos

**Fix:** Fusionar dot product en el bucle: hoistear stride, prefetch N filas adelante, eliminar call overhead.

### P0-2: `elut_gemv.s` — API Ambiguo
**Archivo:** `src/asm/elut_gemv.s` líneas 116-124
**Problema:** El autor documentó su confusión en comentarios:
```
# The mandate: "Accumulate into i16". 
# But wait, earlier I said it's a dot product of ONE row.
# If the user passes *mut i16, we write a single i16 scalar?
```
Escribe un solo i16 en `(%rdx)`. Si el caller espera un array, corrompe memoria.

### P0-3: `ternary_lut.s` — Semántica Add-To-Output
**Archivo:** `src/asm/ternary_lut.s` línea 97
**Problema:** `vaddss (%rcx), %xmm0, %xmm0` **lee `*out` y suma** el dot product. Los otros GEMV **escriben** el resultado. Esta inconsistencia causará doble conteo si se usa esta función.

### P0-4: `elut_gemv.s` — Sin vzeroupper
**Archivo:** `src/asm/elut_gemv.s` líneas 126-127
**Problema:** Termina con `pop %rbp; ret` sin `vzeroupper`. Causa penalidad de transición AVX-SSE en el caller.

### P0-5: `qat_step.s` — No Existe
**Archivo:** Referenciado en `build.rs:42` y `AGENTS.md` pero no está en disco. Engañoso.

---

## 4. OPORTUNIDADES DE OPTIMIZACIÓN POR ARCHIVO

### 4.1 `ternary_gemv.s` (GEMV principal, 283 LOC)
| Línea | Problema | Fix |
|-------|----------|-----|
| 27-30 | Solo 4 accumuladores YMM (ymm0,9,13,14). 8 disponibles. | Usar ymm0-7 como accum, mover constantes a ymm8-15 → 2× menos iteraciones |
| 42 | `prefetchnta 32(%rdx)` — distancia de solo 1 u32 | Cambiar a ≥512 bytes (prefetchnta 512(%rdx)) |
| 269-278 | Reducción: `vextractf128 + vaddps + 2× vshufps` (5 instr) | Usar `vhaddps` (2 instr): `vextractf128$1, %ymm0, %xmm1; vhaddps %xmm1, %xmm0, %xmm0; vhaddps %xmm0, %xmm0, %xmm0` |
| — | Sin NaN guard | Añadir `vminps`/`vmaxps` en output |

### 4.2 `ternary_gemv_4rows.s` (GEMV 4 filas, 150 LOC)
| Línea | Problema | Fix |
|-------|----------|-----|
| 19-22 | `push %r12-%r15` — salva 4 registros, solo necesita 2-3 | Reducir pushes |
| 102-103 | `sub $8, %r9; jnz .loop` — **sin leftover handler**. Si hidden_size no es múltiplo de 8, procesa datos incorrectos | Añadir bucle scalar para remainder |
| — | Excelente: 5 prefetches, shared x load para 4 filas | ✅ Mantener |

### 4.3 `ternary_gemm_batch4.s` (GEMM batch-4, 217 LOC)
| Línea | Problema | Fix |
|-------|----------|-----|
| — | **Cero prefetch** en inner loop que procesa 4×2560×6912 elementos | Añadir `prefetchnta` para pesos, `prefetcht0` para activaciones |
| 58-120 | Inner loop sin prefetch → cache misses garantizados | Ver arriba |

### 4.4 `ternary_backward.s` (Backward GCC, 355 LOC)
| Línea | Problema | Fix |
|-------|----------|-----|
| 179-282 | Unpacking nibble a nibble escalar (`movl`, `shrl`, `andl`) en vez de vectorizado | Usar `vpsrlvd` + `vpand` como en forward pass |
| — | 44 branches spaghetti de GCC | Reescribir a mano siguiendo patrón de ternary_gemv.s |
| — | NaN guards presentes ✅ | Mantener |

### 4.5 `adam_step.s` (Optimizer Adam, 162 LOC)
| Línea | Problema | Fix |
|-------|----------|-----|
| — | **Sin prefetch**: Lee 4 arrays simultáneos (w,m,v,grads) sin prefetch | Añadir `prefetcht0 256(%r8,%r12,4)` para grads y similar para w/m/v |
| 62-63 | Boundary check costoso con `lea rbx, [r12+8]; cmp rbx, rdi` | Simplificar: `cmp rdi, 8; jl .tail` al inicio |
| 30 | **Intel syntax** — único archivo en Intel, todos los demás AT&T | Convertir a AT&T para consistencia |

### 4.6 `silu.s` (SiLU activation, 153 LOC)
| Línea | Problema | Fix |
|-------|----------|-----|
| 73 | `vdivps` (división, 13 ciclos latencia) para sigmoid | Usar `vrcpps` + Newton-Raphson (2× más rápido) |
| — | **Sin NaN guard**: Si entrada es NaN, polinomio propaga basura | Añadir `vminps`/`vmaxps` clamp |

### 4.7 `rmsnorm.s` (RMSNorm simple, 57 LOC)
| Línea | Problema | Fix |
|-------|----------|-----|
| 17 | `shr $3, %rax` — solo procesa múltiplos de 8, resto ignorado | Añadir bucle scalar leftover |
| — | Sin NaN guard en `eps` | Verificar `eps > 0` |

### 4.8 `slime_rmsnorm.s` (RMSNorm SlimeRegister 3-pase, 255 LOC)
| Línea | Problema | Fix |
|-------|----------|-----|
| 46-76 | Comentarios **incorrectos** sobre `vshufps $0x88` (documentan resultado erróneo, el código es correcto) | Corregir o eliminar comentarios |
| — | **3 pases** sobre mismos datos: sum_sq → peak → quantize | Fusionar pase 1+2 (sum_sq + peak simultáneos en ymm0-3 y ymm4-7) |
| — | **Sin leftover handler** para hidden%8 != 0 | Añadir cleanup scalar |
| 159 | `vmaxss` floor clamp ✅ | Mantener |

### 4.9 `math.s` (5 utilidades, 278 LOC)
| Línea | Problema | Fix |
|-------|----------|-----|
| 209-210 | `hadamard_transform_avx2` usa `%rbx` **sin salvar** (callee-saved register) | Añadir `push %rbx`/`pop %rbx` alrededor |
| 14-15 | `dot_product` tiene prefetch ✅ (2 prefetcht0 a 128B) | Mantener |
| 27-34 | Leftover handler de 8 elementos — si n%8 !=0, resto ignorado | Añadir cleanup |
| 145-146 | `apply_gradient` tiene NaN guard ✅ | Mantener |

### 4.10 `sgemm.s` (SGEMM, 246 LOC)
| Línea | Problema | Fix |
|-------|----------|-----|
| — | **Cero prefetch** en inner loops | Añadir `prefetchnta` para matriz B |
| 140-161 | Zero-initialization overhead en C antes de acumular | Acumular en registros locales |

### 4.11 `mamba.s` (SSM scan, 193 LOC)
| Línea | Problema | Fix |
|-------|----------|-----|
| — | **Cero prefetch**: 6 streams simultáneos (x, state, a_bar, b_bar, c, out) | Añadir `prefetcht0` a 512B lookahead |
| 92-93, 99-100 | Usa `vhaddps` ✅ — mejor que `vshufps` | Mantener |

### 4.12 `rope.s` (RoPE, 64 LOC)
| Línea | Problema | Fix |
|-------|----------|-----|
| — | `apply_rope_interleaved_asm` es un **stub vacío** (solo ret) | Implementar o eliminar (P-08) |
| — | Sin prefetch (roce solo 4 arrays pequeños) | Baja prioridad |

### 4.13 `lm_head.s` (LM head, 89 LOC)
| Línea | Problema | Fix |
|-------|----------|-----|
| 48-59 | Stack save/restore de `xmm1` **dentro del bucle** (128k veces) | Mover fuera del bucle |
| 42 | `imul %r13, %rax` stride recalcula cada iteración | Hoistear `stride = hidden*4` fuera del bucle |
| — | **Cero prefetch** para ~1.3GB de pesos | Añadir prefetch N rows ahead |
| — | **Sin FFI declaration** en mod.rs → no se puede llamar | Añadir `extern "C"` |

---

## 5. PATRONES DE REDUCCIÓN HORIZONTAL

| Archivo | Patrón | Instrs | Evaluación |
|---------|--------|--------|------------|
| `math.s`, `ternary_gemv.s`, `q4_0_gemv.s`, `rmsnorm.s`, `slime_rmsnorm.s`, `sgemm.s` | `vextractf128 + vaddps + 2× vshufps` | 5 | ✅ Funciona, ❌ Verboso |
| `mamba.s` | `vextractf128 + 2× vhaddps` | 4 | ✅ Mejor, 20% menos instrs |

**Recomendación:** Unificar todos al patrón `vhaddps`.

---

## 6. ANÁLISIS DE NaN/INF

| Archivo | NaN Guard | Detalle |
|---------|-----------|---------|
| `ternary_backward.s` | ✅ | 9× `vcomiss` contra thresholds |
| `math.s:apply_gradient` | ✅ | `vmaxps`/`vminps` clamp a [-1,1] |
| `ternary_gemv.s` | ❌ | Sin protección |
| `ternary_gemv_4rows.s` | ❌ | Sin protección |
| `ternary_gemm_batch4.s` | ❌ | Sin protección |
| `adam_step.s` | ❌ | Sin protección en grads |
| `silu.s` | ❌ | Sin clamp en entrada |
| Todos los demás | ❌ | Sin protección |

---

## 7. COBERTURA DE TESTS

### Testeado ✅ (6 funciones)
- `ternary_gemv_avx2` — 3 tests
- `ternary_gemv_4rows_avx2` — 1 test
- `ternary_gemm_batch4_avx2` — 2 tests
- `q4_0_gemv_asm` — 1 test
- `rms_norm_scale_asm` — 2 tests
- `sum_squares_avx2` — 2 tests
- `dot_product_avx2` — 1 test
- `sgemm_abt_avx2` — 1 test

### Ignored 🟡 (2 funciones)
- `hadamard_transform_avx2` — `#[ignore]`
- `sgemm_avx2` — `#[ignore]`

### Sin test ❌ (13 funciones — ALTO RIESGO)
- `adam_step_avx2`, `silu_vectorial_avx2`, `apply_rope_asm`, `mamba_scan_avx2`,
  `mamba_delta_fold_avx2`, `lm_head_avx2`, `slime_rmsnorm_i8_avx2`,
  `ternary_gemv_lut_avx2`, `elut_gemv_avx2`, `pext_unpack_ternary`,
  `ternary_gemv_backward_avx2`, `peak_abs_avx2`, `apply_gradient_avx2`

---

## 8. COMPARATIVA SINTÁCTICA

| Syntax | Archivos |
|--------|----------|
| **AT&T** (gas default) | 16 archivos |
| **Intel** (`.intel_syntax noprefix`) | `adam_step.s` |

**Recomendación:** Unificar a AT&T o Intel. El archivo Intel es el optimizer Adam — inconsistencia cognitiva.

---

## 9. REGISTRO DE CONSTANTES RODATA DUPLICADAS

Tres archivos tienen **idénticas** secciones `.rodata`:

| Constante | `ternary_gemv.s` | `ternary_gemv_4rows.s` | `ternary_gemm_batch4.s` |
|-----------|:---:|:---:|:---:|
| `SHIFTS_ELUT` | ✅ | ✅ | ✅ |
| `MASK_ELUT` | ✅ | ✅ | ✅ |
| `VAL_ONE` | ✅ | ✅ | ✅ |
| `VAL_MINUS_ONE` | ✅ | ✅ | ✅ |

**128 bytes × 3 = 384 bytes** de binario desperdiciados. Crear `.include` compartido.

---

## 10. MAPA DE REGISTROS (YMM) POR ARCHIVO

| Archivo | YMM usados | YMM libres | Presión |
|---------|-----------|------------|---------|
| `ternary_gemv.s` | 14/16 | 2 | Media |
| `ternary_gemv_4rows.s` | 14/16 | 2 | Media |
| `ternary_gemm_batch4.s` | 14/16 | 2 | Media |
| `adam_step.s` | **16/16** | **0** | 🔴 **Máxima** |
| `slime_rmsnorm.s` | 14/16 | 2 | Media |
| `silu.s` | 8/16 | 8 | Baja |
| `mamba.s` | 12/16 | 4 | Baja |
| `lm_head.s` | 4/16 | 12 | Muy baja |
| `sgemm.s` | 5/16 | 11 | Baja |
| `math.s` | 5/16 | 11 | Baja |

---

## 11. PLAN DE ACCIÓN PRIORIZADO

### 🔴 DÍA 1: Arreglos Críticos
1. **Añadir FFI declarations** en `mod.rs` para `lm_head_avx2`, `adam_step_avx2`, `slime_rmsnorm_i8_avx2`
2. **Reescribir `lm_head.s`**: Fusionar dot product en bucle, hoistear stride, añadir prefetch
3. **Eliminar/quemar `elut_gemv.s` y `ternary_pext.s`** (P-08)

### 🟠 DÍA 2: Optimizaciones Alto Impacto
4. **Expandir accumuladores** en `ternary_gemv.s` (4→8 YMM)
5. **Añadir prefetch** a `ternary_gemm_batch4.s`, `adam_step.s`, `sgemm.s`
6. **Fusionar pases 1+2** en `slime_rmsnorm.s`

### 🟡 DÍA 3: Correctitud y Tests
7. **Añadir leftover handlers** para dimensiones no-múltiplo-de-8
8. **Añadir NaN guards** en todos los kernels sin protección
9. **Corregir `%rbx` en `hadamard_transform_avx2`** y habilitar test
10. **Escribir tests** para las 13 funciones no testeadas

### 🟢 DÍA 4: Pulido
11. **Unificar reducciones** a patrón `vhaddps`
12. **Convertir `adam_step.s`** a AT&T syntax
13. **Deducir `.rodata` duplicada** con `.include` compartido
14. **Corregir comentarios** en `slime_rmsnorm.s`
15. **Añadir `strip = "symbols"`** al perfil release

---

## TOP 3 INMEDIATOS

1. **FFI declarations faltantes** — `lm_head_avx2`, `adam_step_avx2`, `slime_rmsnorm_i8_avx2` son código muerto sin ellas (5-10× speedup al activarlas)
2. **`lm_head.s` reescritura** — 128k llamadas a función por forward pass es el bottleneck #1 del proyecto
3. **Prefetch en `ternary_gemm_batch4.s`** — 70M elementos sin prefetch = guaranteed cache misses
