# PLAN DE CORRECCIÓN ASM — Forge LLM

**Base:** Auditorías `docs/audits/ASM_AUDIT_REPORT.md` + `docs/audits/AUDIT_REPORT.md` + `docs/STATUS_REPORT.md`
**Principios:** P-00 (raw pointers), P-01 (zero alloc), P-06 (clippy clean), P-07 (Rust-only), P-08 (dead code delete), P-13 (no hardcoded), P-17 (fail-fast null), P-27 (no Rayon)

---

## FASE 1: CONECTAR ASM EXISTENTE (DÍA 1)
*Objetivo: Activar kernels ya compilados pero no llamables desde Rust.*

### 1.1 Añadir FFI declarations en `src/asm/mod.rs`

Las siguientes funciones ASM existen en `.s` pero no tienen `extern "C"` en `mod.rs`:

| Función ASM | Archivo `.s` | Firma |
|------------|-------------|-------|
| `lm_head_avx2` | `lm_head.s` | `(vocab_size: usize, hidden: usize, regs: *const f32, weights: *const u32) -> usize` |
| `adam_step_avx2` | `adam_step.s` | `(n: usize, w: *mut f32, m: *mut f32, v: *mut f32, grads: *const f32, clip_coef: f32, wd: f32, b1: f32, b2: f32, lr_bc1: f32, inv_bc2: f32, eps: f32)` |
| `slime_rmsnorm_i8_avx2` | `slime_rmsnorm.s` | `(regs: *const SlimeRegister, weights: *const f32, out_i8: *mut i8, hidden: usize, eps: f32)` |

**Archivo a modificar:** `src/asm/mod.rs`
**Verificación:** `cargo build --release` + `nm target/release/forge_llm | grep lm_head_avx2`

### 1.2 Conectar `lm_head_avx2` en `main.rs`

Buscar en `main.rs` el loop de argmax del LM head (líneas ~350-400) donde actualmente se calcula `dot_product` en Rust. Reemplazar con llamada a `lm_head_avx2`.

**Archivos a modificar:** `main.rs`, `src/asm/mod.rs`
**Verificación:** `cargo test` + inferencia manual con `--release`

### 1.3 Conectar `slime_rmsnorm_i8_avx2` en `slime_forward.rs`

Buscar en `slime_forward.rs` dónde se aplica RMSNorm + i8 quantization actualmente en Rust. Reemplazar con llamada a `slime_rmsnorm_i8_avx2`.

**Archivos a modificar:** `slime_forward.rs`, `src/asm/mod.rs`
**Verificación:** `cargo test`

### 1.4 Conectar `adam_step_avx2` en entrenamiento

Buscar en `corpus_trainer.rs` `apply_optimizer_cpu_step_and_pack`. Actualmente usa `sgd_step_avx2` de `forge_autograd`. Opción: añadir ruta condicional que use `adam_step_avx2` cuando `OptimizerStrategy::Adam`.

**Archivos a modificar:** `corpus_trainer.rs`, `src/asm/mod.rs`
**Verificación:** `cargo build --release`

---

## FASE 2: REPARAR ASM ROTO (DÍA 2)
*Objetivo: Corregir bugs de funcionamiento y seguridad en ASM existente.*

### 2.1 Corregir `elut_gemv.s` — Sin `vzeroupper`

**Archivo:** `src/asm/elut_gemv.s:126-127`
**Problema:** Termina con `pop %rbp; ret` sin `vzeroupper`.
**Fix:** Insertar `vzeroupper` antes de `pop %rbp`, línea 125.

### 2.2 Corregir `elut_gemv.s` — API ambiguo

**Archivo:** `src/asm/elut_gemv.s:116-127`
**Problema:** Los comentarios (líneas 116-124) documentan confusión del autor sobre si escribe i16 scalar o array. La instrucción actual escribe un solo i16 en `(%rdx)`.
**Fix:** Decidir API correcta basado en callers reales. Si no hay callers, marcar como dead y eliminar en Fase 5.

### 2.3 Corregir `ternary_lut.s` — Semántica add-to-output

**Archivo:** `src/asm/ternary_lut.s:97`
**Problema:** `vaddss (%rcx), %xmm0, %xmm0` suma `*out` al resultado en vez de escribirlo. Inconsistente con `ternary_gemv.s` que escribe.
**Fix:** Cambiar a `vmovss %xmm0, (%rcx)` (write semantics). O, si add-to-output es intencional, documentarlo y mantener consistencia.

### 2.4 Corregir `hadamard_transform_avx2` — `%rbx` sin salvar

**Archivo:** `src/asm/math.s:209-210`
**Problema:** Usa `%rbx` (callee-saved) sin `push`/`pop`.
**Fix:** Añadir `push %rbx` antes de línea 199 y `pop %rbx` después de línea 258.

---

## FASE 3: OPTIMIZACIONES ASM CONFIRMADAS (DÍA 3-4)
*Objetivo: Optimizaciones de las que tenemos certeza por el audit.*

### 3.1 Prefetch en kernels sin él

Añadir `prefetcht0`/`prefetchnta` en los siguientes archivos que actualmente tienen **cero prefetch**:

| Archivo | Dónde añadir | Distancia sugerida |
|---------|-------------|-------------------|
| `ternary_gemm_batch4.s` | Inner loop (entrada) | `prefetchnta 256(%rdx)` (pesos), `prefetcht0 256(%rsi)` (x) |
| `adam_step.s` | `.Ladam_loop8` entrada | `prefetcht0 256(%r8,%r12,4)` (grads), similar para w,m,v |
| `silu.s` | `.Lvec_loop` entrada | `prefetcht0 256(%rsi)`, `prefetcht0 256(%rdx)` |
| `sgemm.s` | Inner loops | `prefetchnta` para B, `prefetcht0` para A |
| `q4_0_gemv.s` | Entrada de cada bloque A/B/C/D | `prefetcht0` para x y qs |
| `rope.s` | Loop principal | `prefetcht0 128(%rsi/%rdx/%rcx)` |
| `mamba.s` | Loop principal | `prefetcht0` para x, state, a_bar, b_bar |

**Archivos a modificar:** 7 archivos `.s`
**Verificación:** `cargo test` + benchmark comparativo

### 3.2 Widen prefetch distance en `ternary_gemv.s`

**Archivo:** `src/asm/ternary_gemv.s:42`
**Problema:** `prefetchnta 32(%rdx)` distancia de solo 1 u32 (~4 bytes).
**Fix:** Cambiar a `prefetchnta 512(%rdx)`.

### 3.3 Añadir leftover handlers

Los siguientes kernels no procesan correctamente dimensiones no-múltiplo-de-8:

| Archivo | Loop | Fix |
|---------|------|-----|
| `ternary_gemv_4rows.s:102-103` | `sub $8, %r9; jnz .loop` | Añadir sección scalar cleanup después del loop |
| `rmsnorm.s:17` | `shr $3, %rax` | Añadir loop scalar para remainder |
| `slime_rmsnorm.s` | Procesa en bloques de 8 regs | Añadir cleanup para hidden%8 != 0 |

**Archivos a modificar:** 3 archivos `.s`
**Verificación:** Tests con dimensiones non-multiple-of-8 (p.ej. hidden=7, 15, 23)

---

## FASE 4: UNIFICAR PATRONES (DÍA 5)
*Objetivo: Consistencia y mantenibilidad.*

### 4.1 Unificar reducciones horizontales a `vhaddps`

Reemplazar el patrón `vextractf128 + vaddps + 2× vshufps` con `vextractf128 + 2× vhaddps` en:

| Archivo | Líneas del patrón actual |
|---------|-------------------------|
| `math.s` | Reducción en dot_product, sum_squares |
| `ternary_gemv.s` | 269-278 |
| `q4_0_gemv.s` | 86-97 |
| `rmsnorm.s` | 45-52 |
| `slime_rmsnorm.s` | Reducciones en pases 1,2,3 |
| `sgemm.s` | Reducciones en inner loops |

**Archivos a modificar:** 6 archivos `.s`
**Verificación:** `cargo test` (comparación contra referencia scalar)

### 4.2 Convertir `adam_step.s` a AT&T syntax

**Archivo:** `src/asm/adam_step.s:30`
**Problema:** Único archivo en Intel syntax (`.intel_syntax noprefix`).
**Fix:** Convertir a AT&T syntax estándar del proyecto.

**Verificación:** `cargo build --release` + test de Adam contra referencia

### 4.3 Deducir `.rodata` duplicada

**Archivos:** `ternary_gemv.s`, `ternary_gemv_4rows.s`, `ternary_gemm_batch4.s`
**Problema:** Las mismas constantes `SHIFTS_ELUT`, `MASK_ELUT`, `VAL_ONE`, `VAL_MINUS_ONE` definidas 3 veces (128 bytes × 3 = 384 bytes).
**Fix:** Crear archivo `src/asm/elut_constants.inc` con las definiciones y usar `.include` en los 3 archivos.

---

## FASE 5: ELIMINAR CÓDIGO MUERTO (DÍA 6)
*Objetivo: P-08 compliance.*

### 5.1 Eliminar ASM huérfano sin callers

| Archivo | Razón | Acción |
|---------|-------|--------|
| `ternary_lut.s` | Sin callers en Rust | Mover a `src/asm/legacy/` o eliminar |
| `ternary_pext.s` | Deprecated (2-bit PEXT, superceded por ELUT 4-bit) | Eliminar |
| `elut_gemv.s` | Sin callers + API confuso + sin vzeroupper | Eliminar |
| `lm_head.s` | Mantener — ver Fase 1.2 para conectar | No eliminar |
| `qat_step.s` | **No existe en disco.** Quitar referencia de `build.rs` | Quitar línea `cargo:rerun-if-changed=src/asm/qat_step.s` |

### 5.2 Eliminar función dummy en `asm/mod.rs`

**Archivo:** `src/asm/mod.rs:147-161`
**Problema:** `ternary_gemv_backward_avx2` con body "I don't care... just make it compile".
**Fix:** Si no se usa, eliminar la función. Si se va a implementar, reemplazar con la llamada ASM real.

### 5.3 Eliminar `sgemm_abt` Rust fallback si no se usa

**Archivo:** `src/asm/mod.rs:165-176`
**Problema:** Fallback Rust triply-nested loop para sgemm_abt. Si el ASM `sgemm_abt_avx2` está linkeado y funcionando, este fallback es dead code.
**Fix:** Verificar si `sgemm_abt_avx2` es llamado desde Rust. Si sí tiene callers, eliminar el fallback. Si no, eliminarlo también al no tener callers.

---

## FASE 6: NaN GUARDS (DÍA 7)
*Objetivo: Robustez numérica en entrenamiento.*

### 6.1 Añadir NaN guards en kernels sin protección

| Archivo | Dónde añadir |
|---------|-------------|
| `ternary_gemv.s` | Antes de escribir `out`: `vcmpps` + `vandps` para detectar NaN en accumuladores |
| `ternary_gemv_4rows.s` | Antes de escribir cada row output |
| `adam_step.s` | `vcmpps` en grads antes de usar (NaN → 0.0) |
| `silu.s` | `vmaxps`/`vminps` clamp en entrada x ([-50, 50]) |
| `ternary_gemm_batch4.s` | En output de cada row |

**Patrón a usar** (consistente con `ternary_backward.s`):
```asm
vcmpps $0x3, %ymm0, %ymm0, %ymm1  # cmpneq — detect NaN
vblendvps %ymm1, %ymm2, %ymm0, %ymm0  # replace NaN with 0 (ymm2=zero)
```

---

## FASE 7: DOCUMENTACIÓN Y AGENTS.MD (DÍA 8)
*Objetivo: Sincronizar documentación con realidad.*

### 7.1 Actualizar AGENTS.md

- Eliminar referencia a `qat_step.s` (no existe)
- Eliminar referencia a `en.txt` (no existe)
- Corregir item 36: `select_optimizer()` solo almacena opciones, no las ejecuta
- Marcar Muon/GaLore como disconnected hasta que se conecte el dispatch

### 7.2 Añadir cabeceras a cada `.s` file

Cada archivo ASM debe tener un bloque de comentario al inicio:
```asm
# Purpose: <qué hace>
# Signature: <parámetros y calling convention>
# Callers: <dónde se llama desde Rust>
# Clobbers: <registros que modifica>
```

---

## TABLA DE ARCHIVOS AFECTADOS POR FASE

| Archivo | F1 | F2 | F3 | F4 | F5 | F6 | F7 |
|----------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `src/asm/mod.rs` | ✅ | — | — | — | ✅ | — | — |
| `main.rs` | ✅ | — | — | — | — | — | — |
| `slime_forward.rs` | ✅ | — | — | — | — | — | — |
| `corpus_trainer.rs` | ✅ | — | — | — | — | — | — |
| `elut_gemv.s` | — | ✅ | — | — | ✅ | — | ✅ |
| `ternary_lut.s` | — | ✅ | — | — | ✅ | — | ✅ |
| `math.s` | — | ✅ | — | ✅ | — | — | ✅ |
| `ternary_gemm_batch4.s` | — | — | ✅ | — | — | ✅ | ✅ |
| `adam_step.s` | — | — | ✅ | ✅ | — | ✅ | ✅ |
| `silu.s` | — | — | ✅ | — | — | ✅ | ✅ |
| `sgemm.s` | — | — | ✅ | ✅ | — | — | ✅ |
| `q4_0_gemv.s` | — | — | ✅ | ✅ | — | — | ✅ |
| `rope.s` | — | — | ✅ | — | — | — | ✅ |
| `mamba.s` | — | — | ✅ | — | — | — | ✅ |
| `ternary_gemv.s` | — | — | ✅ | ✅ | — | ✅ | ✅ |
| `ternary_gemv_4rows.s` | — | — | ✅ | ✅ | — | ✅ | ✅ |
| `rmsnorm.s` | — | — | ✅ | ✅ | — | — | ✅ |
| `slime_rmsnorm.s` | — | — | ✅ | ✅ | — | — | ✅ |
| `ternary_pext.s` | — | — | — | — | ✅ | — | — |
| `ternary_backward.s` | — | — | — | — | — | — | ✅ |
| `build.rs` | — | — | — | — | ✅ | — | ✅ |
| `AGENTS.md` | — | — | — | — | — | — | ✅ |

---

## VERIFICACIÓN FINAL

Después de cada fase, ejecutar:

```bash
cargo clippy --all-targets        # P-06: 0 errors, 0 warnings
cargo test                        # 90+ tests must pass
cargo build --release             # Build exitoso
```

Para Fase 6 (NaN guards), añadir test que inyecte NaN y verifique que el output es finito.

---

## PRIORIDAD DE EJECUCIÓN

1. **Fase 1** (conectar ASM) — mayor impacto inmediato: activa 3 kernels que corren en Rust
2. **Fase 5** (eliminar dead code) — limpia 3-5 archivos, P-08 compliance
3. **Fase 2** (reparar bugs) — corrige bugs de funcionamiento
4. **Fase 3** (prefetch + leftovers) — optimizaciones de rendimiento
5. **Fase 6** (NaN guards) — robustez para entrenamiento
6. **Fase 4** (unificar patrones) — consistencia, mantenibilidad
7. **Fase 7** (documentación) — cierre del ciclo
