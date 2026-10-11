# PLAN DE OPTIMIZACIÓN ASM — CÓMPUTO EXTREMO
*Sin cambios arquitectónicos. Solo hot paths ASM + eliminación de deprecated.*

---

## FASE 1: ELIMINAR DEPRECATED (DÍA 1)

| Archivo | Razón | Acción |
|---------|-------|--------|
| `ternary_pext.s` | 2-bit PEXT, superceded por ELUT 4-bit, sin callers | **Eliminar** + quitar de `build.rs` |
| `elut_gemv.s` | API ambiguo (comentarios autor confundido), sin vzeroupper, sin callers | **Eliminar** + quitar de `build.rs` |
| `ternary_lut.s` | Semántica add-to-output inconsistente, sin callers | **Eliminar** + quitar de `build.rs` |
| `build.rs:42` | Referencia a `qat_step.s` que **no existe** en disco | **Quitar línea** |
| `asm/mod.rs:147-161` | `ternary_gemv_backward_avx2` dummy ("I don't care...") | **Eliminar función** |
| `asm/mod.rs:165-176` | `sgemm_abt` Rust fallback (triple loop puro) | **Eliminar** si `sgemm_abt_avx2` ASM tiene caller |

**Después:** `src/asm/` pasa de 17 → 14 archivos. `build.rs` compila 14 `.s`.

---

## FASE 2: HOT PATH #1 — `lm_head.s` REESCRITURA (DÍA 2-3)
*Mayor bottleneck de inferencia. Estado actual: 128,256 llamadas a `dot_product_avx2` por forward.*

**Problema:** Por cada token del vocabulario (128,256):
```
imul + shl + lea + sub $24,%rsp + vmovaps + mov*3 + call + vmovaps + add $24,%rsp + vcomiss + jae + mov
```
~10 instr + call overhead por token = ~1.28M instr. 128,256 × 2560 ciclos de dot product = **328M ciclos**.

**Fix:** Reescribir `lm_head.s` completo con:

1. **Dot product inlined** — el bucle carga pesos, calcula dot product on the fly, compara vs max. Sin `call`.
2. **Stride hoisteado** — `stride = hidden * 4` calculado una vez fuera del bucle, no por iteración.
3. **Prefetch** — `prefetchnta` N filas adelante para ~1.3GB de pesos.
4. **Registro del mejor candidato** — `%rax` = best_idx, `%xmm0` = max_val, actualizado con `vcomiss` + `cmov`.

**API final:** `lm_head_avx2(vocab_size, hidden, regs_ptr, weights_ptr) -> usize` (misma que ahora, solo que rápida).

**Añadir FFI en `mod.rs`** para que `main.rs` pueda llamarlo.

**Verificación:** Comparar output contra referencia scalar. Ambos deben dar exactamente el mismo argmax.

---

## FASE 3: HOT PATH #2 — `ternary_gemv.s` 8 ACUMULADORES (DÍA 4)
*El kernel más llamado del proyecto (30 capas × ~7 GEMVs por capa = 210+ llamadas por forward).*

**Problema actual:** 4 acumuladores YMM (0, 9, 13, 14) + 4 constantes (8, 10, 11, 12) + 1 escala (15) = 13 registros. Procesa 8 weights × 4 accs = **32 weights/iteración**.

**Fix:** Mover constantes a ymm8-15, usar ymm0-7 como 8 acumuladores. Procesa 8 weights × 8 accs = **64 weights/iteración**. 2× menos iteraciones, 2× menos overhead de loop.

| Antes | Después |
|-------|---------|
| ymm0: acc1 | ymm0: acc1 |
| ymm9: acc2 | ymm1: acc2 |
| ymm13: acc3 | ymm2: acc3 |
| ymm14: acc4 | ymm3: acc4 |
| (ymm1-7 temps) | ymm4: acc5 |
| ymm8: scale (const) | ymm5: acc6 |
| ymm10-12,15: const | ymm6: acc7 |
| | ymm7: acc8 |
| | ymm8: scale (const) |
| | ymm9-15: const |

**Ajustes necesarios:**
- Cambiar `vfmadd231ps` target registers (ymm0-3 → ymm0-7)
- Ajustar punteros de pesos y activaciones (avanzan 64×4=256 bytes vs 128)
- Ajustar leftover handlers (32-63 → 64-127, 16-31 → 32-63, 8-15 → 8-31)
- Ajustar reducción final (8 acc → sum, no 4)

**Verificación:** `cargo test` (test_ternary_gemv_avx2_vs_reference usa comparación contra scalar).

---

## FASE 4: PREFETCH EN KERNELS CRÍTICOS (DÍA 5)
*Costo: ~1 línea por kernel. Impacto: 5-15% en cada op.*

### 4.1 `ternary_gemm_batch4.s` — CRÍTICO (speculative drafter)
Procesa 4×2560×6912 = 70M elementos sin prefetch.

Añadir en entrada del inner loop:
```asm
prefetchnta 256(%rdx)     # pesos N filas adelante
prefetcht0 512(%rsi)      # activaciones L1
prefetcht0 256(%r11)      # token1 activations
prefetcht0 256(%r12)      # token2 activations
prefetcht0 256(%r13)      # token3 activations
```

### 4.2 `adam_step.s` — ALTO (optimizer)
Lee 4 arrays simultáneos (w, m, v, grads) sin prefetch.

Añadir en `.Ladam_loop8`:
```asm
prefetcht0 256(%r8, %r12, 4)   # grads
prefetcht0 256(%rdi, %r12, 4)  # w
prefetcht0 256(%rsi, %r12, 4)  # m
prefetcht0 256(%rdx, %r12, 4)  # v
```

### 4.3 `silu.s` — MEDIO (FFN activation)
```asm
prefetcht0 256(%rsi)      # src
prefetcht0 256(%rdx)      # dst
```

### 4.4 Widen `ternary_gemv.s` weight prefetch
```asm
prefetchnta 32(%rdx)   →   prefetchnta 512(%rdx)
```

---

## FASE 5: NaN GUARDS EN HOT PATHS (DÍA 6)
*Entrenamiento produce gradientes NaN. Sin protección → propagación silenciosa.*

| Kernel | Inserción | Código |
|--------|-----------|--------|
| `ternary_gemv.s` | Antes de escribir out | `vcmpps $0x3, %ymm0, %ymm0, %ymm1; vblendvps %ymm1, %ymm6, %ymm0, %ymm0` (ymm6=zero) |
| `ternary_gemv_4rows.s` | Antes de cada row output | Ídem |
| `adam_step.s` | Antes de usar grads | `vcmpps` en cada ymm de grad → blend con cero |
| `silu.s` | Entrada x | `vmaxps` clamp a [-50, 50] antes de polinomio |

**Patrón único** (consistente con `ternary_backward.s` que ya lo usa):
```asm
# NaN → 0.0
vxorps %ymm6, %ymm6, %ymm6        # zero
vcmpps $0x3, %ymm0, %ymm0, %ymm7  # cmpneq (detecta NaN)
vblendvps %ymm7, %ymm6, %ymm0, %ymm0  # NaN→0
```

---

## FASE 6: ELIMINAR CÓDIGO MUERTO RUST (DÍA 7)

| Archivo | Acción |
|---------|--------|
| `workspace.rs:188-425` | `InferenceWorkspace` — struct + 6 métodos nunca instanciados. **Eliminar.** |
| `scratch.rs`, `scratch2.rs`, `scratch3.rs`, `scratch4.rs` | **Eliminar** |
| `scratch`, `scratch2`, `scratch3`, `scratch4` (binarios) | **Eliminar** |
| `scratch_telemetry.rs` | **Eliminar** |
| `test_affinity.rs` | **Eliminar** |
| `src/main.rs.orig`, `src/main.rs.rej` | **Eliminar** |
| `src/model/tokenizer.rs.orig`, `src/model/tokenizer.rs.rej` | **Eliminar** |

---

## RESUMEN: ARCHIVOS ASM FINALES (14)

| Archivo | LOC | Acción | Estado final |
|---------|-----|--------|-------------|
| `ternary_gemv.s` | 283 | 8 accum + prefetch widen + NaN guard | ✅ Optimizado |
| `ternary_gemv_4rows.s` | 150 | NaN guard | ✅ Optimizado |
| `ternary_gemm_batch4.s` | 217 | Prefetch + NaN guard | ✅ Optimizado |
| `ternary_backward.s` | 355 | (GCC, NaN guard ya tiene) | ⚪ Mantener |
| `adam_step.s` | 162 | Prefetch + NaN guard | ✅ Optimizado |
| `silu.s` | 153 | Prefetch + NaN guard | ✅ Optimizado |
| `rmsnorm.s` | 57 | — | ⚪ Mantener |
| `slime_rmsnorm.s` | 255 | — | ⚪ Mantener |
| `math.s` | 278 | — | ⚪ Mantener |
| `sgemm.s` | 246 | Prefetch | ✅ Optimizado |
| `mamba.s` | 193 | Prefetch | ✅ Optimizado |
| `rope.s` | 64 | Prefetch | ✅ Optimizado |
| `q4_0_gemv.s` | 102 | Prefetch | ✅ Optimizado |
| `lm_head.s` | 89 | **Reescritura completa** | ✅ Optimizado |
| ~~`ternary_pext.s`~~ | ~~69~~ | ~~Eliminar~~ | ❌ Eliminado |
| ~~`elut_gemv.s`~~ | ~~127~~ | ~~Eliminar~~ | ❌ Eliminado |
| ~~`ternary_lut.s`~~ | ~~102~~ | ~~Eliminar~~ | ❌ Eliminado |

---

## ORDEN DE EJECUCIÓN

```
Día 1:  Fase 1 (eliminar deprecated) + Fase 6 (eliminar dead code Rust)
Día 2-3: Fase 2 (lm_head.s reescritura) — mayor impacto
Día 4:   Fase 3 (ternary_gemv.s 8 accum) — segundo mayor impacto
Día 5:   Fase 4 (prefetch en 7 kernels)
Día 6:   Fase 5 (NaN guards en 4 hot paths)
Día 7:   Verificación final + cargo clippy + cargo test
```

**Impacto esperado:**
- `lm_head.s` reescrito: ~3× en inferencia (elimina 128k calls + 328M ciclos)
- `ternary_gemv.s` 8 accum: ~1.5× en GEMV individual
- Prefetch añadido: 5-15% en cada kernel, ~10% agregado en forward completo
- NaN guards: 0 ciclos en path normal, evita NaN propagation en training
