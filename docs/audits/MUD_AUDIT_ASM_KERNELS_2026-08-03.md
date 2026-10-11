# Auditoría de Kernels ASM — src/asm/*.s (2026-08-03)

**Fecha:** 2026-08-03
**Alcance:** 24 archivos `.s` x86-64/AVX2 + FFI en `src/asm/mod.rs` (33 externs) + `build.rs` (21 compilados).
**Método:** Revisión instrucción a instrucción de cada kernel contra la convención SysV (rdi/rsi/rdx/rcx/r8/r9 + xmm0-7), el layout de registros `[im:re]` (im en bits 16-31, `slime.rs:33-50`), y los contratos FFI de `mod.rs`.

**Veredicto global:** el hot-path real (18 kernels) está **CORRECTO** — los fixes críticos de sesiones anteriores (`vpsub→vaddps` en backward dx single y 8rows) están presentes y verificados. Se encontraron **6 bugs** (1 en kernel vivo con FFI pero sin callers, 5 en kernels muertos/test-only) y **4 archivos completos muertos**.

---

## 1. BUGs de numerics / registro

### A-1 — `pack.s` `gate_batch` → gates constantes 0.5 (BUG de registro, kernel vivo con FFI, sin callers)

**Archivo:** `src/asm/pack.s:129-213`
**Severidad:** BUG latente — el único kernel que implementa la FFI `gate_batch` declarada en `mod.rs:30-34`

```asm
# Extract imag f16 using shift + shuffle
vpsrld $16, %ymm0, %ymm1            # ← desplaza im (bits 16-31) a LOW 16 bits
vmovdqa .Limag_extract_mask(%rip), %ymm2
vpshufb %ymm2, %ymm1, %ymm1         # ← la máscara selecciona bytes 2,3,6,7,10,11,14,15
```

La máscara `.Limag_extract_mask` (pack.s:210-214) está construida para el layout **nativo** `[im:re]` (im en bytes 2-3 de cada dword). El `vpsrld $16` se ejecuta **antes** del `vpshufb`, moviendo im a los bytes 0-1 y dejando **ceros** en los bytes 2-3 — que es exactamente lo que la máscara luego selecciona.

**Resultado:** extrae 8 f16 = 0.0 → `gate = sigmoid(0) = 0.5` **constante**, sin importar la entrada.

**Fix:** eliminar el `vpsrld $16` (la máscara trabaja sobre el layout nativo) y añadir test de equivalencia contra `1/(1+exp(-im_f32))`. Nota: `gate_batch_into` (`mod.rs:339-348`) es el wrapper — verificar si algún futuro caller lo usa.

### A-2 — `cmud_cell_v5_avx2.s` `cellv5_phase_repulsion_batch_avx2` — máscara invertida + eps denormal

**Archivo:** `src/asm/cmud_cell_v5_avx2.s:179-193`
**Severidad:** BUG (no compilado — P-08)

```asm
vpxor %ymm11, %ymm11, %ymm15        # ← ymm15 = 0 (intención: ~mask)
vandps %ymm15, ...                  # ← usa ymm11 (la máscara), no su complemento
```

- `ymm15` se pone a **cero** en lugar de `~mask` → el `vandps` conserva las fases donde |s| < eps y anula donde |s| > eps — **exactamente al revés** de lo que debe hacer la repulsión.
- `eps_val = 0x00000001` (denormal ~1.4e-45) → umbral inexistente.

**Fix:** `vpcmpeqd` + `vpxor` para el complemento real (o `vandnps`), y eps normal (>1e-6). O borrar el archivo (P-08, no compilado).

### A-3 — `ternary_gemm_batch4.s` — guard Inf incompleto + overread del leftover

**Archivo:** `src/asm/ternary_gemm_batch4.s:176-179` y `:129-165`
**Severidad:** BUG de numerics + UB (test-only)

- Guard de no-finitos con `vucomiss` + `jnp` solo detecta **NaN**; ±Inf pasa y contamina los acumuladores. Fix: comprobar exponente `0xFF` (patrón de `apply_scales.s`).
- Loop leftover: cuando `in_dim % 8 ∈ [1,7]`, carga **vectores YMM completos** en las últimas filas → lee hasta 7 f32 **fuera del final de la fila** (overread). Solo en tests, pero UB bajo miri. Fix: tail escalar o redondear `in_dim` en el caller.

### A-4 — `rmsnorm.s` — media incorrecta con n%8≠0

**Archivo:** `src/asm/rmsnorm.s:16-48`
**Severidad:** BUG (caller `q4_0_gemv_fused` muerto)

El loop procesa `n/8` bloques pero el divisor es `n` completo: si `n % 8 ≠ 0` la media es incorrecta; si `n < 8` produce `scale = 1/√eps` (datos ignorados). Fix: acumular conteo real, o exigir n%8==0 en el contrato FFI.

### A-5 — `fenwick_avx2.s` — doble convención de índice (latente)

**Archivo:** `src/asm/fenwick_avx2.s` (`fenwick_f32_prefix_sum_batch4`)
**Severidad:** bug de documentación↔código (no compilado)

El comentario documenta `tree[idx-1]` pero el código indexa `tree[idx]`. Si alguien conecta este `.s` a futuro obtendrá resultados distintos a la semántica documentada (la implementación real `FenwickF32` en `fenwick.rs` es 1-indexada en Rust). Fix: alinear o borrar (P-08).

---

## 2. Código muerto (P-08)

| Archivo | Estado | Consumidor |
|---|---|---|
| `rope.s` | **No compilado** (build.rs solo lo lista en rerun-if-changed) | Ninguno — el símbolo `apply_rope_*` no matchea la FFI `apply_rope_interleaved_asm`; RoPE real usa intrinsics |
| `fenwick_avx2.s` | No compilado | Ninguno (ver A-5) |
| `cmud_cell_v5_avx2.s` | No compilado | Ninguno (ver A-2) |
| `slimex_cell_v5_avx2.s` | **Compilado por build.rs:16** | **0 callers** — `slimex_asm.rs` fue eliminado (2026-07-29); peso muerto en cada build |
| `math.s`: `peak_abs_avx2`, `apply_gradient_avx2`, `hadamard_transform_avx2` | Compilados | Sin FFI ni callers. `hadamard` además **OOB si n%8≠0** (test `#[ignore]`) |
| `q4_0_gemv.s` | Compilado | Test-only; resto `n%32` descartado silenciosamente |
| `ternary_gemm_batch4.s` | Compilado | Test-only (ver A-3) |

**`dot_product_avx2` y `sum_squares_avx2` (math.s) SÍ están vivos y son correctos** — no borrar.

---

## 3. PERF

### A-6 — `backward_ternary_gemv_8rows.s` — 3 ineficiencias
- Cadena serial de **8 `vaddps`** en 1 acumulador (latencia ~4 ciclos × 8 por elemento) → usar 2 acumuladores YMM y reducir al final.
- `scale_gy` recargada con `vmovss`+broadcast **por cada fila** (×8) → un solo `vmovups` + broadcast fuera del loop de filas.
- Bases de fila (`lea`) recargadas desde el stack en cada word del loop interior → mantener en registros calientes.

### A-7 — `adam_step.s` — `vdivps` → `vrcpps` + Newton-Raphson
`m/(√v+eps)` con `vdivps` (~13-21 ciclos latencia). Fórmula idéntica a `adam_step_scalar` (`adam_state.rs:246-258`), correcto pero lento. `vrcpps` + 2 iteraciones de Newton dan ~1.7e-7 error relativo (suficiente para f32).

### A-8 — `sgemm.s` — `imul` por elemento, sin tiling
Numéricamente correcto pero `imul` en el triple loop y sin blocking. Test-only — si se va a usar en producción, tiling 4×4 f32 + índices hoisteados.

---

## 4. Kernels VERIFICADOS CORRECTOS (hot-path real)

| Kernel | Archivo | Notas |
|---|---|---|
| `ternary_gemv_avx2` | ternary_gemv.s | NaN→0; requiere n%8==0 (callers garantizan: n_in ∈ {576, 2048, ...} en `gemv_gpu.rs:298/379`) |
| `ternary_gemv_4rows_avx2` / `_8rows` | ternary_gemv_4rows.s / _8rows.s | Paralelismo 4/8 filas, `vpmaddwd`/`vpsign`, correctos |
| `backward_ternary_gemv_avx2` | backward_ternary_gemv.s | **fix `vaddps` presente (línea 98)**; tail escalar OK (168-173) |
| `backward_ternary_gemv_8rows_avx2` | backward_ternary_gemv_8rows.s | **fix `vaddps` presente (~123)**; solo notas PERF (A-6) |
| `jepa_ou_avx2` | jepa_ou.s | OU exacto `z+α(y_norm−z)` (vsubps+vfmadd231ps); `push %rbx` muerto y `.jepa_alpha` sin usar (cosmético) |
| `backward_rmsnorm_avx2` | backward_rmsnorm.s | ABI completa rdi..r9 (r8d=mean_bits, r9=grad_x) ✓ |
| `backward_rope_avx2` | backward_rope.s | Layout split consistente con forward; guards vector+escalar ✓ |
| `mhc_gate_avx2` | mhc_gate.s | Gate mHC ✓ |
| `lm_head_logits_avx2` / `lm_head_argmax_avx2` | lm_head.s | NaN guard; argmax ignora NaN; `−inf` inicial correcto ✓ |
| `dot_product_avx2`, `sum_squares_avx2` | math.s | Correctos y vivos ✓ |
| `silu_avx2` | silu.s | 3 paths (8/4/1-3) ✓ |
| `apply_scales_avx2` | apply_scales.s | NaN/Inf→0 vía exponente + blend; clamp ±1e8 ✓ |
| `cmud_batch_avx2` (gauss_mul, wave_collapse, phase_repulsion) | cmud_batch.s | 6 instrs/par complejo ✓ |
| `registers_to_f32_batch_avx2` / `f32_to_registers_batch_avx2` | pack.s | Layout `[im:re]` correcto — único bug de pack.s es `gate_batch` (A-1) |
| `sgemm_abt_avx2` | sgemm.s | Correcto (PERF: A-8) |

---

## 5. Priorización

1. **A-1 `gate_batch`** — único bug de un kernel con FFI declarada; corregir o eliminar junto al wrapper `gate_batch_into`.
2. **P-08 purge** — borrar `rope.s`, `fenwick_avx2.s`, `cmud_cell_v5_avx2.s`, `slimex_cell_v5_avx2.s` (+ entrada build.rs:16) y las 3 funciones muertas de `math.s`; verificar `q4_0_gemv.s`/`ternary_gemm_batch4.s` (test-only → tail o borrar).
3. **A-6/A-7** — optimizaciones PERF si el perfil las justifica (backward 8rows es hot path real del training).
4. **A-3/A-4/A-5** — solo si los kernels pasan a producción.
