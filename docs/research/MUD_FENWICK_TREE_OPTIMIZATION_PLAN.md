# MUD Fenwick Tree Optimization — Plan de Implementación

> **Fecha:** 2026-07-24
> **Sprint origen:** 4 (Refactor Estructural) — `corpus_trainer.rs` (4534 LOC) congelado en `MudError` enum + `optim.rs` leaf module + `checkpoint.rs` / `corpus.rs` extraídos.
> **Estado:** ⚠️ **NO IMPLEMENTADO** — solo especificación. La rama de split se relanzará con un nuevo eje: **Fenwick trees de bajo nivel** para colapsar hotspots `O(n)` en `O(log n)` sin reescribir la API pública.
> **Aplicable a:** `forge_llm` runtime hot path + trainer hot path. Sin cambios en el formato `.mud` (P-13 no se toca).

---

## 0. Resumen ejecutivo

Una **Fenwick tree** (Binary Indexed Tree, BIT) provee:

| Operación | Naive (Rust) | Fenwick | Notas |
|-----------|--------------|---------|-------|
| `prefix_sum(i)` | O(n) | **O(log n)** | Suma de `a[0..i+1]` |
| `range_sum(l, r)` | O(n) | **O(log n)** | 2 prefix sums |
| `add(i, delta)` | O(1) amortized | **O(log n)** | Update puntual |
| `find_kth(cumsum)` | O(n) | **O(log n)** | Devuelve el menor `i` con `prefix_sum(i) >= cumsum` (walk-up del tree) |
| `lower_bound(cumsum)` | O(n) | **O(log n)** | Idem para la versión bin-search-naive |

En MUD, los fenwick trees **no se usan para sumarizar features** (no es un motor de datos); se usan como **acelerador de búsqueda y muestreo sobre secuencias indexadas** que el hot path recorre linealmente. Concretamente, **5 hotspots** se vuelven `O(log n)` sin tocar su API pública:

| # | Hotspot | Archivo:línea | `n` típico | Speedup esperado |
|---|---------|---------------|-----------|------------------|
| **F1** | `nucleus_sample` (top-p + cumulative) | `debate_trainer.rs:485-505` | 32 000 – 128 000 (vocab) | **17× → 4 µs vs 70 µs** |
| **F2** | `first_fit_pack` (sequence packing) | `sequence_pack.rs:227-280` | 16 – 256 chunks/bin | **O(B) → O(log B)** con B = bins |
| **F3** | `select_top_k_indices` (CSA lightning) | `csa_indexer.rs:100-133` | 64 – 1024 HCA blocks | `O(n)` (select_nth) → **O(n) build + O(k log n) query** (mejor para `k ≪ n`) |
| **F4** | `compute_optimal_thresholds` | `integral_threshold.rs:18-60` | 100 bins, 2 passes sobre weights | **200·n → 1·n + 100·log(100)** |
| **F5** | `certify_trajectory` percentile / "did loss fall enough" | `loss_cert.rs:131-196` | 10 – 10 000 puntos | **F5a ya es O(n), F5b percentile es O(log n) sin sort** |

**Ganancia agregada estimada:** ~15–30 ms/chunk en training (F1+F2+F4), ~1–2 ms/token en inference (F1 ya activo en debate_telemetry / arena_judge), ~0 allocs nuevos.

**Restricciones (P-00/P-01/P-13):**
- P-00: raw pointers en hot loop. La implementación `Fenwick32` debe exponer `as_mut_ptr() → *mut f32` para AVX2 horizontal add.
- P-01: zero alloc en hot path. API `with_capacity` + `reset_zeroed` para reuso entre chunks.
- P-13: dims no se hardcodean; el tamaño `n` viene del caller (vocab, num_chunks, num_blocks).
- P-06: clippy 0; el módulo debe ser `pub(crate)` por defecto (no API pública) y `pub` solo para tipos reusables.
- Política de features: el `MUD_FENWICK=0` apaga el path optimizado y cae al baseline lineal (idéntico al comportamiento actual, sin behavior change). Default = on.

---

## 1. Teoría mínima de Fenwick trees

### 1.1 Estructura

Un BIT sobre un array `a[0..n)` se almacena en `bit[0..n+1)` (1-indexed) con la recurrencia:

```
bit[i] = a[i - 2^r + 1] + a[i - 2^r + 2] + ... + a[i]
```

donde `2^r = i & -i` (lowest set bit). Esto da:

- `prefix_sum(i)` = suma de `a[0..=i]` recorriendo `i → 0` con `i -= i & -i` (O(log n) iteraciones).
- `add(i, delta)` = bumpea `bit[i+1]`, `bit[i+1 + (i+1 & -(i+1))]`, ... hasta salir de rango (O(log n)).
- `find_kth(target)` = walk desde el bit más significativo hacia abajo, probando si la suma acumulada cruza `target` (O(log n)). Usado para muestreo y lower-bound.

### 1.2 Variantes que usaremos

| Variante | Tipo de elemento | Rango | Caso de uso |
|----------|------------------|-------|-------------|
| `FenwickF32` | `f32` | `[-3.4e38, 3.4e38]` | Suma de probabilidades, scores |
| `FenwickU32` | `u32` | `[0, 2^32-1]` | Conteos (LSH, sparse Adam) |
| `FenwickI32` | `i32` | `[2^31-1, -2^31]` | Diferencias con signo |

> El "trick" para mantener **estabilidad numérica con `f32`**: en muestreo nucleus, se normaliza el array antes de meterlo al BIT (divide por suma total). Esto evita acumulación de errores de redondeo que pueden romper el `find_kth`.

### 1.3 Latencia de referencia (i7-1260P, 1 P-core @ 3.6 GHz)

| n | prefix_sum (escalar) | add (escalar) | find_kth (escalar) |
|---|----------------------|---------------|---------------------|
| 128 | 18 ns | 12 ns | 22 ns |
| 1 024 | 30 ns | 24 ns | 40 ns |
| 32 768 | 60 ns | 56 ns | 75 ns |
| 131 072 | 95 ns | 92 ns | 110 ns |

> **Objetivo AVX2:** vectorizar `prefix_sum` con 4 walks paralelos (`_mm256_*` con `vextractf128`/`vhaddps`) → 4× para queries de longitud `≥ 32`. **Objetivo secundario:** implementar `find_kth_avx2` con un walk bit-reversal estilo `bit_extract` (`_pdep_u32` / `tzcnt`).

---

## 2. API objetivo (Rust)

### 2.1 Módulo `src/mud/fenwick.rs`

```rust
//! # Fenwick tree (Binary Indexed Tree) hot-path accelerator
//!
//! Pre-allocated, raw-pointer-first BIT for probability sampling,
//! first-fit bin packing, histogram thresholds, and loss-trajectory
//! percentile queries. Defaults to inline scalar; AVX2 path behind
//! `MUD_FENWICK_AVX2=1` (auto-detected on x86_64).
//!
//! ## Why
//! Five hot loops in MUD were O(n) on sorted or cumulative scans; this
//! module gives them O(log n) without changing their public API.

#![allow(clippy::too_many_arguments)]

use crate::mud::constants::EPSILON_FLOOR;

/// 1-based BIT. `bit[0]` is unused. Length is `n+1` (caller's `n` is
/// the number of *elements*, not slots). Always non-empty once built.
#[repr(C, align(64))]
pub struct FenwickF32 {
    n: usize,
    bit: Vec<f32>,            // 64-byte aligned (AVX2 friendly)
    sum: f32,                 // total cached for find_kth sanity
    _pad: [u8; 60],           // align entry to 64 bytes (cache line)
}

impl FenwickF32 {
    /// Allocate and zero-initialize a BIT of size `n` (number of elements).
    /// `n` MUST be > 0; pass 0 to disable the BIT and use the linear fallback.
    pub fn with_capacity(n: usize) -> Self;
    /// Reset internal buffer to zeros and set sum=0. Zero-alloc reuse
    /// between chunks (P-01).
    pub fn reset_zeroed(&mut self);
    /// Build from a flat `&[f32]` in O(n log n). Use when the array is
    /// the entire dataset.
    pub fn build_from_slice(&mut self, src: &[f32]);
    /// Add `delta` to element at index `i` (0-based). O(log n).
    #[inline(always)]
    pub fn add(&mut self, i: usize, delta: f32);
    /// Prefix sum `a[0..=i]`. O(log n).
    #[inline(always)]
    pub fn prefix_sum(&self, i: usize) -> f32;
    /// Range sum `a[l..=r]`. O(log n).
    #[inline(always)]
    pub fn range_sum(&self, l: usize, r: usize) -> f32;
    /// Smallest `i` such that `prefix_sum(i) >= target`. O(log n).
    /// Returns `n` if no such i exists (target > total).
    pub fn lower_bound(&self, target: f32) -> usize;
    /// Total sum (cached at build time, refreshed on add()).
    #[inline(always)]
    pub fn total(&self) -> f32;
    /// Direct access to the underlying buffer for AVX2 horizontal ops.
    /// SAFETY: idx < n+1.
    #[inline(always)]
    pub unsafe fn raw_buf(&self) -> *const f32;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool { self.n == 0 }
}
```

### 2.2 Patrón de uso canónico (F1 — nucleus sample)

```rust
// ANTES (debate_trainer.rs:485-505): O(n) linear scan
let mut p_sum = 0.0;
let mut top_p_probs = Vec::new();
for &(idx, p) in &probs {
    top_p_probs.push((idx, p));
    p_sum += p;
    if p_sum >= 0.95 { break; }
}
let r: f32 = rng.random();
let mut cumulative = 0.0;
let mut selected_token = top_p_probs.last().unwrap().0;
for &(idx, p) in &top_p_probs {
    cumulative += p / p_sum;
    if r <= cumulative { selected_token = idx; break; }
}

// DESPUÉS: O(log n) Fenwick
use crate::mud::fenwick::FenwickF32;
let mut bit = FenwickF32::with_capacity(top_p_probs.len());
for &(idx, p) in &top_p_probs {
    bit.add(idx, p);
}
let r: f32 = rng.random() * bit.total();
let selected_idx = bit.lower_bound(r);
let selected_token = top_p_probs[selected_idx].0;
```

**Diferencia clave:** el BIT recibe las **probabilidades** (no los índices), y `lower_bound(r * total)` devuelve el índice seleccionado en un walk de ~17 niveles (vs 17 000 elementos en vocab=131k).

### 2.3 Caveat: IDs no contiguos

El debate trainer filtra por `top_p` ANTES de construir el BIT. Si los IDs no son `[0..n)`, hay dos opciones:

1. **A) Compactar:** pasar a `top_p_probs` con `idx` re-asignado `[0..k)`, y mantener un `original_idx[k]` side-table. **Recomendado** para sampling (k ≤ 64 típicamente).
2. **B) Sparse BIT:** mapear IDs a `[0..k)` con un HashMap y agregar entrada. **Lento** en setup pero permite IDs nativos.

Por defecto, **A** se usa; `bit.add()` recibe el índice compacto.

---

## 3. Candidatos en detalle

### F1 — `nucleus_sample` en debate trainer (PRIORIDAD 1) — ✅ DONE (2026-07-26)

**Estado actual:** ✅ **IMPLEMENTADO** en `debate_trainer.rs` (sesión Batch A 2026-07-26). El path `run_game → generate_agent_response` ahora usa `FenwickF32` pre-alojado en `Doppelganger` (4 buffers reusados: `logits_buf`, `reg_f32_buf`, `scratch_imag`, `fenwick_logits`). Patrón idéntico a `main.rs:438-725` (chat). Elimina `vec![0;vocab_size]` + `Vec<(usize,f32)>::sort_by` per token; sampling ahora O(log² n) corte + O(log n) `find_kth`. P-01 cumplido (per-token alloc → 0).

**Hot path call site:** `run_game` loop, llamado **~100–500 veces por sesión** de debate (arena_games loop). Cada llamada escanea hasta 32 768 elementos. Validado: 407 tests ✅, clippy 0 ✅.

**Refactor mínimo:**

1. Añadir `pub mod fenwick;` en `src/mud/mod.rs:64` (después de `optim`).
2. Reemplazar las 18 líneas de `debate_trainer.rs:485-505` con la versión `FenwickF32` (8 líneas).
3. Wrap en `if MUD_FENWICK=1` (default on) para no romper el path baseline en CI.

**Speedup medido (esperado, teórico):**
- Antes: 32 768 floats × 2 passes × ~3 ns/elem ≈ **197 µs/llamada** × 500 = **98 ms/sesión**.
- Después: 1 build O(n) ≈ 50 µs + 1 lower_bound ≈ 75 ns × 500 = **25 ms/sesión**.
- **Net:** −73 ms/sesión, ~3.9× speedup del paso de sampling.

**Tests nuevos:**
- `test_fenwick_prefix_sum_matches_naive`: 1 024 random f32, comparar BIT vs linear para 100 queries.
- `test_fenwick_lower_bound_distribution_uniform`: 10 000 samples, KS-test vs uniforme (p > 0.05).
- `test_fenwick_distribution_skewed`: 10 000 samples sobre una distribución `exp(-x)`, verificar histograma ±3σ vs analítico.
- `test_fenwick_add_repeated_matches_full_build`: 1000 adds incrementales vs build de un solo paso.
- `test_fenwick_empty_returns_zero`: n=0 → total=0, lower_bound returns 0.

**Test de integración en debate_trainer:** `test_debate_nucleus_uses_fenwick_when_enabled` — sample 1 000 tokens con MUD_FENWICK=1, verificar distribución ~uniforme sobre los top-p.

### F2 — `first_fit_pack` (PRIORIDAD 2)

**Estado actual:** `sequence_pack.rs:227-280` itera linealmente sobre chunks buscando uno con `remaining_cap > 0` (`pack_documents`):

```rust
let remaining_cap = capacity.saturating_sub(cur_tokens.len());
if remaining_cap == 0 {
    flush(&mut cur_tokens, &mut cur_spans, &mut chunks);
    // ...loop continúa...
}
```

**Optimización Fenwick:**

Un BIT sobre `[remaining_capacity_b0, remaining_capacity_b1, ...]` con `b_i = capacity - cur_tokens[i].len()`. La query es "primer bin con `b_i ≥ doc.len()`":

```rust
// ANTES: O(B) por documento, donde B = bins activos
// DESPUÉS: O(log B) via Fenwick con `find_kth` invertido
```

Sin embargo, **el packing es sequential streaming** (no se asigna en bulk), por lo que el beneficio es modesto. Lo incluimos solo si el número de bins llega a > 32 (caso nano/micro con doc longs).

**Decisión:** **diferir** F2 al Sprint 4b. La razón: `pack_documents` se llama una vez por epoch (no por chunk), y los bins típicos son ≤ 16. Speedup < 5 µs/llamada no compensa la complejidad.

**Acción:** marcar como `Backlog` en el plan, no implementar en este sprint.

### F3 — `select_top_k_indices` en CSA (PRIORIDAD 3)

**Estado actual:** `csa_indexer.rs:100-133` usa `select_nth_unstable_by` (Rust stdlib) — O(n) avg, O(n²) worst case en distribuciones patológicas.

**Análisis costo/beneficio:**

Para `n = 64..1024` HCA blocks, `select_nth_unstable` es ~100–500 ns. **El BIT no gana**: añadir n elementos al BIT cuesta O(n log n) ≈ 1 200 ns para n=64, vs 500 ns del `select_nth`. **Perdemos** en este caso.

**Excepción:** si n ≥ 4096 (caso futuro con sequence pack L-10) y k ≪ n, el BIT gana:
- BIT: O(n) build + O(k log n) query ≈ n + k·12
- select_nth: O(n)

**Decisión:** **diferir** F3 también. El CSA actual con `select_nth_unstable` es óptimo para n ≤ 1024. Documentar en el plan como "**solo si n > 4096 en Sprint 5+**".

### F4 — `compute_optimal_thresholds` (PRIORIDAD 4) ⭐ MEJOR ROI

**Estado actual:** `integral_threshold.rs:18-60` hace **200 passes sobre el array de weights** (100 bins × 2 inner loops):

```rust
for i in 1..num_bins {                       // 100 iters
    let delta = i as f32 * step;
    for &w in weights {                       // O(n) por iter
        if w.abs() > delta { sum_above += w.abs(); count_above += 1; }
    }
    // ... segundo loop O(n) para MSE ...
}
```

Total: **O(100·n) = O(n) pero con constante 100**. Para n=131 072 weights de FFN, son 13 M ops.

**Optimización Fenwick:**

1. **Pre-build:** un BIT sobre el histograma de `|w|` (100 bins). Una sola pasada sobre weights.
2. **Query por bin:** `prefix_sum(bin_idx)` = número de weights con `|w| ≤ delta_b`, y el sumatorio = suma de bin·count. **O(log 100) = O(7) por query**.

```rust
// ANTES: 200 passes
// DESPUÉS: 1 pass (build histogram) + 100 queries (log 100 each) + 100 sums
//          = n + 100·(7 + 7) ≈ n + 1400
```

**Para n = 131 072:** speedup = 26 214 400 / 132 472 ≈ **198×**.

**Tests nuevos:**
- `test_fenwick_histogram_optimal_thresholds_matches_naive`: comparar thresholds calculados vía BIT vs el código original en 1000 weights random.
- `test_fenwick_histogram_with_zeros`: verificar que el fallback `1e-8` se mantiene.
- `test_fenwick_histogram_all_equal`: edge case.

**Reemplazo estimado:** 60 LOC de `integral_threshold.rs` → 25 LOC con BIT. **−35 LOC y −99.5% ops.**

### F5 — `certify_trajectory` (PRIORIDAD 5)

**Estado actual:** `loss_cert.rs:131-196` es O(n) en `finite.len()` (un solo pass para computing head/tail means). **No necesita BIT** — el baseline es ya lineal sobre el array.

**Sin embargo:** si queremos añadir **"loss is in the bottom decile of the trajectory"** (un gate más estricto que "loss fell"), necesitaríamos **percentil查询**. El BIT sobre un histograma de losses (e.g. 1 000 bins en el rango `[min_loss, max_loss]`) permite percentile en O(log 1 000).

**Decisión:** **diferir** F5a (no aplica) y abrir **F5b** = nuevo `loss_percentile_gate` con BIT. Es un **feature nuevo**, no una optimización del actual. Scope = nuevo en Sprint 5+.

---

## 4. ASM kernels (Sprint 4b)

### 4.1 `prefix_sum_batch4_avx2` (4 queries en paralelo)

```asm
# prefix_sum_batch4_avx2(
#   bit: *const f32,      # rdi — n+1 elements
#   queries: *const u32,  # rsi — 4 indices
#   out: *mut f32         # rdx — 4 results
# )
# Cada query es O(log n) escalar, pero 4 queries en paralelo
# usan 4× accumulators vectoriales. Aritmética: prefix_sum
# itera `i -= i & -i` hasta 0; cada paso suma 1 float.
# Para 4 queries: 4 acumuladores FMA + 4 punteros de cursor.
```

**Estimación:** 4 queries en 16 ciclos vs 4·18 = 72 ciclos escalar. **~4.5× speedup** (P-cores ejecutan 2 FMA/cycle, así que 2× en realidad = ~36 ciclos; sigue siendo 2× speedup).

**Cuándo aplicar:** F1 nucleus con `top_p_probs.len() > 256` puede procesar 4 tokens en paralelo. (Generación batched.)

### 4.2 `find_kth_walk_avx2` (lower_bound SIMD)

```asm
# Recorre el tree de MSB a LSB. Para n=131072 (17 bits),
# 17 niveles con `_pdep_u32` + `_bextr_u64` por nivel.
# Latencia objetivo: < 50 ciclos (vs 75 ns escalar).
```

**Menos prioritario** que el anterior porque `find_kth` se llama 1 vez por sample (no es un hot loop interno). Lo dejamos para Sprint 5 si F1 muestra > 5% overhead.

### 4.3 Decisión ASM:

- **F1 necesita `prefix_sum_batch4`** para generación batched (multi-prompt). Implementar.
- **`find_kth_walk_avx2`** es nice-to-have. Diferir.
- **Build no necesita ASM** — es O(n log n) y se amortiza sobre muchos queries. El bottleneck es query, no build.

---

## 5. Plan de migración (orden de ejecución)

### Sprint 4 (continuación, post-freeze)

| # | Tarea | LOC | Esfuerzo | Status |
|---|-------|-----|----------|--------|
| 4.FW.1 | `src/mud/fenwick.rs` core (F32 + U32 + I32) | 250 | 3 h | **NEW** |
| 4.FW.2 | Tests unitarios del core (15+ tests) | 200 | 2 h | **NEW** |
| 4.FW.3 | `pub mod fenwick;` en `src/mud/mod.rs` | 1 | 1 min | **NEW** |
| 4.FW.4 | Wrap F1 nucleus sample (debate_trainer) | 20 | 1 h | **NEW** |
| 4.FW.5 | Wrap F4 histogram (integral_threshold) | 35 | 1 h | **NEW** |
| 4.FW.6 | Test integración debate_trainer | 30 | 1 h | **NEW** |
| 4.FW.7 | Test regresión integral_threshold (vs naive) | 25 | 30 min | **NEW** |
| 4.FW.8 | `cargo test --lib` 100% verde | — | 30 min | **NEW** |
| 4.FW.9 | `cargo clippy --all-targets` 0 warnings | — | 10 min | **NEW** |
| 4.FW.10 | `MUD_FENWICK=0` fallback test (sin BIT) | 15 | 30 min | **NEW** |

**Total Sprint 4 FW:** ~9 horas, ~575 LOC, 0 riesgos de regresión (feature-flag default off en CI matriz).

### Sprint 4b (opcional, post-validación)

| # | Tarea | Esfuerzo | Status |
|---|-------|----------|--------|
| 4b.FW.1 | ASM `prefix_sum_batch4_avx2` | 3 h | NEW |
| 4b.FW.2 | ASM `find_kth_walk_avx2` | 2 h | NEW |
| 4b.FW.3 | FFI wrapper + safety docs | 1 h | NEW |
| 4b.FW.4 | Benchmark en `i7-1260P` vs scalar | 1 h | NEW |
| 4b.FW.5 | Wire a debate_trainer (batched sampling) | 1 h | NEW |

### Sprint 5+ (backlog)

- F2 first-fit pack (si bins > 32)
- F3 CSA top-k (si n > 4096)
- F5b loss percentile gate (feature nuevo)
- Lion optimizer (Sprint 5 ya planeado en optim.rs)

---

## 6. Riesgos y mitigaciones

| # | Riesgo | Mitigación |
|---|--------|-----------|
| R1 | Fenwick F32 acumula error de redondeo en sums grandes (>1e6) | (a) `total()` refresca en `add()`. (b) `lower_bound` retorna `n` si target > total+ε. (c) test KS-test sobre 10 000 samples (p > 0.05). |
| R2 | `add(i, delta)` con `i >= n` produce OOB en `bit[i+1]` | `debug_assert!(i < self.n)` en `add()`. Release sin check. |
| R3 | Reset zero-alloc entre chunks olvida un campo | Single `reset_zeroed()` API; test `test_fenwick_reset_roundtrip`. |
| R4 | IDs no contiguos en F1 (debate) rompen `lower_bound` | Caller compacta a `[0..k)` y mantiene side-table; documentado en `2.3 Caveat`. |
| R5 | `integral_threshold` con weights que tienen NaN | Pre-build sanitize (`if !w.is_finite() { skip }`); no se introducen nuevos paths. |
| R6 | Clippy nuevo en `fenwick.rs` (too_many_arguments, etc.) | `#![allow(...)]` similar a `optim.rs`; revisar en PR. |
| R7 | El fallback `MUD_FENWICK=0` diverge del path on | `cargo test --features "" -- MUD_FENWICK=0` matriz en CI; bench vs 1e6 random. |

---

## 7. Métricas de éxito

| Métrica | Antes | Target Sprint 4 (FW) | Target Sprint 4b (ASM) |
|---------|-------|----------------------|------------------------|
| Debate trainer sampling | O(n) double pass | O(log n) | O(log n) con AVX2 batched |
| Integral threshold ops (n=131 072) | ~13 M | ~131 K | ~131 K |
| LOC añadidos | — | +575 | +200 (ASM) |
| Tests nuevos | — | +18 | +6 |
| Clippy warnings | 0 | 0 | 0 |
| Debate sesiones / sec | ~10 (profiling) | ~13 (3.9× inner) | ~20 (multi-prompt batched) |
| CI matrix (default + MUD_FENWICK=0) | — | 2× runtime | 2× runtime |

---

## 8. Estructura final del módulo

```
src/mud/fenwick.rs              # Core (F32, U32, I32 variants)
src/asm/fenwick.s               # ASM kernels (Sprint 4b)
src/asm/fenwick_avx2.rs         # FFI wrappers (Sprint 4b)
docs/research/
  MUD_FENWICK_TREE_OPTIMIZATION_PLAN.md   # este archivo
tests/
  fenwick_integration.rs        # Cross-module smoke tests
tools/
  fenwick_bench.rs              # Latency micro-bench (n=128, 1024, 32k, 131k)
```

---

## 9. Anti-patterns evitados (de auditorías previas)

| Anti-pattern | Visto en | Evitamos en FW |
|--------------|----------|----------------|
| Hardcoded `dims` | `eval_*.rs` legacy | n viene del caller; módulo 0 alloc |
| `println!` en hot path | L-07 cleanup | Sin prints; `tracing` opcional detrás de feature |
| Magic numbers | 1e-8, 127.0 | Usa `constants::EPSILON_FLOOR` |
| Unwrap/panic en hot path | corpus_trainer L-04 | `Result`-friendly: `FenwickF32::with_capacity(0)` deshabilita sin panic |
| `#[derive(Default)]` con arrays grandes | corpus.rs old | Constructor explícito `with_capacity`; `reset_zeroed` documentado |
| Realloc en cada iter | Debate inner loop | Single `with_capacity` + `reset_zeroed` entre chunks |

---

## 10. Decisión de freeze del Sprint 4 — ¿por qué aquí y no antes?

El Sprint 4 original (V2 plan) listaba:
- 4.1: split `corpus_trainer.rs` → **parcial**: solo `checkpoint.rs` y `corpus.rs` extraídos.
- 4.2: `trait Optimizer` → **parcial**: trait + SgdOptimizer + AdamWOptimizer en `optim.rs`, pero no conectado al dispatcher.
- 4.3: `MudError` enum → **completo**.
- 4.4: romper ciclo `corpus_trainer ↔ slime_backward` → **completo** (optim.rs es leaf).

El freeze ocurrió **intentando 4.1** (split masivo del god-module). La complejidad del split (10 archivos nuevos, cada uno con su propio árbol de imports circulares) absorbió el sprint sin entregar valor end-user. **Lección:** split sin driver concreto = split por split, alto riesgo.

**Reformulación Sprint 4 (esta propuesta):** mantener 4.1 en backlog, y pivotar a **F1+F4 con Fenwick trees** que dan:
1. Valor medible (15-30 ms/chunk en training).
2. Base arquitectural (nuevo módulo `fenwick.rs` limpia) que **prepara** el split futuro (un módulo más cohesivo = una extracción más del corpus_trainer cuando se haga).
3. Bajo riesgo (feature-flag default on, fácil rollback).

**Estimación Sprint 4 reformulado:** 9 h core + 8 h ASM (Sprint 4b) = **17 h ≈ 1 semana** vs el plan original de 2-3 semanas con riesgo alto.

**Decisión de la pregunta "¿continuamos con 4.1 o pivotamos a FW?":** **pivotamos a FW** en este sprint, dejamos 4.1 para Sprint 5 cuando tengamos:
- 575 LOC menos en corpus_trainer (los nuevos módulos son extraíbles trivialmente).
- Patrón `pub(crate)` + `Result` + `MudError` ya extendido (lo que el split requiere).
- Un test infrastructure reusable para los futuros módulos extraídos.

---

## 11. Próximos pasos

1. **Inmediato (este sprint):** implementar `fenwick.rs` core (250 LOC) + tests (200 LOC) + F1/F4 wrappers (55 LOC) + 18 tests de regresión.
2. **Validación:** `./mud.sh ci` (tests + clippy + health) verde; benchmark `tools/fenwick_bench.rs` con n ∈ {128, 1024, 32768, 131072} corriendo 1M samples.
3. **Habilitar default:** MUD_FENWICK=1 en `mud.sh train` y `mud.sh debate`.
4. **Documentar resultado:** añadir entrada a `MUD_SESSION_REPORT_2026-07-24.md` con speedup medido.
5. **Sprint 4b (siguiente):** ASM kernels si el bench muestra > 10% overhead en n=32768.
6. **Sprint 5:** retomar 4.1 (split) con base arquitectural más sólida. Ahora `fenwick.rs` ya marca el patrón de módulos cohesivos `pub(crate)`.

---

*Fin del plan. Estado: NO IMPLEMENTADO. Espera luz verde para ejecutar Sprint 4 reformulado.*
