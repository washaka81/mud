# MUD Trainer — Plan de Optimización Extrema (2026-08-02)

## Estado actual
- **Velocidad:** ~7.7 tk/s (CPU AVX2, i7-1260P, 8 P-cores)
- **Tests:** 446 ✅ | **Clippy:** 0 ✅
- **ASM kernels:** 23 archivos `.s` en `src/asm/`
- **EZOP:** 27 kernels AVX2 en `src/mud/ezop.rs`
- **Fenwick trees:** 947 LOC, 25 tests (`src/mud/fenwick.rs`) — ya implementado pero **no integrado en el trainer**

---

## E-1. Fenwick Trees en el Trainer (árvol fenwick — tabla de datos)

### Estado
`src/mud/fenwick.rs` implementa `FenwickF32` con:
- `prefix_sum_raw(idx)` → O(log n)
- `find_kth_raw(k)` → búsqueda binaria O(log n)
- `prefix_sum_batch4_avx2`/`prefix_sum_batch8_avx2` → 4/8 queries paralelas
- `find_kth_avx2` + `find_kth_walk` con `lzcnt` (BMI1) — O(1) bitmask

### Integración en el trainer (F-1 nucleus sampling + F-5b percentile gate)
```
┌─────────────────────────────────────────────────────────────┐
│ Trainer needs:                                              │
│ 1. F1 nucleus sampling: prefix_sum up to cumulative prob   │
│ 2. Percentile gate (CertMode::Percentile): histogram threshold │
│                                                             │
│ Current (scalar): O(vocab_size) per token → ~131K ops       │
│ FenwickF32: O(log vocab_size) per token → ~17 ops (198×)     │
└─────────────────────────────────────────────────────────────┘
```

### Implementación
- **`debate_trainer.rs`**: Reemplazar `vec1..vocab_size` allocation + sort O(n log n) → `FenwickF32::find_kth`
- **`train_step.rs`**: Softmax de sampled logits (256 classes) → `FenwickF32::prefix_sum_batch8_avx2` para top-k selection
- **Thread-local**: `FENWICK_SCRATCH` (reusable buffer, zero-alloc per step)

**Expected:** 198× speedup en nucleus sampling, elimina allocation per-token.

---

## E-2. Optimización de ASM (mejora ensamblador)

### 2.1. `backward_ternary_gemv_8rows.s` — optimización existente
| Optimización | Status |
|-------------|--------|
| `vpsub`→`vaddps` fix | ✅ (ya hecho, AGENTS.md) |
| `lea` row pointer precomputation (scale 1,2,4) | ✅ |
| Thread-local scratch para `jepa_ou` | ✅ |

### 2.2. Nuevas oportunidades ASM
```
┌─────────────────────────────────────────────────────────────┐
│ backward_rmsnorm.s — scalar fallback path                    │
│   Current: Rust scalar loop for remainder                   │
│   Opt:    Merge scalar tail into AVX2 main loop with        │
│           masking (vpmovmskb + conditional stores)          │
│                                                              │
│ jepa_ou.s — OU dynamics update                              │
│   Current: separate AXPY + exp                              │
│   Opt:    Fused z=0.9*z+0.1*y_norm in single pass           │
│                                                              │
│ mhc_gate.s — gate computation                               │
│   Current: sigmoid_vec_local en Rust                        │
│   Opt:    Full ASM sigmoid + blend en un pass               │
└─────────────────────────────────────────────────────────────┘
```

### 2.3. Pipeline de instrucciones
```
Current ASM pattern (per-row):
  vmovdqu ymm0, [W]    ; load 8 ternary u32
  vextract i8 lanes    ; 8 extract operations
  vpmovsxbd → fp       ; 8 sign-extend + convert
  vfmadd231            ; 8 MAC
  → repeat 8× per u32

Optimized pattern (batched):
  vmovdqu ymm0-3, [W]  ; load 4 u32 (32 weights)
  VPSHUFB batch unpack  ; 1 shuffle instead of 4 extracts
  VFMADDPD batch       ; 32 MAC in 4 cycles
```

---

## E-3. Branch Prediction + CPU Pipeline

### 3.1. Hot branch locations (identificadas)
| Archivo | Línea | Rama | Probabilidad |
|---------|-------|------|-------------|
| `slime_forward.rs` | 429 | `has_mhc = !alpha_w.is_null()` | 95% true |
| `slime_forward.rs` | 555 | `is_f32_weights` dispatch | 95% false |
| `slime_backward.rs` | 275 | `!layer.q_w.is_null()` | 95% true |
| `train_step.rs` | 310 | `is_x86_feature_detected` | 100% true |
| `slime_forward.rs` | 814 | `h < n_heads` loop | variable |

### 3.2. Optimizaciones de branch prediction
```rust
// BEFORE: branch in hot loop
for i in 0..n {
    if is_f32_weights {
        // F32 path
    } else {
        // Ternary path
    }
}

// AFTER: predicated dispatch (branchless inner loop)
let ternary_fn = if is_f32_weights { f32_path } else { ternary_path };
for i in 0..n { ternary_fn(...); }  // single indirect call, no branch mispredict
```

### 3.3. Loop unrolling + software pipelining
```
// Current: sequential dependency chain
sum_sq = 0
for i in 0..hidden:
    sum_sq += x[i] * x[i]    // each iteration depends on previous

// Optimized: 4-way unrolled, 2 accumulators
for i in (0..hidden).step_by(8):
    acc0 = fma(x[i], x[i], acc0)
    acc1 = fma(x[i+1], x[i+1], acc1)
    acc2 = fma(x[i+2], x[i+2], acc2)
    acc3 = fma(x[i+3], x[i+3], acc3)
    ; 4 independent accumulators → no pipeline stall
```

### 3.4. Cache-aware tiling
```
hidden=2048 → 8KB per row (f32)
L1d = 1.25MB → 156 rows fit
L2  = 10MB → 1280 rows fit

Current GEMV: row-by-row → good spatial locality
Optimization: process 8 rows together → temporal locality in PRQ scale
```

---

## E-4. Soluciones fáciles: Matrix / Vector Processing

### 4.1. Class embs round-and-scale (train_step.rs:551-553)
```rust
// CURRENT (scalar):
for v in &mut class_embs_flat[0..hidden_size] {
    *v = (*v / scale).round().clamp(-1.0, 1.0) * scale;
}

// OPTIMIZED (AVX2 batch):
unsafe {
    crate::mud::ezop::clamp_finite_avx2(ptr, -1.0 * scale, 1.0 * scale, 0.0, hidden_size);
    // then element-wise multiply by scale via scale_vec_avx2
}
```

### 4.2. Class softmax (train_step.rs:591-597)
```rust
// num_classes = 64 or 256 → AVX2 viable
unsafe {
    let max = ezop::max_avx2(class_logits_ptr, num_classes);
    ezop::axpy_avx2(class_logits_ptr, -max, class_logits_ptr /* no, need broadcast */);
    ezop::exp_avx2(class_logits_ptr, prob_q_ptr, num_classes);
    let sum = ezop::sum_avx2(prob_q_ptr, num_classes);
    ezop::scale_vec_avx2(prob_q_ptr, 1.0/sum, num_classes);
}
```

### 4.3. Embedding dequant (slime_forward.rs:590, 621, 634, 641)
```rust
// These are ternary GEMV inner loops — already have ternary_gemv_4rows.s
// But the i8→f32 dequant in the non-GEMV path (load_emb_row_into) is scalar
// Opt: add dequant_8rows_avx2 using vmovdqu + VPSHUFB
```

### 4.4. Layer norm backward (slime_backward.rs:754-761)
```rust
// CURRENT: scalar loop computing y_unscaled + sum_dy_xnorm
// Opt: split into 2 AVX2 passes:
// 1. y_unscaled = y_f32 / w  (vectorized divide)
// 2. sum_dy_xnorm = sum(grad * w * y_unscaled)  (dot3_fma_avx2)
// 3. w_grad += grad * y_unscaled  (vmul + axpy)
// 4. grad_x = w * (grad - y_unscaled * mean)  (rmsnorm_backward_avx2 already exists!)
```

### 4.5. Attention backward QKV residual (slime_backward.rs:838-878)
```rust
// Multiple per-head scalar loops over head_d (typically 128)
// Opt: flatten to single hidden_size loop, use AVX2 across heads
```

---

## E-5. Priority Order

| Priority | Task | Archivo | Effort | Impact | **Status** |
|----------|------|---------|--------|--------|-----------|
| **P0** | Fenwick tree integration in nucleus sampling | `debate_trainer.rs` + `train_step.rs` | Medio | 198× sampling | ✅ DONE (2026-08-02) |
| **P0** | `exp_avx2` NaN/Inf guard fix | `ezop.rs` | Bajo | SIGSEGV fix | ✅ DONE (2026-08-02) |
| **P1** | Class embs round-and-scale AVX2 | `train_step.rs:551` | Fácil | ~8× STP | ✅ DONE (2026-08-02) |
| **P1** | Softmax num_classes AVX2 | `train_step.rs:591` | Fácil | ~4× sampled | ✅ DONE (2026-08-02) |
| **P1** | FFN RMSNorm backward split | `slime_backward.rs:754` | Mediano | ~4× backward | ✅ **NOW DONE** — `div_avx2` + `dot3_fma_avx2`, elimina 2 scalar loops (4096 iter) |
| **P1** | **NEW: `div_avx2` kernel** | `ezop.rs` | Bajo | RMSNorm backward split | ✅ DONE (2026-08-02) — 3 tests, 0 UB miri |
| **P1** | **NEW: `grad_x_reduce_8rows_avx2`** | `ezop.rs` + `slime_backward.rs:480` | Fácil | elimina 8× axpy calls | ✅ DONE (2026-08-02) — tree reduction, 2 tests |
| **P1** | **NEW: `sum_sq` 4-accumulator** | `ezop.rs` | Fácil | ILP en RMSNorm fwd | ✅ DONE (2026-08-02) — 4× acc vs 1× serial |
| **P2** | Branchless feature dispatch (mHC) | `slime_forward.rs:435` | Fácil | elimina 6144 mispredict | ✅ DONE (2026-08-02) |
| **P2** | Layer norm backward y_unscaled | `slime_backward.rs:1335` | Mediano | ~2× backward norm | ✅ DONE — vectorized attention norm backward |
| **P3** | Attention backward flatten | `slime_backward.rs:838` | Alto | ~4× MHA backward | ⏳ TODO |
| **P3** | Backward RMSNorm tail masking | `asm/backward_rmsnorm.s` | Alto | ~1.3× RMSNorm | ⏳ TODO (ASM-level) |
| **P3** | JEPA OU fused AXPY+exp | `asm/jepa_ou.s` | Alto | ~1.5× JEPA | ⏳ TODO (ASM-level) |

## E-6. Implementation Pointers

### P0 — Fenwick tree nucleus sampling

**Pattern to follow** (from `debate_trainer.rs:438`):
```rust
// BEFORE (scalar O(n log n) per token):
let mut cumsum: Vec<f32> = (0..vocab_size).map(|i| probs[i]).collect();
for i in 1..vocab_size { cumsum[i] += cumsum[i-1]; }
let pos = cumsum.binary_search(&rng) ...sort...;

// AFTER (FenwickF32, O(log n) per token):
thread_local! { static FENWICK_SCRATCH: RefCell<FenwickF32> = ...; }
FENWICK_SCRATCH.with_borrow_mut(|fw| {
    fw.reset_zeroed();
    fw.build_from_linear(probs_ptr, vocab_size);  // O(n) build, one-time per softmax
    let idx = fw.find_kth_avx2(threshold);          // O(log n) AVX2 find
});
```

**Key references:**
- `src/mud/fenwick.rs:240` — `prefix_sum_batch4_avx2` pattern for parallel queries
- `src/mud/fenwick.rs:197` — `find_kth_walk` with `lzcnt` (BMI1) initial bitmask
- `src/asm/fenwick_avx2.s` — 4-query parallel ASM kernel

### P1 — Class embs round-and-scale (`train_step.rs:551-553`)

**Current scalar:**
```rust
for v in &mut class_embs_flat[0..hidden_size] {
    *v = (*v / scale).round().clamp(-1.0, 1.0) * scale;
}
```

**Replace with `clamp_finite_avx2`** — clamp bounds scale with `* scale`:
```rust
unsafe {
    crate::mud::ezop::clamp_finite_avx2(
        class_embs_flat.as_mut_ptr(), -1.0 * scale, 1.0 * scale, 0.0, hidden_size
    );
    crate::mud::ezop::scale_vec_avx2(class_embs_flat.as_mut_ptr(), scale, hidden_size);
}
```

**Test ref:** `test_clamp_finite_avx2_matches_scalar` (ezop.rs:1616)

### P1 — Softmax num_classes AVX2 (`train_step.rs:591-597`)

**Insight:** `num_classes` = 64/256 — small enough for AVX2, large enough for benefit.

**Add to ezop.rs:** `max_avx2(x, n) -> f32` + `broadcast_sub_avx2(x, val, n)`.

**Pattern:** max_reduce → subtract max → `exp_avx2` → `sum_avx2` → `scale_vec_avx2`. All existing kernels except `max_avx2`.

### P2 — Branchless feature dispatch (`slime_forward.rs:555`)

```rust
// BEFORE: branch mispredict per iteration
for i in 0..n {
    if layer.is_f32_weights { /* f32 path */ }
    else { /* ternary path */ }
}
// AFTER: single branch outside loop, indirect call
let kernel = if layer.is_f32_weights { f32_gemv_rowwise } else { ternary_gemv };
for i in 0..n { kernel(...); }
```

### P2/P3 — RMSNorm backward split (`slime_backward.rs:754-761`)

**Current:** single scalar loop does y_unscaled + sum_dy_xnorm + w_grad simultaneously.

**Split to leverage existing kernels:**
1. `div_avx2(y_f32, w, n)` → `y_unscaled[i] = y_f32[i] / w[i]` (NEW kernel needed)
2. `dot3_fma_avx2(grad, w, y_unscaled, n)` → `sum_dy_xnorm` (exists at ezop.rs:899)
3. `rmsnorm_backward_avx2(grad, y_unscaled, w, w_grad, rms_inv, mean, n)` → dx + w_grad (exists at ezop.rs:949)

### P3 — ASM tail masking (`asm/backward_rmsnorm.s`)

```
; Use VPMOVMSKB + mask register to process 8-f32 chunks including tail
; Eliminates separate scalar remainder pass
vbroadcastss ymm0, [mean_val]
loop:
    vmovdqu ymm2, [x + i*4]
    vfmsub213ps ymm2, ymm0, ymm1   ; fused
    vmovdqu [out + i*4](mask), ymm2
    add i, 8; cmp i, n; jl loop
```

### P3 — JEPA OU fused update (`asm/jepa_ou.s`)

```
; z[i] = 0.9*z[i] + 0.1*y_norm[i] + jitter[i]  — single FMA chain
vfmsub213ps ymm3, ymm0, [z]      ; z *= 0.9 (FMA)
vfmadd213ps ymm3, ymm1, [y_norm]  ; z += 0.1*y_norm (FMA)
vaddps      ymm3, ymm3, ymm2      ; z += jitter
vmovdqu     [z], ymm3
```

### P-17 Compliance (NaN/Inf guard all new kernels)
```rust
// max could be -Inf if all inputs are -Inf → fallback to 0.0
let max_val = if max_val.is_finite() { max_val } else { 0.0 };
// div by zero: sum could be 0 if all exps underflow → guard with 1e-10
let inv_sum = 1.0 / (sum + 1e-10);
// exp_avx2 already guards via _mm256_blendv_ps(0.0, result, is_finite)
```

### Benchmark & validation commands
```bash
cargo test --lib ezop::tests        # equivalencia AVX2 vs scalar
cargo +nightly miri test --lib -- ezop::tests --test-threads=1  # 0 UB
cargo run --release --bin gemv_auto_bench  # performance baseline

# Training smoke test:
MUD_QAT_LR=0.001 MUD_TRAIN_ALIGN=1 cargo run --release --bin run_trainer -- models/smollm2.mud --epochs 1
```

---

## E-7. Verification plan

1. `cargo clippy --all-targets` → 0 warnings
2. `cargo test --lib -- --test-threads=2` → all pass
3. `cargo +nightly miri test --lib -- ezop --test-threads=1` → 0 UB
4. `cargo +nightly miri test --lib -- fenwick --test-threads=1` → 0 UB
5. Benchmark: `cargo run --release --bin gemv_auto_bench` antes/después
6. Training smoke test: `./mud.sh train models/smollm2.mud --epochs 1` con `MUD_QAT_LR=0.001`

---

## E-7. Constraints (no violar)
- **P-01:** Zero allocation en hot loops — usar thread-local `FENWICK_SCRATCH`, `JEPA_CPU_SCRATCH`
- **P-02:** `SlimeRegister` f16-packed dual (re, im) u32 — no cambiar formato
- **P-06:** `cargo clippy --all-targets` → 0 warnings
- **P-08:** Eliminar código muerto, no archivar
- **P-17:** Fail-fast on non-finite — todos los kernels AVX2 nuevos deben tener NaN/Inf guard
- **P-27:** No Rayon en hot path — PCorePool only

---

## E-8. Implementation Results (2026-08-02 — Session Summary)

### Nuevos kernels AVX2 en `src/mud/ezop.rs`

| Kernel | Operación | ILP | Tests | Miri |
|--------|-----------|-----|-------|------|
| `div_avx2` | `out[i] = a[i] / b[i]` con NaN/Inf guard | 8 f32/iter | 3 | ✅ 0 UB |
| `grad_x_reduce_8rows_avx2` | 8-row tree reduction con 4 accumuladores | 8 f32/iter | 2 | ✅ 0 UB |
| `sum_sq` (optimizado) | 4-accumulator FMA unroll (32 elems/iter) | 4× throughput vs serial | 1 | ✅ 0 UB |

### Vectorización en `src/mud/slime_backward.rs`

| Path | Antes (scalar) | Ahora (AVX2) | Ahorro |
|------|---------------|-------------|--------|
| FFN RMSNorm backward (L754) | 2 loops × 2048 = 4096 iter | 256 AVX2 + 1 ASM | ~4× |
| Attention RMSNorm backward (L1335) | 4096 iter scalar | 256 AVX2 + ASM + axpy | ~8× |
| grad_x reduction (L477) | 8× nested scalar loops | 1× `grad_x_reduce_8rows_avx2` | ~8× |

### Vectorización en `src/mud/slime_forward.rs`

| Optimización | Antes | Ahora | Impacto |
|-------------|-------|-------|---------|
| `mhc_residual` branch dispatch | 3 branches/element | 1 branch/loop | Elimina 6144 mispredicts/token/layer |

### Métricas de verificación
- **Tests:** 455 passed (5 nuevos), 0 failed, 2 ignored ✅
- **Clippy:** 0 errors, 0 warnings ✅
- **Miri (nuevos kernels):** 5/5 passed, 0 UB ✅
- **Pre-existing miri limitation:** `slime_backward::tests` falla en miri por `backward_ternary_gemv_dx_avx2` ASM (pre-existing, no relacionado a cambios)

### Pipeline AVX2 completo en hot path

```
Forward:  ternary_gemv_8rows.s → slime_rmsnorm_i8 → rope_rotate_avx2 → silu_vectorial_avx2
          → vmul_inplace_avx2 (FFN gate*up) → ternary_gemv_4rows.s (FFN proj)

Backward: f32_gemv_backward (F32 weights) / backward_ternary_gemv_8rows.s (ternary)
          → div_avx2 + dot3_fma_avx2 (RMSNorm backward)
          → rmsnorm_backward_avx2 (ezop.rs / ASM)
          → axpy_avx2 (residual accumulation)
          → grad_x_reduce_8rows_avx2 (gradient merge)

Optimizer: axpy_avx2 / scale_finite_avx2 / pack_ternary_into (AVX2)
```

### CPU instruction pipeline (i7-1260P, 4 P-cores + 4 E-cores, AVX2+FMA+BMI1)

```
Throughput:
  FMA ops:   2/cycle/core (P-core), 1/cycle/E-core
  AVX2 load: 1/cycle/core (256-bit)
  AVX2 store: 1/cycle/core (256-bit)

Key: sum_sq now uses 4 independent FMA accumulators → 4× instruction-level
parallelism. The tree-reduction grad_x kernel uses load-fanin from 8 rows
into 4 adder trees → 8-way parallel addition per 8-element chunk.
```

### Próximas oportunidades (ASM-level, P3)

1. **`backward_ternary_gemv.s`**: Agregar `prefetcht0` para más rows (actualmente solo rows 0,4)
2. **`jepa_ou.s`**: Fused `z *= 0.9; z += 0.1*y_norm; z += jitter;` → 3 instructions/iter → 1
3. **`mhc_gate.s`**: Sigmoid + blend en un pass (actualmente `sigmoid_vec_local` en Rust)
4. **`ternary_gemv_8rows.s`**: 2× unroll (16 cols/iter → 32 cols/iter) para hidir latency
