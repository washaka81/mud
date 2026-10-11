# ASM Kernel Documentation

This document describes the AVX2 assembly kernels in `src/asm/` and their algorithms.

## File Index

| File | Algorithm | Hot Path | Lines |
|---|---|---|---|
| `src/asm/ternary_gemv.s` | Ternary GEMV (8 weights/u32, ELUT 4-bit, AVX2) | QKV/FFN GEMV | ~298 |
| `src/asm/ternary_gemv_4rows.s` | 4-row batched ternary GEMV (shared activation) | Forward GEMV | ~290 |
| `src/asm/ternary_gemv_8rows.s` | 8-row batched ternary GEMV | Forward GEMV (large) | ~507 |
| `src/asm/ternary_gemm_batch4.s` | 4-element batch GEMM | GEMV batch | ~234 |
| `src/asm/adam_step.s` | Adam optimizer step (AVX2 + FMA) | Optimizer step | ~153 |
| `src/asm/apply_scales.s` | Apply PRQ scales to f32 accumulators | Post-GEMV scaling | ~71 |
| `src/asm/rmsnorm.s` | RMSNorm (vectorized, rsqrtps) | Pre-attention norm | ~81 |
| `src/asm/rope.s` | RoPE rotation (precomputed cos/sin) | Position encoding | ~64 |
| `src/asm/silu.s` | Vectorized SiLU (Swish) activation | FFN gate | ~194 |
| `src/asm/lm_head.s` | AVX2 LM head logits (vocab scan) | Output projection | ~226 |
| `src/asm/sgemm.s` | Single-precision GEMM (AVX2) | Dense GEMM | ~248 |
| `src/asm/pack.s` | Pack f32 → ELUT u32 (8:1) | Optimizer step | ~232 |
| `src/asm/q4_0_gemv.s` | Q4_0 (GGUF) GEMV | Q4_0 inference | ~102 |
| `src/asm/math.s` | Math constants + helpers | Various | ~313 |
| `src/asm/cmud_batch.s` | AVX2 vectorized CMUD complex multiply | C-MUD thinking | ~53 |
| `src/asm/slimex_cell_v5_avx2.s` | CellV5 u32 unpack (8:8:8:8) | SlimeX memory | ~131 |

## Algorithm Details

### 1. Ternary GEMV (`ternary_gemv.s`)

**Purpose:** Compute `y = W * x` where W is ternary2bit (8 weights per u32, ELUT encoding).

**ELUT encoding:**
- `0x1` → +1.0
- `0xF` → -1.0
- `0x0` → 0.0

**Algorithm:**
1. Load 8 weights from packed u32 (one per nibble)
2. Load 8 input values from x
3. For each weight: if +1, add x; if -1, subtract x; if 0, skip
4. Accumulate into f32 accumulator
5. Apply PRQ scale: `y[i] *= scale[i]`

**AVX2 optimization:**
- Uses `_mm256_i32gather_ps` for indexed gather of x values
- Branchless: precompute sign mask from nibble values
- 8 weights processed per u32 load
- 64 elements per unrolled iteration (8 accumulators)

### 2. 4-Row Batched Ternary GEMV (`ternary_gemv_4rows.s`)

**Purpose:** Compute 4 rows of `y = W * x` simultaneously, sharing the activation load.

**Algorithm:**
1. Load 8 activation values (shared across 4 rows)
2. For each of 4 rows:
   - Load 8 packed weights (u32)
   - Unpack nibbles → sign masks
   - Multiply-accumulate into 4 separate f32 accumulators
3. 5 simultaneous prefetches for next iteration

**AVX2 optimization:**
- Single activation load for 4 rows (4× bandwidth reduction)
- 14 YMM registers (no spill)
- 5 prefetch streams

### 3. 8-Row Batched Ternary GEMV (`ternary_gemv_8rows.s`)

**Purpose:** Compute 8 rows simultaneously for large GEMV operations.

**Algorithm:**
- Extension of 4-row batch to 8 rows
- 16 YMM registers for accumulators
- Shared activation tiling across 8 rows

### 4. Batch4 GEMM (`ternary_gemm_batch4.s`)

**Purpose:** 4-element batch GEMM for small matrix operations.

**Algorithm:**
- Processes 4 independent GEMV operations in batch
- Uses shared tile for activations
- AVX2 with 8-element vectors

### 5. Adam Optimizer Step (`adam_step.s`)

**Purpose:** Adam optimizer update with weight decay and gradient clipping.

**Algorithm:**
```
m = b1 * m + (1 - b1) * grad
v = b2 * v + (1 - b2) * grad^2
m_hat = m / (1 - b1^t)
v_hat = v / (1 - b2^t)
w = w - lr * (m_hat / (sqrt(v_hat) + eps) + wd * w)
```

**AVX2 optimization:**
- 16/16 YMM registers (no spill)
- FMA for fused multiply-add
- All 10 parameters passed in registers (System V AMD64)

**Arguments (System V AMD64):**
- `rdi`: n, `rsi`: w, `rdx`: m, `rcx`: v, `r8`: grads
- `xmm0`: clip_coef, `xmm1`: wd, `xmm2`: b1, `xmm3`: b2
- `xmm4`: lr_bc1, `xmm5`: inv_bc2, `xmm6`: eps

### 6. Apply PRQ Scales (`apply_scales.s`)

**Purpose:** Apply per-row quantization (PRQ) scales to f32 accumulators after ternary GEMV.

**Algorithm:**
1. Load 8 f32 accumulators
2. Load 8 f32 scales
3. `acc[i] *= scale[i]` (element-wise multiply)
4. Store 8 results

**AVX2 optimization:**
- `vmulps` for element-wise multiply
- 8 elements per iteration
- NaN guard: non-finite results → 0.0

### 7. RMSNorm (`rmsnorm.s`)

**Purpose:** Compute `y = x / sqrt(mean(x^2) + eps)`

**Algorithm:**
1. Compute sum of squares: `Σ x[i]^2` (AVX2 horizontal sum)
2. Compute mean: `sum_sq / n`
3. Compute scale: `1 / sqrt(mean + eps)`
4. Scale: `y[i] = x[i] * scale`

**AVX2 optimization:**
- 8 floats per iteration
- Horizontal sum via `vhaddps`
- `vrsqrtps` for fast inverse square root

### 8. RoPE (`rope.s`)

**Purpose:** Apply rotary position embedding to Q/K vectors.

**Algorithm:**
1. For each position pos and head dim i:
   - `theta = pos * inv_freq[i]` (precomputed, no powf)
   - `cos = cos(theta)`, `sin = sin(theta)`
   - `q[2i] = q[2i] * cos - q[2i+1] * sin`
   - `q[2i+1] = q[2i+1] * cos + q[2i] * sin`

**AVX2 optimization:**
- Precomputed `inv_freq` array (no powf in hot path)
- `cos`/`sin` cached per position
- 8-element batch processing

### 9. SiLU (`silu.s`)

**Purpose:** Compute `f(x) = x * sigmoid(x) = x / (1 + exp(-x))`

**Algorithm:**
1. Compute `-x` (negation)
2. Compute `exp(-x)` (via `exp2` + scaling)
3. Compute `1 + exp(-x)`
4. Compute `x / (1 + exp(-x))`

**AVX2 optimization:**
- 8 floats per iteration
- Polynomial approximation for `exp2ps`
- NaN guard: non-finite inputs → 0.0

### 10. LM Head Logits (`lm_head.s`)

**Purpose:** Compute `logits[v] = Σ_h reg[h] * W[v,h]` for all vocab tokens.

**Algorithm:**
1. For each token v in vocab:
   - Load 8 reg values (AVX2)
   - Load 8 weight values (from tied embedding or dequant output.weight)
   - FMA: `acc += reg * weight`
2. Store logits[v]

**AVX2 optimization:**
- Processes 8 hidden elements per iteration
- Uses `_mm256_fmadd_ps` for fused multiply-add
- Outer loop over vocab (not vectorized — too large for L1)

### 11. Single-Precision GEMM (`sgemm.s`)

**Purpose:** Dense single-precision matrix multiplication for non-ternary paths.

**Algorithm:**
- Standard GEMM: `C = α * A * B^T + C`
- AVX2 with 8-element vectors
- Used for dense weight paths (e.g., F32 converted models)

### 12. Pack Ternary (`pack.s`)

**Purpose:** Pack f32 values into ELUT u32 (8:1 compression).

**Algorithm:**
1. For each group of 8 f32 values:
   - Quantize each to {-1, 0, +1}
   - Pack 8 nibbles into one u32
2. Store packed u32

**AVX2 optimization:**
- Branchless quantization via comparison masks
- 8 values per u32
- Used in optimizer step (STE QAT)

### 13. Q4_0 GEMV (`q4_0_gemv.s`)

**Purpose:** Compute GEMV for GGUF Q4_0 quantized weights (4-bit blocks).

**Algorithm:**
1. Load 16 4-bit weights from a single block_q4_0
2. Dequantize using block scale (f32)
3. Multiply-accumulate with activations
4. Store f32 result

**AVX2 optimization:**
- `vpsrlw` / `vpand` for nibble extraction
- Block scale broadcast via `vbroadcastss`

### 14. Math Helpers (`math.s`)

**Purpose:** Math constants and helper functions used across kernels.

**Contents:**
- IEEE 754 constants (pi, e, sqrt(2), etc.)
- Polynomial coefficients for `exp2ps` approximation
- Lookup tables for transcendental functions

### 15. CMUD Batch (`cmud_batch.s`)

**Purpose:** AVX2 vectorized CMUD complex multiply (8 elements at once).

**Algorithm:**
```
void cmud_gauss_mul_batch_avx2(
    const float* x_re, const float* x_im,
    const float* w_re, const float* w_im,
    float* y_re, float* y_im
);

For each of 8 elements:
  Y_R = X_R * W_R - X_I * W_I
  Y_I = X_R * W_I + X_I * W_R
```

**AVX2 optimization:**
- Uses 4 `vmulps` + 1 `vaddps` + 1 `vsubps` per batch (6 instructions for 8 elements)

### 16. CellV5 Unpack (`slimex_cell_v5_avx2.s`)

**Purpose:** Unpack u32 (8:8:8:8 bit allocation) to 4 f32 values.

**Algorithm:**
1. Load u32 word
2. Extract 4 bytes: `(w >> 0) & 0xFF`, `(w >> 8) & 0xFF`, etc.
3. Convert to f32 via lookup table or scaling
4. Store 4 f32 values

**AVX2 optimization:**
- `vpmovzxbd` for zero-extension of 4 bytes to 4x u32
- Single cycle per u32 → 4 f32

## Safety

Every `unsafe` block in the calling Rust code has a `// SAFETY:` comment explaining:
- What invariants the caller must uphold
- Why the pointer operations are valid
- Alignment requirements

## Testing

- `test_ternary_gemv_asm_vs_scalar_random`: ASM vs scalar parity
- `test_ternary_gemv_asm_accumulate`: Accumulation correctness
- `test_ternary_gemv_4rows_consistency`: 4-row batch consistency
- `test_ternary_gemv_8rows_consistency`: 8-row batch consistency
- `test_asm_unpack_cells_v5_matches_scalar`: CellV5 ASM vs scalar
- `test_adam_step_correctness`: Adam step ASM vs scalar
- `test_rmsnorm_asm_matches_scalar`: RMSNorm ASM vs scalar
- `test_silu_asm_matches_scalar`: SiLU ASM vs scalar
