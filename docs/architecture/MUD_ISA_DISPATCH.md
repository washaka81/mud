# ISA Dispatch Framework

## Overview

Every ASM kernel in the MUD engine has an **ISA dispatch wrapper** that routes at runtime to the optimal implementation based on CPU capabilities. This ensures:

- **Portability**: runs on ARM, x86-64 w/o AVX2, or any CPU
- **Correctness by default**: scalar fallbacks are pure Rust, trivially correct
- **Zero overhead on modern x86**: `is_x86_feature_detected!` compiles to a single static bit-check (~1 cycle)
- **Debug safety**: all wrappers include `debug_assert!` for null/non-finite inputs

## Architecture

```
Call site                    Dispatch wrapper                ASM / Fallback
───────────                  ────────────────                ──────────────
pub unsafe fn foo(n, ...)    if avx2_available() {           foo_avx2(n, ...)
─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ▶      foo_avx2(n, ...)        ─ ▶   (assembly)
                             } else {
                                 foo_scalar(n, ...)      ─ ▶   (pure Rust)
                             }
```

## ISA Detection

Defined in `src/asm/mod.rs`:

```rust
#[cfg(target_arch = "x86_64")]
fn avx2_available() -> bool { is_x86_feature_detected!("avx2") }

#[cfg(not(target_arch = "x86_64"))]
fn avx2_available() -> bool { false }

#[cfg(target_arch = "x86_64")]
fn bmi2_available() -> bool { is_x86_feature_detected!("bmi2") }

#[cfg(not(target_arch = "x86_64"))]
fn bmi2_available() -> bool { false }
```

Two gates exist:
| Gate | Checks | Used by |
|------|--------|---------|
| `avx2_available()` | `is_x86_feature_detected!("avx2")` | 15 of 16 kernels |
| `bmi2_available()` | `is_x86_feature_detected!("bmi2")` | `unpack_ternary` (pext/pdep) |

Both return `false` on non-x86_64 (ARM, etc.), forcing scalar fallback.

## Complete Kernel Table

All 16 ASM functions, their dispatch wrapper, ISA gate, scalar fallback, and source files.

| # | ASM Function | Dispatch Wrapper | ISA Gate | Scalar | ASM File | Call Sites |
|---|-------------|-----------------|----------|--------|----------|------------|
| 1 | `rms_norm_scale_asm` | `rms_norm_scale` | AVX2 | ✓ | `rmsnorm.s` | `forward.rs`, `q4_0_gemv_fused` |
| 2 | `ternary_gemv_avx2` | `ternary_gemv` | AVX2 | ✓ | `ternary_gemv.s` | `forward.rs`, `trainer.rs`, `tests` |
| 3 | `ternary_gemv_4rows_avx2` | `ternary_gemv_4rows` | AVX2 | ✓ | `ternary_gemv_4rows.s` | `vulkan_backend.rs`, `tests` |
| 4 | `ternary_gemm_batch4_avx2` | `ternary_gemm_batch4` | AVX2 | ✓ | `ternary_gemm_batch4.s` | `forward.rs` |
| 5 | `dot_product_avx2` | `dot_product` | AVX2 | ✓ | `math.s` | `forward.rs`, `trainer.rs`, `tests` |
| 6 | `sum_squares_avx2` | `sum_squares` | AVX2 | ✓ | `math.s` | `forward.rs`, `rms_norm_scale`, `tests` |
| 7 | `peak_abs_avx2` | `peak_abs` | AVX2 | ✓ | `math.s` | `corpus_trainer.rs` |
| 8 | `apply_gradient_avx2` | `apply_gradient` | AVX2 | ✓ | `math.s` | `corpus_trainer.rs` |
| 9 | `hadamard_transform_avx2` | `hadamard_transform` | AVX2 | ✓ | `math.s` | `forward.rs` |
| 10 | `silu_vectorial_avx2` | `silu_vectorial` | AVX2 | ✓ | `silu.s` | `forward.rs` |
| 11 | `apply_rope_asm` | `apply_rope` | AVX2 | ✓ | `rope.s` | `forward.rs` |
| 12 | `mamba_scan_avx2` | `mamba_scan` | AVX2 | ✓ | `mamba.s` | `forward.rs`, `jamba_benchmark.rs` |
| 13 | `mamba_delta_fold_avx2` | `mamba_delta_fold` | AVX2 | ✓ | `mamba.s` | `inference.rs` |
| 14 | `q4_0_gemv_asm` | `q4_0_gemv` | AVX2 | ✓ | `q4_0_gemv.s` | `inference.rs`, `transformer.rs`, `q4_0_gemv_fused` |
| 15 | `pext_unpack_ternary` | `unpack_ternary` | BMI2 | ✓ | `ternary_pext.s` | `tests.rs` |
| 16 | `ternary_gemv_lut_avx2` | `ternary_gemv_lut` | AVX2 | ✓ | `ternary_lut.s` | `tests.rs` |

## Wrapper ABI Signatures

### Standard FP32 Kernels

```rust
// RMSNorm: computes 1/sqrt(mean(x²) + eps)
pub unsafe fn rms_norm_scale(n: usize, x: *const f32, eps: f32) -> f32;

// Ternary GEMV: out = sum(x[i] * w[i]) * scale   (w ∈ {−1,0,1} packed 2-bit)
pub unsafe fn ternary_gemv(n: usize, x: *const f32, weights: *const u32, out: *mut f32, scale: f32);

// 4-row ternary GEMV: 4 outputs sharing activation load
pub unsafe fn ternary_gemv_4rows(n: usize, x: *const f32, weights: *const u32, out: *mut f32, scale: f32, stride: usize);

// Batch-4 GEMM: tokens × weights → 4×out_dim output matrix
pub unsafe fn ternary_gemm_batch4(out_dim: usize, in_dim: usize, x_ptr: *const f32, w_ptr: *const u32, out_ptr: *mut f32, scales: *const f32);

// Dot product: ∑ a[i]·b[i]
pub unsafe fn dot_product(n: usize, a: *const f32, b: *const f32) -> f32;

// Sum of squares: ∑ x[i]²
pub unsafe fn sum_squares(n: usize, x: *const f32) -> f32;

// Peak absolute value: max(|x[i]|)
pub unsafe fn peak_abs(n: usize, x: *const f32) -> f32;

// SGD update: w = w·(1−decay) + α·g, clamped to [−5, 5]
pub unsafe fn apply_gradient(n: usize, weight: *mut f32, grad: *const f32, alpha: f32, decay: f32);

// In-place Walsh-Hadamard transform (n must be power of 2)
pub unsafe fn hadamard_transform(n: usize, x: *mut f32);

// SiLU activation: x / (1 + e⁻ˣ)
pub unsafe fn silu_vectorial(n: usize, src: *const f32, dst: *mut f32);

// Split RoPE: x[i] = a·c−b·s, x[i+½n] = a·s+b·c   (i in 0..½n)
pub unsafe fn apply_rope(n: usize, x: *mut f32, cos: *const f32, sin: *const f32);

// Mamba SSM scan: hⱼ = hⱼ·ā + x·b̄, out += hⱼ·cⱼ
pub unsafe fn mamba_scan(n: usize, d_state: usize, x: *const f32, a_bar: *const f32, b_bar: *const f32, c: *const f32, state: *mut f32, out: *mut f32);

// State decay: state *= decay
pub unsafe fn mamba_delta_fold(len: usize, state: *mut f32, decay: f32);
```

### Special-Purpose Kernels

```rust
// Q4_0 GEMV: fused dequantize (GGML format) + dot product
pub unsafe fn q4_0_gemv(n: usize, x: *const f32, weights: *const BlockQ4_0, out: *mut f32);

// Ternary unpack: 64-bit packed → 32 i8 values (−1, 0, 1)
pub unsafe fn unpack_ternary(packed: u64, out: *mut i8);

// Int8 ternary LUT GEMV: pre-quantized activations × unpacked i8 weights
pub unsafe fn ternary_gemv_lut(n: usize, x: *const i8, weights: *const i8, out: *mut f32, scale: f32);
```

## Scalar Fallback Implementations

All fallbacks live in `src/asm/mod.rs` as private `unsafe fn X_scalar(...)` functions.
They are plain Rust loops with no SIMD intrinsics or `target_feature` annotations,
guaranteeing they compile on any CPU architecture.

Key implementation notes:

| Kernel | Algorithm | Complexity |
|--------|-----------|------------|
| `ternary_gemv_scalar` | Shift/mask on each u32 to extract 2-bit fields 0/1/2 → 0/+1/−1 | O(n/16 × 16) |
| `ternary_gemv_4rows_scalar` | Same unpack × 4 rows, `stride` controls row pitch | O(4·stride·16) |
| `ternary_gemm_batch4_scalar` | Double loop: tokens (4) × rows (out_dim) × columns (in_dim/16 × 16) | O(4·out_dim·in_dim) |
| `apply_rope_scalar` | `a·c ± b·s` per element | O(½n) |
| `mamba_scan_scalar` | State update + output dot per channel | O(n·d_state) |
| `mamba_delta_fold_scalar` | `state[i] *= decay` | O(len) |
| `apply_gradient_scalar` | FMA + `clamp(−5, 5)` | O(n) |
| `peak_abs_scalar` | `max(|x[i]|)` | O(n) |
| `hadamard_transform_scalar` | Iterative butterfly: `{a+b, a−b}` | O(n·log₂n) |
| `silu_vectorial_scalar` | `x / (1 + e⁻ˣ)` | O(n) |
| `q4_0_gemv_scalar` | Q4_0 dequant: `(nibble−8)·d`, accumulate | O(n) |
| `pext_unpack_scalar` | `out[i] = (low_bit − hi_bit) as i8` | O(32) |
| `ternary_gemv_lut_scalar` | i8×i8→i32 dot product → f32 scale | O(n) |

## NaN/Inf Validation

All dispatch wrappers include `debug_assert!` guards:

```rust
debug_assert!(!x.is_null(), "wrapper: x is null");
debug_assert!(n > 0, "wrapper: n must be > 0");
debug_assert!(scale.is_finite(), "wrapper: scale must be finite");
```

Additional utility functions for debugging:

```rust
// Returns Some(index) of first NaN/Inf, or None
pub unsafe fn check_nan_slice(ptr: *const f32, n: usize) -> Option<usize>;

// Zeroes out NaN/Inf in-place, returns count sanitized
pub unsafe fn sanitize_finite(ptr: *mut f32, n: usize) -> usize;
```

## How to Add a New Kernel

1. **Write ASM function** → `src/asm/foo.s` with global label `foo_avx2`
2. **Add extern declaration** in `src/asm/mod.rs`:
   ```rust
   extern "C" { pub fn foo_avx2(n: usize, x: *const f32, out: *mut f32); }
   ```
3. **Write scalar fallback** in the same file:
   ```rust
   unsafe fn foo_scalar(n: usize, x: *const f32, out: *mut f32) { ... }
   ```
4. **Write dispatch wrapper**:
   ```rust
   pub unsafe fn foo(n: usize, x: *const f32, out: *mut f32) {
       debug_assert!(!x.is_null()); debug_assert!(n > 0);
       if avx2_available() { foo_avx2(n, x, out); } else { foo_scalar(n, x, out); }
   }
   ```
5. **Update call sites** → replace `crate::asm::foo_avx2(...)` with `crate::asm::foo(...)`

## Benchmarking

The `kernel_bench` tool (`tools/kernel_bench.rs`) measures throughput at sizes 128–32768
for each kernel, comparing dispatch wrapper vs direct AVX2 call.

```sh
cargo run --release --bin kernel_bench
```

Baseline throughput at n=32768 on Alder Lake i7-1260p:
| Kernel | Bandwidth | Operations |
|--------|-----------|------------|
| `sum_squares` | 72.9 GB/s | 18.2 Gop/s |
| `dot_product` | 60.9 GB/s | 15.2 Gop/s |
| `ternary_gemv` | 40.4 GB/s | 10.1 Gop/s |
| `silu_vectorial` | 9.4 GB/s | 2.4 Gop/s |
| `rms_norm_scale` | 57.3 GB/s | 14.3 Gop/s |

## Performance Notes

- Dispatch overhead is **negative** on this machine: the Rust dispatch wrapper with `#[inline(always)]` frequently outperforms the direct `extern "C"` call because the compiler optimizes through the wrapper boundary. AVX2 code is identical in both paths.
- Scalar fallbacks are **never called** on AVX2-capable CPUs. They exist only for correctness on other architectures.
- The `pext_unpack_ternary` ASM uses BMI2 (`pext`/`pdep`). The dispatch wrapper `unpack_ternary` uses a separate `bmi2_available()` gate. On CPUs without BMI2 (e.g., older AMD, early Intel), the scalar fallback runs.
