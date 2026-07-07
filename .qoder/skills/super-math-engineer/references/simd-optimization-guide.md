# SIMD Optimization Guide: AVX2 Kernels in MUD

This guide details the technical standards for writing and auditing x86_64 assembly code in `src/asm/`.

## 1. AVX2 Register Layout for Ternary Operations

Ternary weights in MUD are stored in packed formats (e.g., 2 bits per weight). During the forward GEMV/GEMM pass, we unpack these weights and multiply them by activations (`f32`).

- **YMM0 - YMM7:** Reserved for accumulator registers to hold intermediate floating-point sums.
- **YMM8 - YMM11:** Reserved for activations (loaded sequentially or broadcasted).
- **YMM12 - YMM15:** Reserved for ternary weights (packed or unpacked).
- **YMM15:** Commonly used to hold the mask for ternary scaling or bitmask operations.

## 2. Branchless Ternary Multiplexing

To maintain high pipeline efficiency, avoid conditional branches like `jmp` when processing ternary states `{-1, 0, 1}`. Instead, use masked additions and subtractions.

### Example Sequence for Ternary GEMV:
```assembly
# ymm0: accumulator, ymm1: activations, ymm2: weight flags
# Unpack weight flags: bit 0 = sign (1 for negative), bit 1 = mask (1 for non-zero)

# 1. Generate positive mask
vpand ymm3, ymm2, ymm_non_zero_mask
# 2. Generate sign mask
vpand ymm4, ymm2, ymm_sign_mask

# 3. Add positive activations where mask is set
vaddps ymm0, ymm0, ymm1   # (Conditional addition or blend)
```
Always use `vpblendvb` or `vpand`/`vpandn` to mask operations rather than conditional jumps.

## 3. Cache Line Alignment (64 bytes)

All tensor rows and active work buffers should be aligned to **64 bytes** to prevent performance penalties from cache line splits.
- In Rust, use `#[repr(align(64))]` on struct arrays where possible.
- In Assembly, use `.align 64` before function entry points.
- Ensure loop counters are decremented in steps matching the vector width (e.g., steps of 8 for `f32` vectors in YMM registers).
