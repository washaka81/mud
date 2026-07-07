# Forge LLM (MUD) - Session Report
## Date: 13 de junio de 2026

### Focus: Kernel Validation and Policy Audit

#### Overview
A comprehensive audit and validation of the AVX2 arithmetic kernels within `src/asm/` was performed to ensure strict adherence to the project's architectural mandates, particularly memory safety, 0-error/0-warning compilation, and Per-Row Quantization (PRQ) high-fidelity scaling.

#### Accomplishments & Critical Fixes

1.  **Memory Safety Fix (SIGSEGV Resolution):**
    -   **Issue:** The test suite encountered a segmentation fault (`SIGSEGV`) when running the `bench_int8_vs_f32_throughput` test.
    -   **Root Cause:** The AVX2 kernel `ternary_gemv_i8act_avx2` had a malformed epilogue. While the prologue correctly saved callee-saved registers (`%rbx`, `%r12`-`%r15`) and aligned the stack, the epilogue failed to pop them and restore the stack pointer.
    -   **Resolution:** Re-wrote the kernel's epilogue to sequentially `add $8, %rsp` and `pop` all callee-saved registers in reverse order.

2.  **Gradient Sanitization (NaN Propagation Fix):**
    -   **Issue:** Three test cases (`test_ternary_gemv_i8act_correctness`, `test_ternary_gemv_i8act_tail_non_multiple_of_16`, `test_ternary_gemv_i8act_vs_f32_parity`) panicked due to returning `NaN` instead of expected numerical values.
    -   **Root Cause:** The scalar tail accumulator was mapped to `%xmm3`. In the main `.loop_i8`, `%xmm3` was repurposed to hold a 32-bit bitmask (`vpcmpeqd VAL_ONE(%rip), %ymm2, %ymm3`). The bitmask equivalent of all 1s (`0xFFFFFFFF`) represents `NaN` in floating-point format. When the tail was added during the horizontal reduction, the `NaN` polluted the entire scalar sum.
    -   **Resolution:** Migrated the scalar tail accumulator to a pristine register, `%xmm15`, across all tail-processing loops (`.scalar_tail_i8`, `.tail_plus1_i8`, `.tail_minus1_i8`, `.done_accum_i8`).

3.  **High-Fidelity INT8 Scaling (PRQ Validation):**
    -   **Issue:** In hybrid kernels, the hardware-equivalent scaling was mismatched because the `act_scale` argument was being ignored.
    -   **Resolution:** Modified `ternary_gemv_i8act_4rows_avx2` and `ternary_gemv_i8act_avx2` to multiply the weight scale (`%xmm0`) by the activation scale (`%xmm1`) using `vmulss %xmm1, %xmm0, %xmm0` immediately in the prologue before broadcasting the combined scale (`vbroadcastss`).

#### Final Technical State
- **Audit Verification:** `cargo clippy -- -D warnings` completed with **0 Errors, 0 Warnings**.
- **Test Integrity:** `cargo test` successfully passed all 76 unit tests.
- **Performance:** INT8 ternary matrix-vector multiplication is completely stable and mathematically sound for both block-level and scalar-tail workloads.