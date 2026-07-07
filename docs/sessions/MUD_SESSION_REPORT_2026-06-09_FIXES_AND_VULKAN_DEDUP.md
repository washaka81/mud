# MUD Session Report: June 9, 2026 (Part 2)
## Theme: Kernel Bug Fixes, Vulkan Dedup & ISA Dispatch Complete

### 1. `ternary_gemm_batch4.s` — Three Bugs Found & Fixed

**Root cause:** Stack-slot corruption across row iterations.

| Bug | Description | Fix |
|-----|-------------|-----|
| 1. Token 2/3 push order | `push r12` then `push r13` → Token 3 on top, but Token 2 reads from top slot | Replaced stack-slot pointer management with indexed addressing `(%r12, %rbp, 4)` using `rbp` as float-index offset |
| 2. r13 overwrite | `pop %r13` restored Token 3 base, then `mov %r14, %r13` destroyed it for next row | r13 never modified — no push/pop inside row loop needed |
| 3. Stack slot mutation | `mov %r13, 0(%rsp)` write-backs advanced saved pointers to past-end → `pop` restored wrong address | Eliminated entirely — r12/r13 hold original bases, `rbp` tracks column index |

**Result:** 64/64 tests pass (2 new batch4 tests). `cargo check` clean. ✅

### 2. Vulkan Shader Dedup (Phase 2)

**Before:** 3 almost-identical GEMV shaders, only 1 actually loaded:
- `ternary_gemv.comp` (dead code — had do_norm/do_rope)
- `ternary_gemv_igpu.comp` (loaded — std430, no norm/rope)
- `ternary_gemv_zero_copy.comp` (dead code — single scale)

**After:** Single `ternary_gemv_unified.comp` with push-constant flags:
- `do_norm` — RMSNorm fusion (from dead ternary_gemv.comp)
- `do_rope` — RoPE fusion stub (from dead ternary_gemv.comp)
- `single_scale` — uniform scale mode (from dead ternary_gemv_zero_copy.comp)

**Files deleted:** 3 old `.comp` files. `Rust` path updated in `vulkan/mod.rs:1112`. All 5 `PushConstants` call sites updated with new fields. ✅

### 3. ISA Dispatch Wrappers for All 16 ASM Kernels (Phase 3)

**Problem:** 10 of 16 ASM kernels lacked dispatch wrappers. Calling `extern "C"` ASM directly crashes on ARM, AMD64 w/o AVX2, etc.

**Before:** Only 6 dispatched (`ternary_gemv`, `dot_product`, `sum_squares`, `rms_norm_scale`, `silu_vectorial`, `ternary_gemm_batch4`).
**After:** All 16 dispatched with pure-Rust scalar fallbacks.

| Kernel | Dispatch Wrapper | ISA Gate | Weight Location |
|--------|-----------------|----------|----------------|
| `ternary_gemv_4rows` | `ternary_gemv_4rows` | `avx2` | `src/asm/ternary_gemv_4rows.s` |
| `apply_rope` | `apply_rope` | `avx2` | `src/asm/rope.s` |
| `mamba_scan` | `mamba_scan` | `avx2` | `src/asm/mamba.s` |
| `mamba_delta_fold` | `mamba_delta_fold` | `avx2` | `src/asm/mamba.s` |
| `apply_gradient` | `apply_gradient` | `avx2` | `src/asm/math.s` |
| `peak_abs` | `peak_abs` | `avx2` | `src/asm/math.s` |
| `hadamard_transform` | `hadamard_transform` | `avx2` | `src/asm/math.s` |
| `q4_0_gemv` | `q4_0_gemv` | `avx2` | `src/asm/q4_0_gemv.s` |
| `pext_unpack_ternary` | `unpack_ternary` | `bmi2` | `src/asm/ternary_pext.s` |
| `ternary_gemv_lut` | `ternary_gemv_lut` | `avx2` | `src/asm/ternary_lut.s` |

**Scalar fallbacks implemented:**
- `ternary_gemv_4rows`: 4 independent GEMV loops over `stride` u32 blocks
- `apply_rope`: Split RoPE — `x[i]=a·c−b·s`, `x[i+½]=a·s+b·c`
- `mamba_scan`: SSM scan — `hⱼ=hⱼ·ā+ x·b̄`, `out+=hⱼ·cⱼ`
- `mamba_delta_fold`: `state[i] *= decay`
- `apply_gradient`: `w = w·(1−decay) + α·g`, clamped to `[-5, 5]`
- `peak_abs`: `max(|xᵢ|)`
- `hadamard_transform`: Iterative in-place FWHT
- `q4_0_gemv`: Q4_0 dequant `(nibble−8)·d` fused with FP32 dot
- `pext_unpack_scalar` → `unpack_ternary`: 2-bit unpack via bit ops
- `ternary_gemv_lut`: i8×i8 dot product → f32 scale

**ISA detection:** `bmi2_available()` added alongside existing `avx2_available()`.

**Call sites updated:** 27 call sites across 9 source files. ✅

### 4. `kernel_bench` — `[[bin]]` entry in `Cargo.toml`

Added `[[bin]]` entry for `kernel_bench` + `tensor_health` in `Cargo.toml`.

Baseline throughput (dispatch, n=32768):
- `sum_squares`: 72.89 GB/s
- `dot_product`: 60.93 GB/s
- `ternary_gemv`: 40.42 GB/s
- `silu_vectorial`: 9.42 GB/s
- `rms_norm_scale`: 57.26 GB/s

### 5. Build Status

```
cargo check              ✅  0 errors, 0 warnings
cargo test               ✅  64/64 passed, 0 failed
cargo run --release --bin kernel_bench  ✅
```

### 6. RRM/LDT Loop Fix — Brace Imbalance & Missing `while`

**Problem:** After applying RRM edits (max_ldt_iterations formula, lattice_levels, sigma ε-jitter) to the MoE section in `src/mud/forward.rs`, the edit tool dropped the `while` loop header together with `let vk = ...` and `let ffn_hidden = ...` declarations. This caused:

- **Brace cascade:** The `}` at line 657 that closed the `while` body now closed the `else` branch instead, shifting brace depth by −1 for the rest of the file. Final depth was −1 (underflow at line 1575).
- **Lost loop:** The MoE LDT iteration block ran unconditionally (single pass). Variable references `vk` and `ffn_hidden` were undefined.
- **Misleading compiler error:** Rust reported `unexpected closing delimiter '}'` at line 1575, pointing to the wrong location.

**Fix at `forward.rs:456`:**
```rust
                    // BEFORE (edit tool output — broken):
                    let lattice_levels = [3.0f32, 5.0, 7.0, 7.0, 7.0, 7.0];
                        ws.combined_expert_out.write().fill(0.0);

                    // AFTER (re-inserted):
                    let lattice_levels = [3.0f32, 5.0, 7.0, 7.0, 7.0, 7.0];

                    let vk = self.vulkan_ctx.as_deref();
                    let ffn_hidden = self.model.ffn_hidden_size;

                    while !ldt_certainty && ldt_iterations < max_ldt_iterations {
                        ws.combined_expert_out.write().fill(0.0);
```

**Lesson:** The edit tool's replacement text was truncated — everything after `lattice_levels` was silently dropped. Always verify `while` / `for` / `if` headers after bulk replacements.

**Post-fix:** `cargo build` — 0 errors. `cargo test` — passes. `cargo clippy` — 0 new warnings. `ldt_audit` — LDT convergence validation passes. ✅
