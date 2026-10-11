# MUD Session Report — 2026-07-30 (ASM Backward Kernels + Production Wiring)

## Summary

Completed all 4 backward kernel tasks from `docs/sessions/MUD_TODO_2026-07-30.md` Priority 0/1, plus wiring into production backward code, dead-code removal, and verification.

| Task | Status | Details |
|------|--------|---------|
| `backward_silu.s` | ✅ FIXED | 8-element array constants for clamp (was pulling `+Inf` from adjacent `.text`) |
| `jepa_ou.s` | ✅ NEW | `z_next = 0.9*z + 0.1*y_norm` via `vsubps` + `vfmadd231ps` |
| `backward_rmsnorm.s` | ✅ NEW | `grad_x = grad_y*w - y_unscaled*mean` via `vmulps` + `vfmadd231ps` |
| `backward_rope.s` | ✅ NEW | Inverse RoPE rotation: `dx0=dy0*cos+dy1*sin`, `dx1=dy1*cos-dy0*sin` |
| `mhc_gate.s` | ✅ NEW | Sigmoid backward: `d_v = gate*(1-gate)*d_gate` via `vsubps` + `vmulps` |
| `apply_rope_asm` wiring | ✅ WIRED | Replaced scalar RoPE loop in `slime_forward.rs` forward pass |
| `backward_rope_asm` wiring | ✅ WIRED | Un-rotates Q/K gradients in `slime_backward.rs` between SDP backward and QKV projection backward |
| `backward_rmsnorm_avx2` wiring | ✅ WIRED | Both FFN and Attention RMSNorm backward in `slime_backward.rs` |
| `apply_rope_interleaved_asm` | ✅ REMOVED | Dead code (P-08): 0 production call sites, split layout is used everywhere |

**Skipped:** `mhc_gate_backward_avx2` not wired — computes sigmoid derivative for `d_v_jepa`, but current mHC backward computes `d_alpha`/`d_beta` only (no v_jepa gradient needed yet). Building block for future full mHC gradient.

## Key Findings

### backward_silu.s Root Cause

The vector path produced sigmoid ≈ 0.5 for 7/8 lanes because `bw_clamp_lo/hi` were single `.float` (4 bytes), but `vmaxps` with YMM reads **32 bytes**. Adjacent `bw_exp_mask` (0x7F800000 = +Inf) was pulled into lanes 1-7, corrupting the exp(-x) polynomial. **Fix:** 8-element arrays.

### VFMADD Operand Encoding (GAS vs Intel)

| Instruction | GAS syntax | Result |
|-------------|-----------|--------|
| VFMADD231PS | `vfmadd231ps %src1, %src2, %dst` | `dst = dst + src2 * src1` |
| VFMADD213PS | `vfmadd213ps %src1, %src2, %dst` | `dst = dst * src2 + src1` |
| VPANDN | `vpandn %val, %mask, %dst` | `dst = (~mask) & val` |

**Critical bug in mhc_gate.s:** Initially had `vpandn %ymm5, %ymm4, %ymm4` = `(~val) & mask` — zeroed ALL outputs. Fixed to `vpandn %ymm4, %ymm5, %ymm4` = `(~mask) & val`.

### GAS VEX.vvvv encoding

For `vfmadd231ps %src1, %src2, %dst`: VEX.vvvv encodes src1 (inverted), ModR/M.rm encodes src2, ModR/M.reg encodes dst. Verified with `objdump -d` (`c4 e2 71 b8 c2` = 231 form, `c4 e2 71 a8 c2` = 213 form).

## Files Modified

| File | Change | Lines |
|------|--------|-------|
| `src/asm/backward_silu.s` | Fixed clamp constants to 8-element arrays | 2 changes |
| `src/asm/jepa_ou.s` | NEW: JEPA EMA update kernel (vector + scalar + NaN guard) | 85 lines |
| `src/asm/backward_rmsnorm.s` | NEW: RMSNorm backward (FMA path + scalar tail + NaN guard) | ~120 lines |
| `src/asm/backward_rope.s` | NEW: RoPE backward (split layout + scalar tail + NaN guard) | ~135 lines |
| `src/asm/mhc_gate.s` | NEW: Sigmoid backward (vector + scalar + NaN guard) | ~120 lines |
| `src/asm/rope.s` | Added scalar tail to `apply_rope_asm`; removed `apply_rope_interleaved_asm` (dead code) | -133/++37 |
| `src/asm/mod.rs` | Added FFI decls for 4 new kernels; removed `apply_rope_interleaved_asm` | +12/-4 |
| `src/asm/tests.rs` | Added 11 new tests (4 silu + 3 jepa_ou + 3 rmsnorm + 2 rope); removed 2 interleaved tests; fixed tolerance + PI const | +200/-30 |
| `build.rs` | Added 3 .s files to compiler + rerun list (21→24 total) | +3/+3 |
| `src/mud/slime_forward.rs` | Replaced scalar RoPE loop with `apply_rope_asm` call | -16/+15 |
| `src/mud/slime_backward.rs` | Wired `backward_rope_asm` + `backward_rmsnorm_avx2` into FFN/Attn RMSNorm backward | ~+40/-30 |

## Verification Results

- `cargo test --lib -- --test-threads=2` → **431 passed, 0 failed, 2 ignored** ✅
- `cargo clippy --all-targets` → **0 errors, 0 warnings** ✅
- C equivalence tests (gcc -mavx2 -mfma -mbmi2 -mf16c) → all PASS ✅
  - backward_silu: n=8/16/12/1/7/23/137 — max_err < 3e-6 ✅
  - jepa_ou: n=8/16/12/1/7/23/0 — max_err < 2.4e-7 ✅
  - backward_rmsnorm: n=32/10/1 — max_err = 0.0 ✅
  - backward_rope: n=32/10 — max_err < 6e-8 ✅
  - mhc_gate_backward: n=64 — max_err < 1e-7 ✅

## P-08 Dead Code Removal

| Code Removed | Reason |
|-------------|--------|
| `apply_rope_interleaved_asm` | 0 production call sites; codebase uses split layout `[real..., imag...]`, not interleaved. `apply_rope_asm` handles split layout directly. |
| `ROPE_DEINTERLEAVE_IDX`, `ROPE_INTERLEAVE_IDX` | Only used by removed interleaved kernel |
| 2 tests for `apply_rope_interleaved_asm` | No longer applicable |

## Production Wiring

### `apply_rope_asm` (forward pass)
- `slime_forward.rs:1518-1532`: Replaced per-element scalar loop with `apply_rope_asm(head_d, ...)` call for Q and K heads
- ABI: `n=head_d` (head_dim), `x=q_f32[base..]` (split layout), `cos_tbl`, `sin_tbl` (half_d elements)

### `backward_rope_asm` (backward pass)
- `slime_backward.rs:737-757` (new step 6b): Un-rotates Q/K gradients between SDP-attention backward and QKV projection backward
- Same ABI as forward: split layout, per-head call

### `backward_rmsnorm_avx2` (backward pass)
- `slime_backward.rs:675-704` (step 4): FFN RMSNorm backward — pre-computes `y_unscaled` into `scratch_grad_x`, calls ASM for elementwise `grad_x = grad_y*w - y_unscaled*mean`
- `slime_backward.rs:808-827` (step 8): Attention RMSNorm backward — same pattern, accumulates into `grad_out`
- Pre-computes `y_unscaled` and weight gradient in a single pre-pass (P-01: single pass vs original 2-pass loop)
- Weight gradient still scalar (per-element, used by optimizer, not in backward chain)

## Deferred

- `backward_silu_avx2` and `mhc_gate_backward_avx2` not wired into backward pass — they are correct building blocks (verified via C tests) but the current backward uses `relu2_backward_avx2` for the activation derivative chain and doesn't compute `d_v_jepa` for mHC. Can be wired when the backward is refactored to match forward SiLU exactly.
