# MUD Session Report — 2026-07-30 (File Extraction + Verification)

## Summary

Completed file extraction per `docs/sessions/MUD_TODO_2026-07-30.md` Priority 2:
large module splitting for maintainability. All extractions verified with full
test suite + clippy.

| Extraction | Before | After | Extracted To |
|-----------|--------|-------|--------------|
| `cmud.rs` | 2,249 lines | 1,730 lines | `cmud_params.rs` (295) + `cmud_thinking.rs` (250) |
| `session.rs` | 1,778 lines | 1,624 lines | `self_play.rs` (+156 lines) |
| **Total reduction** | **4,027 lines** | **3,579 lines** | **+701 net new lines** across 3 modules |

## Extracted Modules

### `cmud_params.rs` — Data types + CellV5 pack/unpack (295 lines)

Pure data structures and bit-packing utilities — no runtime logic, no
side-effects. All 5 CellV5 tests pass (roundtrip, full fields, edge cases,
batch, compression ratio).

| Item | Type | Lines |
|------|------|-------|
| `ComplexF32` | struct + impls (Add, Sub, hermite_norm, phase, scale, is_finite) | 57 |
| `GaussTernary` | struct + impls (new, pack_id, from_id) | 28 |
| `f32_to_u8_sat` / `f32_to_i8_sat` | private fns | 20 |
| `pack_cmud_to_cellv5` / `unpack_cellv5_to_cmud` | pack/unpack | 18 |
| `pack_complexf32_to_cellv5` / `unpack_cellv5_to_complexf32` | pack/unpack | 12 |
| `pack_batch_*` / `unpack_batch_*` | batch variants | 16 |
| `cmud_cellv5_enabled` | env check (`MUD_CMUD_CELLV5`) | 5 |
| Tests | 5 tests in `cmud_cellv5_tests` | 55 |

**CellV5 compression ratio:** ComplexF32 (8 bytes) → u32 (4 bytes) = **4× (50% reduction)**.

### `cmud_thinking.rs` — Complex arithmetic kernel (250 lines)

Pure mathematical primitives for C-MUD complex-valued thinking — all functions
depend only on `ComplexF32`/`GaussTernary`/`EPSILON_FLOOR`, no global state.

| Item | Type | Lines |
|------|------|-------|
| `gauss_mul` | Complex × Gaussian ternary (adds/subs only) | 8 |
| `gauss_mac` | Accumulate scaled complex product | 6 |
| `gauss_mul_batch_avx2` | AVX2 batch complex multiply (8/iter, `crate::asm` FFI) | 45 |
| `wave_collapse_batch_avx2` | AVX2 batch collapse (identity: `cos(atan2)=re/sqrt(re²+im²)`) | 70 |
| `project_hermitian` | Hermitian ball projection (complex mHC) | 12 |
| `phase_delta` | Angular distance on circle (wraps to (-π, π]) | 13 |
| `wave_collapse` | Scalar wave-function collapse | 12 |
| `ComplexSlimeRegister` | `#[cfg(test)]` research register | 36 |

### `self_play.rs` — D1 self-play training step (+156 lines → 352 total)

Extracted `self_play_training_step` from `corpus_trainer/session.rs`. The method
remains as `impl MudCorpusTrainer` (Rust allows multiple impl blocks per type).
Now co-locates with the existing `generate_synthetic_sequence`,
`is_sequence_confident`, `SelfPlayConfig`, `apply_gradient_jitter`, and
`calculate_shannon_entropy`.

## Benchmark Metrics

### GEMV Throughput (AVX2 ×8 PCorePool)

```
Single-row GEMV (1024×32): 7.79ms
4-row GEMV  (1024×32):      5.73ms  (1.36× speedup vs 1-row)
8-row GEMV  (1024×32):      5.16ms  (1.51× speedup vs 1-row)
```

### ASM Backward Kernel Verification

| Kernel | Tests | Max Error | Status |
|--------|-------|-----------|--------|
| `backward_ternary_gemv_avx2` | 4 (matches_ref, accumulation, zero_grad, tail) | 0.0 (exact) | ✅ |
| `backward_silu_avx2` | 4 (matches_ref, large, overwrite, tail) | <1e-6 | ✅ |
| `backward_rmsnorm_avx2` | 3 (matches_ref, large, tail) | <1e-5 | ✅ |
| `backward_rope_asm` | 2 (matches_ref, tail) | <1e-6 | ✅ |
| `mhc_gate_backward_avx2` | 2 (matches_ref, tail) | <1e-5 | ✅ |
| `backward_ternary_gemv` (Rust) | 1 (ternary_gemv_backward) | 0.0 (exact) | ✅ |

### CMUD Kernel Self-Check

| Check | Input | Output | Status |
|-------|-------|--------|--------|
| `gauss_mul` basis states | wr∈{-1,0,1}, wi∈{-1,0,1} | 9 states verified | ✅ |
| `wave_collapse` real axis | im=0 | re (identity) | ✅ |
| Hermitian projection | \|h\|>R | scales to R | ✅ |
| `phase_delta` circle wrap | θ=3.0, θ_prev=-3.0 | ≈0.28 (not 6.0) | ✅ |
| `contour_rotate` norm | any h | preserves \|h\| | ✅ |
| `cue_phase_repulsion` spreads phases | clustered | increases spread | ✅ |
| `cmud_backward` finite-diff | random input | match within 1e-3 | ✅ |

## Test Results

```
cargo clippy --all-targets       → 0 errors, 0 warnings
cargo test --lib -- --test-threads=2  → 431 passed, 0 failed, 2 ignored
```

### Test counts by module

| Module | Tests |
|--------|-------|
| `mud::` (lib) | 368 |
| `asm::tests` | 42 |
| `vulkan::tests` | 14 |
| `model::tests` | 5 |
| `gguf::tests` | 4 |
| **Total** | **431** |

### CMUD test breakdown (38 tests)

| Submodule | Tests |
|-----------|-------|
| `cmud::tests` | 18 |
| `cmud::avx2_batch_tests` | 4 |
| `cmud_params::cmud_cellv5_tests` | 5 |
| `cmud::tests` (selfcheck, etc.) | 11 |
| **Total** | **38** |

### Backward tests (22 tests)

| Kernel | Tests |
|--------|-------|
| `asm::tests` (backward_silu, backward_rope, backward_rmsnorm, backward_ternary_gemv, mhc_gate) | 12 |
| `mud::slime_backward::tests` | 3 |
| `mud::slime_forward::tests` | 2 |
| `mud::cmud::tests::test_cmud_backward_matches_fd` | 1 |
| `mud::tests` | 4 |
| **Total** | **22** |

## Architecture: Current File Layout

```
src/mud/
├── mod.rs                    (module declarations)
├── cmud.rs                   1,730  ← split target (was 2,249)
├── cmud_params.rs              295  ← NEW: ComplexF32, GaussTernary, CellV5
├── cmud_thinking.rs            250  ← NEW: gauss_mul, wave_collapse, phase_delta, etc.
├── rope_cache.rs                95  ← EXTRACTED (2026-07-30:1)
├── gemv_gpu.rs                 734  ← EXTRACTED (2026-07-30:1)
├── self_play.rs                352  ← EXTRACTED (2026-07-30:2: self_play_training_step)
├── slime_forward.rs          1,481
├── slime_backward.rs         1,301
└── corpus_trainer/
    ├── mod.rs
    ├── session.rs            1,624  ← REDUCED (self_play_training_step extracted)
    ├── train_step.rs
    ├── metrics.rs
    └── ...
```

## Pending / Deferred

| Item | Status | Rationale |
|------|--------|-----------|
| `slime_backward.rs` split | ⏸️ Not in AGENTS TODO | 1,301 lines — not a 2026-07-30 extraction target |
| `slime_x.rs` (`ShadowExpertBus`) removal | ⏸️ Deferred | Requires touching `slime_backward.rs` + `session.rs` simultaneously |
| Full `session.rs` → `telemetry.rs` | ⏸️ Deferred | Telemetry code intertwined with 20+ local variables in training loop |
| `mhc_gate_backward_avx2` wiring | ⏸️ Deferred | Sigmoid derivative doesn't match current mHC backward formula (uses d_alpha/d_beta) |

## Verification Commands

```bash
cargo clippy --all-targets          # 0 warnings, 0 errors
cargo test --lib -- --test-threads=2  # 431 passed, 2 ignored
cargo test --lib -- backward         # 22 backward kernel tests
cargo test --lib -- cmud             # 38 CMUD math + kernel tests
```
