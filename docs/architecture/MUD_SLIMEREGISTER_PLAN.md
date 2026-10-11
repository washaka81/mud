# MUD SlimeRegister Architecture Plan

**Status:** ACTIVE — Priority 27 onward  
**Session:** 2026-06-18  
**Paradigm Shift Audit:** V27

---

## Overview

The MUD engine is transitioning from an FP32-workspace inference model to a **bare-metal SlimeRegister paradigm**. This eliminates all intermediate FP32 buffers from the hot-path, reducing memory bandwidth pressure by ~4× and enabling direct AVX2 i16 accumulation without type conversions.

---

## Dual-System Design Principle

The engine operates under a **strict separation of two systems** that MUST converge to thermodynamic equilibrium:

```
┌────────────────────────────────────────────────────────────┐
│  DETERMINISTIC SYSTEM (JEPA)          — bits 16-31          │
│  Fixed rule. No training. No sampling. No randomness.       │
│  z_next = z - α * (y_final - mu_ctx)                        │
│  Acts as a gravity well — always pulling toward mu_ctx.     │
│                                                              │
│  STATISTICAL SYSTEM (Ternary Compute) — bits 0-15           │
│  Learned weights. QAT/STE training. PRQ scaling.            │
│  y_accum = Σ w_i * x_i  where w_i ∈ {-1, 0, +1}           │
│  Acts as the information carrier — stochastic by nature.    │
│                                                              │
│  EQUILIBRIUM POINT:                                          │
│  E[y_final] → mu_ctx        (mean converges)                │
│  Var[y_final] → 1/inv_sigma² (variance stabilizes)          │
│  JEPA correction → 0        (attractor becomes idle)        │
└────────────────────────────────────────────────────────────┘
```

### Why Equilibrium is Guaranteed

Let the combined output at each register be:

```
y_final = y_accum + z * max(0, 1 - |z - mu| * inv_sigma)
z_next  = z - α * (y_final - mu)
```

**At equilibrium**: `z_next = z` → `y_final = mu_ctx`

This means the ternary computation (`y_accum`) plus the JEPA orbital correction (`z_effect`) must sum to `mu_ctx`. During QAT training, the ternary weights learn to produce activations that naturally approach this equilibrium — reducing the correction burden on JEPA over time.

**Practical convergence**: The weights are trained (statistical side) to center their outputs near zero. JEPA's `mu_ctx` is a running EMA of `y_final`. As the ternary computation improves, `y_accum → mu_ctx`, JEPA correction → 0, and the system settles into a stable orbit. The two systems are **co-dependent**: JEPA cannot orbit without the ternary signal, and the ternary signal cannot self-correct without JEPA's gravity.

### Invariants That Enforce Equilibrium

| Property | Owner | Rule |
|---|---|---|
| `mu_ctx` update | JEPA (deterministic) | EMA: `mu = 0.99 * mu + 0.01 * y_final` |
| `inv_sigma` update | JEPA (deterministic) | EMA over variance: `var = 0.99 * var + 0.01 * (y_final - mu)²` |
| Weight update | Ternary (statistical) | QAT STE gradient descent |
| JEPA correction rule | JEPA (deterministic) | Fixed: `α = JEPA_ATTRACTOR_LR = 0.01`, no learning |
| Equilibrium test | Engine diagnostic | `|mu_ctx - mean(y_final_batch)| < epsilon` per layer |

**JEPA has no trainable parameters.** Its rule is fixed at compile time. Only `mu_ctx` and `inv_sigma` evolve as running statistics — deterministically computed from the data.

---

## The SlimeRegister

```rust
#[derive(Copy, Clone)]
#[repr(C, align(4))]
pub struct SlimeRegister {
    /// Bits 0–15: i16 ternary MatMul accumulator (STATISTICAL side)
    /// Range: ±32,767 — MUST reseat every 256 elements max
    /// Carries the learned ternary computation result
    pub matmul_accum: i16,

    /// Bits 16–31: JEPA orbital state stored as f16 bits (u16) (DETERMINISTIC side)
    /// Updated per block: z_next = z - JEPA_ATTRACTOR_LR * (y_final - mu_ctx)
    /// No gradients. No training. Pure deterministic correction.
    pub jepa_packed: u16,
}
```

### Memory Layout

```
SlimeWorkspace (fixed, pre-allocated at engine init):
  ┌─────────────────────────────────────────────┐
  │  registers[0..hidden_size]  SlimeRegister    │  ← Current layer activations
  │  registers_tmp[0..hidden_size]               │  ← Scratch for block output
  │  kv_cache[0..n_heads][0..max_pos][0..head_d] │  ← KV cache (i16)
  │  jepa_mu: f32                                │  ← Deterministic running mean
  │  jepa_inv_sigma: f32                         │  ← Deterministic running inv-std
  └─────────────────────────────────────────────┘

Total footprint (hidden=2048, max_pos=2048, n_heads=16, head_d=128):
  registers:     2048 × 4B = 8 KB
  registers_tmp: 2048 × 4B = 8 KB
  kv_cache:      16 × 2048 × 128 × 2B = 8 MB  (i16)
  jepa_state:    2 × 4B = 8 B
```

---

## ELUT Weight Format (4-bit Nibble Packing)

Each byte stores **two ternary weights** as 4-bit nibbles:

```
Byte layout:
  [7:4] = weight[2k+1]  (high nibble)
  [3:0] = weight[2k]    (low nibble)

Nibble encoding:
  0x0 = 0    (zero weight — skip accumulation)
  0x1 = +1   (add activation to accumulator)
  0xF = -1   (subtract activation from accumulator)

Storage: hidden_size × hidden_size weights → hidden_size² / 2 bytes
Example: 2048×2048 layer = 2 MB (vs 4 MB for 2-bit ternary, 4 MB for i8)
```

---

## AVX2 ELUT-GEMV Kernel Design

```
elut_gemv_avx2(activations: *const i8, weights_elut: *const u8,
               accumulators: *mut i16, n: usize, scale: f32)

Registers:
  ymm0  = 32× i8 activations (loaded)
  ymm1  = 16× packed nibbles → expanded to 32× 4-bit codes
  ymm2  = low nibbles  (mask & 0x0F)
  ymm3  = high nibbles (>> 4 & 0x0F)
  ymm11 = i32 accumulator A (first 16 elements)
  ymm12 = i32 accumulator B (next 16 elements)

Reseat rule (P-04):
  Every 256 elements: vcvtdq2ps + vmulps(scale) → write partial f32 → vzero accumulators
  Prevents i16/i32 overflow on rows longer than 256.

Final step:
  vpaddd ymm11, ymm12 → vcvtdq2ps → vmulps(scale) → horizontal reduce → store
```

---

## JEPA Orbital Attractor (Deterministic — Zero-EXP)

**No training. No gradients. Fixed rule only.**

Runs **per `SlimeRegister`** at every block boundary:

```
Input:
  z     = f16_to_f32(register.jepa_packed)       // Current orbital state
  accum = register.matmul_accum as f32 * prq_scale // Statistical dequantized result

Phase 1 — Statistical output read:
  y_accum = accum                                   // From ternary GEMV (bits 0-15)

Phase 2 — Deterministic JEPA gate (linear, zero-EXP):
  delta        = |z - mu_ctx|
  gate         = max(0.0, 1.0 - delta * inv_sigma) // Linear ReLU approximation
  jepa_effect  = z * gate                           // Orbital pull

Phase 3 — Fusion (where deterministic meets statistical):
  y_final = y_accum + jepa_effect                   // Single point of convergence

Phase 4 — Orbital update (deterministic rule, fixed α):
  z_next = z - JEPA_ATTRACTOR_LR * (y_final - mu_ctx)  // Always pulls toward mu

Phase 5 — Running statistics update (deterministic EMA):
  mu_ctx    = 0.99 * mu_ctx    + 0.01 * y_final
  inv_sigma = recompute from running variance EMA

Phase 6 — Write back to register:
  register.matmul_accum = (y_final / prq_scale) as i16
  register.jepa_packed  = f32_to_f16(z_next)
```

**Equilibrium test**: When `|z - mu_ctx| → 0`, `gate → 1.0`, `y_final → y_accum + z`. The ternary system is said to be "in orbit" when the JEPA correction becomes negligible (z remains near mu_ctx naturally).

---

## Forward Pass Block Structure

Each Transformer block operates over `registers: &mut [SlimeRegister]`:

```
1. ELUT-GEMV (Q, K, V)        → i16 accumulation (AVX2, statistical)
2. PRQ dequant anchor          → one f32 multiply per row (momentary FP32)
3. JEPA attractor fusion       → deterministic orbital correction per register
4. MHA (attention)             → i16 dot products over KV cache
5. ELUT-GEMV (O projection)    → i16 accumulation (AVX2, statistical)
6. Residual add                → i16 add (statistical)
7. RMSNorm                     → f32 momentary, result back to i16
8. ELUT-GEMV (FFN: gate+up)    → i16 accumulation (AVX2, statistical)
9. SiLU gate                   → f32 momentary (deterministic function)
10. ELUT-GEMV (FFN: down)      → i16 accumulation (AVX2, statistical)
11. JEPA attractor fusion      → deterministic orbital correction again
12. Residual add               → i16 add (statistical)
```

FP32 appears ONLY at steps 2, 7, 9 (PRQ anchor, RMSNorm, SiLU). All other steps are integer. JEPA runs at steps 3 and 11.

---

## Migration Plan

### Phase A — New data structures (no regressions)
- [ ] `src/mud/slime.rs` — `SlimeRegister`, `SlimeWorkspace`, pack/unpack, f16↔f32 helpers
- [ ] `src/mud/slime_jepa.rs` — deterministic attractor, EMA mu/sigma tracking
- [ ] Unit tests for both modules (equilibrium test mandatory)
- [ ] `tools/slime_bench.rs` — throughput benchmark

### Phase B — ELUT kernel
- [ ] `src/asm/elut_gemv.s` — ELUT nibble-unpack AVX2 kernel with partial reseat
- [ ] `src/asm/tests.rs` — correctness tests vs reference scalar
- [ ] `tools/elut_bench.rs` — benchmark reporting ops/cycle

### Phase C — Forward pass migration (layer by layer)
- [ ] `src/mud/slime_forward.rs` — SlimeRegister-based forward, one block at a time
- [ ] Integration test: cosine similarity ≥ 0.99 vs reference FP32 forward
- [ ] Keep old `forward.rs` alive until all layers pass integration test
- [ ] Equilibrium diagnostic: report JEPA correction magnitude per layer

### Phase D — Vulkan adaptation
- [ ] `assets/shaders/elut_gemv_i16.comp` — i16 accumulator shader
- [ ] `tools/vulkan_slime_bench.rs`

### Phase E — Dead code purge
- [ ] Delete `src/mud/forward.rs` (old FP32 forward)
- [ ] Delete `src/mud/inference.rs` (old FP32 inference)
- [ ] Delete `src/mud/jepa.rs` (old neural JEPA — replaced by deterministic attractor)
- [ ] Delete orphan tools that served old FP32 pipeline
- [ ] `cargo clippy -- -D dead_code` → 0 warnings

---

## Constraints & Known Risks

| Constraint | Value | Mitigation |
|---|---|---|
| i16 max accumulation | ±32,767 | Reseat every 256 elements (P-04) |
| f16 precision for JEPA z | ~3 decimal digits | Sufficient for orbital correction (not weights) |
| ELUT nibble encoding | 3 states in 4 bits | Spare codes reserved for future sparsity hints |
| KV cache in i16 | Half FP16 range | Monitor attention overflow on long contexts |
| JEPA convergence rate | α = 0.01 (fixed) | Slow enough to not oscillate, fast enough to correct |
| EMA mu_ctx lag | ~100 tokens for 99% accuracy | Warmup phase in first context window |

---

## Policy Compliance Checklist (per module)

```
[ ] #[cfg(test)] mod tests { ... } with ≥2 tests per public fn
[ ] Edge-case test (zero input, max-size input, overflow boundary)
[ ] tools/<name>_bench.rs registered in Cargo.toml [[bin]]
[ ] ./mud.sh bench <name> entry in Benchmarks section
[ ] docs/architecture/ entry updated
[ ] cargo clippy -- -D dead_code passes clean
[ ] No Python used anywhere in this module's pipeline
[ ] Every unsafe block has // SAFETY: comment
[ ] All constants named and justified (no magic numbers)
[ ] JEPA functions marked deterministic in docs (no training params)
```
