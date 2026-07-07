# MUD Audit Report V30: JEPA Scale Mismatch — Frozen Signal & Intermittent Collapse

**Date:** 2026-06-20 (updated 2026-06-21)
**Severity:** HIGH
**Component:** `slime_jepa.rs` (`jepa_stabilizer`), `corpus_trainer.rs` (embedding init)
**Status:** RESOLVED
**Follows:** V29 (Ternary2Bit Format Mismatch)

---

## 1. Summary

After fixing the Ternary2Bit format mismatch (V29), the forward pass and backward pass now
compute correctly. However, the JEPA stabilizer exhibits a **scale mismatch** between its
per-dimension tracker `z` and the cross-dimension mean `mu_ctx`, causing the multiplicative
gate to kill ~99% of the signal on every token after the first.

This manifests as:
- "Frozen" signal: Sigma converges to a constant value (~227) across all tokens
- Intermittent collapse: some segments drop to Sigma=0 before recovering
- VarJ = 0.00: JEPA tracking has zero variance (all dimensions converge to same z)
- Rho = 0.00: no correlation between hard activations and JEPA state

---

## 2. Observed Telemetry

### Segment 2 (post-fix, healthy GEMV)
```
Step  Token     Sigma  E_JEPA  Rho    VarH      VarJ   Mode  Delta   Eps   Omega
4     _has      227    1.00    0.00   0.000423  0.00   211   0.02    1.00  1.00
5     _no       227    1.00    0.00   0.000423  0.00   211   0.02    1.00  1.00
6     _equal    220    0.86   -0.38   0.004161  0.20   -55   0.02    1.00  1.00
```

Most tokens show Sigma=227 (frozen), VarJ=0.00 (JEPA dead), E_JEPA=1.00 (maximum energy =
complete divergence between hard and JEPA domains). Occasional tokens like `_equal` break
the pattern with different Sigma and non-zero Rho.

### Segment 3 (intermittent collapse)
```
Step  Token     Sigma  E_JEPA  Rho    VarH      VarJ   Mode  Delta   Eps   Omega
0     _gods     0      1.00    0.00   0.000000  0.00   0     0.00    1.00  1.00
...
6     :\n       0      1.00    0.00   0.000000  0.00   0     0.00    1.00  1.00
7     Be        136    1.26    0.26   0.006015  0.18   113   0.00    1.00  1.00
8     -m        134    1.37    0.17   0.010865  0.26   110   0.00    1.00  1.00
```

Tokens 0-6 collapse to Sigma=0, then tokens 7+ recover with Sigma=134-179 and healthy Rho.

---

## 3. Root Cause: Three Interacting Issues

### Issue A: z/mu_ctx Scale Mismatch

The JEPA stabilizer compares per-dimension tracker `z` against cross-dimension mean `mu_ctx`:

```rust
let y_accum = reg.matmul_accum as f32 * prq_scale;   // quantized signal ≈ 0.02
let z = half_to_float_bits(reg.jepa_packed);           // embedding magnitude ≈ 14.8
let delta = (z - *mu_ctx).abs();                       // ≈ 14.78
let gate = (1.0 - delta * (*inv_sigma_ctx)).clamp(0.01, 1.0);  // = 0.01
```

- `z` is initialized to `float_to_half_bits(emb_val.abs())` — the raw embedding magnitude (0 to ~14.8)
- `mu_ctx` converges to `batch_mu` — the mean of `y_final` across all 2560 dimensions (~0.02)
- `delta ≈ 14.78` for high-magnitude dimensions → gate clamps to 0.01

**The gate kills 99% of the signal** for any dimension where |emb_val| > 1.02.

### Issue B: var_ema Floor Too Aggressive

```rust
*var_ema = batch_var.max(1.0);  // Forces var_ema ≥ 1.0
*inv_sigma_ctx = (1.0 / (*var_ema + EPSILON_FLOOR).sqrt()).min(1.0);  // = 1.0
```

The `max(1.0)` floor was justified as "unit-variance in normalized embedding space," but
`batch_var` is measured in post-residual quantized space where typical variance is ~0.001.
The floor forces `var_ema = 1.0` → `inv_sigma = 1.0` → kill radius = 1.0.

With z ranging 0-14.8 and kill radius 1.0, most dimensions are classified as "anomalous"
and gated to 1%.

### Issue C: z Update Tracks Gated Signal

```rust
let z_next = z * 0.99 - JEPA_ATTRACTOR_LR * (y_final - *mu_ctx);
```

`z_next` is updated using `y_final` (the GATED signal), not `y_accum` (the raw signal).
When the gate kills y_final to 1%, the attractor term becomes negligible:

```
z_next = 14.8 * 0.99 - 0.01 * (0.148 - 0.02) ≈ 14.65 - 0.001 ≈ 14.65
```

z barely moves toward mu_ctx. After 16 tokens: z ≈ 14.8 * 0.99^16 ≈ 12.6.
After 100 tokens: z ≈ 5.4. It takes hundreds of tokens for z to converge.

---

## 4. Numerical Trace

### First token (Layer 0, attention JEPA):
```
inv_sigma = 0.0 (initialized) → gate = 1.0 (pass-through) ✓
y_final = y_accum ≈ 14.8 (signal preserved)
batch_mu = mean(y_final) ≈ 0.02
batch_var = var(y_final) ≈ 0.001
var_ema = max(0.001, 1.0) = 1.0  ← FLOOR ACTIVATES
inv_sigma = min(1.0, 1/sqrt(1.0)) = 1.0
```

### Second token:
```
inv_sigma = 1.0, mu_ctx = 0.02
For dimension with z = 14.8:
  delta = |14.8 - 0.02| = 14.78
  gate = (1 - 14.78 * 1.0).clamp(0.01, 1.0) = 0.01  ← KILLED
  y_final = 14.8 * 0.01 = 0.148

For dimension with z = 0.02:
  delta = |0.02 - 0.02| = 0.0
  gate = 1.0  ← PASSES

Result: only dimensions near the mean survive → signal collapses to constant
```

---

## 5. Proposed Fixes (for future implementation)

### Option A: Soften var_ema floor
```rust
// Current:
*var_ema = batch_var.max(1.0);

// Proposed:
*var_ema = batch_var.max(0.01);  // Allow smaller variance
```
**Risk:** inv_sigma = 1/sqrt(0.01) = 10.0, but min(1.0) clamp still limits to 1.0.
Would need to also relax the min(1.0) clamp on inv_sigma.

### Option B: Initialize z to match y_accum scale
```rust
// Current (corpus_trainer.rs):
ws.registers[h].jepa_packed = float_to_half_bits(emb_val.abs());

// Proposed:
let quantized_val = (emb_val / iscale).clamp(-32767.0, 32767.0) as i16 as f32 * iscale;
ws.registers[h].jepa_packed = float_to_half_bits(quantized_val);
```
**Note:** Since `quantized_val ≈ emb_val` (within quantization error), this may not
significantly change behavior. The real issue is the cross-dimension mean vs per-dimension
comparison.

### Option C: Sigmoid gate instead of linear
```rust
// Current:
let gate = (1.0 - delta * (*inv_sigma_ctx)).clamp(0.01, 1.0);

// Proposed:
let gate = 1.0 / (1.0 + (delta * (*inv_sigma_ctx)).powi(2));
```
**Effect:** Smooth decay instead of hard clamp. gate = 0.5 at delta = 1/inv_sigma.
Never reaches exactly 0.01, preserving some signal for all dimensions.

### Option D: Per-dimension statistics (major redesign)
Replace single `mu_ctx`/`var_ema` with per-dimension arrays of size `hidden`.
Track z's own history instead of comparing to cross-dimension mean.
**Cost:** 3x memory for JEPA state (3 * hidden * f32 per layer instead of 3 scalars).

---

## 6. Current Impact

- Training loss decreases slowly because the gate kills most signal dimensions
- The model can still learn from the ~1% of signal that passes through
- Intermittent collapses recover when the gradient update shifts embeddings enough
  to change the z/mu_ctx relationship
- No NaN or permanent death (V29 fixes hold)

---

## 7. Files Involved

| File | Location | Issue |
|------|----------|-------|
| `src/mud/slime_jepa.rs` | `jepa_stabilizer()` L138 | Linear gate + var_ema floor |
| `src/mud/slime_jepa.rs` | `jepa_stabilizer()` L145 | z_next uses y_final not y_accum |
| `src/mud/corpus_trainer.rs` | L950 | z init to emb_val.abs() |
| `src/main.rs` | L312 | z init to emb_val.abs() |

---

## 8. Resolution (2026-06-21)

Applied combination of Options A + C with z EMA redesign:

### Changes to `jepa_stabilizer()` in `src/mud/slime_jepa.rs`:

1. **Sigmoid gate** (Option C):
   ```rust
   // Old:
   let gate = (1.0 - delta * (*inv_sigma_ctx)).clamp(0.01, 1.0);
   // New:
   let d = delta * (*inv_sigma_ctx);
   let gate = 1.0 / (1.0 + d * d);
   ```
   - delta*inv_sigma = 0 → gate = 1.0 (pass-through)
   - delta*inv_sigma = 1 → gate = 0.5 (soft suppression)
   - delta*inv_sigma = 3 → gate = 0.1 (strong suppression)
   - No hard kill floor — every dimension retains signal

2. **z EMA tracking y_accum** (replaces broken formula):
   ```rust
   // Old: z_next = z * 0.99 - 0.01 * (y_final - mu_ctx)  // tracked gated signal
   // New: z_next = z * 0.9 + 0.1 * y_accum                // tracks raw signal
   ```
   - z converges to y_accum within ~10 tokens (decay 0.9)
   - No dependence on gated y_final — avoids feedback loop

3. **var_ema tracks z_var** (replaces batch_var):
   ```rust
   // Old: var_ema = (0.99 * var_ema + 0.01 * batch_var).max(1.0)  // batch_var of gated signal
   // New: var_ema = 0.99 * var_ema + 0.01 * z_var                 // variance of z around mu_ctx
   ```
   - No arbitrary floor — adapts to actual z spread
   - inv_sigma capped at 10.0 (was 1.0) — allows natural adaptation

4. **Dead code cleanup** (P-08):
   - Removed unused `JEPA_ATTRACTOR_LR` constant
   - Removed unused `NEURAL_KICK_JITTER` constant

### Verification
- `cargo test` — 85/85 pass
- `cargo clippy --all-targets` — 0 warnings (P-06)
- `cargo check --release` — clean
