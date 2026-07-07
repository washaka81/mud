# MUD Audit: JEPA Death Spiral Resolution

**Date:** 2026-06-20
**Component:** `slime_jepa.rs`

## The Problem
The forward pass was producing `LAYER PRODUCED ALL ZEROS FOR POS X` and `APPLY_OUTPUT_NORM PRODUCED ALL ZEROS!`.
This issue was colloquially named the **"Death Spiral"**.

### Root Cause Analysis
1. **EMA Initialization:** The variables `mu_ctx` and `var_ema` were initialized to `0.0`.
2. **First Token Impact:** The first token processed had a very large `batch_var` because `matmul_accum` operates in integer space (values around 1400).
3. **EMA Overshoot:** Because `var_ema` started at `0.0`, the update `var_ema = 0.99 * 0.0 + 0.01 * batch_var` meant that `var_ema` was 100x smaller than the true variance for the first batch.
4. **Enormous `inv_sigma_ctx`:** Since `inv_sigma_ctx = 1 / sqrt(var_ema)`, the resulting inverse sigma was artificially 10x larger than it should have been.
5. **Gate Clamping:** For the subsequent tokens, `delta * inv_sigma_ctx` evaluated to values around 10.0 instead of 1.0. This caused the Multiplicative Gate to evaluate to `1.0 - 10.0 = -9.0`, which clamped to `0.01`.
6. **Integer Truncation:** With a gate of `0.01`, the signal in Layer 0 shrank from 1400 to 14. In Layer 1, another gate of `0.01` shrank the signal to 0.14, which truncated to `0_i16`.
7. **Permanent Coma:** Once the signal truncated to `0`, `batch_var` became `0.0`. Consequently, `var_ema` remained stuck at `0.0` forever, preventing the network from ever recovering.

## The Resolution
We modified `jepa_stabilizer()` to conditionally detect if `var_ema` is uninitialized (`0.0`). If so, it fully absorbs `batch_mu` and `batch_var` directly, avoiding the EMA warm-up period altogether:

```rust
if *var_ema == 0.0 {
    *mu_ctx = batch_mu;
    *var_ema = batch_var.max(0.01);
} else {
    *mu_ctx = 0.99 * (*mu_ctx) + 0.01 * batch_mu;
    *var_ema = (0.99 * (*var_ema) + 0.01 * batch_var).max(0.01);
}
```

## Results
The `mud_train_metrics.log` now confirms that `gate` operates smoothly, the activations do not zero out, and the model gradients are computing correctly across all 30 layers.
