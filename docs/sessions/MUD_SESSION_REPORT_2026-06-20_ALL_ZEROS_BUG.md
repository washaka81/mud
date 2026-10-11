# MUD Session Report: Resolving the "All Zeros" JEPA Attractor Collapse
**Date:** 2026-06-20
**Module:** `SlimeWorkspace`, `jepa_stabilizer`
**Status:** RESOLVED

## The Problem: The "All Zeros" Anomaly
During the QAT STE training loop (`corpus_trainer.rs`), the diagnostic telemetry (`mud_train_metrics.log`) exhibited a catastrophic anomaly where tokens would completely collapse to `0.0` variance across all 2560 dimensions. 

When this happened, the logs flooded with:
`LAYER PRODUCED ALL ZEROS FOR POS X`

This cascading failure meant that:
1. `Sigma` became 0.
2. `E_JEPA` reached exactly 1.00.
3. The FFN sublayer received exactly zero input and outputted exactly zero.
4. The residual connection simply passed a constant stream of zeros up to the final LM head.
5. Gradients collapsed completely.

Initially, we suspected:
- Data corruption in `shadow_emb` (`token_embd.weight`).
- `NaN` or `Inf` floating-point overflow during `iscale` quantization or RMSNorm.

Extensive validation (`check_emb.rs`) confirmed that all static tensors (`.weight`, `.prq_scale`) and dynamic embeddings were strictly finite and within valid dynamic ranges (`max_abs ~ 14.8`).

## The Root Cause: Cross-Layer Variance Bleed (JEPA State Mismanagement)
The true bug was found in the definition of the `SlimeWorkspace` variables used to track the JEPA (Joint Embedding Predictive Attractor) deterministic statistical state.

```rust
// Old definition in SlimeWorkspace
pub jepa_mu: f32,
pub jepa_inv_sigma: f32,
pub jepa_var_ema: f32,
```

Because these metrics were implemented as **single, global scalars** passed sequentially to all 30 layers for every token, they were suffering from extreme cross-layer bleeding:
1. `Layer 1` processed a token with highly energetic raw embeddings and updated `jepa_mu` and `jepa_var_ema`.
2. As the forward pass progressed down to `Layer 30`, the activations became extremely processed and small. The single `jepa_var_ema` adjusted downward (shrinking variance) to match Layer 30's statistics.
3. When the **next token** arrived, the loop restarted at `Layer 1`. 
4. `jepa_stabilizer` mathematically compared the raw, highly energetic embedding of `Layer 1` (Token N) against the extremely tight, low-variance EMA of `Layer 30` (Token N-1).
5. The `delta` (`|z - jepa_mu|`) exploded relative to `jepa_inv_sigma`.
6. This caused the multiplicative JEPA gate `1.0 - (delta * inv_sigma)` to saturate fully at `0.0`.
7. **Complete Signal Blackout:** The entire activation vector was zeroes out. In the next iteration, the variance shrank even more towards 0, causing `inv_sigma` to explode mathematically (hitting the `EPSILON_FLOOR`), permanently bricking the forward pass for all subsequent tokens.

## The Resolution
The fix required treating the JEPA EMA statistics not as global variables, but as layer-specific context.

1. **Vectorization in SlimeWorkspace:** 
   Converted the scalars into pre-allocated vectors of size `128` (enough for any reasonable depth):
   ```rust
   pub jepa_mu: std::vec::Vec<f32>,
   pub jepa_inv_sigma: std::vec::Vec<f32>,
   pub jepa_var_ema: std::vec::Vec<f32>,
   ```

2. **Layer-Indexed Access:**
   Updated the signature of `evaluate_slime_block` to accept `layer_idx: usize`:
   ```rust
   pub fn evaluate_slime_block(layer: &SlimeLayer, layer_idx: usize, ws: &mut SlimeWorkspace, ...)
   ```

3. **Targeted JEPA Stabilization:**
   Passed the layer-specific slices into the `jepa_stabilizer` function:
   ```rust
   jepa_stabilizer(&mut ws.registers, &mut ws.jepa_mu[layer_idx], &mut ws.jepa_inv_sigma[layer_idx], &mut ws.jepa_var_ema[layer_idx], iscale);
   ```

4. **Integration:** 
   Updated both `corpus_trainer.rs` and `main.rs` loops to pass the exact layer index during the sequential evaluation iteration.

## Result
Recompilation (`cargo check --release`) succeeded. Rerunning `mud.sh train` showed that the `LAYER PRODUCED ALL ZEROS` warning was completely eliminated from the output. The QAT STE trainer now correctly propagates signal depth without catastrophic attenuation or collapse.
