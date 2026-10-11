# MUD Audit Report V27: All Zeros Anomaly & JEPA Variance Bleed
**Date:** 2026-06-20
**Focus:** QAT STE Trainer & SlimeWorkspace Mathematical Stability

## 1. The Anomaly
During QAT STE training (`run_trainer`), the model completely collapsed into a state where every layer after the first few produced identically zero outputs for every token in the sequence. 

The logs were flooded with:
`LAYER PRODUCED ALL ZEROS FOR POS X`

This resulted in:
- No gradients propagating backward.
- Total loss of signal through the FFN and residual streams.
- The `mud_train_metrics.log` showed `Sigma = 0.00`, `E_JEPA = 1.00`, and `VarJ = 0.00` across the board, alongside an infinitely spiking `inv_sigma` parameter.

## 2. The Investigation
We initially hypothesized:
1. Dead or NaNs in the model weights (checked via `check_emb.rs` - proven clean).
2. NaNs or zeros in the `shadow_emb` embedding matrix (proven clean with `max_abs ~ 14.8`).
3. Precision overflow in the `iscale` quantization logic (already fixed and bounded safely).

## 3. The Root Cause: "Cross-Layer Variance Bleed"
The bug resided in the definition and usage of the JEPA state variables inside the pre-allocated `SlimeWorkspace`.
```rust
// Old Implementation
pub jepa_mu: f32,
pub jepa_inv_sigma: f32,
pub jepa_var_ema: f32,
```
Because these were stored as single, global scalars, the statistics were being mixed across all layers and sequence positions. 
- `Layer 30` (with extremely tiny, processed activations) would shrink `jepa_var_ema` toward zero.
- When the next sequence position started, `Layer 1` (with raw, high-magnitude embeddings) was evaluated against `Layer 30`'s microscopic variance.
- This resulted in an enormous `delta` (`|z - mu|`), which when multiplied by a bloated `inv_sigma`, forced the JEPA mathematical gate to evaluate to `0.0`.
- The gate clamped the entire activation vector to zero, destroying all signal and reinforcing the variance collapse.

## 4. The Resolution
We refactored `SlimeWorkspace` to store JEPA statistics as layer-specific arrays:
```rust
pub jepa_mu: std::vec::Vec<f32>,
pub jepa_inv_sigma: std::vec::Vec<f32>,
pub jepa_var_ema: std::vec::Vec<f32>,
```
Initialized with size 128 (to support all models safely). 

The `evaluate_slime_block` function signature was updated to accept a `layer_idx: usize` parameter, ensuring that the JEPA stabilizer strictly uses the state slice corresponding to the active layer:
```rust
jepa_stabilizer(&mut ws.registers_tmp, &mut ws.jepa_mu[layer_idx], &mut ws.jepa_inv_sigma[layer_idx], &mut ws.jepa_var_ema[layer_idx], iscale);
```

Both `run_trainer.rs` and `main.rs` iteration loops were patched to pass `l_idx` through.
Finally, we integrated a real-time `ETA` and `Loss` tracker that prints every 10 batches in `run_trainer.rs` (previously hardcoded to every 100 batches, obscuring progress during the anomaly).

**Status: FULLY RESOLVED. The model can now undergo STE QAT without mathematical collapse.**
