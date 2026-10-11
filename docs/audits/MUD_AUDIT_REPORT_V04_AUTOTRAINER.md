# MUD Audit Report V4: Autotrainer Zero-Sigma Collapse

**Date:** 29 de mayo de 2026
**Subject:** Total weight destruction ("Zero-Sigma Collapse") during Hot Ternary SGD.
**Status:** RESOLVED

## 1. The Phenomenon
During the initial seating cycles of the newly generated `blank_micro.mud` hybrid model, the `mud_autotrainer` reported a critical failure in the cognitive structure:
```text
Expert 0.w1 | Sigma: 0.0000 | Pos/Neg Ratio: 0.0% / 0.0%
```
This indicated that 100% of the weights in the trained layers had collapsed to exactly `0.0`. The model had suffered complete "Ternary Amnesia" after a single batch of gradient descent.

## 2. Root Cause Analysis
Through deep inspection of the `forge_autograd` computational tape and the `save_shadows_to_mud` persistence logic, two cascading failures were identified:

### A. NaN Gradient Propagation (The Poison)
In a completely untrained, randomized model, the initial forward pass produces highly chaotic logits. The resulting Cross-Entropy loss can be extreme, leading to extremely large or `NaN` (Not a Number) gradients during the backward pass.
When these `NaN` gradients were accumulated into the high-precision `GradAccum` buffers and subsequently flushed into the FP32 `ExpertShadow` / `MambaShadow` weights, the entire weight matrix became poisoned with `NaN`s.

### B. Unquantized Packing (The Collapse)
The `save_shadows_to_mud` function is responsible for converting the trained FP32 shadow weights back into the 1.58-bit packed `u32` format required by the `.mud` file.
The function was calculating the Per-Row Quantization (PRQ) scales correctly, but it was passing the raw FP32 data directly to the `pack_ternary_from_f32` function. 
Since `pack_ternary_from_f32` expects values strictly clustered around `{-1.0, 0.0, 1.0}`, feeding it unscaled, potentially `NaN`-poisoned raw floats caused the packing logic to default every value to `0` (the fallback branch).

## 3. The Resolution

### Fix 1: Gradient Sanitization & Clamping
The `flush_expert_grads` and `flush_mamba_grads` functions were modified to strictly validate gradients before applying them to the shadow weights:
```rust
for (w, g) in s.shadow_w1.iter_mut().zip(&a.grad_w1) { 
    if g.is_finite() { *w -= f * g.clamp(-1.0, 1.0); } 
}
```
This acts as a "Numerical Firewall", preventing chaotic early-training gradients from destroying the structural integrity of the network.

### Fix 2: Forced Hot PRQ Quantization
The persistence logic in `save_shadows_to_mud` was refactored. Before packing, the FP32 shadow weights are now explicitly scaled and quantized into a temporary buffer:
```rust
let mut ternary_data = vec![0.0f32; data.len()];
for r in 0..rows {
    let s = scales[r];
    let start = r * cols;
    for j in 0..cols {
        ternary_data[start + j] = (data[start + j] / s).round().clamp(-1.0, 1.0);
    }
}
```
This guarantees that the data passed to the packer perfectly adheres to the 1.58-bit manifold.

## 4. Verification
Post-fix, a new `blank_micro.mud` was generated and subjected to the seating phase over 201 unassimilated facts.
**Result:**
```text
Expert 0.w1 | Sigma: 0.8603 | Pos/Neg Ratio: 37.0% / 37.1%
```
The model assimilated the knowledge while perfectly maintaining its structural sparsity and standard deviation. The training loop is now fully stable for both Transformer and Mamba architectures.
