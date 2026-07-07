# QAT & PRQ Integration Guide: Mathematical Specifications

This document outlines the numerical standards for Per-Row Quantization (PRQ) and Quantization-Aware Training (QAT) in the MUD engine.

## 1. Per-Row Quantization (PRQ) Formulation

In MUD, weights $W$ are quantized row-by-row into ternary values $W_q \in \{-1, 0, 1\}$ using a scale factor $\gamma_i$ computed for each row $i$:

$$\gamma_i = \frac{\text{DEPTH\_DAMPENING\_FACTOR}}{N_c} \sum_{j=1}^{N_c} |W_{i, j}|$$

Where:
- $N_c$ is the number of columns (input size).
- $\text{DEPTH\_DAMPENING\_FACTOR} = 0.7071$ (corrects variance limits).

Ternary mapping is performed as follows:

$$W_{q, i, j} = \text{clamp}\left( \text{round}\left( \frac{W_{i, j}}{\gamma_i} \right), -1.0, 1.0 \right)$$

Weights with absolute values below the sparsity threshold are truncated to zero. The threshold is defined as:

$$\text{Threshold}_i = \text{SPARSITY\_THRESHOLD\_RATIO} \times \gamma_i$$

Where $\text{SPARSITY\_THRESHOLD\_RATIO} = 0.7$.

## 2. Straight-Through Estimator (STE) in QAT

Since the rounding and clamping functions have zero gradients almost everywhere, we use the Straight-Through Estimator during backpropagation. The forward pass simulates the quantized weights, but the backward pass acts as if the weights were continuous FP32 shadow weights:

$$\frac{\partial \mathcal{L}}{\partial W_{shadow}} = \frac{\partial \mathcal{L}}{\partial W_q}$$

## 3. Gradient Sanitization Protocol

To prevent catastrophic model collapse ("Zero-Sigma" matrix collapse) where all weights converge to a single value (e.g., zero), apply the following checks to gradients before updating the shadow weights:

1. **Check Finiteness:** Assert `grad.is_finite()`.
2. **Apply Neural Kick:** If a weight row's variance (Sigma) drops below `0.10`, inject a small jitter:
   $$\text{jitter} \sim \mathcal{N}(0, \text{NEURAL\_KICK\_JITTER})$$
   Where $\text{NEURAL\_KICK\_JITTER} = 1e-5$.
3. **Clip Gradients:** Clip the gradient values to $[-1.0, 1.0]$ or a dynamically calculated maximum norm.
