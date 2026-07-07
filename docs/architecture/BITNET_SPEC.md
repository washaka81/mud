# Microsoft BitNet (1.58b) - Architectural Specification

This document extracts the exact implementation details from the official Microsoft BitNet repository (`src/README.md`) to guide the bit-exact parity implementation in the MUD engine.

## 1. Quantization Scheme: W2A8
BitNet utilizes a **W2A8** (2-bit Weight, 8-bit Activation) matrix multiplication paradigm.

*   **Weights (W):** Constrained strictly to the ternary grid: `{-1, 0, +1}`.
    *   **Scale ($\gamma$):** The weight scaling factor is computed as the mean of the absolute values of the entire weight matrix (or row):
        $$ \gamma = \frac{1}{n \cdot m} \sum |W_{ij}| $$
*   **Activations (X):** Quantized token-wise to 8-bit integers `[-127, 127]` using Absolute Maximum Scaling (AbsMax).
    *   $$ Q_{max} = \max(|X|) $$
    *   $$ X_q = \text{round}\left(X \cdot \frac{127}{Q_{max}}\right) $$

## 2. The Final Projection (Scaling)
Because the inner loop operates entirely in integers (`i8 * i8`), the final accumulated result must be de-quantized back to FP32 before proceeding to the next layer or residual connection.

The final de-quantization formula is:
$$ Y = Y_{int32} \cdot \left( \frac{\gamma \cdot Q_{max}}{127} \right) $$

Where:
*   $Y_{int32}$ is the raw accumulation of `X_q * W`.
*   $\gamma$ is the pre-computed weight scale.
*   $Q_{max}$ is the dynamic absolute maximum of the current activation vector.

## 3. Packing & Memory Layout (I2_S)
*   **Format (`I2_S`):** Weights are packed into 2 bits per element.
*   **Tiling / Blocks:** The official engine optimizes L1/L2 cache by traversing the matrix in blocks of `ROW_BLOCK_SIZE = 4` and `COL_BLOCK_SIZE = 128`.
*   **Embeddings:** The `token_embd` is often kept at a higher precision (e.g., `Q6_K` or `FP16`) to maintain perplexity, while the core Transformer blocks use `I2_S`.

## 4. Inference Mechanics (SIMD Parity)
To achieve parity with Microsoft's CPU kernels:
*   The kernel must be an **Activation Parallel** kernel.
*   The inner loop must execute **zero multiplications**. Since weights are `{-1, 0, 1}`, the dot product logic should use branchless `add` and `sub` integer operations based on the unpacked bits.
*   SIMD instructions (AVX2/NEON) must execute these integer additions before casting the accumulated block to float.

## MUD Engine Adaptation
In the `MudInference::gemv_vulkan_or_cpu_scaled` fallback loop, the logic must perfectly mirror this process:
1.  Iterate `x` to find `Q_{max}`.
2.  Quantize `x` to an `i8` array.
3.  Perform integer-based dot products (via `ternary_gemv_lut_avx2`).
4.  Multiply the output by `(Q_{max} / 127.0) * gamma`.

*Note: Microsoft's bit packing order maps `01 -> +1`, `10 -> -1`, and `00 -> 0`.*
