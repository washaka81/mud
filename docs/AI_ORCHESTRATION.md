# Transformer Orchestration: MUD MoE

## 1. Sparse Mixture of Experts (MoE) Flow
MUD implements a **Sparse MoE** architecture. Instead of processing every token through every parameter, it uses a learned **Gate (Router)** to select only the top `K` most relevant experts.

### Execution Pipeline (Per Token):
1. **Input:** Word embedding is retrieved from the Ternary Embedding table.
2. **RMSNorm:** Input is normalized using AVX2-optimized kernels.
3. **Causal Attention:**
    - **Q, K, V Projections:** Executed on the iGPU (Vulkan) using Ternary Subgroup kernels.
    - **RoPE:** Rotary Positional Embeddings are applied in Rust to encode sequence order.
    - **Softmax:** Standard attention scores calculated with causal masking.
    - **Output Projection:** Multi-head results are fused and projected back to `hidden_size`.
4. **MoE Block:**
    - **Gate Projection:** A ternary layer (`gate_w`) predicts which experts are best suited for the current token.
    - **Routing:** A Top-2 selection mechanism (Softmax + Top-K) selects the experts.
    - **Expert Execution:** Selected experts execute their internal SwiGLU blocks (`w1`, `w2`, `w3`) on the iGPU.
    - **Weighted Sum:** The outputs of the selected experts are multiplied by their routing probabilities and summed.
5. **Residual Connection:** The MoE output is added back to the original input (damped by `1/sqrt(num_layers)`).

## 2. Hardware Mapping (Intel i7-1260p)
- **Gate & Orchestration:** Handled by CPU (Sequential logic).
- **Matrix Multiplications (GEMV):** Offloaded to Intel Iris Xe iGPU.
- **Ternary Logic:** Both CPU and GPU paths use optimized kernels that avoid floating-point multiplications, relying instead on additions and subtractions based on the `{-1, 0, 1}` bit-packed weights.

## 3. Training Protocol (Kaggle)
- **Loss Function:** Combined Cross-Entropy (Language) + **Auxiliary Balance Loss** (ensures all experts are utilized and prevents expert collapse).
- **Quantization-Aware Training (QAT):** Weights are kept as FP32 during training but quantized to ternary in the forward pass using Straight-Through Estimators (STE).
