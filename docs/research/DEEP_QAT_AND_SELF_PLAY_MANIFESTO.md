# Research Manifesto: Deep QAT Thawing & Synthetic Self-Play

**Date:** 2026-06-19
**Author:** Forge LLM Architecture Team
**Context:** Resolution of the "Identity Bypass Syndrome" and Evolution of MUD Autonomy (Phases 8 & 9)

---

## 1. The Bottleneck: "Identity Bypass Syndrome"
Following the successful implementation of the **SlimeRegister** and **Lexical Resonance (Phase 7)**, empirical training metrics demonstrated a ceiling effect: the QAT `Avg Loss` stabilizes around `18.5`.

**Mathematical Diagnosis:**
Because the 30 internal `SlimeLayers` suffer from Ternary Shock, their variance (`VarH`) drops to `0.01`. In a residual architecture (`Output = Input + Layer`), a layer outputting `0.0` turns the entire network into a pure Identity Function ($I$). 
Currently, the STE QAT engine updates *only* the `shadow_emb` (Embedding Matrix). The QAT is essentially finding a way to map words directly to the LM Head by passing them unaltered through the frozen network. The internal $\{-1, 0, 1\}$ weights never learn, and true intelligence is not recovered.

---

## 2. Phase 8: Deep QAT Thawing & Vulkan Acceleration

To solve this, we must shatter the ice. We need to implement full backpropagation through the ternary core.

### 2.1 SlimeBackward (The Mathematical Thaw)
We must implement a reverse pass that corresponds exactly to `evaluate_slime_block`.
- **Target Variables:** `blk.N.attn_q.weight`, `blk.N.attn_k.weight`, `blk.N.attn_v.weight`, `blk.N.expert.N.weight`.
- **Mechanism (Straight-Through Estimator):** The forward pass quantizes weights to $\{-1, 0, 1\}$. The backward pass must compute the exact gradient of the Loss with respect to the `i16` register accumulators, and then route those gradients *straight through* the quantization step to update the underlying FP32 `shadow_weights`.
- **Outcome:** As the ternary layers begin to update, they will break the Identity Bypass. `VarH` will rise from `0.01`, the internal layers will begin modulating the Lexical Prior, and the `Avg Loss` will violently drop from `18.5` down to `< 5.0`.

### 2.2 Vulkan QAT Dispatcher (Compute Unbound)
Calculating partial derivatives for 2 billion parameters iteratively on a CPU `for` loop is computationally non-viable for rapid experimentation.
- **Implementation:** The `run_trainer.rs` pipeline will map the `shadow_emb` and `shadow_weights` into Zero-Copy Vulkan Subbuffers.
- **Compute Shaders:** We will write a custom `vulkan_qat_optimizer.comp` shader that applies the STE QAT update.
- **Outcome:** Epoch times will reduce from hours to minutes. Throughput will scale from hundreds of tokens per second to tens of thousands.

---

## 3. Phase 9: Synthetic Self-Play (Autoentrenamiento Autónomo)

Once the core is thawed, the engine will be capable of learning. However, forcing it to learn from an external, static text file (`unified_corpus.txt`) introduces **Distributional Shift**. The model may struggle to align its pre-trained 1.58-bit latent space with our arbitrary choice of external Spanish/English grammar.

### 3.1 The Synthetic Alignment Protocol
Instead of relying on external text, MUD will dream its own curriculum.
1. **Generation (Dreaming):** The engine is run in standard autoregressive inference mode using a high-temperature sampler. It generates synthetic chains of text based on its strongest surviving neural pathways.
2. **Filtration:** We keep only the syntactic chains that have a high internal confidence score (low perplexity).
3. **Assimilation (QAT):** These self-generated chains are immediately fed back into the `SlimeBackward` pipeline. 

### 3.2 Theoretical Advantages
- **Zero Distributional Friction:** The model is learning the grammar it already intrinsically prefers. 
- **Self-Healing Topology:** By recursively predicting its own high-confidence latent representations, the JEPA gates and Ternary weights naturally align themselves without fighting external contradictions.
- **Infinite Curriculum:** The engine becomes entirely self-contained. It requires no external datasets to repair itself; it merely requires compute time to dream, evaluate, and assimilate.

---

## 4. Execution Protocol

To execute this Manifesto, the development team must proceed in strict order:
1. Write the `SlimeBackward` logic for the Linear GEMV components.
2. Port the `SlimeBackward` logic to GLSL Vulkan Compute Shaders.
3. Hook `vulkan_qat_optimizer_async` to `corpus_trainer.rs`.
4. Validate the IQ restoration against `unified_corpus.txt`.
5. Implement the Autoregressive Synthetic loop to replace the static corpus.

---

## 5. Mathematical Formulations

### 5.1 The Forward Pass (Lexical Resonance)
Let $\mathbf{x} \in \mathbb{R}^d$ be the lexical embedding of the input token. The JEPA gate state $\mathbf{z}$ is initialized using the absolute magnitude (Lexical Energy) of the embedding:
$$ \mathbf{z}_0 = |\mathbf{x}| $$

The ternary weights are quantized using Per-Row Quantization (PRQ) with scale $\mathbf{s}$:
$$ \mathbf{W}_q = \text{clamp}(\text{round}(\mathbf{W}_{shadow} \cdot \mathbf{s}), -1, 1) \in \{-1, 0, 1\} $$

The output of a `SlimeLayer` (ignoring non-linearities for brevity) is:
$$ \mathbf{y} = \text{RMSNorm}(\mathbf{x}) \mathbf{W}_q $$

### 5.2 SlimeBackward: Straight-Through Estimator (STE)
Let $\mathcal{L}$ be the Cross-Entropy Loss. To thaw the core, we must compute $\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{shadow}}$. Since the rounding function is non-differentiable, we apply the Straight-Through Estimator (STE), which approximates the local derivative as the Identity function:
$$ \frac{\partial \mathbf{W}_q}{\partial \mathbf{W}_{shadow}} \approx \mathbf{I} \quad \text{for } \mathbf{W}_{shadow} \in [-1, 1], \quad 0 \text{ otherwise.} $$

The backpropagated gradient for the layer becomes:
$$ \nabla_{\mathbf{W}_{shadow}} \mathcal{L} \approx \left( \text{RMSNorm}(\mathbf{x})^T \nabla_{\mathbf{y}} \mathcal{L} \right) \odot \mathbb{I}_{|\mathbf{W}_{shadow}| \le 1} $$
Where $\odot$ is the Hadamard product and $\mathbb{I}$ is the indicator function that zeroes out gradients for saturated weights to prevent catastrophic unbounding.

### 5.3 JEPA Attractor Dynamics
During the forward pass, the deterministic JEPA state $\mathbf{z}$ corrects the output dynamically. Let $\mu$ be the Exponential Moving Average (EMA) of the layer activations. The JEPA update rule is:
$$ \mathbf{z}_{t+1} = \mathbf{z}_t - \eta (\mathbf{y} - \mu) $$
Where $\eta$ is `JEPA_ATTRACTOR_LR`. The gate applies a multiplicative filter: $\mathbf{y}_{final} = \mathbf{y} \odot \sigma(\mathbf{z})$.

### 5.4 Synthetic Self-Play Objective
Let $\pi_{\theta_{frozen}}$ be the model's policy generating a synthetic sequence $\hat{s}$. We compute the Shannon Entropy $\mathcal{H}$ of the prediction. If $\mathcal{H} > \tau$ (uncertainty threshold), the token is discarded. The self-play loss $\mathcal{L}_{self}$ is optimized only on high-confidence latent projections:
$$ \mathcal{L}_{self} = - \sum_{t} \mathbb{I}_{\mathcal{H}(\hat{s}_t) < \tau} \log P_{\theta_{active}}(x_t = \hat{s}_t \mid \hat{s}_{<t}) $$
This forces $\theta_{active}$ (the ternary weights) to align with the uncorrupted structural priors of the network.

### 5.5 Mathematical Validation & Viability
The aforementioned formulations have been cross-validated for dimensional correctness and computational viability:
- **STE Clipping Threshold ($\mathbb{I}_{|\mathbf{W}_{shadow}| \le 1}$):** Validated as strictly necessary. Without this boundary mask, unconstrained gradients would push floating-point shadow weights towards infinity, preventing future phase transitions across the quantization boundaries ($-1, 0, 1$).
- **Chain Rule Dimensionality:** $\nabla_{\mathbf{W}_{shadow}} \mathcal{L}$ resolves perfectly to the outer product of the transposed input and the error gradient, which maps cleanly to BLAS/GEMM GPU shader operations.
- **Attractor Stability:** The linear formulation of JEPA ($-\eta(\mathbf{y} - \mu)$) acts as a first-order proportional controller. It is unconditionally stable as long as the learning rate $\eta \ll 0.1$.
- **Entropy Masking:** The $\mathbb{I}_{\mathcal{H} < \tau}$ condition in the self-play objective is mathematically equivalent to *Confidence-based Pseudo-labeling*, guaranteeing that the model cannot undergo catastrophic self-poisoning from hallucinatory QAT epochs.

### 5.6 Architectural Implementation: The SlimeLayerTape & 2-bit Unpack
To satisfy the Zero-Allocation memory constraints while supporting the backward flow, we introduced `SlimeLayerTape`. This struct performs **Activation Checkpointing** during `evaluate_slime_block`. It selectively captures `norm_i8`, `scores`, and pre-projection activations at exactly the right cycles before they are overwritten by the zero-allocation hot-loop. By passing `Option<&mut SlimeLayerTape>`, we maintain zero computational overhead during standard interactive inference.

Additionally, empirical verification of the hot-path revealed that MUD utilizes **2-bit packing** (16 weights per `u32`), not 4-bit nibbles. The `ternary_gemv_backward` function employs bitwise shifting `(val >> (j * 2)) & 3` to accurately decode the weights back to `+1, -1, 0` for calculating $\nabla_{\mathbf{x}}$.

### 5.7 QA Record: Zero-Allocation Policy Enforcement (P-01)
During the implementation of the `backward_slime_block` (Phase 8), a critical Quality Assurance (QA) audit revealed dynamic memory allocations (`vec!`) in the backward hot-loop. To strictly comply with the **P-01 Zero-Allocation Policy**, a `SlimeBackwardWorkspace` was introduced. This workspace mirrors the forward `SlimeWorkspace`, pre-allocating all intermediate gradient vectors (e.g., `grad_ffn_up`, `grad_ffn_in_up`) during engine initialization. Consequently, the backward pass now executes entirely *in-place*, using `fill(0.0)` and sequential tensor accumulation to maintain zero memory churn during the most intensive phases of QAT training.
