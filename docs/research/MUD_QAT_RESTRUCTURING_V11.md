# MUD Architecture Restructuring Study: Resolving the Deterministic Attractor Collapse & QAT Illusion

**Date:** 2026-06-29
**Subject:** Deep restructuring of mHC residual mapping and QAT Loss landscape to cure Semantic Aphasia and Bipartite Attractor Collapse.

---

## 1. The Core Pathologies

Recent training and inference telemetry have revealed three converging failures that completely cripple the engine's capability to learn or generate meaningful text, despite showing a deceptively low "Training Loss".

### Pathology A: The Contrastive Loss Illusion (QAT False Positive)
* **Symptom:** The QAT trainer reports a Loss of `2.0633`, while inference entropy explodes to `H = 10.38`.
* **Root Cause:** In `corpus_trainer.rs`, the Cross-Entropy loss uses a hardcoded Negative Sampling strategy with `NUM_NEG = 7`. This reduces the classification space from 128,256 tokens down to exactly **8 classes** (1 positive, 7 negatives).
* **Mathematical Reality:** The absolute maximum entropy (worst-case random guessing) for 8 classes is `ln(8) ≈ 2.0794`. A loss of `2.0633` means the model is performing at exactly random chance ($p \approx 0.127$). The model learns nothing, but the loss metric appears mathematically "stable" and low.
* **Inference Consequence:** When deployed to 128,256 tokens, the distribution remains perfectly flat, causing the generation entropy to skyrocket.

### Pathology B: Context Amnesia via Exponential Decay (mHC Fix)
* **Symptom:** Inference ignores the user prompt entirely (`hola`, `como te va`), generating an identical stream of words starting with `Ġstrugg`.
* **Root Cause:** During the "NaN-ASM collapse recovery" (P-13), a fallback was implemented in `main.rs` that overrides collapsed mHC weights with `alpha = 0.85` and `beta = 0.15`.
* **Mathematical Reality:** A residual stream operates as $h_n = \alpha h_{n-1} + \beta f(h_{n-1})$. By setting $\alpha = 0.85$ strictly, the residual connection becomes an exponential low-pass filter. Over 30 layers, the initial token embedding (the user's prompt) decays by $0.85^{30} \approx 0.0076$ (99.24% loss of signal). The context is physically deleted from the network.

### Pathology C: Vocab Manifold Drift (Bipartite Attractor)
* **Symptom:** The generation falls into an endless loop of high-norm Latin-origin roots (`Ġstrugg Ġenthus Ġinterpre Ġpsychiat`).
* **Root Cause:** Because the context is destroyed (Pathology B) and the weights are untrained (Pathology A), the network drifts randomly. Greedy search picks tokens with the largest geometric bias in the embedding space (typically rare abstract terms). 

---

## 2. Restructuring Plan

To resolve this, we must completely restructure the training and inference hot-paths. Patches are insufficient; the mathematical equations governing the forward and backward loops must be rebuilt.

### Phase 1: Re-architecting `mHC` (Manifold-Constrained Hyper-Connections)
**Target:** `src/mud/slime_forward.rs` and `src/main.rs`
* **Objective:** Prevent unbounded `VarH` growth without causing Exponential Decay.
* **Restructuring:**
  1. Remove the `0.85/0.15` fallback in `main.rs`. $\alpha$ must default strictly to `1.0`.
  2. Rewrite `mhc_residual()` in `slime_forward.rs`. Instead of scaling down the sum $h + f(h)$, we should project $f(h)$ geometrically before addition, or use a strict clipping norm on $f(h)$ to guarantee bounded updates.
  3. Formulation: $h_{n} = h_{n-1} + \text{clipNorm}(f(h_{n-1}), \text{radius})$. This guarantees `VarH` doesn't explode while preserving the identity mapping perfectly.

### Phase 2: Restructuring the QAT Loss Landscape
**Target:** `src/mud/corpus_trainer.rs`
* **Objective:** Abolish the 8-class illusion and restore true gradient pressure to the QAT optimizer (Muon/Adam).
* **Restructuring:**
  1. Remove `NUM_NEG = 7` negative sampling logic.
  2. Implement **Full-Vocabulary Cross-Entropy** or a massive sampled softmax (e.g., `8192` classes). While full vocab (128k) is computationally expensive on CPU, a dynamic batch of 8192 negative samples guarantees that a loss of `2.06` genuinely represents $12\%$ confidence over 8000 tokens (an incredibly strong signal) instead of random guessing over 8.
  3. Ensure that the loss calculation respects the `output_norm` natively.

### Phase 3: JEPA Covariance Thawing
**Target:** `src/mud/slime_jepa.rs`
* **Objective:** Prevent `VarJ = 0.00` EMA death.
* **Restructuring:**
  1. If `VarJ` decays to 0, the JEPA gate effectively acts as a global scalar, breaking the gating mechanism.
  2. Inject structural jitter (`NEURAL_KICK_JITTER = 1e-5`) actively into the EMA vector `z` during training, preventing the dimensions from collapsing into a singular cross-dimensional mean.

---

## 3. Implementation Order

1. **Modify `main.rs`**: Eliminate the `MhcFix` fallback that sets $\alpha = 0.85$.
2. **Modify `slime_forward.rs`**: Re-code `mhc_residual()` to implement $\alpha=1.0$ identity-preserving projection.
3. **Modify `corpus_trainer.rs`**: Nuke `NUM_NEG=7`. Write a `full_vocab_loss` or a `large_sampled_softmax_loss` (e.g., 4096-8192 negative samples).
4. **Train**: Run `./mud.sh train --epochs 1` to observe the true loss (which should start at ~`9.00` for 8192 samples and slowly drop).
