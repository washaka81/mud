---
lang: en
---

> **[HISTÓRICO — Reemplazado por `MUD_AUDIT_LATEST.md`]**
> El baseline estadístico y análisis de sigmas/deltas está consolidado en las secciones 5 y 7 de `docs/MUD_AUDIT_LATEST.md`.

# MUD Deep Statistical & Delta Audit Report

## 1. Nanometric Weight Sigma Analysis (Conversion Deltas)
We audited the weight distributions comparing the FP32 dense reference vs. the 1.58b Ternary implementation.

| Tensor Layer | Type | Sigma (StdDev) | Mean | Sparsity |
| :--- | :--- | :--- | :--- | :--- |
| `token_embd.weight` | Float32 | **0.130534** | -0.000520 | 0.00% |
| `blk.0.attn_q.weight` | Ternary | **0.568650** | -0.000030 | 67.70% |
| `blk.15.expert.0.w2.weight` | Ternary | **0.680365** | 0.000290 | 53.70% |
| `blk.29.attn_output.weight` | Ternary | **0.688847** | 0.000931 | 52.55% |

### Analysis:
*   **Variance Inflation:** The Sigma of ternary weights is **~5x higher** than the dense embedding reference. This indicates that the 1.58b rounding aggressively amplifies weight magnitude to compensate for lost precision, injecting significant noise into the forward pass.
*   **Sparsity Manifold:** Sparsity hovers around 50-60%. This is healthy for BitNet 1.58b, but without QAT, the specific zeros are chosen by magnitude rather than semantic importance.

## 2. Runtime Neural Autopsy (Engine Sigmas & Deltas)
We tracked the hidden state deltas (`X-Move`) and activation variances (`LogitVar`) during a 48-step inference cycle.

| Metric | Baseline (Step 1) | Peak / Range | State Status |
| :--- | :--- | :--- | :--- |
| **LogitVar (Sigma^2)** | 19.15 | 15.31 - **49.35** | **Explosion detected** |
| **Entropy** | 5.95 | 0.08 - 6.52 | **Instability / Collapse** |
| **X-Move (Delta)** | 0.00 (Init) | 41.31 - **58.01** | **Chaotic Trajectory** |

### Critical Findings:
1.  **Logit Explosion:** The `LogitVar` frequently exceeds 30.0. This causes the Softmax function to produce near-one-hot distributions even when the model is mathematically "guessing." This leads to the "and and and" stuttering because the model locks into a local noise peak.
2.  **Chaotic Manifold:** The `X-Move` (Euclidean distance between sequential hidden states) remains extremely high (>50.0). A healthy converged model should show `X-Move` decaying or stabilizing as it follows a semantic path. Here, each token jump is a violent re-orientation of the entire hidden vector.
3.  **Entropy Collapse:** At steps like #12, entropy drops to `0.08`. The model becomes 99.9% certain of a token that is statistically noise, indicating that the ternary quantization has created "dead ends" in the logic graph.

## 3. Conclusion & Remediation
The audit confirms that the **structural MUD engine is robust** (it handles the dynamic ranges without crashing), but the **PTQ (Post-Training Quantization) math is insufficient** for Dense-to-Ternary conversion.

**Immediate Remediation (Roadmap):**
1.  **Layer-wise Scale Dampening:** Use `mud_calibrator` to manually lower the `.scale` of late-layer experts to bring `LogitVar` below 15.0.
2.  **Activation-Aware Distillation:** Implement the QAT loop to align the Ternary Sigmas with the original FP32 activation manifold.
3.  **Delta Clipping:** Introduce a dampening factor in the residual connections to prevent `X-Move` from exceeding 30.0.
