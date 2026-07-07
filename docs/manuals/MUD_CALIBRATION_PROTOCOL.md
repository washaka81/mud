# MUD Universal Calibration Protocol (UCP)
**Version:** 1.0 (High-Fidelity)
**Date:** 26 de mayo de 2026

## 1. Overview
The **Universal Calibration Protocol (UCP)** is a model-agnostic framework designed to bridge the gap between full-precision (FP16/BF16) architectures and the MUD Ternary (1.58-bit) execution environment. This protocol ensures that any transformer-based model (Qwen, Llama, Mistral, DeepSeek, etc.) can be "domesticated" into a high-performance ternary state without losing semantic coherence.

## 2. Calibration Tiers

### Tier 1: Static Per-Row Quantization (PRQ)
- **Tool:** `universal_converter`
- **Method:** Each weight row is quantized using an `absmean` heuristic.
- **Use Case:** Initial conversion and rapid prototyping.
- **Accuracy:** Moderate. Sufficient for small models (<0.5B) but prone to "Ternary Shock" in deep architectures.

### Tier 2: Bayesian Quality Control (Projection)
- **Tool:** `recalibration_projector.rs`
- **Method:** Post-conversion statistical analysis of the weight manifold. It identifies "dead" neurons or exploding variances and adjusts scales using a Bayesian-inspired boost factor.
- **Use Case:** Fixing "Zero-IQ" outputs immediately after conversion.

### Tier 3: Activation-Aware Calibration (Data-Driven)
- **Tool:** `mud_calibrator.rs` (Manual) / `recalibration_projector.rs --data` (Planned)
- **Method:** Runs a small calibration corpus (e.g., 1000 tokens of high-quality text) through the model. It records activation distributions and optimizes the per-row scales to minimize the Mean Squared Error (MSE) between full-precision and ternary activations.
- **Use Case:** Mandatory for models >7B or production-grade deployments.

### Tier 4: Cognitive Restoration (Restore-IQ)
- **Tool:** `./mud.sh restore-iq`
- **Method:** Live Stochastic Gradient Descent (SGD) on the ternary manifold. The model's weights are allowed to drift slightly to compensate for the discrete mapping error.
- **Use Case:** Final stage of the calibration pipeline.

## 3. Universal Applicability
The UCP is designed to be **Architecture-Agnostic**:
1. **Normalization:** Handles RMSNorm, LayerNorm, and custom norms through automatic parameter detection.
2. **MoE Complexity:** Calibrates expert-specific scales individually, ensuring that specialized experts (e.g., coding, math) maintain their magnitude relative to common-knowledge experts.
3. **Vocabulary Scaling:** Adjusts embedding scales based on token frequency if a frequency map is provided.

## 4. Execution Workflow
For any new model, the recommended workflow is:
1. **Convert:** `universal_converter input.safetensors output.mud --ternarize-emb`
2. **Analyze:** `recalibration_projector output.mud`
3. **Calibrate:** `recalibration_projector output.mud --boost` (if Signal-to-Noise is low)
4. **Restore:** `./mud.sh restore-iq --model output.mud`

---
*MUD: Intelligence is not in the precision of the bits, but in the calibration of the manifold.*
