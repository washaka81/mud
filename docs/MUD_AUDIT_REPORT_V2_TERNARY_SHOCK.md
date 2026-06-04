# MUD Audit Report V2: Ternary Shock & Signal Decay
**Date:** 26 de mayo de 2026
**Subject:** Diagnostic of catastrophic semantic failure in Qwen v3 (0.5B) and similar ternary models.

## 1. Executive Summary
The transition to Qwen v3 (0.5B) revealed a critical failure mode designated as **"Ternary Shock"**. Despite correct mathematical implementation of the AVX2 GEMV kernels, the model produces unintelligible output (gibberish). Investigation confirms this is not a code bug, but a statistical collapse caused by insufficient quantization granularity.

## 2. Technical Diagnosis: Signal Flow Analysis
Using `diagnose_layers.rs`, we traced the signal standard deviation (std) and ranges through the 24-layer stack:

| Stage | std | Range [min, max] | Status |
| :--- | :--- | :--- | :--- |
| **Embedding** | 0.0098 | [-0.012, 0.012] | **CRITICAL:** Too low. |
| **RMS Norm Scale** | 101.09 | — | **AMPLIFICATION:** High noise. |
| **Layer 0 (Q Projection)** | 0.39 | [-1.5, 1.5] | Loss of variance. |
| **After Layer 24** | 7.54 | [-31, 27] | **EXPLOSION:** Drift accumulation. |
| **Logits (Raw)** | 172.65 | [-566, 581] | Saturated. |
| **Logits (Scaled)** | 2.13 | [-6.9, 8.1] | **FLAT:** No semantic peak. |

### Root Cause: Global Tensor Scaling
The current `universal_converter` uses a single `scale` factor for the entire weight tensor.
- **Problem:** Different rows in a weight matrix (e.g., Attention Heads or FFN neurons) have vastly different magnitudes.
- **Effect:** A single global scale "squashes" rows with smaller weights to zero or forces rows with large weights into high-error ternary approximations.
- **Accumulation:** Across 24 layers and hundreds of matrix multiplications, the cumulative quantization error (Noise-to-Signal ratio) exceeds 100%, destroying the learned weights' structure.

## 3. The "Multi-Model" Problem
This is not specific to Qwen. Any model converted to 1.58-bit (ternary) without **Per-Row Scaling** will suffer from this collapse. The more layers the model has, the faster the signal decays into noise.

## 4. Remediation Strategy: The High-Fidelity Pipeline
To restore IQ across the entire MUD ecosystem, we are implementing a three-pillar fix:

### Phase A: Per-Row Quantization (PRQ)
- Modify `universal_converter/quantizer.rs` to calculate and store a separate `scale` for each row of every weight matrix.
- Update the `.mud` format to support scale vectors instead of scalars.
- Update `MudInference` to apply scales per-row during the GEMV pass.

### Phase B: Bayesian Quality Control (Project)
- After conversion, run `recalibration_projector.rs` to adjust scales based on actual activation statistics, not just static weights.

### Phase C: Live SGD (Restore-IQ)
- Standardize the `./mud.sh restore-iq` command which automates:
    1. **Align:** Mapping corpus tokens to the new expanded vocabulary.
    2. **Project:** Weight adjustment.
    3. **Train:** A short, high-learning-rate "Fine-Tuning" burst to let the ternary weights settle into their new quantized manifolds.

## 5. Conclusion
Ternary weights are viable but require surgical precision in scaling. Moving from **Global-Scale** to **Per-Row-Scale** is the mandatory evolution for the MUD engine to support models larger than 0.1B with high intelligence.
