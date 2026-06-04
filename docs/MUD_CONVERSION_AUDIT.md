# MUD Conversion Audit: Ternary Precision Analysis
**Last Update:** 26 de mayo de 2026

## History of Quantization
Initially, MUD used a **Global Scaling** approach for ternary weights (1.58-bit). This worked for small experimental models (v37) but failed catastrophically when applied to larger architectures like Qwen2-0.5B (24 layers).

## The Collapse: "Ternary Shock"
Models converted with global scaling exhibit "Ternary Shock"—a state where the mathematical operations are correct, but the semantic information is lost.

### Diagnostic Findings
1.  **Noise Accumulation:** Each layer adds quantization noise. With 24 layers, the noise standard deviation (std) grows to ~7.5, completely drowning out the signal (std ~0.4).
2.  **Symmetry Loss:** Global scaling assumes all "heads" or "neurons" in a matrix have similar weight distributions. They do not.
3.  **Logit Flattening:** Final logits show a flat distribution where every token is equally probable, resulting in gibberish text output.

## Corrective Strategy: Per-Row Scaling (PRQ)
To resolve this for **all models** (Qwen, Llama, Gemma, Mistral, etc.), the conversion protocol is being upgraded to Per-Row Quantization.

### PRQ Specification
- **Granularity:** 1 floating-point scale per output dimension (row) of the weight matrix.
- **Storage:** For a matrix of size `[N, M]`, we store `[N, M/16]` packed u32s and `[N]` f32 scales.
- **Overhead:** Negligible (adds only 4 bytes per row, ~0.1% increase in total model size).
- **Gain:** Preserves the relative importance of different features (neurons/heads), reducing quantization error by up to 10x per layer.

## The Multi-Model Restoration Pipeline
This protocol is model-agnostic and must be applied sequentially to ensure intelligence survival:

1.  **Conversion:** `universal_converter` extracts weights and generates initial row-wise scales.
2.  **Deployment:** Model is loaded into the Static Workspace engine.
3.  **Agnostic Calibration:** `recalibration_projector` performs Bayesian checks across all layers.
4.  **Linguistic Seating:** `restore-iq` performs a short-burst retraining cycle to adapt the model to the ternary discrete space and the MUD vocabulary.

## Status: RECALIBRATION REQUIRED
All models (regardless of architecture) converted prior to 26 May 2026 are considered "Low-Fidelity" and must be re-converted using the new PRQ-enabled `universal_converter`.
