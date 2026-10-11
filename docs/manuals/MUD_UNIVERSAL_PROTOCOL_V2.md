# MUD Universal Protocol V2: Overcoming Ternary Shock

**Date:** May 27, 2026
**Scope:** Universal Multi-Model Standard (Agnostic for Qwen, Llama, Mistral, DeepSeek, etc.)

## 1. The Problem: "Ternary Shock" and Signal Decay

During the conversion of high-parameter models to the 1.58-bit ternary format (`-1, 0, 1`), the system experienced catastrophic signal degradation ("Ternary Shock"). 

### 1.1 Root Cause Analysis
An extensive audit of the signal flow revealed the following:
- **Global Scaling Failure:** Applying a single, global scaling factor per tensor destroys the internal structure of the weights. A single scale factor introduces ~50% quantization error per element.
- **Variance Explosion:** While the initial ternary embedding starts small (e.g., standard deviation `~0.01`), RMS Norm compensates by amplifying the signal (e.g., `101x`). This amplification multiplies the quantization noise.
- **Residual Accumulation:** Over the course of deep architectures (e.g., 24 layers), these residual errors accumulate exponentially. Despite delta clipping, the output normalization scales the errors out of bounds, resulting in featureless, homogenous logits (gibberish output).

## 2. The Solution: Per-Row Quantization (PRQ)

To preserve the cognitive health of the model without altering the binary memory footprint, MUD enforces **Per-Row Quantization (PRQ)** across all ternary matrices (Attention Q/K/V/O, FFN). 

Instead of one scale per tensor, each row maintains its own distinct floating-point scale. This provides fine-grained fidelity that mirrors the performance of the FP16 originals while maintaining zero-allocation ternary inference.

## 3. The Universal Pipeline

Because this issue transcends individual architectures, MUD now enforces a strict **Universal Calibration Protocol (UCP)** for *all* supported models, spanning from initial conversion to continuous retraining.

### Step 1: Universal Conversion (PRQ)
All models must be ingested through the universal converter using the PRQ standard.
```bash
./mud.sh convert [input_safetensors] [output_mud] --ternarize-emb
```
*Note: Ensure the target architecture is correctly mapped in `parser.rs`. The converter automatically applies row-wise scaling to all linear layers.*

### Step 2: Calibration & Analysis
Before inference, the model must be audited for signal health and calibrated to minimize Mean Squared Error (MSE) between the ternary and original floating-point activations.
```bash
./mud.sh project [output_mud] --boost
```
*The `recalibration_projector` assesses the Signal-to-Noise Ratio (SNR) and adjusts the PRQ scales to optimize activation preservation.*

### Step 3: Linguistic Restoration (Restore-IQ)
Even with PRQ, models experience initial misalignment. To fully "seat" the weights onto the ternary manifold, a short-burst Stochastic Gradient Descent (SGD) retraining cycle is mandatory.
```bash
./mud.sh restore-iq [output_mud]
```
*This step uses latent FP representations or scale adjustments to fine-tune the model against the `knowledge.db`, recovering coherence and semantic structure.*

## 4. Retraining & Evolution
The pipeline guarantees that models can continuously evolve. Using the local Rust `MudAutoTrainer` or the Kaggle sync scripts (`training/push_to_kaggle.sh`), the PRQ-seated models absorb new chunks from the PageRank-driven knowledge database without suffering from cumulative quantization degradation.
