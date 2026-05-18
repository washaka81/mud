# MUD Training Pipeline: Modular Understanding Dynamics

This directory contains the training scripts to create Ternary MoE models for the Forge LLM engine.

## Core Concepts
- **Ternary Weights:** All linear weights are quantized to `{-1, 0, 1}` during training using the Straight-Through Estimator (STE).
- **BitNet 1.58b Logic:** We follow the scaling and quantization logic from BitNet 1.58b to maximize performance on our additions-only ASM kernels.
- **MoE Routing:** Dynamic expert selection (Top-K) to increase model capacity without increasing per-token compute.

## Setup
1. **Local:** Use `ternary_moe_logic.py` to prototype.
2. **Kaggle / Unsloth:** 
   - Upload the logic scripts to a Kaggle environment.
   - Use Unsloth for optimized training speed on available GPUs.
   - Example Notebook coming soon: `mud_training_v1.ipynb`.

## Exporting to .ai
Once trained, use the `exporter.py` (planned) to convert PyTorch weights into the custom `.ai` binary format.
The exporter will:
- Pack 16 ternary weights (2 bits each) into `u32` blocks.
- Organize expertise into Skill Modules.
- Embed any static knowledge indexes.
