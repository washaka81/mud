---
lang: en
---

# MUD Training Pipeline: Modular Understanding Dynamics

This directory contains the scripts and data to train Ternary MoE models for the Forge LLM engine.

## Core Concepts
- **Ternary Weights:** All linear weights are quantized to `{-1, 0, 1}` during training using the Straight-Through Estimator (STE).
- **BitNet 1.58b Logic:** We follow the scaling and quantization logic from BitNet 1.58b to maximize performance on our additions-only ASM kernels.
- **MoE Routing:** Dynamic expert selection (Top-K) to increase model capacity without increasing per-token compute.

## Pipeline Overview
Training is managed via `./mud.sh train` (local) or `./mud.sh train --kaggle` (cloud):
1. **Ingest:** Use `/ingest` in the MUD chat to populate `knowledge.db`.
2. **Sync:** Run `bash training/push_to_kaggle.sh` to upload datasets to Kaggle.
3. **Train:** The Kaggle notebook (`training/notebook*.ipynb`) trains using the uploaded data.
4. **Pull:** Run `bash training/pull_from_kaggle.sh` to download the trained model.

For local training, the Rust `MudAutoTrainer` (built into the engine) processes knowledge base chunks directly, without Python dependencies.

## Files
- `push_to_kaggle.sh` / `pull_from_kaggle.sh`: Cloud sync scripts.
- `kaggle_config.sh`: Kaggle API configuration.
- `dataset-metadata.json` / `kernel-metadata.json`: Kaggle dataset/kernel definitions.
- `KAGGLE_COMMANDS.md`: Step-by-step guide for Kaggle setup.
- `*.txt`: Knowledge corpora (vocabulary, RAE, statistics, synthetic reasoning pairs, etc.).
