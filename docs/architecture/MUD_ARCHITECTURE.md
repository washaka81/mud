# MUD Architecture - High-Fidelity Hybrid Engine

## Overview
The MUD (Modular Understanding Dynamics) engine is a hardware-aware, zero-allocation inference system optimized for ternary (1.58-bit) weights. It is natively designed to run **Jamba Hybrid Architectures**, combining the infinite context scaling of Selective State Space Models (Mamba) with the relational logic of Transformer MoEs.

## 1. Jamba Hybrid Design
MUD interweaves sequence and relational processing to maximize efficiency:
- **Transformer Layers ($O(N^2)$):** Sparse attention blocks equipped with Mixture of Experts (MoE). Responsible for logical leaps and deep relational understanding.
- **Mamba Layers ($O(N)$):** Fast, recurrent State Space Models. Responsible for sequence scanning and memorization with an $O(1)$ memory footprint, eliminating KV-cache explosion.

## 2. Static Workspace (Zero-Allocation)
To achieve extreme performance (>160 TPS on CPU), MUD pre-allocates all necessary memory buffers upon model loading.
- **InferenceWorkspace:** A monolithic structure containing all intermediate tensors, Attention KV-cache, and Mamba SSM recurrent states.
- **Static Pointers:** All weight access is done via raw pointers to mmap'ed memory, eliminating lookup overhead.

## 3. High-Fidelity Quantization (PRQ)
The core innovation of version 1.5 is **Per-Row Quantization (PRQ)**.
- **Problem:** Global scaling causes SNR collapse across deep layers.
- **Solution:** Every output dimension (row) of every matrix (Attention, FFN, Mamba Projections) has a dedicated 32-bit float scale.
- **Format:** `.mud` files store packed u32 weights and a separate f32 scale vector for each tensor.

## 4. Hardware Acceleration Layers
MUD dynamically selects the best execution path:
- **AVX2 ASM (CPU):** Hand-written assembly for ternary GEMV, RMSNorm, SiLU, and the **Mamba Parallel Scan** algorithm.
- **Vulkan (iGPU) & Asynchronous Heartbeat:** Zero-copy compute shaders for parallelized batch processing. Uses an async heartbeat mechanism to keep the GPU active on output projections without blocking the CPU's sequence loop.
- **Rayon:** Multi-threaded expert execution for MoE layers.

## 5. Linguistic Restoration Pipeline
Since ternary weights are discrete, they require "seating" after conversion:
- **Align:** Maps weights to the bilingual MUD vocabulary.
- **Project:** Bayesian recalibration of scales via activation analysis.
- **Train:** Live Hot Ternary SGD to minimize quantization drift across both Attention and Mamba layers.

## 6. Safety & Integrity
- **Pointer Alignment:** Strict adherence to `(n + 15) / 16` block alignment.
- **Sanitization:** Auto-detection and neutralization of NaNs/Infs in the hidden state.
- **Diagnostic CHI:** Continuous monitoring of the Cognitive Health Index during inference.

---
*Architecture is the foundation of intelligence.*
