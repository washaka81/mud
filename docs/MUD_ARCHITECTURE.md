---
lang: en
---

# MUD Architecture - Static Workspace & MoE Robustness

## Overview
The MUD (Modular Understanding Dynamics) engine has been refactored to achieve >100 TPS by eliminating dynamic memory allocations in the inference hot loop. This ensures consistent execution time, critical for high-latency tasks like complex reasoning.

## Core Design Principles
1. **Zero-Allocation Hot-Loop:** The inference engine follows a "Zero-Allocation" policy. All operational buffers (Q/K/V, attention scores, logits, gate states) are pre-allocated within the `InferenceWorkspace` at startup. This eliminates memory fragmentation and kernel-mode overhead during token generation, achieving >50 TPS on mobile-grade CPUs.
2. **Split RoPE Implementation:** To maintain compatibility with LLaMA and SmolLM2 architectures, MUD uses a "Split" Rotary Position Embedding strategy. Rotations are applied across halves of the head dimension, ensuring accurate positional encoding and preventing linguistic incoherence.
3. **Hardware-Agnostic Acceleration:** The engine supports optional Vulkan acceleration for large models, with a high-performance AVX2/ASM CPU fallback for smaller, latency-sensitive deployments.
4. **MoE Stability:** Expert gating uses a stabilized Top-K selection with temperature-adjusted softmax. Models with a single expert bypass the router entirely to minimize latency.
5. **Skill-Aware Routing:** Each MUD skill module is dynamically routed through the MoE gate, ensuring compute resources are prioritized for the requested domain.
6. **Resilient Tokenization:** The tokenizer uses a validated, zero-copy byte mapping strategy with automatic GPT/SentencePiece space concordance detection.

## Roadmap (Post-Refactor)
- [x] Static Workspace implementation.
- [ ] MoE Gating refinement for high-IQ reasoning.
- [ ] Skill module integration optimization.
- [ ] Multi-thread contention resolution (Rayon pool tuning).
- [ ] Local Trainer in Rust (Native training engine optimized for database chunks).
- [ ] Embedding Pointer Validation (Bounds-checking raw pointer offsets to prevent segmentation faults).
- [ ] Dynamic Attention Integration (Replace causal attention projection placeholder with full scaled dot-product and pos pos cache).
