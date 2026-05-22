# MUD Architecture - Static Workspace & MoE Robustness

## Overview
The MUD (Modular Understanding Dynamics) engine has been refactored to achieve >100 TPS by eliminating dynamic memory allocations in the inference hot loop. This ensures consistent execution time, critical for high-latency tasks like complex reasoning.

## Core Design Principles
1. **Static Pre-allocation:** All buffers for logits, KV-cache, and hidden states are pre-allocated at `InferenceWorkspace` initialization.
2. **MoE Stability:** Expert gating uses a stabilized Top-K selection with temperature-adjusted softmax to avoid logit collapse.
3. **Skill-Aware Routing:** Each MUD skill module is dynamically routed through the MoE gate, ensuring compute resources are prioritized for the requested domain (Math, Logic, Programming, Language).
4. **Resilient Tokenization:** The tokenizer uses a validated, zero-copy byte mapping strategy to prevent `<unk>` corruption.

## Roadmap (Post-Refactor)
- [x] Static Workspace implementation.
- [ ] MoE Gating refinement for high-IQ reasoning.
- [ ] Skill module integration optimization.
- [ ] Multi-thread contention resolution (Rayon pool tuning).
