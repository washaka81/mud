# Forge LLM (MUD) Technical Specifications

## 1. Architectural Mandates
- **Zero-Allocation Policy:** The inference hot-loop MUST NOT perform dynamic memory allocations. Use `InferenceWorkspace` pre-allocated buffers.
- **Jamba Hybrid Engine:** Weights are stored in 1.58-bit (ternary) format. Support for interleaved Transformer Attention and Mamba SSM layers.
- **O(1) Context Scaling:** Mamba layers MUST utilize the fixed-state SSM scan to ensure constant memory footprint regardless of context length.
- **High-Fidelity Scaling (PRQ):** All ternary tensors MUST use **Per-Row Quantization**. This standard applies to both Attention and Mamba layers.
- **Gradient Sanitization:** All gradients MUST be checked for finiteness (`is_finite()`) and clamped before being applied to shadow weights.
- **Forced Hot PRQ:** Shadow weights in FP32 MUST be explicitly scaled, rounded, and clamped to `[-1.0, 0.0, 1.0]` before being packed back into the `.mud` format.

## 2. Technical Standards
- **SIMD Priority:** Math kernels are implemented in AVX2 assembly (`src/asm/*.s`).
- **0-Error, 0-Warning Policy:** The Rust codebase MUST strictly maintain 0 compilation errors and 0 warnings via `cargo clippy`.
- **Memory Safety:** Use `unsafe` blocks only when necessary for performance, clearly marked and audited.
- **Bilingual Core:** The tokenizer and training corpus are optimized for Spanish and English.
- **Universal Agnosticism:** Every tool and pipeline step (Conversion, Calibration, Training) must be designed to work across multiple model architectures without hardcoded dependencies.

## 3. Justified Constants
- `DEPTH_DAMPENING_FACTOR = 0.7071` (1/sqrt(2)): Dampens absmean to resolve Target Sigma paradox.
- `SPARSITY_THRESHOLD_RATIO = 0.7`: Maps normal distribution to the 26.0% sparsity boundary.
- `NEURAL_KICK_JITTER = 1e-5`: Neural Kick v2 intensity to prevent deterministic attractor collapse.
- `EPSILON_FLOOR = 1e-8`: Absolute numerical floor for stability-critical divisions.

## 4. Universal Calibration Protocol (UCP v2)
1. **Convert (PRQ):** Depth-dampened row-wise scaling.
2. **Verify Conversion:** SQNR $\ge 10.5$ dB and HiPPO eigenvalue stability.
3. **Verify Boundary Security:** Ternary grid conformity and scale boundary safety.
4. **Estimate Workload:** Calculate optimal hyperparameters.
5. **Restore IQ:** Warmup + Cosine LR Schedule + STE QAT.
6. **Assert Effectiveness:** Composite score $\ge 96\%$.
