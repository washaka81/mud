# Forge LLM (MUD) - Project Instructions

## 1. Architectural Mandates
- **Zero-Allocation Policy:** The inference hot-loop MUST NOT perform dynamic memory allocations. Use `InferenceWorkspace` pre-allocated buffers.
- **Jamba Hybrid Engine:** Weights are stored in 1.58-bit (ternary) format. Support for interleaved Transformer Attention and Mamba SSM layers.
- **O(1) Context Scaling:** Mamba layers MUST utilize the fixed-state SSM scan to ensure constant memory footprint regardless of context length.
- **High-Fidelity Scaling (PRQ):** All ternary tensors MUST use **Per-Row Quantization**. This standard applies to both Attention and Mamba layers.
- **Gradient Sanitization:** All gradients MUST be checked for finiteness (`is_finite()`) and clamped before being applied to shadow weights to prevent catastrophic "Zero-Sigma" matrix collapse during autonomous training.
- **Forced Hot PRQ:** Shadow weights in FP32 MUST be explicitly scaled, rounded, and clamped to `[-1.0, 0.0, 1.0]` before being packed back into the `.mud` format.

## 2. Technical Standards
- **SIMD Priority:** Math kernels are implemented in AVX2 assembly (`src/asm/*.s`).
- **Memory Safety:** Use `unsafe` blocks only when necessary for performance, clearly marked and audited.
- **Bilingual Core:** The tokenizer and training corpus are optimized for Spanish and English.
- **Universal Agnosticism:** Every tool and pipeline step (Conversion, Calibration, Training) must be designed to work across multiple model architectures without hardcoded dependencies.

## 3. Calibration & Restoration Pipeline (UCP)
Every new model converted to MUD format MUST follow the **Universal Calibration Protocol (UCP v2)**:
1. **Convert (PRQ):** Use `universal_converter` with depth-dampened row-wise scaling.
2. **Verify Conversion:** Run `conversion_verifier` to assert SQNR $\ge 10.5$ dB and HiPPO eigenvalue stability (negative eigenvalues).
3. **Verify Boundary Security:** Run `boundary_validator` to assert ternary grid conformity (no fractional weights) and scale boundary safety (all scales $\ge 10^{-8}$, scale COV < 0.12).
4. **Estimate Workload:** Run `training_estimator` to calculate optimal hyperparameters, required alignment tokens, dynamic weight decay $\lambda$, and SGD seating steps.
5. **Restore IQ:** Run `./mud.sh restore-iq` (Warmup + Cosine LR Schedule + STE QAT) to seat the model into the ternary grid.
6. **Assert Effectiveness:** Run `iteration_validator` to verify the composite score is **$\ge 96\%$**, certifying the model for production.

## 4. Development Workflow
- **Research:** Check `docs/` and `GEMINI.md` before making architectural changes.
- **Validation:** Run `boundary_validator` and `iteration_validator` after model changes or training chunks.
- **Testing:** Add unit tests in `src/asm/tests.rs` or `src/model/tokenizer_test.rs` for core logic changes.

---
## 5. Architectural Audits & Research Pivots
- **Ternary Shock (Audit V3 & V5):** Empirical testing demonstrated that PTQ (Post-Training Quantization) directly to 1.58-bit causes irreversible semantic aphasia despite mathematical stability (Sigma > 0.4). The models produce random BPE tokens.
- **Current Development Focus (Audit V6 & V7 - Resolution):** The engine has successfully pivoted to **Quantization-Aware Training (QAT) via Straight-Through Estimator (STE)** inside the native corpus trainer. The forward pass now forcefully simulates ternary truncation on-the-fly (`[-1, 0, 1] * scale`), calculating strict gradients that are then applied to the FP32 shadow weights. This natively develops structural resilience to the ternary boundaries, resolving the need for Knowledge Distillation or INT8 fallbacks.
- **Statistical Health & Code Polish (Audit V8):** The engine enforces mathematical homeostasis using metrics like **Sigma (σ)** (Variance), **Delta (Δσ)** (Entropy Deviation), **Epsilon (ε)** (RMS Stabilization), and **Lambda (λ)** (Dynamic Weight Decay penalization) to guarantee coherent models. The Rust codebase maintains a strict **0-warning, 0-error** policy via `cargo clippy`, ensuring structural integrity, memory safety (no unsafe raw pointer casts), and rigorous algebraic clamps on all loss and scale calculations.
- **Mathematical Paradox & Attention Looping (Audit V9):** The inference engine now requires a strict `1e-8` floor (Epsilon) across all Key/Value caching and RMS Normalization steps to prevent massive repetition loops. Furthermore, PRQ quantization mandates a `0.707` depth-dampening factor against the absmean to accurately achieve 26.0% Sparsity and resolve the Target Sigma paradox (which corrects the true variance limit of the ternary grid to $\sigma=0.86$).

---
## 6. Neuro-Symbolic & Recursive Architectures (Future Roadmap)
The engine aims to transition from traditional "Width Scaling" (probabilistic brute-force parameter expansion) to **Recursive Reasoning Models (RRMs)** to decouple computational depth from parameter scale:
- **Depth vs Scale:** Favor iterative refinement of latent states (deliberative "slow thinking") over massive single-pass networks ("fast thinking").
- **Lattice-based Deduction (LDT):** Utilize structured, mathematically grounded deterministic pathways (similar to the LDT architecture, which projects states onto a logical lattice) to ensure 100% accuracy in reasoning tasks while operating with sub-2M parameter footprints, actively avoiding LLM hallucinations.
- **Probabilistic Width Scaling (PTRM):** When deploying probabilistic paths, use mechanisms like Q-head selection to inject stochasticity solely during inference, preventing "Single Attractor" cognitive loops.
- **Goal:** Small, hyper-efficient models capable of running locally on constrained hardware while executing complex symbolic logic that outperforms multi-billion parameter Frontier LLMs.

---
*MUD: Static, Ternary, High-Fidelity.*
