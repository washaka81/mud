# Forge LLM (MUD) - Project Instructions

## 1. Architectural Mandates

- **Pointer Mastery Mandate (Mandatory):** "El que domina los punteros domina el núcleo de la programación". Raw pointer mastery (`*mut T`, `*const T`) is the supreme truth of the engine. Abstractions must never block direct memory access.
- **Zero-Allocation Policy:** The inference hot-loop MUST NOT perform dynamic memory allocations. Use pre-allocated fixed-size workspace buffers (now `SlimeWorkspace`).
- **SlimeRegister Paradigm (NEW — Priority 27+):** The fundamental compute unit is `SlimeRegister` (`u32` value) mapping directly to physical registers processed in FP32 inside the AVX2 and Vulkan Iris Xe Compute Shaders. The memory representation is split: bits [15:0] carry the accumulated learned `ternary` state, and bits [31:16] are divided between the JEPA and cognitive functions (carrying the running integral and embedded derivatives). These embedded calculus/differential states are directly incrusted in the model's structural parameters and computed natively by the engine. All activations MUST flow through these registers. Storing values as IEEE f16 provides self-contained scaling and range (±65504) without fixed-point scaling, structurally bypassing i16 saturation.
- **Jamba Hybrid Engine:** Weights are stored in 1.58-bit (ternary) format, with ELUT (4-bit nibble) packing as the new hot-path wire format for AVX2 ingestion. Supports interleaved Transformer Attention and Mamba SSM layers.
- **O(1) Context Scaling:** Mamba layers MUST utilize the fixed-state SSM scan to ensure constant memory footprint regardless of context length.
- **High-Fidelity Scaling (PRQ):** All ternary tensors MUST use **Per-Row Quantization**. This standard applies to both Attention and Mamba layers.
- **Gradient Sanitization:** All gradients MUST be checked for finiteness (`is_finite()`) and clamped before being applied to shadow weights to prevent catastrophic "Zero-Sigma" matrix collapse during autonomous training.
- **Forced Hot PRQ:** Shadow weights in FP32 MUST be explicitly scaled, rounded, and clamped to `[-1.0, 0.0, 1.0]` before being packed back into the `.mud` format.
- **Anti-Hardcoding Mandate:** MUD must be strictly agnostic to architectural dimensions. Under NO circumstances shall network dimensions (e.g., hidden size, vocabulary size, attention heads) be hardcoded or fallback to default magic numbers. The engine and its tools must dynamically interrogate physical tensor boundaries to infer dimensions or panic explicitly. Fail-fast is mandatory over silent corruption.
- **JEPA-Integrated Compute (Deterministic):** JEPA is a **pure deterministic system** — no trainable parameters, no gradients, no randomness. The bits [31:16] of every `SlimeRegister` carry the running JEPA integral `I` and cognitive differential functions. The attractor update `I_next = 0.99·I + 0.01·v_jepa` is a fixed low-pass filter running per-register at every block boundary to yield a smooth control gate, integrating derivatives directly embedded in the structural model weights. The underlying JEPA orbital state tracking `z` (along with its mean `mu_z` and `sigma_z`) is maintained as a raw `f32` buffer in `SlimeWorkspace.jepa_z`, avoiding direct register pollution.
- **Ternary Compute (Statistical):** Bits [15:0] carry the accumulated result of learned ternary GEMV — stochastic, QAT-trained, PRQ-scaled. This is the statistical system.
- **Equilibrium Mandate:** The deterministic (JEPA) and statistical (ternary) systems MUST converge to equilibrium: `E[y_final] → mu_ctx`, JEPA correction → 0, ternary weights learn to naturally center their outputs near `mu_ctx`. Every forward pass diagnostic MUST report JEPA correction magnitude per layer. Divergence is a critical bug.
- **i16 Partial-Accumulation Mandate (DEPRECATED):** Previously, due to i16 overflow at `|accum| > 32,767`, every ELUT-AVX2 GEMV kernel was required to perform a mid-row reseat/normalize every 256 elements maximum. With the migration to the dual-f16 `SlimeRegister` (accumulating in f16/f32), this constraint is deprecated.
- **Rust-Only Hardware Saturation Mandate (Intel Core i7-1260P Target):** No Python and no PyTorch are allowed under any circumstances. All execution paths must be implemented natively in Rust. Computation is split between handwritten x86_64 AVX2 Assembly (compiled exclusively for Performance cores / P-cores of the Intel i7-1260P CPU) and asynchronous Vulkan compute shaders running on the integrated Intel Iris Xe GPU (iGPU) to maximize memory-bandwidth and avoid E-core latency overhead.

---

## 2. Technical Standards

- **SIMD Priority:** All compute kernels MUST be implemented in AVX2 assembly (`src/asm/*.s`). Rust intrinsics are permitted only for runtime feature detection. No scalar fallback in the hot-path.
- **No Python — Rust-only Toolchain (Mandatory):** ALL tools, converters, validators, and pipeline steps MUST be implemented in Rust. Python and PyTorch are strictly forbidden for any part of the runtime, training, validation, or conversion pipeline. The codebase must be 100% Rust.
- **Rayon Prohibited (Mandatory):** Rayon is strictly prohibited across the entire project (including all sub-crates like `forge_autograd`). Rayon's thread pool scheduler introduces runtime thread contention and E-core latency that disrupts P-core execution pinning. All loops must run sequentially or use custom ASM/Vulkan threading.
- **0-Error, 0-Warning Policy (Mandatory):** The Rust codebase MUST strictly maintain 0 compilation errors and 0 warnings via `cargo clippy`. Any introduced warning is treated as a critical bug and fixed immediately.
- **Test Mandate (NEW — Mandatory):** Every new module (`*.rs`) MUST include an inline `#[cfg(test)] mod tests { ... }` block with at minimum:
  - One unit test per public function.
  - One edge-case test (e.g., zero-length input, max-size input, overflow boundary).
  - Tests MUST pass under `cargo test --lib` with no panics.
- **Benchmark Mandate (NEW — Mandatory):** Every new compute module (`asm/*.s`, `src/mud/*.rs`, `forge_autograd/*.rs`) MUST ship with a corresponding benchmark binary in `tools/` (e.g., `tools/slime_bench.rs`) registered in `Cargo.toml` as a `[[bin]]`. Benchmarks MUST report: throughput (elements/sec), latency (ns/call), and SIMD utilization (ops/cycle). Benchmarks are run via `./mud.sh bench <name>`.
- **Dead Code Elimination Mandate (NEW — Mandatory):** Any module, function, struct, or file that is no longer referenced by any active code path MUST be deleted. Dead code is NOT archived, NOT commented out, NOT gated behind `#[allow(dead_code)]`. It is removed from the repository. `cargo clippy` with `-D dead_code` enforces this automatically. If a component is removed, its `[[bin]]` entry in `Cargo.toml` and its `./mud.sh` command MUST also be removed.
- **Memory Safety:** Use `unsafe` blocks only when necessary for performance (AVX2 intrinsics, raw pointer arithmetic). Every `unsafe` block MUST have a `// SAFETY:` comment explaining the invariant that makes it sound.
- **Bilingual Core:** The tokenizer and training corpus are optimized for Spanish and English.
- **Universal Agnosticism:** Every tool and pipeline step (Conversion, Calibration, Training) must be designed to work across multiple model architectures without hardcoded dependencies. All constants MUST be mathematically justified and, where possible, derived from model metadata.
- **Justified Constants (Mandatory):**
    - `DEPTH_DAMPENING_FACTOR = 0.7071` (1/sqrt(2)): Dampens absmean to resolve Target Sigma paradox.
    - `SPARSITY_THRESHOLD_RATIO = 0.7`: Maps normal distribution to the 26.0% sparsity boundary.
    - `NEURAL_KICK_JITTER = 1e-5`: Neural Kick v2 intensity to prevent deterministic attractor collapse.
    - `EPSILON_FLOOR = 1e-8`: Absolute numerical floor for stability-critical divisions.
    - `JEPA_ATTRACTOR_LR = 0.01`: JEPA orbital correction rate (linear approximation, zero-EXP).
    - `SLIME_RESEAT_STRIDE = 256`: Maximum i16 accumulation stride before mandatory partial reseat.
- **Dynamic Context Scaling:** All buffer allocations and KV-cache dimensions MUST be derived from `max_position_embeddings` metadata to ensure universal compatibility.

---

## 3. Calibration & Restoration Pipeline (UCP v2)

Every new model converted to MUD format MUST follow the **Universal Calibration Protocol (UCP v2)**:
1. **Convert (PRQ + ELUT):** Use `universal_converter` with depth-dampened row-wise scaling AND ELUT 4-bit nibble packing for the hot-path wire format.
2. **Verify Conversion:** Run `conversion_verifier` to assert SQNR ≥ 10.5 dB and HiPPO eigenvalue stability (negative eigenvalues).
3. **Verify Boundary Security:** Run `boundary_validator` to assert ternary grid conformity (no fractional weights) and scale boundary safety (all scales ≥ 10⁻⁸, scale COV < 0.12).
4. **Estimate Workload:** Run `training_estimator` to calculate optimal hyperparameters, required alignment tokens, dynamic weight decay λ, and SGD seating steps.
5. **Restore IQ:** Run `./mud.sh restore-iq` (Warmup + Cosine LR Schedule + STE QAT via `SlimeRegister` forward pass).
6. **Assert Effectiveness:** Run `iteration_validator` to verify the composite score is **≥ 96%**, certifying the model for production.

---

## 4. Development Workflow

- **Research first:** Check `docs/` and `GEMINI.md` before making architectural changes.
- **Module checklist:** New module → test block → benchmark binary → `mud.sh` entry → doc entry. All four are mandatory.
- **Dead code pass:** After every session, run `cargo clippy -- -D dead_code`. Remove all flagged items.
- **Validation:** Run `boundary_validator` and `iteration_validator` after model changes or training chunks.
- **No orphan tools:** Every `[[bin]]` in `Cargo.toml` MUST have a corresponding `./mud.sh <command>`. Orphan binaries are dead code and must be deleted.

---

## 5. Architectural Audits & Research Pivots

- **Ternary Shock (Audit V3 & V5):** Empirical testing demonstrated that PTQ directly to 1.58-bit causes irreversible semantic aphasia despite mathematical stability (Sigma > 0.4). The models produce random BPE tokens.
- **Current Development Focus (Audit V6 & V7 — Resolution):** The engine pivoted to **QAT via Straight-Through Estimator (STE)** inside the native corpus trainer.
- **Statistical Health & Code Polish (Audit V8):** Enforces mathematical homeostasis using Sigma (σ), Delta (Δσ), Epsilon (ε), Lambda (λ). 0-warning/0-error policy via `cargo clippy`.
- **Mathematical Paradox & Attention Looping (Audit V9):** Requires `1e-8` floor across all KV-caching and RMS Normalization. PRQ mandates 0.707 depth-dampening against absmean.
- **Hybrid INT8 Pipeline — "Beast Engine" (Audit V10):** Inference integrates 1.58-bit Ternary weights with INT8 Activations via AVX2 `VPMADDUBSW`. Now superseded by the **SlimeRegister i16 accumulator** which operates natively on i16 without FP32 intermediate conversion.
- **SlimeRegister Paradigm Shift (Audit V27 — ACTIVE):** The inference workspace is rebuilt around the dual-f16 `SlimeRegister` (`u32`). The accumulated ternary state and the JEPA integral are stored as 16-bit floats (`ternary_f16` and `integral_f16`). ELUT 4-bit nibble packing replaces 2-bit ternary pack in the AVX2 hot-path. Linear JEPA attractor (zero-EXP, 2 cycles) replaces neural JEPA module.

---

## 6. Architecture Roadmap: The "Fast, Efficient, Super Intelligent" Mandate

**THE ULTIMATE MANDATE:** MUD must be *Fast* (bare-metal AVX2 + Vulkan, compute-bound), *Efficient* (SlimeRegister 32-bit dual-state, zero FP32 in hot-path), and *Super Intelligent* (JEPA orbital correction embedded per register). It MUST run locally on low-end hardware (no discrete GPU, low RAM) while outperforming multi-billion parameter Frontier LLMs in complex reasoning.

### Phase 1–4: COMPLETED (Priorities 1–26)
See session reports in `docs/sessions/`.

### Phase 5: SlimeRegister Paradigm (COMPLETED — Priorities 27-31)
27. **Priority 27: [COMPLETED] SlimeRegister Core Substrate**
28. **Priority 28: [COMPLETED] ELUT-AVX2 Kernel (4-bit Nibble GEMV)**
29. **Priority 29: [COMPLETED] Integrated JEPA Attractor (Linear, Zero-EXP)**
30. **Priority 30: [COMPLETED] SlimeForward Pass**
31. **Priority 31: [COMPLETED] Vulkan SlimeShader**

### Phase 6: Code Purge (COMPLETED — Priority 32+)
32. **Priority 32: [COMPLETED] Dead Code Purge**
    After SlimeForward is operational: delete `src/mud/inference.rs` (old FP32 inference), `src/mud/forward.rs` (old FP32 forward), `src/mud/jepa.rs` (old neural JEPA), and all orphan tools. `cargo clippy -- -D dead_code` MUST pass clean.

33. **Priority 33: [COMPLETED] Unified Agentic UI (carry-forward from Priority 19)**
    Interactive orchestration dashboard consolidating engine logs, RLVR validations, and subagent system. Built using `crossterm` and async `mpsc` channels.

### Phase 7: Lexical Resonance (Semantic Attractor)
34. **Priority 34: [COMPLETED] JEPA Lexical Prior Initialization**
    Instead of initializing the `jepa_packed` state to 0 at Layer 0, inject the absolute magnitude (Lexical Energy) of the token embedding. This forces the multiplicative gate to filter Ternary Shock noise against the original semantic intent of the word, effectively making Semantic Aphasia mathematically impossible and hyper-accelerating STE QAT convergence. Documented in `docs/research/JEPA_LEXICAL_RESONANCE.md`.

### Phase 8: Deep QAT Thawing & Vulkan Acceleration (The Final IQ Restoration)
35. **Priority 35: [COMPLETED] SlimeBackward (Thawing the Core)**
    Resolve the "Identity Bypass Syndrome" by implementing the full backward pass (`SlimeBackward`) for the residual 1.58-bit layers. Currently, only the embedding table receives gradient updates. Computing partial derivatives for `attn_q.weight`, `expert.N.weight`, etc., will force the deep ternary structure to adapt to the Lexical Resonance, driving the `Avg Loss` from ~18.5 down to typical converged ranges (< 5.0).
36. **Priority 36: [COMPLETED] Vulkan QAT Dispatcher & Compute Optimization**
    Wired up the QAT gradient accumulation directly into the `VulkanContext`. Replaced CPU loops in `corpus_trainer.rs` with `run_qat_optimizer_async` Compute Shaders.
    - **VRAM Ephemeral Reuse**: Avoided a catastrophic 17.5GB VRAM OOM by dynamically reusing `grad_w`, `scales`, and `packed_w` buffers based on tensor type (e.g., `attn_q`) instead of unique tensor names across all 30 layers. This drops ephemeral VRAM requirements to ~275MB.
    - **Subgroup Math Optimization**: Rewrote `shadow_optimizer.comp` to map 1 Workgroup = 1 Row, guaranteeing perfectly coalesced memory accesses (eliminating strided accesses). Integrated `subgroupAdd` for native silicon-level parallel reductions.
36.5. **Priority 36.5: [PROPOSED] Real-Time Regression Telemetry (TUI Graph)**
    Build a real-time TUI regression graph using the collected metrics (`mud_train_metrics.log`) to visually study the training convergence behavior. This will allow us to observe if the `Avg Loss` projects correctly or if there are any divergence/loss spikes. This tool will integrate into the existing Agentic UI.
37. **Priority 37: [COMPLETED] Full UCP Validation**
    Once the core is thawed and accelerated, execute the complete Universal Calibration Protocol (UCP v2). The model passed the `iteration_validator` with a composite score of ≥ 96%, effectively completing the engine's capability to repair any Ternary Shock completely autonomously on consumer hardware.

### Phase 9: Synthetic Self-Play (Autoentrenamiento Autónomo)
38. **Priority 38: [COMPLETED] JEPA Synthetic Alignment**
    Instead of relying strictly on an external static text corpus (`unified_corpus.txt`), use the model's own autoregressive generation (sampling) to produce high-confidence syntactic chains. These self-generated chains are then fed back into the JEPA QAT pipeline. This forces the ternary weights and the JEPA gates to align with the model's *intrinsic* latent space representations, smoothing out edge cases where the external corpus conflicts with the pre-trained structural priors.

### Phase 10: DeepSeek-V4 Algorithm Integration (2026-06-28 — From arXiv:2606.19348)

Research on DeepSeek-V4 identified 4 algorithms directly applicable to the MUD ternary engine. See `docs/research/DEEPSEEK_V4_TERNARY_INTEGRATION.md` for full technical analysis.

39. **Priority 39: [COMPLETED] DSpark Speculative Decoding**
    Implement a lightweight ternary "drafter" model (2–4 SlimeLayers) that proposes K token candidates. The main model verifies all K in a single forward pass. Mathematically lossless (identical output to greedy decoding). Expected throughput gain: +60–85% for inference. New module: `src/mud/speculative.rs`. Open-source reference: https://github.com/deepseek-ai/DeepSpec (MIT).
    - **Input:** `SlimeWorkspace` state + current position
    - **Output:** K candidate token IDs + acceptance mask
    - **Integration:** Called from `main.rs` autoregressive loop when `--speculative` flag is set

40. **Priority 40: [COMPLETED] mHC Residual — Manifold-Constrained Hyper-Connections**
    Replace the current adaptive-clipping residual in `evaluate_slime_block` with a geometrically-bounded projection. Resolves the VarH explosion and the `safe_ceiling` hardcoding violations (P-13) documented in AGENTS.md §9.
    - **Phase 1 & 2 (COMPLETED):** Implemented manifold-constrained residual and parameterized `alpha`/`beta` dynamic weights per layer in `slime_forward.rs`. Eliminates VarH unbounded growth structurally.
    - **Phase 3 (learned radius) [COMPLETED]:** `radius` is a per-layer learnable scalar, loaded and trained dynamically.
    - **Mathematical guarantee:** `∀ layer i: ||h_i|| ≤ radius_i` — eliminates VarH unbounded growth structurally.

41. **Priority 41: [COMPLETED] Muon Optimizer**
    Replace Adam (`adam_step_avx2`) with Newton-Schulz orthogonalization for weight matrix updates in QAT. Reduces training time from ~27h/epoch to an estimated ~10h/epoch. New module: `src/mud/muon.rs`.
    - **Algorithm:** 5 iterations of `X = 1.5X − 0.5X·Xᵀ·X` on the gradient matrix, then apply as SGD
    - **Scope:** Applied only to `attn_q/k/v/o` and `ffn_up/gate/down` (matrix parameters). Adam retained for embeddings and norm weights (vector parameters where orthogonalization is undefined).
    - **Compatibility:** Fully compatible with STE QAT — preserves gradient direction while removing inter-parameter correlations.
    - **P-13 compliance:** No hardcoded shapes — operates on `&[f32]` slices with runtime `rows`/`cols`.

42. **Priority 42: [PROPOSED — FUTURE] CSA/HCA KV Cache Compression**
    Implement Compressed Sparse Attention for long-context inference (32k+ tokens). Current KV cache for a typical 2B parameter model is ~10MB (5 heads × 4096 × 128 × 4B) — manageable. CSA becomes essential at 32k+ context where KV cache would reach ~1.3GB.
    - **CSA:** Learned compression projection `W_compress` reduces KV entries before storage. Lightning Indexer selects top-K most relevant past tokens via sparse attention.
    - **HCA:** Extreme compression + Sliding Window Attention for recent tokens.
    - **Prerequisite:** Priority 22 (Dynamic Context Scaling from metadata) must be fully implemented first.

43. **Priority 43: [COMPLETED] Tequila STE & Dynamic Optimizer**
    - **Tequila Anti-Deadzone STE:** Custom straight-through estimator adapting gradient clipping based on JEPA variance to revive dead neurons without breaking ternary stability.
    - **Dynamic Optimizer Selection:** Adapts between Adam, Muon, and Sparse Adam depending on matrix shape and sparsity.

47. **Priority 47: [DEFERRED] Gradient Checkpointing**
    - Attempted to implement activation recomputation per-block to reduce memory usage during QAT and increase batch size.
    - **Status:** Deferred due to codebase stability issues during complex chunking loops inside `corpus_trainer.rs`. The focus is maintained on keeping the architecture simple and relying on Vulkan's out-of-core offloading instead.

### Phase 11: Heterogeneous Multi-Processing (HMP) / Asynchronous Vulkan Offloading
48. **Priority 48: [COMPLETED] HMP Vulkan Async Offloading on Integrated GPUs**
    Isolate CPU P-Cores for memory-bound sequential tasks (like AVX2 GEMV) to saturate shared RAM bandwidth (e.g., DDR4-2666), and offload isolated **compute-bound (O(N³))** or **asynchronous** tasks to the Vulkan execution units on the iGPU (e.g., Intel Iris Xe) to prevent bus contention.
    - **Muon Newton-Schulz Offload:** [COMPLETED] Shift the 5 dense orthogonalization steps of `X = 1.5X - 0.5X * X^T * X` to Vulkan.
    - **Thermodynamic Telemetry Offload:** [COMPLETED] Use `subgroupAdd` inside Vulkan to reduce large tensors into `VarH`, `VarJ`, and `Z_Entrop` scalars asynchronously, saving CPU cycles.
    - **DSpark Asynchronous Drafter:** [COMPLETED] Run the lightweight ternary drafter model inside the GPU while the CPU verifies the K candidate tokens.
    See `docs/research/HMP_VULKAN_ASYNC_OFFLOAD.md` for full implementation details.

### Phase 12: Advanced Homeostasis & Deep Scaling
49. **Priority 49: [PROPOSED] Adaptive JEPA Attractor Scaling (Dynamic `jepa_alpha`)**
    Implement an adaptive, non-linear `jepa_alpha` that scales with the derivative of `y_norm`. During a "Ternary Shock" (massive variance spike), temporarily increase `jepa_alpha` to 0.1 to instantly stabilize the dimension, then revert to 0.01 to preserve variance.
50. **Priority 50: [PROPOSED] DSpark-Vulkan Asynchronous Ring Buffer**
    Create a shared `VkBuffer` (Ring Buffer) with unified memory (Zero-Copy) to decouple the Vulkan Drafter from the CPU Verifier. The Drafter constantly pushes token candidates in the background, providing true zero-latency speculative decoding.
51. **Priority 51: [PROPOSED] HCA (Hyper-Compressed Attention) for KV Cache**
    Implement sliding window + learned compression projection `W_compress` for historical KV elements (DeepSeek-V4 algorithm). Compresses historical KV cache by 10x, enabling 32k+ context scaling on low-bandwidth memory systems without crushing the DDR4 bus.

### Phase 13: Test-Time Compute (TTC) & Dynamic Inference
52. **Priority 52: [PROPOSED] Integral Saturation Stop-Anchor (TTC)**
    Replace the fixed depth `N_layers` inference pass with a deterministic Early-Exit/Recurrent loop. Evaluate the integral (accumulated sum) of `spring_force` (the JEPA latent derivative). Stop computation dynamically when the integral saturates (predicting diminishing returns), granting instant answers for simple tokens and deep recurring steps for complex logic.

---

## 7. Documentation Mandate

All project documentation MUST be strictly organized within the `docs/` directory:
- **`audits/`**: Sequential audit reports (`MUD_AUDIT_REPORT_VXX*.md`) and `MUD_AUDIT_LATEST.md`.
- **`sessions/`**: Chronological session reports (`MUD_SESSION_REPORT_YYYY-MM-DD.md`) and technical logs.
- **`architecture/`**: Manifestos, architecture overviews, and component specifications.
- **`research/`**: External papers, academic research notes, and ecosystem analysis.
- **`manuals/`**: User guides, protocols, roadmaps, and directory structure files.
- **`dumps/`**: Raw text dumps and temporary debugging outputs.
- **No loose files:** Documentation files placed outside `docs/` subdirectories are treated as dead code and must be moved or deleted.

---

## 8. Orchestration Mandate (mud.sh v3.1 Optimized)

`mud.sh` is the **single canonical entry point** for all tool binaries. Rules:
- **No raw `cargo run`**: Every tool invocation goes through `./mud.sh <command>`.
- **Section discipline**: 9 color-coded sections — Recovery, Training, Conversion, Diagnostics, Benchmarks, Audits, Interaction, Safety, Meta.
- **Consolidation**: Commands MUST be logically grouped into domain namespaces (e.g., `audit [type]`, `bench [type]`, `diag [tool]`, `util [tool]`) instead of polluting the menu with flat command lists.
- **Auto-Selection**: The `select_model()` helper MUST prioritize existing trained checkpoints (`weights/checkpoints/model_latest_checkpoint.mud` or `*trained*.mud`) to bypass interactive prompts and reduce UX friction for everyday workflows like `chat` or `train`.
- **UCP v2 ordering**: `restore-iq` MUST execute: Bound → Estimate → L-QAT → Full-QAT → Project → Validate in strict order.
- **Discoverability**: `./mud.sh tools` prints a symptom-driven catalog.
- **Safety**: Destructive actions (`deep-clean`, `restore`) MUST require explicit user confirmation.
- **No orphan commands**: Any `./mud.sh` subcommand without a corresponding `[[bin]]` in `Cargo.toml` is dead code and must be removed.

---

## 9. Complete Policy Index (Quick Reference)

| ID | Policy | Severity |
|----|--------|----------|
| P-01 | Zero-Allocation hot-loop (SlimeWorkspace) | CRITICAL |
| P-02 | SlimeRegister [f16 / f16] (dual-f16 u32 layout) as fundamental compute unit | CRITICAL |
| P-03 | ELUT 4-bit nibble packing for AVX2 hot-path | CRITICAL |
| P-04 | [DEPRECATED] i16 partial reseat every ≤256 elements | DEPRECATED |
| P-05 | JEPA attractor linear (zero-EXP) at every block boundary | CRITICAL |
| P-06 | 0-Error, 0-Warning (cargo clippy) | CRITICAL |
| P-07 | Rust-only toolchain — No Python, No PyTorch | CRITICAL |
| P-08 | Dead code MUST be deleted (not archived, not commented) | CRITICAL |
| P-09 | Every module MUST have inline unit tests | MANDATORY |
| P-10 | Every compute module MUST have a benchmark binary | MANDATORY |
| P-11 | Every [[bin]] MUST have a ./mud.sh entry | MANDATORY |
| P-12 | x86_64 ASM optimized for P-cores & Vulkan shaders for iGPU (no E-core/scalar hot-path) | MANDATORY |
| P-13 | Anti-hardcoding — no magic dimension constants | MANDATORY |
| P-14 | Gradient sanitization (is_finite + clamp) | MANDATORY |
| P-15 | Forced Hot PRQ clamp to [-1, 0, 1] | MANDATORY |
| P-16 | SAFETY: comment on every unsafe block | MANDATORY |
| P-17 | Fail-fast over silent corruption | MANDATORY |
| P-18 | Documentation in docs/ subdirectories only | MANDATORY |
| P-19 | mud.sh single entry point — no raw cargo run | MANDATORY |
| P-20 | UCP v2 ordering for restore-iq pipeline | MANDATORY |
| P-21 | Universal agnosticism (no architecture-specific hardcodes) | MANDATORY |
| P-22 | Dynamic Context Scaling from max_position_embeddings | MANDATORY |
| P-23 | Bilingual corpus (Spanish + English) | STANDARD |
| P-24 | PRQ Per-Row Quantization on all ternary tensors | CRITICAL |
| P-25 | SQNR ≥ 10.5 dB on conversion verification | CRITICAL |
| P-26 | iteration_validator composite score ≥ 96% for production cert | CRITICAL |
| P-27 | Prohibición absoluta de Rayon (Scheduler e hilos de fondo interfieren con el pinning de P-cores) | CRITICAL |

---

*MUD: Bare-Metal, Ternary, SlimeRegister, High-Fidelity.*
