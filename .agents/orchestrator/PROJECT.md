# Project: Forge LLM (MUD) Cleanup & Optimization

## Architecture
- **Inference Engine**: Jamba hybrid engine with interleaved Jamba Attention and Mamba SSM layers.
- **Hardware Backend**: CPU (AVX2 asm in `src/asm/`) and Vulkan GPU backend (`src/vulkan/mod.rs`).
- **Quantization**: 1.58-bit ternary quantized weights stored in `.mud` format with Per-Row Quantization (PRQ).

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Vulkan Dispatch Deduplication | Refactor `src/vulkan/mod.rs` to deduplicate descriptor set, pipeline, and push constants setup | None | PLANNED |
| 2 | Dead Code Cleanup | Remove `sample_probs` from `InferenceWorkspace` and `_cos_sim`/`_l2_shift` from `src/mud/forward.rs` | None | PLANNED |
| 3 | Vulkan iGPU Latency Optimization | Analyze and optimize iGPU Vulkan memory barriers and synchronization flags | None | PLANNED |

## Interface Contracts
- **Vulkan Refactoring**: Deduplicated helper functions in `src/vulkan/mod.rs` must preserve the exact external behaviors of `run_ternary_gemm_cached`, `run_ternary_gemm_cached_async`, `pulse_heartbeat`, and `dispatch_imagination_async`.
- **InferenceWorkspace**: Removal of `sample_probs` must compile clean and pass tests.
- **Forward Logic**: Removal of unused variables in `src/mud/forward.rs` must not affect correct forward execution.

## Code Layout
- `src/vulkan/mod.rs` — Vulkan GPU backend, command recording, pipeline configuration, barriers/synchronization.
- `src/mud/workspace.rs` — InferenceWorkspace struct definition and memory buffers.
- `src/mud/forward.rs` — Model forward pass, hybrid Attention/Mamba layers.
