# Original User Request

## 2026-06-10T19:25:32Z

# Teamwork Project Prompt — Draft

> Status: Launched
> Goal: Craft prompt → get user approval → delegate to teamwork_preview

Complete the remaining cleanup and optimization tasks from the V13 Audit Action Plan for Forge LLM (MUD) to deduplicate Vulkan dispatches, remove dead code, and investigate/optimize Vulkan iGPU latency.

Working directory: /home/ale/proyectos/forge_llm
Integrity mode: development

## Requirements

### R1. Vulkan Dispatch Code Deduplication (DRY)
Refactor src/vulkan/mod.rs to deduplicate descriptor set creation, pipeline binding, and push constant setup.
- Extract common logic from `run_ternary_gemm_cached` and `run_ternary_gemm_cached_async`.
- Extract common logic from `pulse_heartbeat` and `dispatch_imagination_async`.

### R2. Dead Code and Unused Variables Cleanup
Clean up the following dead code identified in the V13 Audit:
- Remove the unused field `sample_probs` from InferenceWorkspace in src/mud/workspace.rs.
- Remove the unused computed variables `_cos_sim` and `_l2_shift` from src/mud/forward.rs.

### R3. Vulkan iGPU Latency Profiling and Optimization
Analyze the iGPU vs CPU latency discrepancy (currently measured at +575.02 ms in diagnostics). Review Vulkan barriers and synchronization flags in src/vulkan/mod.rs to optimize memory transfers and command execution flow.

## Acceptance Criteria

### Compilation & Tests
- [ ] Code builds with 0 errors and 0 warnings: `cargo clippy --all-targets --features tools -- -D warnings`
- [ ] Code passes all unit tests: `cargo test --release --lib`

### Code Quality (DRY & Cleanup)
- [ ] Boilerplate code for Vulkan command recording and descriptor set configuration in src/vulkan/mod.rs is extracted into helper functions.
- [ ] Field `sample_probs` is completely removed from InferenceWorkspace without compiler issues.
- [ ] `_cos_sim` and `_l2_shift` are removed from src/mud/forward.rs.
