# BRIEFING — 2026-06-10T19:29:10Z

## Mission
Investigate Vulkan dispatch code deduplication, dead code candidates in workspace/forward, and Vulkan iGPU latency profiling and optimization.

## 🔒 My Identity
- Archetype: Explorer
- Roles: Read-only investigator, codebase analyst
- Working directory: /home/ale/proyectos/forge_llm/.agents/explorer_cleanup_1/
- Original parent: 6edeff00-d954-42fd-bb6c-2ee02b3386e8
- Milestone: Explorer Cleanup Analysis

## 🔒 Key Constraints
- Read-only investigation — do NOT implement
- Must follow 5-component handoff report
- No external web search (code-only mode)

## Current Parent
- Conversation ID: 6edeff00-d954-42fd-bb6c-2ee02b3386e8
- Updated: 2026-06-10T19:29:10Z

## Investigation State
- **Explored paths**:
  - `src/vulkan/mod.rs` (Vulkan context, GEMM dispatches, heartbeat, synchronization)
  - `src/vulkan/vulkan_backend.rs` (C-Ffi entrypoints, memory allocation)
  - `src/mud/workspace.rs` (InferenceWorkspace struct definition)
  - `src/mud/forward.rs` (Forward pass, hybrid layer loops, trace propagation)
- **Key findings**:
  - `run_ternary_gemm_cached` and `run_ternary_gemm_cached_async` differ only in blocking wait.
  - `pulse_heartbeat` and `dispatch_imagination_async` differ only in dispatch size and returning future vs discarding.
  - Descriptor set, command builder, and execution synchronization logic are heavily duplicated across all dispatch functions.
  - The dead code fields `sample_probs` (in `InferenceWorkspace`) and `_cos_sim` / `_l2_shift` (in `forward.rs`) are already absent/cleaned up in the current codebase state.
  - Vulkan iGPU latency overhead is caused by:
    1. Dynamic memory allocations on the hot-loop (`allocate_zero_copy_buffer` on every GEMM).
    2. Reading from CPU write-combined memory types instead of cached memory.
    3. Missing pipeline barriers in sequential dispatches with dependencies (e.g., `run_chained_ffn`).
    4. Sync execution waits on supposedly async functions.
- **Unexplored areas**: None.

## Key Decisions Made
- Identified optimal helper functions (`create_descriptor_set`, `create_command_builder`, `bind_pipeline_and_set`, and `execute_command_buffer`) to achieve DRY without compiler type errors.
- Formulated Vulkan iGPU memory caching and barrier placement strategies.

## Artifact Index
- /home/ale/proyectos/forge_llm/.agents/explorer_cleanup_1/ORIGINAL_REQUEST.md — Original user request
- /home/ale/proyectos/forge_llm/.agents/explorer_cleanup_1/BRIEFING.md — Briefing file
- /home/ale/proyectos/forge_llm/.agents/explorer_cleanup_1/progress.md — Progress tracker
