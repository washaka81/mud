# Plan: Forge LLM (MUD) Cleanup & Optimization

We are running a single-iteration Explorer -> Worker -> Reviewer cycle to address Vulkan deduplication, dead code cleanup, and Vulkan iGPU latency optimization.

## Verification Gate Criteria
- Zero errors and warnings on cargo clippy: `cargo clippy --all-targets --features tools -- -D warnings`
- Passes all tests: `cargo test --release --lib`
- Verification by Reviewers, Challengers, and Forensic Auditor.

## Steps

### Step 1: Exploration
- **Action**: Spawn 3 Explorer subagents to analyze the codebase.
- **Objectives**:
  - Identify redundant Vulkan command recording / descriptor set setup logic in `src/vulkan/mod.rs`.
  - Locate `sample_probs` in `src/mud/workspace.rs` and its usages.
  - Locate `_cos_sim` and `_l2_shift` in `src/mud/forward.rs` and verify they are safe to remove.
  - Analyze Vulkan synchronization, memory barriers, and command queue usage in `src/vulkan/mod.rs` to find the source of the +575.02 ms iGPU latency.
- **Verification**: Receive 3 independent Explorer reports with analysis and recommended fix strategies.

### Step 2: Analysis Synthesis
- **Action**: Read and synthesize the reports. Reconcile any conflicting ideas.
- **Objectives**:
  - Draft a unified refactoring strategy.
  - Draft Vulkan barrier optimization strategy.

### Step 3: Implementation
- **Action**: Spawn 1 Worker subagent.
- **Objectives**:
  - Implement refactoring in `src/vulkan/mod.rs` to deduplicate Vulkan dispatches.
  - Clean up dead code in `src/mud/workspace.rs` and `src/mud/forward.rs`.
  - Apply optimized Vulkan synchronization/memory barriers.
  - Run `cargo clippy` and `cargo test` to ensure compliance and correctness.
- **Verification**: Worker provides passing build and test outputs with no warnings.

### Step 4: Review and Challenge
- **Action**: Spawn 2 Reviewers and 2 Challengers.
- **Objectives**:
  - Reviewers: verify code correctness, safety, and readability.
  - Challengers: empirically verify execution performance, safety, and behavior.

### Step 5: Integrity Auditing
- **Action**: Spawn 1 Forensic Auditor.
- **Objectives**:
  - Validate that the implementation is genuine and does not circumvent tests or criteria.

### Step 6: Gate & Closeout
- **Action**: Review all feedback. If passes all gates, update status and report completion to the parent agent.
