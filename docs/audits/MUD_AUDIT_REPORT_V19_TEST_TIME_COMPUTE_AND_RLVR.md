# MUD Audit Report V19: The Pivot to Test-Time Compute & Verifiable Rewards (RLVR)

## 1. Context & Motivation
Following the successful completion of the initial 5-priority mandate (culminating in the integration of JEPA, GRPO, and LDT), the MUD engine has established a robust baseline for low-memory, high-efficiency "Slow Thinking." However, to achieve the ultimate goal of autonomous software engineering, the model must transition from generating code that *looks* right to generating code that is *mathematically and syntactically sound*.

A comprehensive research review established that 1.58-bit ternary models are uniquely positioned to excel at **Test-Time Compute (TTC)** because their extreme efficiency allows for massive parallel sampling (Monte Carlo Tree Search) without memory exhaustion.

## 2. Expanded Architectural Mandate (Priorities 6-8)

To address these findings, the architecture roadmap (`GEMINI.md`) has been formally expanded with three new pillars:

### 2.1 Priority 6: Test-Time Compute (TTC) & MCTS
Traditional LLMs spend the vast majority of their compute budget during pre-training. MUD will now focus on allocating compute during the inference phase.
- **Mechanism:** Utilizing the fast Discrete Text Diffusion blocks, the engine will spawn multiple (e.g., G=8 or G=16) parallel reasoning trajectories.
- **Monte Carlo Tree Search (MCTS):** These branches will be evaluated through the LDT framework, and only the most logically consistent paths will be expanded before collapsing into final output tokens.

### 2.2 Priority 7: Reinforcement Learning from Verifiable Rewards (RLVR)
Instead of relying on human feedback (RLHF) or an AI judge (RLAIF), MUD will utilize the ultimate arbiter of code truth: the compiler.
- **SCoRe (Self-Correction via Reinforcement Learning):** The LDT will interface directly with the Rust compiler (`rustc`), Python linters, or test suites.
- **Reward Loop:** 
  - Compilation Success = Positive Reward (+1.0)
  - Compilation Failure = Negative Reward (-1.0), and the exact error log is fed back into the engine.
  - The model adjusts its internal `policy_weights` dynamically to learn from its immediate failures and correct its own syntax.

### 2.3 Priority 8: Sparse-BitNet Integration
While 1.58-bit models are highly efficient, they often lack the character-level exactness required for coding (leading to "Linguistic Aphasia").
- **N:M Sparsity:** We will introduce semi-structured sparsity (forcing specific weight patterns to zero) on top of the ternary constraints.
- **Benefit:** This stabilizes the mathematical activations, improving syntactical precision while maintaining the sub-2M parameter memory footprint necessary for L3 cache seating.

## 3. Next Steps in Development
The immediate implementation focus will be on **Priority 6**. The `src/mud/ldt_micro.rs` module, which currently evaluates a single latent wave with GRPO drift, must be expanded to manage a tree of parallel waves (MCTS node expansion). This will require extending the `workspace::UnifiedBuffer` allocations to support multi-branch trajectory tracking without violating the Zero-Allocation Policy.
