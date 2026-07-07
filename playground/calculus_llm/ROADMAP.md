# Roadmap — SLIME Engine (Selective Latent Integral Model Engine)

## Current State (June 2026)

### Done
- **BPE Tokenizer** (C++): 32K vocab, trained on 139K Spanish words via HuggingFace `tokenizers` Whitespace BPE. Fast O(1) merge-priority via `pair_rank_` hash map. `encode/decode` operational.
- **EmbeddingMatrix**: 32K×128D trainable matrix, Xavier init, forward/lookups/logits/sgd_step/save/load. Trainable via SGD.
- **Embedding training**: CBOW-style negative sampling (K=5). Loss 3.03→2.09, accuracy 84.8% after 10 epochs on 1.5MB Spanish Wikipedia corpus (337K examples/epoch). Weights saved to `embedding_weights.bin`.
- **16D→128D migration**: All model dimensions upgraded (embedding, ODE, SSM). Space partitioned into Intent(0-31), Semantic(32-95), Form(96-127).
- **ODE/SSM co-processor architecture**: SSM (Mamba-style) integrated into ODE dynamics with input-dependent Δ, B, C and ZOH discretization.
- **Eigen Optimization**: Migrated from `Eigen::VectorXd` to fixed-size `Eigen::Matrix<double, 128, 1>`. Eliminated all heap allocations during ODE integration. Training speed increased by ~15x.
- **Kinetic Regularization**: Implemented $||\dot{y}||^2$ penalty in ODE training to stabilize trajectories and allow larger time steps.
- **ODE Training Unblocked**: Successfully trained `W` on 243K word-pairs. Progress reporting implemented.
- **ContinuousLLM**: Full generation pipeline with syntactic node traversal, semantic attractors, grammar routing, positronic censorship.

### Known Issues
- **Hardcoded responses**: Greeting/casual-mode responses are static strings, not generated.
- **Similarity threshold (0.38)** for math routing may be too high with trained embeddings (non-math prompts get 0.04-0.23).
- **Small corpus**: 1.5MB (100 Wikipedia articles) is insufficient for meaningful ODE training.

### Next Session Priorities

### P1: Train ODE & Test Generation
- Now that `W` is trainable and optimized, run deep training (50+ epochs) on larger corpus.
- Tune similarity threshold and keyword detection for better routing.
- Validate that the ODE predicts next-word embeddings better than random.

### P2: Expand Corpus
- Download larger Spanish corpus (100MB+) for better coverage.
- Retrain BPE tokenizer with larger data.
- Retrain embeddings on larger corpus.

### P4: Hybrid LLM Integration
- Integrate llama.cpp GGUF as external co-processor for non-math prompts.
- ODE/SSM handles discourse planning/structuring; external LLM handles fluent generation.
- This is the path to human-level coherence given the small ODE capacity.

## SOTA Implementation (Advanced Research)

### R1: Refined Discretization (Mamba-3)
- Implement **Exponential-Trapezoidal Discretization** to replace RK4/ZOH in the SSM evolution.
- Aim for higher stability in long sequences with larger $\Delta t$.

### R2: Latent Reasoning Loop (PromptCoT)
- Implement a "Thought Phase" where the ODE evolves for $T_{\text{thought}}$ before generating the first token of a response.
- Allow the latent state to "stabilize" on a reasoning attractor.

### R3: Neural Controlled Differential Equations (CDEs)
- Transition from $\dot{y} = f(y)$ to $\dot{y} = f(y) \dot{X}$ to treat input embeddings as a continuous path $X(t)$.
- This will enable true continuous-time processing of math formulas.

### R4: SLiCE Architecture
- Refactor $W$ into a **Structured Linear** form (Block-Diagonal) to reduce parameter count and increase training speed further.

## Architecture Notes

```
Input Text → BPE Tokenizer → EmbeddingMatrix.lookup(token_ids)
  → Average context → ODE evolution (128D) → nearest-neighbor decode → word

ODE dynamics: dy/dt = 0.05·tanh(W·y) + f_mamba(SSM) + attractors + grammar + positronic
```

All vectors are 128D. Vocab: 32K BPE tokens + 139K Spanish words. Training uses negative sampling (binary cross-entropy with sigmoid).

## File Layout

```
src/
  nlp/
    bpe_tokenizer.h/.cpp    — BPE encode/decode with pair_rank_ optimization
    embedding.h/.cpp         — 32K×128D trainable embedding matrix
    embedding_trainer.h/.cpp — Negative-sampling CBOW trainer
    tokenizer.h/.cpp         — Integrates BPE + EmbeddingMatrix + word_to_vec
  models/
    neural_ode.h/.cpp        — ODE dynamics + SSM + training (SPSA+Adam)
    continuous_llm.h/.cpp    — Generation pipeline (syntactic nodes, attractors)
    trainer.h/.cpp           — ODE trainer (train_ode_epoch)
  embed_train_main.cpp       — Embedding training binary
  ode_train_main.cpp         — ODE training binary
  config.h                   — All hyperparameters (EMBEDDING_DIM=128, etc.)
build/
  embedding_weights.bin      — Trained embedding weights (10 epochs)
  model_weights.bin          — ODE weights (currently untrained/uninitialized)
  bpe_vocab.txt / bpe_merges.txt — BPE vocabulary and merge rules
```
