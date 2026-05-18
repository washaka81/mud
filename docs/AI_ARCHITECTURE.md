# Architecture: Forge LLM (MUD)
## Target: Intel i7-1260p & Iris Xe iGPU (Hybrid Ternary Engine)

### 1. MUD: Modular Understanding Dynamics
Forge LLM has evolved into the MUD architecture, focusing on modularity, ternary inference, and intrinsic skill growth.

- **The .mud Format:** A proprietary binary format replacing GGUF.
    - **Ternary Packing:** Weights are packed strictly as `{-1, 0, 1}` using 2-bits per weight. This achieves 16x compression over FP32 and eliminates multiplications in the forward pass.
    - **Skill Modules:** The format organizes weights and logic into "Skills" (e.g., Logic, Grammar, Retrieval).
- **Modular Skills:** Intrinsic capabilities are encapsulated as Rust modules that influence routing, preprocessing, and output.

### 2. Hybrid Inference (CPU + Vulkan)
The engine dynamically divides the workload to maximize hardware efficiency:

- **CPU (AVX2):**
    - **Ternary Kernels:** Custom assembly (`ternary_gemv_avx2`) using only SIMD additions and subtractions.
    - **Routing:** Orchestrates the MoE Top-K expert selection.
    - **Sequential Prompting:** Optimized for low-latency token generation.
- **iGPU (Vulkan):**
    - **Ternary Subgroup Kernels:** Optimized GEMV using `GL_KHR_shader_subgroup_arithmetic` for parallel reductions within the SIMD32 units of the Iris Xe. This eliminates VRAM bottlenecks during dot product calculations.
    - **Complex Attention:** Masively parallel multi-head attention with causal masking.
    - **Knowledge Indexing:** High-speed vector searches in the MUD knowledge base.

### 3. Mixture of Experts (MoE) & Transformer Upgrades
Instead of a single dense network, MUD uses a sparse MoE approach:
- **Router:** A Top-K enforcer that activates specialized experts per token.
- **Sparsity:** Scalable parameter count without linear increase in compute latency.
- **Articulated Logic:** Balanced expert utilization enforced via auxiliary loss during training.
- **Self-Attention & RoPE:** Initially a feed-forward only design, the architecture was upgraded to include Multi-Head Causal Self-Attention and Rotary Positional Embeddings (RoPE). This upgrade provides the model with sequential memory, preventing infinite repetition loops ("parrot mode") by allowing it to attend to previously generated tokens.
- **Thinking Mode (Chain of Thought):** The architecture supports an internal reasoning loop. The model can output reasoning steps within `<thinking>` tags before providing the final `<answer>`, a process enhanced by specialized training data.

### 4. Knowledge Graph & RAG (Autonomous Retrieval)
MUD replaces traditional flat indexes with a **Modular Knowledge Graph (MKG)**:
- **Neural Bridges & Google Algorithm:** Uses a PageRank-style importance ranking to identify "hubs" of knowledge.
- **Autonomous Ingestion:** Supports `.txt` and `.pdf` (via `pdftotext`) parsing.
- **Dynamic Loading:** To prevent memory collapse, MUD only keeps high-rank hubs in RAM, dynamically querying the SQLite **Persistent Store** when similarity thresholds fall, ensuring infinite knowledge capacity on limited hardware.

### 5. Interactive Interface & Dashboard
- **REPL Pro:** Fully autonomous natural language console.
- **Typing Animation:** Intelligent typewriter effect that styles thinking blocks in dimmed cyan/italic.
- **Sticky Dashboard:** Fixed status bar at the terminal bottom showing hardware telemetry and **Tokens per Second (t/s)**.
- **Shortcuts:** Supports Ctrl+Q (ASCII 17) for session exit.

### 6. Training Pipeline
- **Ecosystem:** Integration with **Trainer** and **Kaggle** (GPU P100/T4) for high-speed training.
- **Cumulative Learning:** Checkpoint system (`.pt`) to inherit weights between versions.
- **BitNet 1.58b:** Full implementation of ternary quantization and 8-bit activation scaling.
