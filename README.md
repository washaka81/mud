# Forge LLM (MUD: Modular Understanding Dynamics)

Ultra-optimized 1.58-bit (Ternary) Mixture of Experts (MoE) inference engine. Specifically designed for Intel i7-1260p and Iris Xe iGPU.

## 🚀 Key Features

- **Ternary Transformer MoE:** Full support for Multi-Head Attention and RoPE (Rotary Positional Embeddings), allowing complex sequential reasoning in a ternary format.
- **Thinking Mode (CoT):** Integrated reasoning protocol using `<thinking>` and `<answer>` tags with dedicated CLI styling (Dim/Italic/Cyan).
- **Subgroup Vulkan Optimization:** SPIR-V 1.3 kernels utilizing subgroup arithmetic for parallel reductions on Intel Iris Xe.
- **Sliding Window KV-Cache:** Infinite-loop-safe context management with a 2048-token circular buffer.
- **Intelligent Sampling:** Advanced Top-K (40), Top-P (0.9), and Temperature (0.7) algorithms for creative and human-like output.
- **Semantic RAG & PDF Ingestion:** Autonomous knowledge retrieval from SQLite using model embeddings. Support for `/ingest` of `.txt` and `.pdf` files.
- **Sticky Hardware Dashboard:** Real-time footer showing CPU/RAM/VRAM and **Tokens per Second (t/s)**.

## 📁 Project Structure

- `src/mud/`: Core MUD engine (inference.rs, graph.rs, store.rs, ingester.rs).
- `src/vulkan/`: iGPU Subgroup kernels for GEMV offloading.
- `training/`: High-speed Kaggle training pipeline (Mixed Precision, 6-layer MoE).
- `docs/`: Technical specifications (AI_ARCHITECTURE.md, AI_ORCHESTRATION.md, AI_AUDIT.md).
- `tools/`: Diagnostic and auditing utilities.
- `models/`: Ready-to-use `.mud` models and knowledge bases.
- `debug/`: Disassembly and tensor audit files.
- `logs/`: Execution and training logs.

## 🛠️ Quick Start

1. **Generate Test Model:**
   ```bash
   python3 training/exporter.py
   mv test_model.mud models/core_skills.mud
   ```

2. **Run Inference:**
   ```bash
   cargo run --release --bin forge_llm
   ```

3. **Verify Kernels:**
   ```bash
   cargo run --release --bin ternary_audit
   ```

## 📜 Documentation

See [AI_ARCHITECTURE.md](docs/AI_ARCHITECTURE.md) for low-level details on ternary packing and skill modularity.
