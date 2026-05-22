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

## 📂 Project Structure

For a detailed breakdown of the official layout, see **[MUD_DIRECTORY_STRUCTURE.md](docs/MUD_DIRECTORY_STRUCTURE.md)**.

- `src/mud/`: Core MUD engine (inference.rs, graph.rs, store.rs, ingester.rs).
- `src/asm/`: High-performance AVX2 Ternary Kernels.
- `training/`: Stable training pipeline (DataLoader-based, 4-layer MoE).
- `docs/`: Technical specifications and reports.
- `weights/`: PyTorch checkpoints and training tensors.
- `models/`: Optimized `.mud` deployment models and SQLite knowledge base.

## 🛠️ Quick Start & Command Reference

The project is entirely managed via the **MUD Command Center** (`mud.sh`). This script provides a unified entry point for inference, training, synchronization, and testing.

### Core Operations
- `./mud.sh chat` : Launch the interactive MUD terminal (Rust inference engine).
- `./mud.sh train` : Launch or resume local V1-MASTER training.
- `./mud.sh step` : Run step-by-step inference analysis (Useful for debugging tokens).

### Weights & Cloud Synchronization (Kaggle)
- `./mud.sh pull` : Pull the latest trained weights from your Kaggle notebook.
- `./mud.sh export` : Convert the latest PyTorch checkpoints to the highly-optimized `.mud` format for inference.
- `./mud.sh test-handoff` : Run the asynchronous Kaggle-to-Local synchronization test. Evaluates checkpoint compatibility between environments.

### Optimization, Profiling & Audit
- `./mud.sh profile` : Analyze the local hardware (RAM, Threads, CPU support) to suggest optimal training parameters.
- `./mud.sh bench` : Run performance & memory benchmarks.
- `./mud.sh iq` : Calculate the current Digital IQ Score of the model based on its training metrics.
- `./mud.sh audit` : Run the full cognitive & structural audit suite (Rust-based tests).
- `./mud.sh clean` : Organize the workspace by clearing temporary logs and moving files to their respective folders.

## 📜 Documentation

See [MUD_ARCHITECTURE.md](docs/MUD_ARCHITECTURE.md) for low-level details on ternary packing and skill modularity.
