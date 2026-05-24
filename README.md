# Forge LLM (MUD: Modular Understanding Dynamics)

Ultra-optimized **1.58-bit (Ternary) Mixture of Experts (MoE)** inference and training engine. Specifically designed for hybrid architectures, extracting maximum performance from Intel i7 CPUs (P-Cores) and Intel Iris Xe iGPUs using Vulkan and AVX2.

## 🚀 Key Features

- **Ternary Transformer MoE (1.58b):** Full support for Multi-Head Attention and RoPE (Rotary Positional Embeddings), allowing complex sequential reasoning using purely ternary states (-1, 0, 1) to massively reduce RAM bandwidth.
- **Hybrid Zero-Copy Training (MuonCANS):** Innovative local training pipeline that uses the Intel Iris Xe GPU for INT8 Forward Passes and the CPU P-Cores for continuous FP32 Backward Passes using a highly optimized Newton-Schulz (Muon) algorithm. 
- **Thinking Mode (CoT):** Integrated reasoning protocol using `<thinking>` and `<answer>` tags with dedicated CLI styling (Dim/Italic/Cyan).
- **Subgroup Vulkan Optimization:** SPIR-V 1.3 kernels utilizing subgroup arithmetic for parallel reductions on Intel Iris Xe.
- **Sliding Window KV-Cache:** Infinite-loop-safe context management with a 2048-token circular buffer.
- **Intelligent Sampling:** Advanced Top-K (40), Top-P (0.9), and Temperature (0.7) algorithms for creative and human-like output.
- **Autonomous RAG & DB Ingestion:** Knowledge retrieval from an SQLite database using model embeddings. The Rust engine automatically switches to a beautiful **purple italic UI** when injecting real-time facts into its generation. Support for `/ingest` of `.txt` and `.pdf`.
- **Hardware-Agnostic Cloud Sync:** Train your MoE model on Kaggle or Google Colab (using A100/T4 GPUs) and pull the perfectly quantized ternary `.mud` file to run instantly on your local hardware.

## 📂 Project Structure

For a detailed breakdown of the official layout, see **[MUD_DIRECTORY_STRUCTURE.md](docs/MUD_DIRECTORY_STRUCTURE.md)**.

- `src/mud/`: Core MUD engine (inference.rs, graph.rs, store.rs, ingester.rs).
- `src/asm/`: High-performance AVX2 Ternary Kernels.
- `training/`: Advanced training pipeline (MuonCANS Optimizer, MoE Load Balancer, Dataset metadata).
- `docs/`: Technical specifications and reports.
- `weights/`: PyTorch checkpoints and raw FP32/FP16 training tensors.
- `models/`: Optimized `.mud` deployment models and SQLite knowledge base (`knowledge.db`).

## 🛠️ Quick Start & Command Reference

The project is entirely managed via the **MUD Command Center** (`mud.sh`). This script provides a unified entry point for inference, training, synchronization, and testing.

### Core Operations
- `./mud.sh chat` : Launch the interactive MUD terminal (Rust inference engine).
- `./mud.sh train` : Launch or resume local training (utilizing your local CPU/GPU hardware).
- `./mud.sh step` : Run step-by-step inference analysis (Useful for debugging tokens).

### Weights & Cloud Synchronization (Kaggle/Colab)
- `./mud.sh pull` : Pull the latest trained weights from your Kaggle notebook directly to your local `models/` directory.
- `./mud.sh export` : Convert PyTorch checkpoints (`.pt`) to the highly-optimized `.mud` format (Zero-Copy Ternary mapping) for inference.
- `./mud.sh test-handoff` : Run the asynchronous Kaggle-to-Local synchronization test. Evaluates checkpoint compatibility between environments.

### Optimization, Profiling & Audit
- `./mud.sh profile` : Analyze the local hardware (RAM, Threads, CPU support) to suggest optimal training parameters.
- `./mud.sh bench` : Run performance & memory benchmarks.
- `./mud.sh iq` : Calculate the current Digital IQ Score of the model based on its training metrics (Sigma, Skew, and Steps).
- `./mud.sh audit` : Run the full cognitive & structural audit suite (Rust-based tests).
- `./mud.sh clean` : Organize the workspace by clearing temporary logs and moving files to their respective folders.

## ⚙️ Cloud Training Setup (Kaggle)

If you plan to use Kaggle for mass training, ensure you add your Kaggle username to the `training/dataset-metadata.json` and `training/kernel-metadata.json` files before running the pipeline. The system uses the standard Kaggle API to push/pull models seamlessly.

## 📜 Documentation

See [MUD_ARCHITECTURE.md](docs/MUD_ARCHITECTURE.md) for low-level details on ternary packing and skill modularity.
