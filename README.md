---
lang: en
---

# Forge LLM (MUD: Modular Understanding Dynamics)

Ultra-optimized **1.58-bit (Ternary) Jamba Hybrid** inference and training engine. Designed for the next generation of LLMs, combining **Selective State Space Models (Mamba SSM)** and **Transformer MoE** to achieve linear context scaling and massive CPU throughput.

## 🚀 Key Features

- **Jamba Hybrid Architecture:** Interleaved Attention and Mamba layers. Combines the logical reasoning of Transformers with the $O(1)$ memory efficiency of SSMs.
- **Ternary Engine (1.58b):** Ultra-compressed weights with specialized AVX2/ASM kernels for high-speed inference on CPU.
- **Context Efficiency:** Native support for massive context windows (128k+) with constant memory usage in Mamba layers.
- **160+ TPS Throughput:** Optimized AVX2 Parallel Scan kernel achieves over 180% speedup compared to traditional Attention-only architectures.
- **Native Token Streaming:** Real-time generation feedback delivered instantly word-by-word.
- **Holographic Wave Distillation:** A revolutionary alignment method utilizing Cosine Similarity. M.U.D. preserves an 88.02% baseline semantic phase geometry despite discarding 90% of numeric precision, forcing continuous precision alignment to achieve 99.9% fidelity without expensive corpus re-training.
- **Linguistic Restoration Pipeline:** Unified `restore-iq` command to recover models from ternary quantization shock via straight-through estimator (STE) QAT.
- **Hybrid Zero-Copy Training:** Innovative local training pipeline enforcing asymmetric CPU/Vulkan delegation for mathematically perfect backward passes.
- **Sliding Window KV-Cache:** Infinite-loop-safe context management with a circular buffer.
- **Intelligent Sampling:** Advanced Top-K, Top-P, and Temperature algorithms for creative and human-like output.
- **Autonomous RAG & DB Ingestion:** Knowledge retrieval from an SQLite database using model embeddings. Support for `/ingest` of `.txt` and `.pdf`.

## 📂 Project Structure

For a detailed breakdown of the official layout, see **[MUD_DIRECTORY_STRUCTURE.md](docs/MUD_DIRECTORY_STRUCTURE.md)**.

- `src/mud/`: Core MUD engine (inference.rs, graph.rs, store.rs, ingester.rs).
- `src/asm/`: High-performance AVX2 Ternary Kernels.
- `training/`: Advanced training pipeline (MuonCANS Optimizer, MoE Load Balancer).
- `docs/`: Technical specifications and reports.
- `weights/`: PyTorch checkpoints and raw training tensors.
- `models/`: Optimized `.mud` deployment models and SQLite knowledge base (`knowledge.db`).

## 🛠️ Quick Start & Command Reference

The project is managed via the **MUD Command Center** (`mud.sh`).

### Core Operations
- `./mud.sh chat` : Launch the interactive MUD terminal (Native streaming UI).
- `./mud.sh restore-iq` : Unified restoration: Align (Corpus) -> Project (Bayes) -> Train (Live).
- `./mud.sh diag` : Comprehensive health dashboard (Hardware + Cognitive Audit).
- `./mud.sh train` : Launch the Rust AutoTrainer daemon (Memory-mapped SGD).
- `./mud.sh convert [INPUT] [OUTPUT]` : Universal Converter to zero-copy ternary format.

### Optimization & Diagnostics
- `./mud.sh bench` : Run performance & memory benchmarks.
- `./mud.sh audit` : Run the full cognitive & structural audit suite.
- `./mud.sh clean` : Clear temporary logs and organize workspace.

## 📜 Documentation

- **[MUD_ROADMAP_v4.md](docs/MUD_ROADMAP_v4.md):** Consolidates the feasibility and benchmarks matrix for ultra-modest PCs.
- **[MUD_COMPREHENSIVE_RESEARCH.md](docs/MUD_COMPREHENSIVE_RESEARCH.md):** Consolidated research & feasibility study across 6 domains.
- **[MUD_VS_OXILLAMA.md](docs/MUD_VS_OXILLAMA.md):** Architectural comparison report against OxiLLaMa.
- **[RESEARCH_PAPERS.md](docs/RESEARCH_PAPERS.md):** Master index of all 53 research papers powering MUD.
- **[MUD_USER_MANUAL.md](docs/MUD_USER_MANUAL.md):** Detailed guide on commands and operating modes.
- **[MUD_ARCHITECTURE.md](docs/MUD_ARCHITECTURE.md):** Low-level details on ternary packing and skill modularity.
- **[MUD_WHITE_PAPER.md](docs/MUD_WHITE_PAPER.md):** Deep mathematical theory and the concept of Holographic Wave Distillation.

## ⚖️ License

This project is officially licensed under the **GNU General Public License v3.0 (GPLv3)**.
Any commercial entity modifying or distributing this system must release their source code under the same terms. See the `LICENSE` file for more details.
