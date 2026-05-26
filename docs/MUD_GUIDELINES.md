---
lang: en
---

# Forge LLM (MUD) - Project Guidelines

## Project Structure

- `src/`: Rust source code.
    - `asm/`: SIMD assembly kernels for CPU (AVX2).
    - `vulkan/`: iGPU compute kernels and pipeline.
    - `mud/`: Core MUD engine and modular skill system.
    - `model/`: Transformer architecture and inference logic.
    - `gguf/`: Support for GGUF format (Legacy/Reference).
- `docs/`: High-level documentation.
    - `hardware/`: Low-level specs, cache strategies, and ISA details.
- `tools/`: Utility scripts for auditing, fusion, and benchmarking.
- `training/`: Python ecosystem for model training (Trainer/Kaggle).
- `models/`: Binary `.mud` files and knowledge databases.
- `tests/data/`: Sample data and documents for testing.
- `debug/`: Temporary disassembly and tensor dump files (Git ignored).
- `logs/`: Training and execution logs.

## Engineering Standards

- **Performance First:** All core matrix operations must have a SIMD (AVX2) path and a Vulkan offload path.
- **Ternary Logic:** Weights are strictly `{-1, 0, 1}`. Avoid floating point multiplications in the hot path.
- **Modularity:** New capabilities should be implemented as "Skills" in `src/mud/skills/`.
- **Documentation:** Architectural changes must be reflected in `docs/` before implementation.

## Tooling

- Use `cargo run --release --bin <tool_name>` for auditing and benchmarks.
- Use `python3 training/exporter.py` to convert `.pt` checkpoints to `.mud` format.
