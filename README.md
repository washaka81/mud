# Forge MUD (Modular Understanding Dynamics)

**Forge MUD** is an ultra-optimized, bare-metal mathematical inference engine designed specifically for **1.58-bit (Ternary) Tensor Algebra**. Built entirely in Rust with zero Python or external high-level frameworks, MUD is engineered to maximize hardware saturation on consumer x86_64 CPUs and Integrated GPUs through aggressive manual vectorization and strict memory management.

## ⚡ Core Philosophy

The MUD engine discards conventional matrix paradigms in favor of raw pointer manipulation and deterministic memory arenas. By enforcing a strict **Zero-Allocation Protocol (P-01)** in all hot loops, the engine completely eliminates bounds-checking overhead and OS memory latency, ensuring exact cache-line alignment for peak SIMD performance.

*"He who masters the pointers, masters the core of the machine."*

## 🧬 Architectural Pillars

### 1. The SlimeRegister Paradigm
The fundamental compute unit is the `SlimeRegister`, a radical memory structure that hijacks standard 32-bit registers (`u32`). It transparently splits the register into two distinct 16-bit floats (`f16`):
- **Lower 16-bits (Statistical State):** Accumulates the core ternary matrix multiplications (GEMV).
- **Upper 16-bits (Deterministic Attractor):** Carries embedded running integrals and differential logic for real-time statistical homeostasis.
This allows the engine to compute two concurrent mathematical systems simultaneously within a single AVX2 pass, without needing complex data structures.

### 2. Bare-Metal AVX2 Vectorization
All critical ternary matrix multiplications are executed via handwritten x86_64 Assembly (`src/asm/*.s`). Ternary matrices are compressed using **ELUT (4-bit Nibble)** packing, allowing the CPU to ingest and process dense mathematical states at theoretical memory-bandwidth limits.

### 3. Heterogeneous Vulkan Offloading (HMP)
Sequential, memory-bound workloads (like the ELUT-AVX2 GEMV) are strictly pinned to the CPU's P-Cores. Simultaneously, asynchronous or purely compute-bound $O(N^3)$ operations (like Newton-Schulz matrix orthogonalizations) are offloaded to the Integrated GPU (Intel Iris Xe) via custom Vulkan Compute Shaders, guaranteeing zero bus contention.

### 4. Dynamic Context & Caching
- **O(1) Memory Profiles:** Support for fixed-state sequential scan layers, guaranteeing a constant memory footprint regardless of the sequence length.
- **AOT Binary Caching:** Ahead-Of-Time flat binary translation to prevent string parsing bottlenecks, ensuring the CPU math pipelines are never starved for data.

## 🛠️ Project Constraints & Standards

MUD enforces a draconian development standard to ensure absolute stability and speed:
- **Rust-Only Toolchain:** Python is strictly forbidden. 
- **Zero-Warning Policy:** Code must compile with 0 errors and 0 warnings under `cargo clippy`.
- **No Thread Pools in Hot Paths:** Rayon is banned to avoid E-Core latency and OS thread-contention. We use explicitly pinned `PCorePool` logic for manual hyper-threading saturation.
- **Fail-Fast Agnosticism:** Dimensions and tensor boundaries are inferred dynamically. Hardcoding magic numbers results in an immediate panic.

## 🚀 Building & Running

**Prerequisites:**
- Rust Nightly toolchain
- Intel CPU with AVX2 support (optimized for i7-1260P P-Cores)
- Vulkan SDK (for iGPU acceleration)

```bash
# 1. Verify code integrity
cargo clippy --all-targets

# 2. Run test suite
cargo test

# 3. Compile for maximum performance
cargo build --release
```

## 📜 Repository Structure

- `src/asm/`: Handwritten AVX2 assembly kernels.
- `src/mud/`: Core inference runtime, `SlimeRegister` logic, and parallel dispatchers.
- `src/vulkan/`: Asynchronous compute shaders and GPU memory management.
- `forge_autograd/`: Mathematical gradient calculation module (isolated).
- `tools/`: Utility binaries for serialization, tensor diagnostics, and benchmarking.

---
*Developed for maximum cycle efficiency and pure statistical mathematics.*
