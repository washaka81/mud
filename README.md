# Forge MUD (Modular Understanding Dynamics)

**Forge MUD** is an ultra-optimized, bare-metal mathematical inference engine designed specifically for **1.58-bit (Ternary) Tensor Algebra**. Built entirely in Rust with zero Python or external high-level frameworks, MUD is engineered to maximize hardware saturation on consumer x86_64 CPUs and Integrated GPUs through aggressive manual vectorization and strict memory management.

## ⚡ Core Philosophy

The MUD engine discards conventional matrix paradigms in favor of raw pointer manipulation and deterministic memory arenas. By enforcing a strict **Zero-Allocation Protocol (P-01)** in all hot loops, the engine completely eliminates bounds-checking overhead and OS memory latency, ensuring exact cache-line alignment for peak SIMD performance.

*"He who masters the pointers, masters the core of the machine."*

**Canonical docs:** policies → [`GEMINI.md`](./GEMINI.md) · agents → [`AGENTS.md`](./AGENTS.md) · vision → [`ROADMAP.md`](./ROADMAP.md) · compute → [`docs/architecture/MUD_COMPUTE_STACK.md`](./docs/architecture/MUD_COMPUTE_STACK.md) · launch → [`docs/manuals/LAUNCH_COUNTDOWN.md`](./docs/manuals/LAUNCH_COUNTDOWN.md) · improvements → [`docs/research/MUD_IMPROVEMENTS_POST_AE.md`](./docs/research/MUD_IMPROVEMENTS_POST_AE.md) · index → [`docs/README.md`](./docs/README.md)

> ### 🥇 Prioridad 0 (2026-10-11)
>
> El backlog activo del proyecto es el **plan maestro de correcciones/limpieza/mejoras**: [`docs/planning/MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md`](./docs/planning/MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md) (`R-##`, 7 olas), derivado de la auditoría completa [`docs/audits/MUD_AUDIT_2026-10-11_LOW_COMPUTE.md`](./docs/audits/MUD_AUDIT_2026-10-11_LOW_COMPUTE.md).
>
> **Estado verificado:** `cargo clippy --all-targets` limpio y `cargo test --lib` (debug) verde, **pero** `cargo test --lib --release` falla (UB en `fenwick.rs`), el workflow de CI es inválido (nunca se ejecuta), el trainer no entrena la cabeza que usa la inferencia y el backward del FFN/mHC es incorrecto (el loss no baja). Las afirmaciones de rendimiento de este README describen el **diseño**, no el estado medido: la inferencia medida hoy es ~46 tok/s en `models/ale.mud` con salida incoherente. Detalle en `GEMINI.md` §0.3.
>
> Ejecutar `./mud.sh plan` para ver el backlog y su estado.

## 🧬 Architectural Pillars

### 1. The SlimeRegister Paradigm
The fundamental compute unit is the `SlimeRegister`, a high-efficiency 4-byte packed dual-component register (`raw: u32` packing `f16` real + `f16` imaginary). The real component holds the ternary GEMV accumulator; the imaginary component holds the JEPA integral derivative (or C-MUD complex phasor in thinker mode). Compute happens in f32 (via 1-cycle F16C `vcvtph2ps` reads) and is stored as f16 (via `vcvtps2ph`), packing 2× density vs naive f32 while enabling real-time statistical homeostasis without conversion overhead.

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
- Rust toolchain
- Intel CPU with AVX2 support (optimized for i7-1260P P-Cores)
- Vulkan SDK (for iGPU acceleration)

```bash
# 1. Verify code integrity & test battery (257 tests)
./mud.sh ci

# 2. Run C-MUD Manifold & Cognition Audit
./mud.sh cmud-manifold models/smollm2.mud

# 3. Launch interactive circuit training
./mud.sh circuit models/smollm2.mud
```

## 📜 Repository Structure

- `src/asm/`: Handwritten AVX2 assembly kernels.
- `src/mud/`: Core inference runtime, `SlimeRegister` logic, and parallel dispatchers.
- `src/vulkan/`: Asynchronous compute shaders and GPU memory management.
- `forge_autograd/`: Mathematical gradient calculation module (isolated).
- `tools/`: Utility binaries for serialization, tensor diagnostics, and benchmarking.

---
*Developed for maximum cycle efficiency and pure statistical mathematics.*
