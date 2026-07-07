# 🔬 MUD vs. OxiLLaMa (Comparative Analysis)
**Context:** forge_llm (MUD) — Sovereign Rust Ternary Engine  
**Analysis Date:** June 4, 2026  

---

## 🧭 Overview
`OxiLLaMa` (`cool-japan/oxillama`) is a high-performance, pure-Rust alternative to `llama.cpp`. Built without any C/C++ dependencies or FFI, it implements its own tensor mathematical primitives (`scirs2`, `oxiblas`, `oxifft`). It is notable for being one of the very few engines in the Rust ecosystem supporting **Jamba** (hybrid Transformer-Mamba) and **Mamba-2** architectures, along with **GGUF ternary quantization** (`TQ1_0`, `TQ2_0`).

This analysis contrasts the architecture of MUD with OxiLLaMa to identify potential optimization ports, alignment targets, and unique design advantages.

---

## 📊 Architectural Feature Comparison

| Architectural Vector | OxiLLaMa | MUD (Forge) |
|---|---|---|
| **Programming Language** | 100% Pure Rust | Rust + x86-64 AVX2 Assembly |
| **Model Ingestion Format** | GGUF (v3) | Custom `.mud` Format |
| **Ternary Quantization** | `TQ1_0` (1.6 bpw) & `TQ2_0` (2 bpw) | Packed BMI2 Ternary (2 bpw, PRQ) |
| **Math Kernels Backend** | CPU (OxiBLAS, SciRS2) | CPU (Hand-tuned AVX2) + Vulkan Compute |
| **iGPU Acceleration** | ❌ No (CPU-centric) | ✅ Yes (Vulkan Zero-Copy on Intel Iris Xe) |
| **SSM Layers Support** | Mamba-2 & Jamba | Mamba-1 (moving to Mamba-3 MIMO) |
| **State Memory Model** | `SequenceState` Trait (dynamic buffers) | `InferenceWorkspace` (Zero-Allocation static) |
| **Engine Scope** | Inference-only | **Inference + STE QAT Training** |

---

## 🗜️ Quantization Layouts: GGUF TQ vs. MUD `.mud`

OxiLLaMa supports GGUF's specialized ternary formats:

```
                  ┌─────────────────────────────────────┐
                  │      Ternary Weight Quantization    │
                  └──────────────────┬──────────────────┘
                                     │
                  ┌──────────────────┴──────────────────┐
                  ▼                                     ▼
       ┌─────────────────────┐               ┌─────────────────────┐
       │   GGUF TQ1_0/TQ2_0  │               │   MUD pext Unpack   │
       └──────────┬──────────┘               └──────────┬──────────┘
                  │                                     │
      ┌───────────┴───────────┐               ┌─────────┴─────────┐
      ▼                       ▼               ▼                   ▼
 ┌──────────┐            ┌──────────┐    ┌──────────┐        ┌──────────┐
 │  TQ1_0   │            │  TQ2_0   │    │  LowBit  │        │ HighBit  │
 │ 5 trits/ │            │ 4 trits/ │    │ (Bit 0)  │        │ (Bit 1)  │
 │   byte   │            │   byte   │    └────┬─────┘        └────┬─────┘
 └──────────┘            └──────────┘         │                   │
                                              └─────────┬─────────┘
                                                        ▼
                                                  LowBit - HighBit
                                                   (BMI2 pext/pdep)
```

### 1. GGUF TQ1_0 (Ternary Quantization 1.0)
* **Density:** ~1.6 bits per weight.
* **Packing:** Packs **5 trits per 8-bit byte** ($3^5 = 243 \le 256$).
* **Unpacking Overhead:** High on CPU. Requires modulo-3 divisions or magic multiplications (`reciprocal multiplication`) to extract the 5 ternary states.
* **MUD Takeaway:** MUD prioritizes CPU clock cycles over disk size. TQ1_0 is too computationally intensive for an engine targeting 80+ tokens/second on ultra-modest CPUs.

### 2. GGUF TQ2_0 (Ternary Quantization 2.0)
* **Density:** 2.0 bits per weight.
* **Packing:** Packs **4 trits per 8-bit byte** (exactly 2 bits per weight).
* **Unpacking Overhead:** Low. Easily maps to powers of 2.
* **MUD Takeaway:** MUD's packed format is logically equivalent to `TQ2_0`. We store 32 ternary weights inside a `u64` (2 bits per weight).

### 3. The MUD BMI2 Unpacking Sequence
MUD optimizes unpacking using x86-64 BMI2 intrinsics. `pext_unpack_ternary` extracts 32 weights into a byte array in a few clock cycles:
1. Two `pext` calls extract the low bits (`0x5555555555555555`) and high bits (`0xAAAAAAAAAAAAAAAA`) of the 32 weights into two 32-bit registers.
2. Four `pdep` operations spread these bits into 32 byte lanes (aligned to byte boundaries).
3. Byte-wise vector subtraction (`vpsubb`) computes `LowBit - HighBit` independently for each byte, generating weight values in $\{-1, 0, 1\}$.
   ⚠ **Historical note:** An earlier version used 64-bit `sub` instead of `vpsubb`. The 64-bit subtraction propagates borrows across byte boundaries, corrupting mixed weight patterns (fixed in v0.1.0).
This design avoids all branches, loops, and divisions, making MUD's unpacking execution significantly faster than standard GGUF TQ1_0/TQ2_0 CPU loops.

---

## 🧠 State Buffers & Memory Layouts

### OxiLLaMa: `SequenceState`
OxiLLaMa implements a trait-based state management engine. When running SSMs (which require state convolution buffers and recurrent state matrices), it allocates sequence-specific state matrices on the heap. While flexible for multi-model serving, it introduces allocation steps in the hot loop.

### MUD: `InferenceWorkspace`
MUD enforces a strict **Zero-Allocation Policy**. All buffers (convolution state, SSM recurrent state matrix, GQA grouped KV-caches) are pre-allocated inside `InferenceWorkspace` at startup. During the forward pass, MUD operates strictly in-place, bypassing the Rust heap allocator completely.

---

## 🖥️ Hardware Acceleration & Compute Backends

* **OxiLLaMa Core:** Uses `SciRS2` and `OxiBLAS` for matrix calculations. These are optimized CPU BLAS runtimes. To accelerate ternary formats, it relies on compiler autovectorization, ARM NEON (`vcntq_u8` popcount), and AVX-512 vector pipelines.
* **MUD Core:** Avoids general-purpose BLAS engines. MUD's matrix-vector operations are hand-rolled in AVX2 assembly (`src/asm/ternary_lut.s`, `src/asm/ternary_pext.s`).
* **The Vulkan Differentiator:** OxiLLaMa has no GPU compute shaders. MUD implements raw Vulkan compute pipelines (`src/vulkan/vulkan_backend.rs`) to dispatch ternary matrix multiplication directly to the integrated GPU (Intel Iris Xe) using unified shared memory (Zero-Copy), bypassing CPU-GPU PCIe bus transfers entirely.

---

## 🏆 Key MUD Advantage: Integrated Training (STE QAT)

OxiLLaMa is strictly an **inference engine**. It consumes pre-quantized models (from GGUF conversion scripts).

MUD, conversely, integrates its own compiler and trainer inside the Rust core (`forge_autograd` + `mud_corpus_trainer.rs`). This is critical because **Post-Training Quantization (PTQ) directly to 1.58-bit causes semantic aphasia** (repetition loops, random BPE outputs). By embedding a Straight-Through Estimator (STE) training loop, MUD can:
1. Convert any model (such as Phi-4-mini) using the Depth-Dampened Per-Row Quantization (PRQ) pipeline.
2. Verify boundary conditions and eigenvalue stability.
3. Automatically run SGD/STE training epochs over a local bilingual corpus (`restore-iq` script) to "seat" the model weights into the ternary grid, restoring cognitive performance to $\ge 96\%$ composite score.

This allows MUD to be self-calibrating and autonomous on consumer hardware.

---

## 🚀 Feasibility Porting Opportunities for MUD
From our research of OxiLLaMa, we can port three architectural patterns:
1. **GGUF Gearing (`oxillama-gguf`):** Study OxiLLaMa's GGUF parser as a reference to expand `src/gguf/` to support parsing GGUF's experimental `TQ1_0` and `TQ2_0` layouts directly, allowing MUD to import official BitNet GGUF checkpoints.
2. **SciRS2 Alignments:** Check OxiBLAS cache-tiling strategies to improve L2/L3 cache residency in our manual AVX2 GEMV assembly loops.
3. **Mamba-2 Trait Alignment:** Study OxiLLaMa's implementation of the Mamba-2 parallel scan state matrix to accelerate our planned upgrade from Mamba-1 sequential scans.
