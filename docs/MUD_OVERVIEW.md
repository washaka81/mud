# ARCHITECTURAL_OVERVIEW: MUD (Modular Understanding Dynamics)
## Hybrid Ternary AI Architecture (BitNet 1.58b + MoE)

MUD is a high-performance infrastructure designed to integrate **Trainable Ternary AI** on consumer hardware (Intel i7 + Iris Xe).

### 1. The Ternary Paradigm (BitNet 1.58b)
Unlike traditional models using 16-bit or 32-bit floating-point weights, MUD utilizes the **BitNet 1.58b** architecture.
- **Weights in {-1, 0, 1}:** Every parameter is reduced to three possible states.
- **Multiplication-Free Inference:** This eliminates traditional matrix multiplications, converting them into SIMD additions and subtractions.
- **Energy Efficiency:** Drastic reduction in power consumption and memory bandwidth, enabling large models to run on integrated iGPUs.

### 2. Sparse Mixture of Experts (MoE)
MUD implements a dynamic **Expert** system. Instead of processing a word through all parameters, an intelligent **Gate (Router)** routes the signal to only the two experts (Top-2) best suited for the specific task (Logic, Code, Science, etc.).
- **Scalability:** Allows for models with 30-50M parameters while maintaining the response speed of a 5M parameter model.

### 3. Hybrid Orchestration & Subgroups
The architecture splits the AI workload between two domains:
- **CPU (AVX2):** Handles sequential logic, expert routing, and post-processing.
- **iGPU (Vulkan):** Executes **Multi-Head Attention** and MoE blocks using **Subgroup Arithmetic**. GPU threads collaborate directly to minimize dot-product latency.

### 4. Lifecycle: Assimilation & Dreaming
MUD is **fully trainable**. It uses a Python-based pipeline (Kaggle) for **Quantization-Aware Training (QAT)**.
- **Real Ingestion:** Reads books (PDF/TXT) and indexes them into a Knowledge Graph.
- **Synthetic Dreaming:** Converts this data into reasoning pairs (`<thinking>`) to re-train its own ternary weights, permanently assimilating knowledge.

---

## Hardware Compatibility & ISA Requirements
To achieve target performance, the host system must support the following Instruction Set Architecture (ISA) features:

### CPU (Minimum: Intel 4th Gen / AMD Zen 1)
- **AVX2:** 256-bit Vector Extensions (Required for Ternary Kernels).
- **FMA3:** Fused Multiply-Add (Required for Embeddings and Norm scaling).
- **BMI1 / BMI2:** Bit Manipulation Instructions (Required for efficient 2-bit weight unpacking).
- **SSE 4.2:** Required for base memory operations.
- **AVX-VNNI (Recommended):** Accelerates 8-bit integer operations (Alder Lake and newer).

### iGPU / GPU (Minimum: Vulkan 1.1)
- **Vulkan 1.1+:** Core runtime requirement.
- **GL_KHR_shader_subgroup_arithmetic:** Required for extreme iGPU optimization (Subgroup reductions).
- **Shader Int8 / Float16:** Required for quantized attention paths.
- **MemoryTypeFilter::PREFER_DEVICE:** Support for Unified Memory Architecture (UMA) for zero-copy transfers.
