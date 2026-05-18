# Implementation Plan: Core GEMV & Dequantization Kernel

## 1. Goal: ASM-Level Q4_0 Dequantization + GEMV
To maximize the i7-1260p, we will implement a kernel that reads 4-bit weights and scales directly into YMM registers, dequantizes them to FP32 "on-the-fly," and performs the dot product using FMA instructions.

## 2. Kernel Specification (Q4_0)
- **Block Size:** 32 elements.
- **Structure:** 
  - 1x FP16 scale (`d`).
  - 16 bytes of nibbles (32 weights of 4 bits each).
- **AVX2 Implementation Details:**
  - Load 16 bytes (32 weights) into a YMM register.
  - Use bit-masking (`vpand`) and shifts (`vpsrlq`/`vpsllq`) to isolate high and low nibbles.
  - Subtract the bias (usually 8 for Q4_0) to center the values.
  - Convert to FP32 (`vcvtdq2ps`).
  - Multiply by the scale `d` (broadcasted).
  - Accumulate into a running sum YMM register using `vfmadd231ps`.

## 3. Optimizing for Alder Lake (P-core vs E-core)
- **P-cores (Golden Cove):** High frequency, 48KB L1d. Can handle larger unrolling.
- **E-cores (Gracemont):** 32KB L1d. Requires tighter loops and more frequent cache prefetching.
- **Affinity:** We will implement a thread pool that pins workers to specific cores, avoiding the OS scheduler moving them across the hybrid boundary mid-inference.

## 4. Fused Operations Roadmap
1. **Stage 1:** Simple GEMV (Q4_0 weights * FP32 input vector).
2. **Stage 2:** Fused RMSNorm -> GEMV (Reduce RAM reads of the input vector).
3. **Stage 3:** Fused GEMV -> SwiGLU (Apply activation while weights are still in cache).

## 5. Vulkan Offloading (Iris Xe)
- We will target the iGPU for the **Attention mechanism** and larger **MoE expert batches**, where the 96 EUs can parallelize the KV cache lookups better than the CPU, provided we use the Unified Memory Architecture (UMA) to avoid explicit PCI-e style copies.
