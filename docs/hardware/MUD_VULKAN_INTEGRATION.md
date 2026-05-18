# Vulkan Integration & Iris Xe Optimization

## 1. iGPU Compute Pipeline

The engine leverages the Intel Iris Xe iGPU (ADL GT2) through a highly optimized Vulkan compute pipeline. 

### Why Vulkan for Inference?
While the i7-1260p CPU is excellent for sequential operations and small batch sizes using AVX2, the Attention mechanism and large Feed-Forward Networks (FFNs) in MoE models are highly parallelizable but memory-bound. The Iris Xe's 96 Execution Units (EUs) provide a wider SIMD execution path.

### 2. Unified Memory Architecture (UMA)
The most critical advantage of integrated graphics is UMA. The iGPU and CPU share the same physical 2666MHz DDR4 RAM.
- **Zero-Copy Transfer:** Unlike discrete GPUs, we do not need to transfer weights over a PCIe bus. We use `MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE` to allocate buffers that both the CPU and GPU can access efficiently.
- **Pointer Sharing:** By mapping the GGUF file directly into host memory, we can instruct the Vulkan driver to read from these mapped regions, effectively achieving zero-copy inference on the iGPU.

### 3. Shader Optimizations
- **Workgroup Size:** The compute shaders (`assets/shaders/*.comp`) use a `local_size_x = 32` to align perfectly with the execution characteristics of Intel Gen12 graphics.
- **Subgroup Operations:** Future iterations of the Attention shader will utilize `VK_KHR_subgroup_ballot` to allow threads within a subgroup to share data (like maximum values for Softmax) without writing to VRAM, bypassing the 2666MHz bottleneck.

### 4. Hybrid Dispatch Strategy
1. **CPU (AVX2):** Token generation, RoPE (Rotary Positional Embeddings), Fused RMSNorm, and sequential GEMV for small layers.
2. **GPU (Vulkan):** KV-Cache lookups (FlashAttention), large matrix multiplications (Prefill phase), and MoE expert routing.
