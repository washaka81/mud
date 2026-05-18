# Hardware Specification: Intel Core i7-1260p (Alder Lake) & Iris Xe

## CPU Architecture: Alder Lake-P (Hybrid)
- **Cores:** 4 Performance-cores (P-cores) + 8 Efficient-cores (E-cores).
- **Topology:**
  - P-cores: Golden Cove microarchitecture.
  - E-cores: Gracemont microarchitecture (grouped in 4-core clusters).
- **Cache Hierarchy:**
  - **L1 Instruction/Data:** 
    - P-core: 32KB I-cache, 48KB D-cache per core.
    - E-core: 64KB I-cache, 32KB D-cache per core.
  - **L2 Cache:**
    - P-core: 1.25MB private per core.
    - E-core: 2MB shared per 4-core cluster.
  - **L3 Cache (LLC):** 18MB shared across all cores and iGPU.
- **Memory:** DDR4/LPDDR4x-2666MHz (Current Constraint).
  - Bandwidth: ~42.6 GB/s (Dual Channel).
  - Strategy: Strict zero-copy, temporal prefetching.

## Instruction Set Architecture (ISA) - CPU
Targeting extreme optimization requires utilizing these specific instruction groups:

### 1. Vector Extensions (AVX2 / AVX-VNNI)
- **AVX2:** 256-bit wide registers (`ymm0`-`ymm15`).
  - `VPMADDUBSW`: Multiply and Add Packed Signed and Unsigned Bytes (critical for int8 quantization).
  - `VPADDD` / `VPMULLD`: Parallel integer addition and multiplication.
- **AVX-VNNI (Vector Neural Network Instructions):**
  - `VPDPBUSD`: Multiply and Accumulate (MAC) for int8. This is the "killer feature" for inference on Alder Lake.
  - `VPDPWSSD`: MAC for int16.
- **FMA3:** `VFMADD213PS`, `VFMADD231PS` for floating point 32-bit dot products.

### 2. Cache Control & Prefetching
- `PREFETCHT0`: Fetch data into all levels of cache.
- `PREFETCHT1`: Fetch into L2 and higher.
- `PREFETCHNTA`: Non-temporal prefetch (minimizes cache pollution for weights used only once).
- `CLFLUSHOPT`: Optimized cache line flush (if needed for synchronization).

### 3. Synchronization & Atomic Ops
- `PAUSE`: Spin-lock hint (improved in Alder Lake for power/efficiency).
- `LOCK XADD`: Atomic exchange and add for multi-threaded dispatch.

## iGPU Architecture: Intel Iris Xe (96 EUs)
- **Architecture:** Gen12 (Xe-LP).
- **Execution Units (EUs):** 96.
- **Threads per EU:** 7.
- **Total Threads:** 672.
- **SIMD Width:** SIMD8, SIMD16, SIMD32.
- **Vulkan Extensions for Inference:**
  - `VK_KHR_shader_float16_int8`: Native fp16 and int8 support.
  - `VK_KHR_subgroup_ballot`: Fast cross-lane communication.
  - `VK_INTEL_subgroup_matrix_multiply_accumulate`: Hardware-accelerated matrix ops (DPAS - Dot Product Accumulate Systolic).

## Strategy for "Forge LLM" ASM Kernels
1. **L1 Blocking:** Tile matrix-vector multiplication into 32KB chunks to stay within E-core L1d limits.
2. **Hybrid Dispatch:** 
   - P-cores handle critical path/low latency.
   - E-cores handle batch/background processing.
3. **Loop Unrolling:** Manual unrolling in `.s` files to saturate execution ports.
4. **Register Pressure:** Careful management of 16 YMM registers to avoid spills.
