# Vulkan Compute Integration Guide: Shader Optimizations

This document details the GPU compute standards and Vulkan backend architecture in MUD.

## 1. Subgroup Operations in GLSL

To maximize hardware utilization, MUD compute shaders (`assets/shaders/*.comp`) require subgroup support. We leverage hardware-accelerated subgroup reduction for dot products and normalization loops.

### Enabling Subgroup Extensions:
```glsl
#version 450
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
```

### Example: Subgroup Sum reduction
```glsl
layout(local_size_x = 256) in;

void main() {
    float local_sum = calculate_partial_dot_product();
    
    // Subgroup-level reduction (no barriers needed)
    float subgroup_sum = subgroupAdd(local_sum);
    
    if (subgroupElect()) {
        // Only one invocation per subgroup writes to shared memory/output
        atomicAdd(global_sum, subgroup_sum);
    }
}
```

## 2. Memory-Mapped Files (mmap) to Vulkan

To achieve near-instant model loading times, we memory-map (`mmap`) the `.mud` files directly into CPU space. When copying to GPU:
1. **Device-Local Host-Visible Memory:** If the GPU supports unified memory (like integrated iGPUs or Apple Silicon), we map the `.mud` tensor pointers directly to the Vulkan buffer.
2. **Staging Buffer Pipeline:** On discrete GPUs (with separate VRAM), we allocate a staging buffer, map it, write the `.mud` bytes into it, and then execute a `vkCmdCopyBuffer` command to transfer the data to device-local VRAM.

## 3. Shader Execution Guidelines

- **Workgroup Sizes:** Always use multiples of 32 (typically `local_size_x = 256` or `128`) to prevent execution divergence on common AMD, NVIDIA, and Intel architectures.
- **Push Constants:** Keep push constants compact (less than 128 bytes) to guarantee compatibility across all Vulkan devices. Use them for passing matrix dimensions and scaling coefficients.
