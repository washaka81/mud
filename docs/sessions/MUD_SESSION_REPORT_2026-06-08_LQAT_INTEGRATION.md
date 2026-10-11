# MUD Session Report: L-QAT Vulkan Shader Integration
**Date:** 8 de junio de 2026 (Late Night)
**Focus:** Connecting dead-code Vulkan compute shaders to the FULL-QAT training pipeline

## 1. Executive Summary
The Vulkan compute shaders `run_qat_optimizer_async` (SGD on GPU) and `run_ghost_align_async` (stochastic alignment on GPU) were defined in `src/vulkan/mod.rs` but had zero callers — pure dead code. The FULL-QAT trainer (`corpus_trainer.rs`) performed all operations on CPU via `ghost_align_cpu` and `apply_gradient_avx2`, even when Vulkan shadow buffers were available.

This session connected `run_ghost_align_async` to the `deep_local_alignment` pipeline, enabling GPU-accelerated stochastic alignment when the shadow model resides in VRAM. The `run_qat_optimizer_async` integration was evaluated and deliberately skipped due to sparse-vs-dense gradient incompatibility.

## 2. Ghost Align GPU Integration

### Before (CPU-only)
`deep_local_alignment` always copied shadow weights to a CPU `Vec<f32>`, ran `ghost_align_cpu` 5 times per row, then ternary-packed and wrote back. Even when shadow tensors lived on GPU, this forced a VRAM→RAM→VRAM round-trip.

### After (Vulkan when available)
In `deep_local_alignment`, the trainer now detects whether the shadow tensor is a `ShadowTensor::Vulkan(Subbuffer<[f32]>)` AND `self.vk` is `Some`. If both conditions hold:

1. For each of 5 iterations, generate random `x` on CPU, upload to a temp GPU `Subbuffer<[f32]>` via `upload_to_gpu`
2. Compute `scale` and `jittered_delta` from absmean (read from GPU buffer once)
3. Dispatch `run_ghost_align_async(rows, cols, scale, jittered_delta, shadow_buf, x_buf)` — the shader runs entirely on GPU
4. After all iterations, read back the updated FP32 weights from GPU for ternary packing on CPU

If conditions are not met, the existing CPU path (`ghost_align_cpu`) runs unchanged.

### Helper Added
```rust
fn upload_to_gpu(&self, data: &[f32]) -> Option<Subbuffer<[f32]>> {
    let vk = self.vk.as_ref()?;
    let buf = vk.create_host_visible_buffer::<f32>(data.len()).ok()?;
    buf.write().unwrap().copy_from_slice(data);
    Some(buf)
}
```

## 3. QAT Optimizer (SGD) — Skipped

The `run_qat_optimizer_async` shader applies dense SGD with weight decay to **all** elements of a buffer: `w = w*(1 - lr*decay) - lr*g`. However, the trainer uses **sparse** per-row updates — only the rows corresponding to batch tokens receive gradient updates. Applying the dense shader would incorrectly decay untouched embedding rows, causing weight drift toward zero in inactive vocabulary regions.

This is a fundamental algorithmic mismatch, not a wiring issue. The shader would need a per-element mask or a sparse dispatch to be correct. Decision: **skip** until a sparse variant is implemented.

## 4. Build Health
- `cargo clippy --release`: **0 errors, 0 warnings**
- `cargo test --release --lib`: **57/57 tests passed**
- `cargo build --release`: **compiles clean** (~33s)

## 5. Files Modified
| File | Change |
|------|--------|
| `src/mud/corpus_trainer.rs` | Added `upload_to_gpu` helper; added Vulkan path in `deep_local_alignment` using `run_ghost_align_async`; removed unused `delta` variable |

## 6. Status
- **Ghost Align (Vulkan):** Connected and active when shadow tensors are on GPU.
- **QAT Optimizer (Vulkan):** Remains dead code — awaiting sparse gradient shader variant.
- **FULL-QAT CPU path:** Unchanged, serves as fallback when no Vulkan device is available.
