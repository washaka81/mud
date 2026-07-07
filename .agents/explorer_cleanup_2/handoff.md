# Handoff Report — Explorer Cleanup 2

## 1. Observation

### R1. Vulkan Dispatch Code Deduplication (src/vulkan/mod.rs)
- **Descriptor Set Creation Boilerplate**: Creating descriptor sets is repeated across several dispatch functions. For instance, in `build_ternary_gemm_command_buffer` (lines 445-456):
```rust
        let layout = self.pipeline.layout().set_layouts().first().unwrap();
        let set = PersistentDescriptorSet::new(
            &*self.descriptor_set_allocator,
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, buffer_x.clone()),
                WriteDescriptorSet::buffer(1, buffer_w.clone()),
                WriteDescriptorSet::buffer(2, buffer_y.clone()),
                WriteDescriptorSet::buffer(3, buffer_scales.clone()),
            ],
            [],
        )?;
```
And similarly in `run_qat_backward_async` (lines 842-852), `run_qat_optimizer_async` (lines 907-923), and `run_ghost_align_async` (lines 973-987).
- **Synchronous vs Asynchronous GEMM Executions**: `run_ternary_gemm_cached` (lines 495-514) and `run_ternary_gemm_cached_async` (lines 539-557) share the exact command buffer build step:
```rust
        let command_buffer = self.build_ternary_gemm_command_buffer(
            key, batch_size, n_in, n_out, buffer_x, packed_w, scales, buffer_y,
        )?;
```
They differ only in their execution blocks:
```rust
        // In run_ternary_gemm_cached:
        sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)?
            .then_signal_fence_and_flush()?
            .wait(None)?;
```
```rust
        // In run_ternary_gemm_cached_async:
        let _future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)?
            .then_signal_fence_and_flush()?;
```
- **Heartbeat & Imagination Dispatches**: Both `pulse_heartbeat` (lines 318-324) and `dispatch_imagination_async` (lines 334-342) call `build_heartbeat_command_buffer`:
```rust
        // pulse_heartbeat:
        let command_buffer = self.build_heartbeat_command_buffer(1);
        let _ = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush();

        // dispatch_imagination_async:
        let command_buffer = self.build_heartbeat_command_buffer(64);
        let future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        Box::new(future)
```

### R2. Dead Code and Unused Variables
- **sample_probs**: The struct `InferenceWorkspace` (src/mud/workspace.rs, lines 199-236) has fields like `sample_candidates: parking_lot::Mutex<Vec<(usize, f32)>>` but **no `sample_probs` field**. Similarly, in `InferenceWorkspace::new` (lines 287-372), there is no allocation of `sample_probs`.
- **_cos_sim & _l2_shift**: In `src/mud/forward.rs` (lines 885-918), there are no variables named `_cos_sim` or `_l2_shift`. The variables are actually named `cos_sim` and `l2_shift_val` (with `l2_shift` holding the sum of squares):
```rust
889:                 let mut l2_shift = 0.0;
...
911:                 let cos_sim = dot / ((norm_in * norm_out).sqrt() + EPSILON_FLOOR);
912:                 let l2_shift_val = l2_shift.sqrt();
913:                 println!(
914:                     "  [TRACE FINAL OUT] cos_sim={:.6} l2_shift={:.6} min={:.6} max={:.6}",
915:                     cos_sim, l2_shift_val, min_out, max_out
916:                 );
```
These variables are fully used inside the `if self.trace_propagation { ... }` logging block.

### R3. Vulkan iGPU Latency Profiling and Optimization (src/vulkan/mod.rs)
- **Lack of Pipeline Barriers**: In `run_chained_ffn` (lines 584-827), 4 separate compute dispatches (W1 GEMV, W3 GEMV, SiLU gate, W2 GEMV) are recorded sequentially in the same command buffer. No Vulkan pipeline barriers (`pipeline_barrier`) are ever recorded between these dispatches.
- **Asynchronous Execution with Immediate CPU Read**: In `src/mud/forward.rs`:
  - Line 414: `gemv_vulkan_or_cpu` for `key_o` is called with `is_async = true`.
  - Line 418: The CPU immediately accesses `final_attn_out.read()`.
  - Line 1180: `gemv_vulkan_or_cpu` for `key_out` (Mamba output projection) is called with `is_async = true`.
  - Line 816: The CPU immediately reads `final_attn_out.read()`.
  No fences or future wait calls are performed on the CPU before these read operations.

---

## 2. Logic Chain

### R1. Vulkan Dispatch Code Deduplication
1. Proposing a common helper `create_descriptor_set(&self, pipeline: &ComputePipeline, writes: impl IntoIterator<Item = WriteDescriptorSet>) -> anyhow::Result<Arc<PersistentDescriptorSet>>` will consolidate all repetitive descriptor set layout extraction and allocation boilerplate.
2. Proposing a generic command buffer builder `build_compute_command_buffer<Pc>(&self, pipeline: Arc<ComputePipeline>, writes: impl IntoIterator<Item = WriteDescriptorSet>, push_constants: Pc, grid: [u32; 3])` will deduplicate binding and dispatching.
3. Proposing `execute_sync` and `execute_async` helpers will clean up command buffer submission across GEMM, GEMV, QAT, and Heartbeat/Imagination dispatches.

### R2. Dead Code and Unused Variables
1. **sample_probs**: This field is completely absent from `InferenceWorkspace` and its initializer, indicating it was already removed.
2. **_cos_sim and _l2_shift**: These variables are actually named `cos_sim` and `l2_shift_val` in the codebase. Since they are used inside `if self.trace_propagation`, they cannot be removed unless the entire debug logging block is removed. If the compiler issues warnings about unused variables under release configurations where `trace_propagation` might be optimized, they are safely enclosed in the conditional block.

### R3. Vulkan iGPU Latency Profiling and Optimization
1. **GPU Data Hazards**: In `run_chained_ffn`, Dispatch 3 (SiLU) reads buffers written by Dispatches 1 & 2 (W1/W3 GEMV), and Dispatch 4 (W2 GEMV) reads the buffer written by Dispatch 3. Without pipeline barriers, the GPU will execute these concurrently/out-of-order, causing read-after-write (RAW) races. On iGPUs, cache coherency between GPU cores relies on explicit barriers; lacking them causes heavy GPU-side execution bubbles and memory latency spikes.
2. **CPU-GPU Synchronization Races**: When `gemv_vulkan_or_cpu` is called asynchronously (for attention output `key_o` and Mamba output `key_out`), the command buffer is submitted to the GPU queue and the CPU thread proceeds immediately. Reading `final_attn_out` on the CPU right after submission causes either:
   - Reading stale values (0.0), causing incorrect mathematical propagation.
   - Blocking the CPU thread implicitly inside the driver's memory mapper to wait for the GPU, which incurs massive synchronization and thread-scheduling overhead, directly contributing to the +575.02 ms latency discrepancy.
3. **Coherency**: Host-visible allocations should be explicitly host-coherent via `MemoryTypeFilter::HOST_COHERENT` to prevent CPU cache invalidation overhead.

---

## 3. Caveats
- Command executions (`run_command`) timed out waiting for user permission, so the codebase was investigated via static analysis only. Direct compiler behavior/warning verification could not be run, but the files were thoroughly traced.

---

## 4. Conclusion

### Proposed Vulkan Helper Functions (`src/vulkan/mod.rs`):
```rust
impl VulkanContext {
    fn create_descriptor_set(
        &self,
        pipeline: &ComputePipeline,
        descriptor_writes: impl IntoIterator<Item = WriteDescriptorSet>,
    ) -> anyhow::Result<Arc<PersistentDescriptorSet>> {
        let layout = pipeline.layout().set_layouts().first().unwrap();
        Ok(PersistentDescriptorSet::new(
            &*self.descriptor_set_allocator,
            layout.clone(),
            descriptor_writes,
            [],
        )?)
    }

    unsafe fn execute_sync(&self, command_buffer: Arc<vulkano::command_buffer::PrimaryAutoCommandBuffer>) -> anyhow::Result<()> {
        sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)?
            .then_signal_fence_and_flush()?
            .wait(None)?;
        Ok(())
    }

    unsafe fn execute_async(&self, command_buffer: Arc<vulkano::command_buffer::PrimaryAutoCommandBuffer>) -> anyhow::Result<vulkano::sync::FenceSignalFuture<Box<dyn GpuFuture>>> {
        let future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)?
            .then_signal_fence_and_flush()?;
        Ok(future)
    }
}
```

### Proposed Pipeline Barrier Insertion in `run_chained_ffn`:
To resolve memory coherency and execution hazards on the iGPU, insert pipeline barriers between dispatches in `run_chained_ffn`:
```rust
use vulkano::sync::{DependencyInfo, BufferMemoryBarrier, PipelineStages, AccessFlags};

// Before Dispatch 3 (SiLU):
builder.pipeline_barrier(DependencyInfo {
    buffer_memory_barriers: vec![
        BufferMemoryBarrier {
            src_stages: PipelineStages::COMPUTE_SHADER,
            src_access: AccessFlags::SHADER_WRITE,
            dst_stages: PipelineStages::COMPUTE_SHADER,
            dst_access: AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE,
            queue_family_transfer: None,
            buffer: buffer_w1_out.clone(),
            range: 0..buffer_w1_out.size(),
            ..Default::default()
        },
        BufferMemoryBarrier {
            src_stages: PipelineStages::COMPUTE_SHADER,
            src_access: AccessFlags::SHADER_WRITE,
            dst_stages: PipelineStages::COMPUTE_SHADER,
            dst_access: AccessFlags::SHADER_READ,
            queue_family_transfer: None,
            buffer: buffer_w3_out.clone(),
            range: 0..buffer_w3_out.size(),
            ..Default::default()
        },
    ],
    ..Default::default()
})?;

// Before Dispatch 4 (W2 GEMV):
builder.pipeline_barrier(DependencyInfo {
    buffer_memory_barriers: vec![
        BufferMemoryBarrier {
            src_stages: PipelineStages::COMPUTE_SHADER,
            src_access: AccessFlags::SHADER_WRITE,
            dst_stages: PipelineStages::COMPUTE_SHADER,
            dst_access: AccessFlags::SHADER_READ,
            queue_family_transfer: None,
            buffer: buffer_w1_out.clone(),
            range: 0..buffer_w1_out.size(),
            ..Default::default()
        },
    ],
    ..Default::default()
})?;
```

---

## 5. Verification Method
1. **Compilation**: After refactoring, run:
   `cargo clippy --all-targets --features tools -- -D warnings`
   This asserts that the codebase compiles with 0 warnings/errors.
2. **Execution Integrity**: Run:
   `cargo test --release --lib`
   This verifies correctness of calculations, matching attention scores, and model forward execution sanity.
