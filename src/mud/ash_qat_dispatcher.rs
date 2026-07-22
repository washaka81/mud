//! QAT Dispatcher — Phase 15: bare-metal ash backend
//!
//! This module exposes a single `AshQatDispatcher` that replaces `VulkanQatDispatcher`.
//! Key improvements over the vulkano path:
//!
//! 1. **Zero-copy gradient upload**: grad buffers are HOST_VISIBLE mapped permanently.
//!    `write_f32()` is a raw `ptr::copy_nonoverlapping` — no lock, no allocation.
//! 2. **Batch command buffer**: all optimizer dispatches for a full forward chunk
//!    go into ONE VkCommandBuffer and ONE vkQueueSubmit.
//! 3. **L-05 True Double-Buffer**: `step_async` submits without waiting; packed-weight
//!    readback is deferred until `flush_pending` / next step start so Forward N+1
//!    overlaps GPU Optimizer N (CPU packed vs VRAM packed are distinct buffers).
//! 4. **Persistent VRAM**: shadow weights never leave VRAM between steps.

use crate::vulkan::ash_backend::{AshContext, AshOptimizerUpdate};
use std::collections::HashMap;

/// Key suffixes for buffer naming inside AshContext's cache.
const S_SHADOW: &str = "shadow";
const S_GRAD: &str = "grad";
const S_SCALES: &str = "scales";
const S_PACKED: &str = "packed";

/// One deferred packed/scales readback target after an async optimizer step (L-05).
/// Pointers must remain valid until `flush_pending` (owned by MudFile layer tensors).
pub struct PendingReadback {
    pub name: String,
    pub packed_ptr: *mut u8,
    pub scales_ptr: *mut f32,
    pub packed_len: usize,
    pub rows: usize,
}

// SAFETY: Training is single-threaded w.r.t. these pointers; they are plain addresses
// into Mud tensors that outlive the pending queue for the duration of a train run.
unsafe impl Send for PendingReadback {}

/// Ash-based QAT dispatcher.
/// Owns the `AshContext` and coordinates all GPU dispatches for training.
pub struct AshQatDispatcher {
    pub ctx: AshContext,

    /// Tracks which buffer names have been created, to avoid re-allocation.
    known_buffers: HashMap<String, (usize, usize)>, // name → (elements, rows)

    /// L-05: last async step's readback targets; flushed at next step start / checkpoint.
    pending_readbacks: Option<Vec<PendingReadback>>,
}

impl AshQatDispatcher {
    pub fn new() -> anyhow::Result<Self> {
        let ctx = AshContext::new()?;
        Ok(Self {
            ctx,
            known_buffers: HashMap::new(),
            pending_readbacks: None,
        })
    }

    pub fn is_available(&self) -> bool {
        self.ctx.is_available()
    }

    /// True if an optimizer submit is in flight and packed readback has not been applied.
    pub fn has_pending(&self) -> bool {
        self.pending_readbacks.is_some()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Buffer management — all buffers are HOST_VISIBLE (Intel UMA = zero-copy)
    // ─────────────────────────────────────────────────────────────────────────

    fn shadow_key(name: &str) -> String {
        format!("{}.{}", name, S_SHADOW)
    }
    fn grad_key(name: &str) -> String {
        format!("{}.{}", name, S_GRAD)
    }
    fn scales_key(name: &str) -> String {
        format!("{}.{}", name, S_SCALES)
    }
    fn packed_key(name: &str) -> String {
        format!("{}.{}", name, S_PACKED)
    }

    /// Ensure all 4 buffers for a tensor exist in VRAM. Creates them on first call.
    /// On subsequent calls: returns immediately (O(1) HashMap lookup).
    pub fn ensure_buffers(
        &mut self,
        name: &str,
        elements: usize,
        rows: usize,
        initial_shadow: &[f32],
    ) -> anyhow::Result<()> {
        if self.known_buffers.contains_key(name) {
            return Ok(());
        }

        let packed_elements = elements.div_ceil(8); // 8 ternary weights per u32

        self.ctx
            .alloc_host_visible(&Self::shadow_key(name), elements * 4)?;
        self.ctx
            .alloc_host_visible(&Self::grad_key(name), elements * 4)?;
        self.ctx
            .alloc_host_visible(&Self::scales_key(name), rows * 4)?;
        self.ctx
            .alloc_host_visible(&Self::packed_key(name), packed_elements * 4)?;

        // P-00: Upload the initial CPU shadow weights to the newly created VRAM buffer.
        // Without this, the optimizer starts from uninitialized memory (all zeros) and collapses.
        let skey = Self::shadow_key(name);
        if let Some(buf) = self.ctx.get_buffer(&skey) {
            unsafe {
                buf.write_f32(initial_shadow);
            }
        }

        self.known_buffers
            .insert(name.to_string(), (elements, rows));
        Ok(())
    }

    /// Zero-copy upload: writes shadow weights to their pre-mapped VRAM pointer.
    ///
    /// # SAFETY
    /// Buffer must have been created via `ensure_buffers`. `data.len() == elements`.
    pub unsafe fn upload_shadow(&self, name: &str, data: &[f32]) {
        let key = Self::shadow_key(name);
        if let Some(buf) = self.ctx.get_buffer(&key) {
            buf.write_f32(data);
        }
    }

    /// Zero-copy upload: writes gradients to their pre-mapped VRAM pointer.
    ///
    /// # SAFETY
    /// Buffer must have been created via `ensure_buffers`. `data.len() == elements`.
    pub unsafe fn upload_grad(&self, name: &str, data: &[f32]) {
        let key = Self::grad_key(name);
        if let Some(buf) = self.ctx.get_buffer(&key) {
            buf.write_f32(data);
        }
    }

    /// Zero-copy readback: reads updated shadow weights from VRAM mapped pointer.
    ///
    /// # SAFETY
    /// Buffer must have been created via `ensure_buffers`. `out.len() == elements`.
    pub unsafe fn readback_shadow(&self, name: &str, out: &mut [f32]) {
        let key = Self::shadow_key(name);
        if let Some(buf) = self.ctx.get_buffer(&key) {
            buf.read_f32(out);
        }
    }

    /// Zero-copy readback: reads quantized packed weights from VRAM.
    ///
    /// # SAFETY
    /// Buffer must have been created via `ensure_buffers`. `out.len() == packed_elements`.
    pub unsafe fn readback_packed(&self, name: &str, out: &mut [u8]) {
        let key = Self::packed_key(name);
        if let Some(buf) = self.ctx.get_buffer(&key) {
            // SAFETY: casting u8 output to read from u32 VRAM — same memory, different view.
            let out_u32_len = out.len() / 4;
            let out_f32 = std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut f32, out_u32_len);
            buf.read_f32(out_f32);
        }
    }

    /// # Safety
    /// VRAM buffer must exist and `out.len()` must match.
    pub unsafe fn readback_scales(&self, name: &str, out: &mut [f32]) {
        let key = Self::scales_key(name);
        if let Some(buf) = self.ctx.get_buffer(&key) {
            buf.read_f32(out);
        }
    }

    /// ZERO-COPY UNIFIED MEMORY: Get the raw mapped pointer for the packed buffer (Priority 62).
    pub fn get_packed_ptr(&self, name: &str) -> Option<*mut u8> {
        let key = Self::packed_key(name);
        self.ctx.get_buffer(&key).and_then(|buf| buf.mapped_ptr)
    }

    /// ZERO-COPY UNIFIED MEMORY: Get the raw mapped pointer for the scales buffer (Priority 62).
    pub fn get_scales_ptr(&self, name: &str) -> Option<*mut f32> {
        let key = Self::scales_key(name);
        self.ctx
            .get_buffer(&key)
            .and_then(|buf| buf.mapped_ptr.map(|p| p as *mut f32))
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Batch dispatch — EDGE-08 async (returns before GPU finishes)
    // ─────────────────────────────────────────────────────────────────────────

    /// Dispatch the optimizer for all tensors in `updates` in ONE command buffer.
    /// Returns immediately — GPU continues in the background (L-05 DoubleFrame).
    ///
    /// # SAFETY
    /// All named buffers must exist in VRAM (call `ensure_buffers` first).
    pub unsafe fn dispatch_optimizer_batch_async(
        &mut self,
        updates: &[AshOptimizerUpdate],
    ) -> anyhow::Result<()> {
        self.ctx.dispatch_optimizer_batch_async(updates)
    }

    /// Block until all GPU work is complete (both DoubleFrame slots).
    /// Does **not** apply deferred packed readbacks — prefer `flush_pending` / `sync_all`.
    ///
    /// # SAFETY
    /// device and queue must be valid.
    pub unsafe fn sync(&self) -> anyhow::Result<()> {
        self.ctx.sync()
    }

    /// # Safety
    /// The caller must ensure the Vulkan context is valid and not in a destroyed state.
    pub unsafe fn dispatch_heartbeat_sync(&self) -> anyhow::Result<()> {
        self.ctx.dispatch_heartbeat_sync()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // High-level training step: upload → dispatch → (async) → readback next step
    // ─────────────────────────────────────────────────────────────────────────

    /// Full optimizer step for a batch of layer matrices.
    /// - Uploads gradients zero-copy
    /// - Dispatches ONE command buffer covering all matrices (async, no wait)
    /// - Returns immediately so the CPU can start the next forward pass
    ///
    /// Prefer `step_async_deferred` so packed readback is queued for L-05 overlap.
    ///
    /// # Safety
    /// Caller guarantees `tensor_updates` point to valid memory arrays with sizes matching their `elements` definitions.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn step_async(
        &mut self,
        tensor_updates: &[AshTensorStep],
        lr: f32,
        weight_decay: f32,
        num_tokens: f32,
    ) -> anyhow::Result<()> {
        let mut vk_updates: Vec<AshOptimizerUpdate> = Vec::with_capacity(tensor_updates.len());

        for tu in tensor_updates {
            self.ensure_buffers(&tu.name, tu.elements, tu.rows, tu.shadow)?;
            self.upload_grad(&tu.name, tu.grad);

            vk_updates.push(AshOptimizerUpdate {
                shadow_key: Self::shadow_key(&tu.name),
                grad_key: Self::grad_key(&tu.name),
                scales_key: Self::scales_key(&tu.name),
                packed_key: Self::packed_key(&tu.name),
                total_elements: tu.elements,
                cols: tu.cols,
                learning_rate: lr,
                weight_decay,
                num_tokens,
            });
        }

        self.ctx.dispatch_optimizer_batch_async(&vk_updates)
    }

    /// L-05: like `step_async`, then queue packed/scales readbacks without waiting.
    /// Call `flush_pending` at the **start** of the next chunk (or checkpoint) so
    /// Forward N+1 can run while GPU finishes Optimizer N.
    ///
    /// # Safety
    /// Readback pointers must stay valid until `flush_pending`.
    pub unsafe fn step_async_deferred(
        &mut self,
        tensor_updates: &[AshTensorStep],
        readbacks: Vec<PendingReadback>,
        lr: f32,
        weight_decay: f32,
        num_tokens: f32,
    ) -> anyhow::Result<()> {
        // Ensure any previous step is fully applied before overwriting VRAM shadows/grads.
        self.flush_pending()?;
        self.step_async(tensor_updates, lr, weight_decay, num_tokens)?;
        self.pending_readbacks = Some(readbacks);
        Ok(())
    }

    /// L-05: wait for in-flight optimizer and copy packed+scales into CPU layer tensors.
    /// No-op if nothing is pending.
    ///
    /// # Safety
    /// Pending pointers must still be valid.
    pub unsafe fn flush_pending(&mut self) -> anyhow::Result<()> {
        if let Some(readbacks) = self.pending_readbacks.take() {
            let tuples: Vec<(String, *mut u8, *mut f32, usize, usize)> = readbacks
                .into_iter()
                .map(|r| (r.name, r.packed_ptr, r.scales_ptr, r.packed_len, r.rows))
                .collect();
            self.sync_and_readback_all(&tuples)?;
        }
        Ok(())
    }

    /// Sync GPU and read back packed weights + scales for all tensors.
    ///
    /// # SAFETY
    /// All named buffers must exist. Raw pointers must point to valid memory.
    pub unsafe fn sync_and_readback_all(
        &self,
        readbacks: &[(String, *mut u8, *mut f32, usize, usize)], // (name, packed_ptr, scales_ptr, packed_len_bytes, rows)
    ) -> anyhow::Result<()> {
        // Wait for all async shader execution to finish before reading back memory.
        // Otherwise, CPU reads garbage while GPU is still writing.
        self.sync()?;

        for (name, packed_ptr, scales_ptr, packed_len, rows) in readbacks {
            let pk = Self::packed_key(name);
            let sk = Self::scales_key(name);
            if let Some(buf) = self.ctx.get_buffer(&pk) {
                // SAFETY: packed_ptr is valid for packed_len bytes.
                let out_f32 =
                    std::slice::from_raw_parts_mut(*packed_ptr as *mut f32, packed_len / 4);
                buf.read_f32(out_f32);
            }
            if let Some(buf) = self.ctx.get_buffer(&sk) {
                // SAFETY: scales_ptr is valid for rows * 4 bytes.
                let out = std::slice::from_raw_parts_mut(*scales_ptr, *rows);
                buf.read_f32(out);
            }
        }
        Ok(())
    }

    /// Drain pending readbacks (if any) and wait until GPU is idle.
    /// Call at epoch end / checkpoint / process exit.
    pub fn sync_all(&mut self) -> anyhow::Result<()> {
        unsafe {
            if self.pending_readbacks.is_some() {
                self.flush_pending()?;
            } else {
                self.ctx.sync()?;
            }
        }
        Ok(())
    }
}

/// Describes one tensor to update in a training step.
pub struct AshTensorStep<'a> {
    pub name: String, // e.g. "blk.0.q"
    pub shadow: &'a [f32],
    pub grad: &'a [f32],
    pub elements: usize,
    pub cols: usize,
    pub rows: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_key_format() {
        assert_eq!(AshQatDispatcher::shadow_key("blk.0.q"), "blk.0.q.shadow");
        assert_eq!(AshQatDispatcher::grad_key("blk.0.q"), "blk.0.q.grad");
        assert_eq!(AshQatDispatcher::scales_key("blk.0.q"), "blk.0.q.scales");
        assert_eq!(AshQatDispatcher::packed_key("blk.0.q"), "blk.0.q.packed");
    }

    #[test]
    fn test_ash_dispatcher_disabled() {
        // When MUD_USE_VULKAN=0, dispatcher should fail gracefully.
        unsafe {
            std::env::set_var("MUD_USE_VULKAN", "0");
        }
        let result = AshQatDispatcher::new();
        unsafe {
            std::env::set_var("MUD_USE_VULKAN", "1");
        }
        assert!(
            result.is_err(),
            "Should fail gracefully when Vulkan is disabled"
        );
    }
}
