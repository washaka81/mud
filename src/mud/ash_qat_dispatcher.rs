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
//!
//! ## SAFETY contract (R-36)
//! The `unsafe` sites here are raw VRAM access plus deferred readback pointers.
//! `write_f32`/`read_f32` target `AshBuffer`s created by `ensure_buffers` /
//! `ensure_mhc_buffers` with the tensor's own `elements`/`rows`, and callers
//! pass the matching shadow/grad slices (`tu.shadow.len() == tu.elements`,
//! enforced by the `step_*` `# Safety` contracts). Deferred packed/scales
//! targets are plain addresses into MudFile layer tensors that outlive the
//! queue until `flush_pending`/`sync_all` (see `PendingReadback`); reading them
//! is only done after `sync()` fences GPU work. The dispatcher itself is owned
//! by the single training thread (`PendingReadback: Send` documents the
//! single-thread hand-off).

use crate::vulkan::ash_backend::{AshContext, AshOptimizerUpdate};
use std::collections::HashMap;

/// Key suffixes for buffer naming inside AshContext's cache.
const S_SHADOW: &str = "shadow";
const S_GRAD: &str = "grad";
const S_SCALES: &str = "scales";
const S_PACKED: &str = "packed";
const S_MHC_SHADOW: &str = "mhc_shadow";
const S_MHC_GRAD: &str = "mhc_grad";

/// One mHC scale SGD update for `step_mhc_async`.
/// `shadow` is the current mHC alpha/beta weights; `grad` is the gradient.
/// Both must have length `elements`.
pub struct AshMhcStep<'a> {
    pub name: String,
    pub shadow: &'a [f32],
    pub grad: &'a [f32],
    pub elements: usize,
}

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
            // SAFETY: `skey` was just allocated with `elements * 4` bytes
            // (4 lines above); every caller passes `initial_shadow` as the
            // shadow tensor whose length defines `elements`
            // (`shadow_w.len()` in session.rs, `tu.shadow.len() == tu.elements`
            // for `AshTensorStep`), so the copy fits the mapping.
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

    // ─────────────────────────────────────────────────────────────────────────
    // mHC scale SGD buffer management (Vulkan delegation, safe on Iris Xe)
    // ─────────────────────────────────────────────────────────────────────────

    fn mhc_shadow_key(name: &str) -> String {
        format!("{}.{}", name, S_MHC_SHADOW)
    }
    fn mhc_grad_key(name: &str) -> String {
        format!("{}.{}", name, S_MHC_GRAD)
    }

    /// Ensure mHC shadow + grad buffers exist in VRAM. Creates on first call.
    ///
    /// # Safety
    /// `initial_shadow.len() == elements`.
    pub unsafe fn ensure_mhc_buffers(
        &mut self,
        name: &str,
        elements: usize,
        initial_shadow: &[f32],
    ) -> anyhow::Result<()> {
        let skey = Self::mhc_shadow_key(name);
        let gkey = Self::mhc_grad_key(name);
        if self.ctx.get_buffer(&skey).is_none() {
            self.ctx.alloc_host_visible(&skey, elements * 4)?;
            if let Some(buf) = self.ctx.get_buffer(&skey) {
                buf.write_f32(initial_shadow);
            }
        }
        if self.ctx.get_buffer(&gkey).is_none() {
            self.ctx.alloc_host_visible(&gkey, elements * 4)?;
        }
        Ok(())
    }

    /// Zero-copy upload of mHC gradients.
    ///
    /// # Safety
    /// Buffer must exist via `ensure_mhc_buffers`. `data.len() == elements`.
    pub unsafe fn upload_mhc_grad(&self, name: &str, data: &[f32]) {
        let key = Self::mhc_grad_key(name);
        if let Some(buf) = self.ctx.get_buffer(&key) {
            buf.write_f32(data);
        }
    }

    /// Zero-copy readback of updated mHC weights.
    ///
    /// # Safety
    /// Buffer must exist via `ensure_mhc_buffers`. `out.len() == elements`.
    pub unsafe fn readback_mhc(&self, name: &str, out: &mut [f32]) {
        let key = Self::mhc_shadow_key(name);
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

    /// D5: Dispatch backward pass (dL/dW = X^T * dL/dY) via `ternary_backward.comp`.
    ///
    /// Uses `submit_and_wait` (synchronous). For signal keep-alive, batch with
    /// mHC/STP/JEPA via `dispatch_optimizer_batch_async` (future: combined shader).
    ///
    /// # Safety
    /// `x.len() >= batch_size * n_in`, `grad_y.len() >= batch_size * n_out`,
    /// `grad_w.len() >= n_out * n_in`.
    pub unsafe fn step_backward_sync(
        &mut self,
        x: &[f32],
        grad_y: &[f32],
        grad_w: &mut [f32],
        n_in: usize,
        n_out: usize,
        batch_size: usize,
    ) -> anyhow::Result<()> {
        self.ctx
            .dispatch_ternary_backward_host_sync(x, grad_y, grad_w, n_in, n_out, batch_size)
    }

    /// D5: Async backward pass using DoubleFrame (L-05) — no `submit_and_wait`.
    /// Uploads X + grad_y, dispatches `ternary_backward.comp`, returns immediately.
    /// Call `sync` then `readback_backward_grad` to retrieve grad_w.
    ///
    /// **Signal keep-alive:** batch with mHC/STP/JEPA via `dispatch_optimizer_batch_async`
    /// for a single command buffer covering all ops (future: combined backward+optimizer shader).
    ///
    /// # Safety
    /// `x.len() >= batch_size * n_in`, `grad_y.len() >= batch_size * n_out`.
    pub unsafe fn step_backward_async(
        &mut self,
        x: &[f32],
        grad_y: &[f32],
        n_in: usize,
        n_out: usize,
        batch_size: usize,
    ) -> anyhow::Result<()> {
        self.ctx
            .dispatch_ternary_backward_async(x, grad_y, n_in, n_out, batch_size)
    }

    /// D5: Readback grad_w after `sync`. Must be called after GPU completes.
    ///
    /// # Safety
    /// `grad_w.len() >= n_out * n_in`. Caller must have called `sync` first.
    pub unsafe fn readback_backward_grad(
        &self,
        grad_w: &mut [f32],
        n_in: usize,
        n_out: usize,
    ) -> anyhow::Result<()> {
        self.ctx.readback_backward_grad(grad_w, n_in, n_out)
    }

    /// D5: Backward pass — dL/dX = dL/dY * W (gradient w.r.t. input) via `ternary_backward_dx.comp`.
    ///
    /// Computes the chain-rule backward through Y = X * W^T (forward GEMV).
    /// Uses `submit_and_wait` (synchronous). For signal keep-alive, batch with
    /// mHC/STP/JEPA via `dispatch_optimizer_batch_async` (future: combined shader).
    ///
    /// # Safety
    /// `grad_y.len() >= batch_size * n_out`, `w.len() >= n_out * n_in`,
    /// `grad_x.len() >= batch_size * n_in`.
    pub unsafe fn step_backward_dx_sync(
        &mut self,
        grad_y: &[f32],
        w: &[f32],
        grad_x: &mut [f32],
        n_in: usize,
        n_out: usize,
        batch_size: usize,
    ) -> anyhow::Result<()> {
        self.ctx
            .dispatch_ternary_backward_dx_host_sync(grad_y, w, grad_x, n_in, n_out, batch_size)
    }

    /// D5: Async backward pass — dL/dX (no `submit_and_wait`).
    /// Uploads grad_y + W, dispatches `ternary_backward_dx.comp`, returns immediately.
    /// Call `sync` then `readback_backward_dx_grad` to retrieve grad_x.
    ///
    /// **Signal keep-alive:** batch with mHC/STP/JEPA via `dispatch_optimizer_batch_async`
    /// for a single command buffer covering all ops.
    ///
    /// # Safety
    /// `grad_y.len() >= batch_size * n_out`, `w.len() >= n_out * n_in`.
    pub unsafe fn step_backward_dx_async(
        &mut self,
        grad_y: &[f32],
        w: &[f32],
        n_in: usize,
        n_out: usize,
        batch_size: usize,
    ) -> anyhow::Result<()> {
        self.ctx
            .dispatch_ternary_backward_dx_async(grad_y, w, n_in, n_out, batch_size)
    }

    /// D5: Readback grad_x after `sync`. Must be called after GPU completes.
    ///
    /// # Safety
    /// `grad_x.len() >= batch_size * n_in`. Caller must have called `sync` first.
    pub unsafe fn readback_backward_dx_grad(
        &self,
        grad_x: &mut [f32],
        n_in: usize,
        n_out: usize,
        batch_size: usize,
    ) -> anyhow::Result<()> {
        self.ctx.readback_backward_dx_grad(grad_x, n_in, n_out, batch_size)
    }

    /// D5: Generic element-wise ops for STP + JEPA (safe on Iris Xe).
    /// Modes: 0=add, 1=subtract, 2=clamp, 3=sigmoid, 4=ema, 5=normalize, 6=axpy
    ///
    /// # Safety
    /// Buffers must have length >= n.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn step_elementwise_sync(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        target: &[f32],
        n: usize,
        mode: u32,
        param0: f32,
        param1: f32,
    ) -> anyhow::Result<()> {
        self.ctx
            .dispatch_elementwise_sync(input, output, target, n, mode, param0, param1)
    }

    /// D5: Combined batch — backward + mHC + element-wise in ONE command buffer (L-05 DoubleFrame).
    ///
    /// This is the signal keep-alive core: all ops go into a single `vkQueueSubmit`,
    /// keeping the Iris Xe in boost clock. Uses `frame.acquire` + `frame.submit` (no `submit_and_wait`).
    ///
    /// Readback: call `sync` then `readback_backward_grad` / `readback_mhc` / `readback_elementwise`.
    ///
    /// # Safety
    /// All named buffers must exist in VRAM (call `ensure_buffers`/`ensure_mhc_buffers` first).
    pub unsafe fn step_combined_batch_async(
        &mut self,
        backward_updates: &[crate::vulkan::ash_backend::AshBackwardUpdate],
        mhc_updates: &[crate::vulkan::ash_backend::AshOptimizerUpdate],
        elementwise_updates: &[crate::vulkan::ash_backend::AshElementwiseUpdate],
    ) -> anyhow::Result<()> {
        self.ctx.dispatch_combined_batch_async(
            backward_updates,
            mhc_updates,
            elementwise_updates,
        )
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
                is_mhc: false,
            });
        }

        self.ctx.dispatch_optimizer_batch_async(&vk_updates)
    }

    /// Batch mHC scale SGD step via Vulkan (safe on Iris Xe — no QKV shader, no submit_and_wait).
    ///
    /// Each update runs the optimizer shader in mHC mode: `w = clamp(w - lr * g / num_tokens, 0.0, 4.0)`.
    /// All updates go into ONE command buffer (async, L-05 DoubleFrame overlap).
    ///
    /// # Safety
    /// Caller guarantees `shadow.len() == grad.len() == elements` for each update.
    /// VRAM buffers are created/updated zero-copy via `ensure_mhc_buffers` + `upload_mhc_grad`.
    pub unsafe fn step_mhc_async<'a>(
        &mut self,
        updates: &[AshMhcStep<'a>],
        lr: f32,
        num_tokens: f32,
    ) -> anyhow::Result<()> {
        let mut vk_updates: Vec<AshOptimizerUpdate> = Vec::with_capacity(updates.len());

        for tu in updates {
            self.ensure_mhc_buffers(&tu.name, tu.elements, tu.shadow)?;
            self.upload_mhc_grad(&tu.name, tu.grad);

            vk_updates.push(AshOptimizerUpdate {
                shadow_key: Self::mhc_shadow_key(&tu.name),
                grad_key: Self::mhc_grad_key(&tu.name),
                scales_key: String::new(), // unused in mHC mode
                packed_key: String::new(), // unused in mHC mode
                total_elements: tu.elements,
                cols: tu.elements, // mHC is element-wise, cols == total_elements (1 row)
                learning_rate: lr,
                weight_decay: 0.0, // mHC SGD has no weight decay
                num_tokens,
                is_mhc: true,
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

    /// D5: Combined batch — optimizer step + mHC scale SGD in ONE `vkQueueSubmit`.
    ///
    /// Uploads all non-mHC gradients + mHC gradients, then dispatches
    /// `dispatch_combined_batch_async` with all updates (both mHC and non-mHC)
    /// in a single command buffer. Keeps Iris Xe in boost clock (signal keep-alive).
    ///
    /// Packed readbacks are queued for deferred readback (L-05 overlap).
    ///
    /// # Safety
    /// Readback pointers must stay valid until `flush_pending`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn step_combined_async_deferred(
        &mut self,
        tensor_updates: &[AshTensorStep],
        mhc_updates: &[AshMhcStep],
        readbacks: Vec<PendingReadback>,
        lr: f32,
        weight_decay: f32,
        num_tokens: f32,
    ) -> anyhow::Result<()> {
        self.flush_pending()?;

        let mut vk_updates: Vec<AshOptimizerUpdate> =
            Vec::with_capacity(tensor_updates.len() + mhc_updates.len());

        // Upload non-mHC gradients + build optimizer updates
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
                is_mhc: false,
            });
        }

        // Upload mHC gradients + build mHC updates
        for tu in mhc_updates {
            self.ensure_mhc_buffers(&tu.name, tu.elements, tu.shadow)?;
            self.upload_mhc_grad(&tu.name, tu.grad);

            vk_updates.push(AshOptimizerUpdate {
                shadow_key: Self::mhc_shadow_key(&tu.name),
                grad_key: Self::mhc_grad_key(&tu.name),
                scales_key: String::new(), // unused in mHC mode
                packed_key: String::new(), // unused in mHC mode
                total_elements: tu.elements,
                cols: tu.elements, // mHC is element-wise, cols == total_elements
                learning_rate: lr,
                weight_decay: 0.0, // mHC SGD has no weight decay
                num_tokens,
                is_mhc: true,
            });
        }

        // Dispatch ALL updates in ONE command buffer (D5 signal keep-alive)
        self.ctx.dispatch_combined_batch_async(
            &[], // no backward updates
            &vk_updates, // both mHC and non-mHC optimizer updates
            &[], // no elementwise updates
        )?;

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
        // SAFETY: `flush_pending` holds the same pointer-validity contract as
        // the queue that produced the readbacks (layer tensors alive until the
        // end of training); `ctx.sync()` fences both double-buffer slots before
        // any subsequent reuse of mapped memory.
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

/// D4: Pipeline overlap orchestrator for full CPU→GPU→CPU overlap.
///
/// Wraps `AshQatDispatcher` and orchestrates a 3-stage pipeline:
///   1. CPU readback (drain previous slot) — flush pending readbacks from chunk N-1
///   2. GPU GEMV (submit without wait) — dispatch optimizer batch for chunk N
///   3. Swap slots — rotate double-buffer for next iteration
///
/// The CPU Forward N+1 runs concurrently with GPU Optimizer N because
/// `step_async_deferred` returns immediately (L-05 DoubleFrame).
///
/// Env: `MUD_PIPELINE_OVERLAP=1` to activate.
pub struct PipelineOverlap {
    /// Current double-buffer slot (0 or 1)
    current_slot: usize,
    /// Whether a GPU dispatch is pending in the current slot
    gemv_pending: bool,
    /// Whether a CPU readback is pending from the previous slot
    readback_pending: bool,
    /// Whether CPU prepare is running in background (telemetry)
    prepare_running: bool,
    /// Stream I: Whether a KV block decompression is pending (overlap with GPU work)
    kv_decompress_pending: bool,
    /// Stream I: Block index of the pending KV decompression
    kv_decompress_block: usize,
}

impl PipelineOverlap {
    pub fn new() -> Self {
        Self {
            current_slot: 0,
            gemv_pending: false,
            readback_pending: false,
            prepare_running: false,
            kv_decompress_pending: false,
            kv_decompress_block: 0,
        }
    }

    /// Current slot index (0 or 1).
    #[inline]
    pub fn current_slot(&self) -> usize {
        self.current_slot
    }

    /// Previous slot index (0 or 1).
    #[inline]
    pub fn prev_slot(&self) -> usize {
        1 - self.current_slot
    }

    /// D4: Full pipeline step — orchestrates CPU readback → GPU GEMV → slot swap.
    ///
    /// This is the core of the D4 pipeline overlap:
    /// 1. `flush_pending` — drain CPU readback from chunk N-1 (while GPU works on N)
    /// 2. `step_async_deferred` — submit GPU GEMV for chunk N (returns immediately)
    /// 3. Swap slots — rotate double-buffer for chunk N+1
    ///
    /// The CPU Forward N+1 can start immediately after this returns, overlapping
    /// with GPU Optimizer N.
    ///
    /// # Safety
    /// All tensor update pointers and readback pointers must be valid.
    pub unsafe fn step_pipeline(
        &mut self,
        dispatcher: &mut AshQatDispatcher,
        tensor_updates: &[AshTensorStep],
        readbacks: Vec<PendingReadback>,
        lr: f32,
        weight_decay: f32,
        num_tokens: f32,
    ) -> anyhow::Result<()> {
        // 1. CPU readback: drain previous slot's pending readbacks.
        // This runs while GPU is still working on the current slot.
        if self.readback_pending {
            dispatcher.flush_pending()?;
            self.readback_pending = false;
        }

        // 2. GPU GEMV: submit optimizer batch without waiting (L-05 DoubleFrame).
        // Returns immediately so CPU can start Forward N+1.
        dispatcher.step_async_deferred(
            tensor_updates,
            readbacks,
            lr,
            weight_decay,
            num_tokens,
        )?;
        self.gemv_pending = true;
        self.readback_pending = true;

        // 3. Swap slots for next iteration.
        self.swap_slots();

        Ok(())
    }

    /// D4: Swap double-buffer slots (0↔1).
    #[inline]
    pub fn swap_slots(&mut self) {
        self.current_slot = 1 - self.current_slot;
    }

    /// D4: Mark that CPU prepare is running in background (for telemetry).
    pub fn mark_prepare_started(&mut self) {
        self.prepare_running = true;
    }

    /// D4: Mark that CPU prepare has finished (for telemetry).
    pub fn mark_prepare_done(&mut self) {
        self.prepare_running = false;
    }

    /// D4: Check if CPU prepare is running in background.
    pub fn is_prepare_running(&self) -> bool {
        self.prepare_running
    }

    /// D4: Check if GPU GEMV is pending.
    pub fn is_gemv_pending(&self) -> bool {
        self.gemv_pending
    }

    /// D4: Check if CPU readback is pending.
    pub fn is_readback_pending(&self) -> bool {
        self.readback_pending
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Stream I: KV cache decompression overlap (Delta + LZ4)
    // ─────────────────────────────────────────────────────────────────────────

    /// Stream I: Mark that a KV block decompression should run in the background
    /// while GPU optimizer work is in flight. The caller should call
    /// `CompressedKvCache::decompress_block` for `block_idx` during the next
    /// `step_pipeline` call.
    #[inline]
    pub fn mark_kv_decompress(&mut self, block_idx: usize) {
        self.kv_decompress_pending = true;
        self.kv_decompress_block = block_idx;
    }

    /// Stream I: Check if a KV block decompression is pending.
    #[inline]
    pub fn is_kv_decompress_pending(&self) -> bool {
        self.kv_decompress_pending
    }

    /// Stream I: Block index of the pending KV decompression.
    #[inline]
    pub fn kv_decompress_block(&self) -> usize {
        self.kv_decompress_block
    }

    /// Stream I: Clear the KV decompression pending flag (call after decompressing).
    #[inline]
    pub fn clear_kv_decompress(&mut self) {
        self.kv_decompress_pending = false;
    }

    /// D4: Final drain — flush all pending readbacks and wait for GPU idle.
    /// Call at epoch end / checkpoint / process exit.
    ///
    /// # Safety
    /// The dispatcher must have valid Vulkan context and buffers.
    pub unsafe fn drain_all(&mut self, dispatcher: &mut AshQatDispatcher) -> anyhow::Result<()> {
        if self.gemv_pending {
            dispatcher.sync_all()?;
            self.gemv_pending = false;
            self.readback_pending = false;
        }
        Ok(())
    }
}

impl Default for PipelineOverlap {
    fn default() -> Self {
        Self::new()
    }
}

/// D4: Check if pipeline overlap is enabled (MUD_PIPELINE_OVERLAP=1).
/// Hot-path optimization: cached via `thread_local` (P-01) — avoids
/// `std::env::var` syscall on every `step_pipeline` call.
pub fn pipeline_overlap_enabled() -> bool {
    PIPELINE_OVERLAP_CACHE.with(|cache| {
        if let Some(v) = *cache.borrow() {
            return v;
        }
        let v = std::env::var("MUD_PIPELINE_OVERLAP")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        *cache.borrow_mut() = Some(v);
        v
    })
}

thread_local! {
    #[allow(clippy::missing_const_for_thread_local)]
    static PIPELINE_OVERLAP_CACHE: std::cell::RefCell<Option<bool>> = std::cell::RefCell::new(None);
}

/// D4: Reset the pipeline overlap cache (test-only).
#[cfg(test)]
pub fn reset_pipeline_overlap_cache() {
    PIPELINE_OVERLAP_CACHE.with(|c| *c.borrow_mut() = None);
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

    #[test]
    fn test_pipeline_overlap_slots() {
        let p = PipelineOverlap::new();
        assert_eq!(p.current_slot(), 0);
        assert_eq!(p.prev_slot(), 1);
        assert!(!p.is_gemv_pending());
        assert!(!p.is_readback_pending());
        assert!(!p.is_prepare_running());
        assert!(!p.is_kv_decompress_pending());
    }

    #[test]
    fn test_pipeline_overlap_swap() {
        let mut p = PipelineOverlap::new();
        assert_eq!(p.current_slot(), 0);
        p.swap_slots();
        assert_eq!(p.current_slot(), 1);
        assert_eq!(p.prev_slot(), 0);
        p.swap_slots();
        assert_eq!(p.current_slot(), 0);
    }

    #[test]
    fn test_pipeline_overlap_prepare_flags() {
        let mut p = PipelineOverlap::new();
        p.mark_prepare_started();
        assert!(p.is_prepare_running());
        p.mark_prepare_done();
        assert!(!p.is_prepare_running());
    }

    #[test]
    fn test_pipeline_overlap_kv_decompress() {
        let mut p = PipelineOverlap::new();
        assert!(!p.is_kv_decompress_pending());
        assert_eq!(p.kv_decompress_block(), 0);

        p.mark_kv_decompress(3);
        assert!(p.is_kv_decompress_pending());
        assert_eq!(p.kv_decompress_block(), 3);

        p.clear_kv_decompress();
        assert!(!p.is_kv_decompress_pending());
    }

    #[test]
    fn test_pipeline_overlap_env() {
        // Reset cache before each env var change (thread_local cache)
        // Default: disabled
        std::env::remove_var("MUD_PIPELINE_OVERLAP");
        reset_pipeline_overlap_cache();
        assert!(!pipeline_overlap_enabled());

        // Enabled
        std::env::set_var("MUD_PIPELINE_OVERLAP", "1");
        reset_pipeline_overlap_cache();
        assert!(pipeline_overlap_enabled());

        // Enabled (case-insensitive)
        std::env::set_var("MUD_PIPELINE_OVERLAP", "TRUE");
        reset_pipeline_overlap_cache();
        assert!(pipeline_overlap_enabled());

        // Disabled
        std::env::set_var("MUD_PIPELINE_OVERLAP", "0");
        reset_pipeline_overlap_cache();
        assert!(!pipeline_overlap_enabled());

        std::env::remove_var("MUD_PIPELINE_OVERLAP");
        reset_pipeline_overlap_cache();
    }
}
