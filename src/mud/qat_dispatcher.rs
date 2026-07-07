use std::sync::Arc;
use std::collections::HashMap;
use crate::vulkan::VulkanContext;
use vulkano::buffer::Subbuffer;

/// Manages persistent VRAM buffers for FULL-QAT training.
pub struct VulkanQatDispatcher {
    pub vk: Arc<VulkanContext>,
    
    // Persistent shadow weights (remains in VRAM across steps to avoid PCIe bottleneck)
    pub shadow_w_cache: HashMap<String, Subbuffer<[f32]>>,
    
    // Ping-pong buffers for ephemeral data (One per tensor to allow overlapping GPU execution)
    pub grad_w_cache: HashMap<String, Subbuffer<[f32]>>,
    pub scales_cache: HashMap<String, Subbuffer<[f32]>>,
    pub packed_w_cache: HashMap<String, Subbuffer<[u32]>>,
}

impl VulkanQatDispatcher {
    pub fn new(vk: Arc<VulkanContext>) -> Self {
        Self {
            shadow_w_cache: HashMap::new(),
            grad_w_cache: HashMap::new(),
            scales_cache: HashMap::new(),
            packed_w_cache: HashMap::new(),
            vk,
        }
    }
    
    pub fn get_or_create_shadow_buffer(&mut self, name: &str, elements: usize, initial_data: Option<&[f32]>) -> Subbuffer<[f32]> {
        if let Some(buf) = self.shadow_w_cache.get(name) { return buf.clone(); }
        let buf = self.vk.allocate_zero_copy_buffer(elements);
        if let Some(data) = initial_data {
            buf.write().unwrap().copy_from_slice(data);
        }
        self.shadow_w_cache.insert(name.to_string(), buf.clone());
        buf
    }

    pub fn get_or_create_grad(&mut self, name: &str, elements: usize) -> Subbuffer<[f32]> {
        if let Some(buf) = self.grad_w_cache.get(name) { return buf.clone(); }
        let buf = self.vk.allocate_zero_copy_buffer(elements);
        self.grad_w_cache.insert(name.to_string(), buf.clone());
        buf
    }

    pub fn get_or_create_scales(&mut self, name: &str, rows: usize) -> Subbuffer<[f32]> {
        if let Some(buf) = self.scales_cache.get(name) { return buf.clone(); }
        let buf = self.vk.allocate_zero_copy_buffer(rows);
        self.scales_cache.insert(name.to_string(), buf.clone());
        buf
    }

    pub fn get_or_create_packed(&mut self, name: &str, elements: usize) -> Subbuffer<[u32]> {
        if let Some(buf) = self.packed_w_cache.get(name) { return buf.clone(); }
        let buf = self.vk.allocate_zero_copy_buffer_u32(elements.div_ceil(8));
        self.packed_w_cache.insert(name.to_string(), buf.clone());
        buf
    }
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_optimizer(
        &self,
        shadow_buf: &Subbuffer<[f32]>,
        grad_buf: &Subbuffer<[f32]>,
        scales_buf: &Subbuffer<[f32]>,
        packed_buf: &Subbuffer<[u32]>,
        elements: usize,
        cols: usize,
        lr: f32,
        decay: f32
    ) -> anyhow::Result<()> {
        unsafe {
            self.vk.run_qat_optimizer_async(
                elements,
                cols,
                lr,
                decay,
                shadow_buf,
                grad_buf,
                scales_buf,
                packed_buf,
            )
        }
    }

    pub fn dispatch_optimizer_batch(
        &self,
        updates: &[(usize, usize, f32, f32, Subbuffer<[f32]>, Subbuffer<[f32]>, Subbuffer<[f32]>, Subbuffer<[u32]>)]
    ) -> anyhow::Result<()> {
        unsafe {
            self.vk.run_qat_optimizer_batch(updates)
        }
    }

    pub fn dispatch_newton_schulz(
        &mut self,
        grad_buf: &Subbuffer<[f32]>,
        rows: usize,
        cols: usize,
        n_iters: usize,
    ) -> anyhow::Result<()> {
        let tmp_name = format!("ns_tmp_{}_{}", rows, cols);
        let next_x_name = format!("ns_nextx_{}_{}", rows, cols);
        
        let tmp_buf = self.get_or_create_grad(&tmp_name, cols * cols);
        let next_x_buf = self.get_or_create_grad(&next_x_name, rows * cols);

        unsafe {
            self.vk.run_newton_schulz_async(rows, cols, n_iters, grad_buf, &tmp_buf, &next_x_buf)
        }
    }

    pub fn dispatch_telemetry(
        &mut self,
        regs_buf: &Subbuffer<[f32]>,
        out_buf: &Subbuffer<[f32]>,
        elements: usize,
    ) -> anyhow::Result<()> {
        unsafe {
            self.vk.run_telemetry_async(elements, regs_buf, out_buf)
        }
    }
}
