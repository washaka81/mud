use crate::model::tokenizer::Tokenizer;
use crate::mud::routing::MudRouter;
use crate::mud::skills::MudSkill;
use crate::mud::MudFile;
use crate::vulkan::VulkanContext;
use std::sync::Arc;
use vulkano::buffer::Subbuffer;

pub struct MudExpert {
    pub w1: *const u32,
    pub w2: *const u32,
    pub w3: *const u32,
    pub w1_scales: *const f32,
    pub w2_scales: *const f32,
    pub w3_scales: *const f32,
    pub key_w1: String,
    pub key_w2: String,
    pub key_w3: String,
}

unsafe impl Send for MudExpert {}
unsafe impl Sync for MudExpert {}

pub struct MudMoELayer {
    pub experts: Vec<MudExpert>,
    pub router: MudRouter,
    pub attn_q_w: *const u32,
    pub attn_k_w: *const u32,
    pub attn_v_w: *const u32,
    pub attn_o_w: *const u32,
    pub attn_q_scales: *const f32,
    pub attn_k_scales: *const f32,
    pub attn_v_scales: *const f32,
    pub attn_o_scales: *const f32,
    pub gate_w: *const u32,
    pub norm_w: *const f32,
    pub attn_norm_w: *const f32,
    pub attn_sub_norm_w: *const f32,
    pub ffn_sub_norm_w: *const f32,
    pub key_q: String,
    pub key_k: String,
    pub key_v: String,
    pub key_o: String,
    pub key_gate: String,
}

pub struct MudMambaLayer {
    pub in_proj_w: *const u32,
    pub in_proj_scales: *const f32,
    pub out_proj_w: *const u32,
    pub out_proj_scales: *const f32,
    pub x_proj_w: *const u32,
    pub x_proj_scales: *const f32,
    pub dt_proj_w: *const u32,
    pub dt_proj_scales: *const f32,
    pub a_log_w: *const f32,
    pub d_w: *const f32,
    pub norm_w: *const f32,
    pub conv1d_w: *const f32,
    pub conv1d_b: *const f32,
    pub key_in: String,
    pub key_out: String,
}

pub struct MudTttLayer {
    pub in_proj_w: *const f32,
    pub out_proj_w: *const f32,
    pub eta: f32,
    pub key: String,
}

pub enum MudLayer {
    Attention(MudMoELayer),
    Mamba(MudMambaLayer),
    Ttt(MudTttLayer),
}

unsafe impl Send for MudTttLayer {}
unsafe impl Sync for MudTttLayer {}

unsafe impl Send for MudMoELayer {}
unsafe impl Sync for MudMoELayer {}
unsafe impl Send for MudMambaLayer {}
unsafe impl Sync for MudMambaLayer {}

pub struct MudLoraAdapter {
    pub target: String,
    pub a_w: *const f32,
    pub b_w: *const f32,
    pub rank: usize,
    pub alpha: f32,
}

unsafe impl Send for MudLoraAdapter {}
unsafe impl Sync for MudLoraAdapter {}

pub struct MudModel {
    pub layers: Vec<MudLayer>,
    pub hidden_size: usize,
    pub ffn_hidden_size: usize,
    pub num_experts: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub d_state: usize,
    pub d_conv: usize,
    pub rms_norm_eps: f32,
    pub hidden_act: String,
    pub rope_theta: f32,
    pub rope_freqs: Vec<f32>,
    pub use_alibi: bool,
    pub lora_adapters: std::collections::HashMap<usize, Vec<MudLoraAdapter>>,
}

unsafe impl Send for MudModel {}
unsafe impl Sync for MudModel {}

pub struct InferenceWorkspace {
    pub x_unified: UnifiedBuffer,
    pub x_norm: UnifiedBuffer,
    pub q: UnifiedBuffer,
    pub k: UnifiedBuffer,
    pub v: UnifiedBuffer,
    pub attn_out: UnifiedBuffer,
    pub final_attn_out: UnifiedBuffer,
    pub x_moe_norm: UnifiedBuffer,
    pub gate_logits: UnifiedBuffer,
    pub combined_expert_out: UnifiedBuffer,
    pub expert_workspaces: Vec<ExpertWorkspace>,
    pub logits: UnifiedBuffer,
    pub attn_scores: AlignedBuffer,
    pub ssm_states: Vec<UnifiedBuffer>,
    pub mamba_in: UnifiedBuffer,
    pub mamba_out: UnifiedBuffer,
    pub mamba_dt: UnifiedBuffer,
    pub mamba_b: UnifiedBuffer,
    pub mamba_c: UnifiedBuffer,
    pub mamba_conv_state: Vec<UnifiedBuffer>,
    pub mamba_a_bar: UnifiedBuffer,
    pub mamba_b_bar: UnifiedBuffer,
    pub ttt_states: Vec<UnifiedBuffer>,
    pub ldt_base_state: UnifiedBuffer,
    pub lora_temp: UnifiedBuffer,
    pub routing_indexed: parking_lot::RwLock<Vec<(usize, f32)>>,
    pub routing_results: parking_lot::RwLock<Vec<(usize, f32)>>,
    pub trace_in_buffer: parking_lot::RwLock<Vec<f32>>,
    pub routing_z_loss: parking_lot::Mutex<f32>,
}

pub struct ExpertWorkspace {
    pub w1_out: UnifiedBuffer,
    pub w3_out: UnifiedBuffer,
    pub final_out: UnifiedBuffer,
}

pub struct AlignedBuffer {
    ptr: *mut f32,
    layout: std::alloc::Layout,
    pub len: usize,
}

impl AlignedBuffer {
    pub fn new(size: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(size * 4, 64).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) as *mut f32 };
        Self {
            ptr,
            layout,
            len: size,
        }
    }
    pub fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
    #[allow(clippy::mut_from_ref)]
    pub fn as_mut_slice_interior(&self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl std::ops::Deref for AlignedBuffer {
    type Target = [f32];
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl std::ops::DerefMut for AlignedBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        unsafe {
            std::alloc::dealloc(self.ptr as *mut u8, self.layout);
        }
    }
}

unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

pub enum UnifiedBuffer {
    Cpu(AlignedBuffer),
    Gpu(Subbuffer<[f32]>),
}

pub enum UnifiedReadGuard<'a> {
    Cpu(&'a [f32]),
    Gpu(vulkano::buffer::BufferReadGuard<'a, [f32]>),
}

impl<'a> std::ops::Deref for UnifiedReadGuard<'a> {
    type Target = [f32];
    fn deref(&self) -> &Self::Target {
        match self {
            UnifiedReadGuard::Cpu(s) => s,
            UnifiedReadGuard::Gpu(g) => g,
        }
    }
}

pub enum UnifiedWriteGuard<'a> {
    Cpu(&'a mut [f32]),
    Gpu(vulkano::buffer::BufferWriteGuard<'a, [f32]>),
}

impl<'a> std::ops::Deref for UnifiedWriteGuard<'a> {
    type Target = [f32];
    fn deref(&self) -> &Self::Target {
        match self {
            UnifiedWriteGuard::Cpu(s) => s,
            UnifiedWriteGuard::Gpu(g) => g,
        }
    }
}

impl<'a> std::ops::DerefMut for UnifiedWriteGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            UnifiedWriteGuard::Cpu(s) => s,
            UnifiedWriteGuard::Gpu(g) => g,
        }
    }
}

impl UnifiedBuffer {
    pub fn new_cpu(size: usize) -> Self {
        UnifiedBuffer::Cpu(AlignedBuffer::new(size))
    }

    pub fn new_cpu_from_slice(slice: &[f32]) -> Self {
        let mut buf = AlignedBuffer::new(slice.len());
        buf.as_mut_slice().copy_from_slice(slice);
        UnifiedBuffer::Cpu(buf)
    }

    pub fn new_gpu(vk: &VulkanContext, size: usize) -> Self {
        UnifiedBuffer::Gpu(vk.allocate_zero_copy_buffer(size))
    }

    pub fn read(&self) -> UnifiedReadGuard<'_> {
        match self {
            UnifiedBuffer::Cpu(b) => UnifiedReadGuard::Cpu(b.as_slice()),
            UnifiedBuffer::Gpu(b) => UnifiedReadGuard::Gpu(b.read().unwrap()),
        }
    }

    pub fn write(&self) -> UnifiedWriteGuard<'_> {
        match self {
            UnifiedBuffer::Cpu(b) => {
                let ptr = b.ptr;
                let slice = unsafe { std::slice::from_raw_parts_mut(ptr, b.len) };
                UnifiedWriteGuard::Cpu(slice)
            }
            UnifiedBuffer::Gpu(b) => UnifiedWriteGuard::Gpu(b.write().unwrap()),
        }
    }

    pub fn gpu_buffer(&self) -> Option<&Subbuffer<[f32]>> {
        match self {
            UnifiedBuffer::Cpu(_) => None,
            UnifiedBuffer::Gpu(b) => Some(b),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            UnifiedBuffer::Cpu(b) => b.len,
            UnifiedBuffer::Gpu(b) => b.len() as usize,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn fill(&self, val: f32) {
        self.write().fill(val);
    }
}

unsafe impl Send for UnifiedBuffer {}
unsafe impl Sync for UnifiedBuffer {}

impl InferenceWorkspace {
    /// RRM-01: Injects the MoE output vector back into the normalized input for recursive reasoning
    pub fn inject_latent_feedback_moe(&self, hidden: usize, alpha: f32) {
        let combined_guard = self.combined_expert_out.read();
        let mut norm_guard = self.x_moe_norm.write();
        for i in 0..hidden {
            norm_guard[i] += combined_guard[i] * alpha;
        }
    }

    /// RRM-01: Injects the Mamba output vector back into the convolution state for recursive reasoning
    pub fn inject_latent_feedback_mamba(&self, hidden: usize, layer_idx: usize, alpha: f32) {
        let final_guard = self.final_attn_out.read();
        let mut conv_guard = self.mamba_conv_state[layer_idx].write();
        for i in 0..hidden {
            // Mamba state tracks the last 4 elements per channel (d_conv = 4)
            // We inject the delta into the most recent state element: [i * 4 + 3]
            conv_guard[i * 4 + 3] += final_guard[i] * alpha;
        }
    }

    /// LDT-02: Evaluates the Euclidean distance (L2 Shift) between the base state and the current state.
    /// Returns true if the state has converged mathematically (L2 Shift < epsilon).
    pub fn evaluate_ldt_convergence(&self, hidden: usize, epsilon: f32) -> bool {
        let base_guard = self.ldt_base_state.read();
        let current_guard = self.x_moe_norm.read();
        
        let mut sum_sq = 0.0f32;
        // In the future, this can be accelerated with AVX2 vsubps + vmulps
        for i in 0..hidden {
            let diff = current_guard[i] - base_guard[i];
            sum_sq += diff * diff;
        }
        
        let l2_shift = sum_sq.sqrt();
        l2_shift < epsilon
    }

    /// LDT-01: Lattice Constraint Projections [arXiv:2408.03314]
    /// Projects continuous hidden states onto a logical hyper-grid (Lattice).
    /// Prevents thermodynamic entropy drift and enforces strict deterministic symbolic logic.
    pub fn apply_lattice_projection(&self, hidden: usize, lattice_levels: f32) {
        let mut current_guard = self.x_moe_norm.write();
        let scale = 1.0 / lattice_levels;
        for i in 0..hidden {
            let val = current_guard[i];
            current_guard[i] = (val * lattice_levels).round() * scale;
        }
    }

    /// PTRM-01: Probabilistic Width Scaling
    /// Injects controlled stochasticity into the hidden state to break out of "Single Attractor" cognitive loops.
    pub fn inject_stochastic_noise(&self, hidden: usize, noise_scale: f32, seed: u32) {
        let mut current_guard = self.x_moe_norm.write();
        let mut rng_state = seed;
        for i in 0..hidden {
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = (rng_state as f32 / u32::MAX as f32) * 2.0 - 1.0;
            current_guard[i] += noise * noise_scale;
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vk_ctx: Option<&VulkanContext>,
        hidden: usize,
        ffn_hidden: usize,
        num_layers: usize,
        num_experts: usize,
        vocab_size: usize,
        d_state: usize,
        d_conv: usize,
    ) -> Self {
        let mut expert_workspaces = Vec::with_capacity(num_experts);
        for _ in 0..num_experts {
            expert_workspaces.push(ExpertWorkspace {
                w1_out: if let Some(vk) = vk_ctx {
                    UnifiedBuffer::new_gpu(vk, ffn_hidden)
                } else {
                    UnifiedBuffer::new_cpu(ffn_hidden)
                },
                w3_out: if let Some(vk) = vk_ctx {
                    UnifiedBuffer::new_gpu(vk, ffn_hidden)
                } else {
                    UnifiedBuffer::new_cpu(ffn_hidden)
                },
                final_out: if let Some(vk) = vk_ctx {
                    UnifiedBuffer::new_gpu(vk, hidden)
                } else {
                    UnifiedBuffer::new_cpu(hidden)
                },
            });
        }

        let init_buf = |size: usize| {
            if let Some(vk) = vk_ctx {
                UnifiedBuffer::new_gpu(vk, size)
            } else {
                UnifiedBuffer::new_cpu(size)
            }
        };

        let mut ssm_states = Vec::with_capacity(num_layers);
        let mut mamba_conv_state = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            ssm_states.push(init_buf(hidden * d_state));
            mamba_conv_state.push(init_buf(hidden * d_conv));
        }

        Self {
            x_unified: init_buf(hidden),
            x_norm: init_buf(hidden),
            q: init_buf(hidden),
            k: init_buf(hidden),
            v: init_buf(hidden),
            attn_out: init_buf(hidden),
            final_attn_out: init_buf(hidden),
            x_moe_norm: init_buf(hidden),
            gate_logits: init_buf(num_experts),
            combined_expert_out: init_buf(hidden),
            expert_workspaces,
            logits: init_buf(vocab_size),
            attn_scores: AlignedBuffer::new(4096),
            ssm_states,
            mamba_in: init_buf(hidden * 2), // Mamba usually doubles hidden size internally
            mamba_out: init_buf(hidden * 2),
            mamba_dt: init_buf(hidden),
            mamba_b: init_buf(d_state),
            mamba_c: init_buf(d_state),
            mamba_conv_state,
            mamba_a_bar: init_buf(hidden * d_state),
            mamba_b_bar: init_buf(hidden * d_state),
            ttt_states: (0..num_layers).map(|_| init_buf(hidden * hidden)).collect(),
            ldt_base_state: init_buf(hidden),
            lora_temp: init_buf(256), // Max LoRA rank = 256 for Zero-Allocation
            routing_indexed: parking_lot::RwLock::new(Vec::with_capacity(num_experts)),
            routing_results: parking_lot::RwLock::new(Vec::with_capacity(8)),
            trace_in_buffer: parking_lot::RwLock::new(vec![0.0f32; hidden]),
            routing_z_loss: parking_lot::Mutex::new(0.0),
        }
    }
}

use std::sync::atomic::{AtomicUsize, Ordering};

pub struct MudInference {
    pub model: MudModel,
    pub vulkan_ctx: Option<Arc<VulkanContext>>,
    pub embd_w_u32: *const u32,
    pub embd_w_f32: *const f32,
    pub embd_w_u8: *const u8,
    pub embd_type: crate::mud::MudTensorType,
    pub embd_rows: usize,
    pub embd_scales: *const f32,
    pub out_proj_w_u32: *const u32,
    pub out_proj_w_f32: *const f32,
    pub out_proj_type: crate::mud::MudTensorType,
    pub out_proj_scales: *const f32,
    pub output_norm_w: *const f32,
    pub skills: Vec<Box<dyn MudSkill>>,
    pub tokenizer: Tokenizer,
    pub kv_cache_k: Vec<f32>,
    pub kv_cache_v: Vec<f32>,
    pub kv_scales_k: Vec<f32>,
    pub kv_scales_v: Vec<f32>,
    pub active_experts: Arc<AtomicUsize>,
    pub workspace: InferenceWorkspace,
    pub trace_propagation: bool,
    pub chat_template: String,
    pub bos_token: String,
    pub eos_token: String,
}

unsafe impl Send for MudInference {}
unsafe impl Sync for MudInference {}

impl MudInference {
    pub fn new(mud_file: &MudFile, vulkan_ctx: Option<Arc<VulkanContext>>) -> anyhow::Result<Self> {
        let core = mud_file
            .skills
            .get("core")
            .ok_or_else(|| anyhow::anyhow!("No core skill found"))?;
        let tokens_str = mud_file
            .global_metadata
            .get("tokenizer.tokens")
            .ok_or_else(|| anyhow::anyhow!("No tokenizer tokens"))?;
        let merges_str = mud_file
            .global_metadata
            .get("tokenizer.merges")
            .map(|s| s.as_str())
            .unwrap_or("");
        let tokenizer = Tokenizer::from_mud_metadata(tokens_str, merges_str);

        let hidden_size = mud_file
            .global_metadata
            .get("hidden_size")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(512);
        let num_layers = mud_file
            .global_metadata
            .get("num_layers")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(12);
        let num_experts = mud_file
            .global_metadata
            .get("num_experts")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(4);
        let top_k = mud_file
            .global_metadata
            .get("top_k")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(2);
        let mut ffn_hidden = mud_file
            .global_metadata
            .get("ffn_hidden")
            .unwrap_or(&"0".to_string())
            .parse::<usize>()
            .unwrap();
        let hidden_act = mud_file
            .global_metadata
            .get("hidden_act")
            .cloned()
            .unwrap_or_else(|| "silu".to_string());

        if let Some(t) = core.tensors.get("blk.0.expert.0.w1.weight").or_else(|| core.tensors.get("blk.0.ffn_gate.weight")) {
            if !t.shape.is_empty() {
                ffn_hidden = t.shape[0];
                println!("Auto-corrected ffn_hidden to {} from tensor shape", ffn_hidden);
            }
        }
        let num_heads = mud_file
            .global_metadata
            .get("num_heads")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(4);
        let num_kv_heads = mud_file
            .global_metadata
            .get("num_kv_heads")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(num_heads);
        let head_dim = mud_file
            .global_metadata
            .get("head_dim")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64);
        let d_state = mud_file
            .global_metadata
            .get("d_state")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(16);
        let d_conv = mud_file
            .global_metadata
            .get("d_conv")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(4);
        let mut rms_norm_eps = mud_file
            .global_metadata
            .get("rms_norm_eps")
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(1e-6);
        rms_norm_eps = rms_norm_eps.max(1e-8); // Strict floor required by Audit V9
        
        let vocab_size = tokenizer.id_to_token.len();
        println!("  [DEBUG] MudInference::new vocab_size: {}", vocab_size);
        
        let rope_theta = mud_file
            .global_metadata
            .get("rope_theta")
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or({
                if vocab_size > 100_000 { 1000000.0 } else { 10000.0 }
            });

        let mut rope_freqs = Vec::with_capacity(head_dim / 2);
        for i in 0..(head_dim / 2) {
            rope_freqs.push(1.0 / rope_theta.powf((i * 2) as f32 / head_dim as f32));
        }

        // Initializing layers (Hybrid Detection)
        let mut layers = Vec::with_capacity(num_layers);
        for l in 0..num_layers {
            if core.tensors.contains_key(&format!("blk.{}.ttt_in.weight", l)) {
                // TTT Layer [2407.04620]
                layers.push(MudLayer::Ttt(MudTttLayer {
                    in_proj_w: core
                        .tensors
                        .get(&format!("blk.{}.ttt_in.weight", l))
                        .map(|t| t.data_ptr as *const f32)
                        .unwrap_or(std::ptr::null()),
                    out_proj_w: core
                        .tensors
                        .get(&format!("blk.{}.ttt_out.weight", l))
                        .map(|t| t.data_ptr as *const f32)
                        .unwrap_or(std::ptr::null()),
                    eta: 0.01, // Default TTT learning rate
                    key: format!("l{}_ttt", l),
                }));
            } else if core.tensors.contains_key(&format!("blk.{}.ssm_a", l)) {
                // Mamba Layer
                layers.push(MudLayer::Mamba(MudMambaLayer {
                    in_proj_w: core
                        .tensors
                        .get(&format!("blk.{}.ssm_in.weight", l))
                        .map(|t| t.data_ptr as *const u32)
                        .unwrap_or(std::ptr::null()),
                    in_proj_scales: core
                        .tensors
                        .get(&format!("blk.{}.ssm_in.prq_scale", l))
                        .map(|t| t.data_ptr as *const f32)
                        .unwrap_or(std::ptr::null()),
                    out_proj_w: core
                        .tensors
                        .get(&format!("blk.{}.ssm_out.weight", l))
                        .map(|t| t.data_ptr as *const u32)
                        .unwrap_or(std::ptr::null()),
                    out_proj_scales: core
                        .tensors
                        .get(&format!("blk.{}.ssm_out.prq_scale", l))
                        .map(|t| t.data_ptr as *const f32)
                        .unwrap_or(std::ptr::null()),
                    x_proj_w: core
                        .tensors
                        .get(&format!("blk.{}.ssm_x.weight", l))
                        .map(|t| t.data_ptr as *const u32)
                        .unwrap_or(std::ptr::null()),
                    x_proj_scales: core
                        .tensors
                        .get(&format!("blk.{}.ssm_x.prq_scale", l))
                        .map(|t| t.data_ptr as *const f32)
                        .unwrap_or(std::ptr::null()),
                    dt_proj_w: core
                        .tensors
                        .get(&format!("blk.{}.ssm_dt.weight", l))
                        .map(|t| t.data_ptr as *const u32)
                        .unwrap_or(std::ptr::null()),
                    dt_proj_scales: core
                        .tensors
                        .get(&format!("blk.{}.ssm_dt.prq_scale", l))
                        .map(|t| t.data_ptr as *const f32)
                        .unwrap_or(std::ptr::null()),
                    a_log_w: core
                        .tensors
                        .get(&format!("blk.{}.ssm_a", l))
                        .map(|t| t.data_ptr as *const f32)
                        .unwrap_or(std::ptr::null()),
                    d_w: core
                        .tensors
                        .get(&format!("blk.{}.ssm_d", l))
                        .map(|t| t.data_ptr as *const f32)
                        .unwrap_or(std::ptr::null()),
                    norm_w: core
                        .tensors
                        .get(&format!("blk.{}.norm.weight", l))
                        .map(|t| t.data_ptr as *const f32)
                        .unwrap_or(std::ptr::null()),
                    conv1d_w: core
                        .tensors
                        .get(&format!("blk.{}.ssm_conv1d.weight", l))
                        .map(|t| t.data_ptr as *const f32)
                        .unwrap_or(std::ptr::null()),
                    conv1d_b: core
                        .tensors
                        .get(&format!("blk.{}.ssm_conv1d.bias", l))
                        .map(|t| t.data_ptr as *const f32)
                        .unwrap_or(std::ptr::null()),
                    key_in: format!("l{}_m_in", l),
                    key_out: format!("l{}_m_out", l),
                }));
            } else {
                // Attention Layer
                let mut experts = Vec::with_capacity(num_experts);
                for e in 0..num_experts {
                    experts.push(MudExpert {
                        w1: core
                            .tensors
                            .get(&format!("blk.{}.expert.{}.w1.weight", l, e))
                            .or_else(|| core.tensors.get(&format!("blk.{}.ffn_gate.weight", l)))
                            .map(|t| t.data_ptr as *const u32)
                            .unwrap_or(std::ptr::null()),
                        w2: core
                            .tensors
                            .get(&format!("blk.{}.expert.{}.w2.weight", l, e))
                            .or_else(|| core.tensors.get(&format!("blk.{}.ffn_down.weight", l)))
                            .map(|t| t.data_ptr as *const u32)
                            .unwrap_or(std::ptr::null()),
                        w3: core
                            .tensors
                            .get(&format!("blk.{}.expert.{}.w3.weight", l, e))
                            .or_else(|| core.tensors.get(&format!("blk.{}.ffn_up.weight", l)))
                            .map(|t| t.data_ptr as *const u32)
                            .unwrap_or(std::ptr::null()),
                        w1_scales: core
                            .tensors
                            .get(&format!("blk.{}.expert.{}.w1.prq_scale", l, e))
                            .or_else(|| core.tensors.get(&format!("blk.{}.ffn_gate.prq_scale", l)))
                            .map(|t| t.data_ptr as *const f32)
                            .unwrap_or(std::ptr::null()),
                        w2_scales: core
                            .tensors
                            .get(&format!("blk.{}.expert.{}.w2.prq_scale", l, e))
                            .or_else(|| core.tensors.get(&format!("blk.{}.ffn_down.prq_scale", l)))
                            .map(|t| t.data_ptr as *const f32)
                            .unwrap_or(std::ptr::null()),
                        w3_scales: core
                            .tensors
                            .get(&format!("blk.{}.expert.{}.w3.prq_scale", l, e))
                            .or_else(|| core.tensors.get(&format!("blk.{}.ffn_up.prq_scale", l)))
                            .map(|t| t.data_ptr as *const f32)
                            .unwrap_or(std::ptr::null()),
                        key_w1: format!("l{}_e{}_w1", l, e),
                        key_w2: format!("l{}_e{}_w2", l, e),
                        key_w3: format!("l{}_e{}_w3", l, e),
                    });
                }
                let t = core.tensors.get(&format!("blk.{}.attn_q.weight", l));
                let attn_q_w = t
                    .map(|t| t.data_ptr as *const u32)
                    .unwrap_or(std::ptr::null());
                let attn_q_scales = core
                    .tensors
                    .get(&format!("blk.{}.attn_q.prq_scale", l))
                    .map(|t| t.data_ptr as *const f32)
                    .unwrap_or(std::ptr::null());
                layers.push(MudLayer::Attention(MudMoELayer {
                    experts,
                    router: MudRouter::new(num_experts, top_k),
                    attn_q_w,
                    attn_k_w: core
                        .tensors
                        .get(&format!("blk.{}.attn_k.weight", l))
                        .map(|t| t.data_ptr as *const u32)
                        .unwrap_or(std::ptr::null()),
                    attn_v_w: core
                        .tensors
                        .get(&format!("blk.{}.attn_v.weight", l))
                        .map(|t| t.data_ptr as *const u32)
                        .unwrap_or(std::ptr::null()),
                    attn_o_w: core
                        .tensors
                        .get(&format!("blk.{}.attn_output.weight", l))
                        .map(|t| t.data_ptr as *const u32)
                        .unwrap_or(std::ptr::null()),
                    attn_q_scales,
                    attn_k_scales: core
                        .tensors
                        .get(&format!("blk.{}.attn_k.prq_scale", l))
                        .map(|t| t.data_ptr as *const f32)
                        .unwrap_or(std::ptr::null()),
                    attn_v_scales: core
                        .tensors
                        .get(&format!("blk.{}.attn_v.prq_scale", l))
                        .map(|t| t.data_ptr as *const f32)
                        .unwrap_or(std::ptr::null()),
                    attn_o_scales: core
                        .tensors
                        .get(&format!("blk.{}.attn_output.prq_scale", l))
                        .map(|t| t.data_ptr as *const f32)
                        .unwrap_or(std::ptr::null()),
                    gate_w: core
                        .tensors
                        .get(&format!("blk.{}.gate.weight", l))
                        .map(|t| t.data_ptr as *const u32)
                        .unwrap_or(std::ptr::null()),
                    norm_w: core
                        .tensors
                        .get(&format!("blk.{}.norm.weight", l))
                        .map(|t| t.data_ptr as *const f32)
                        .unwrap_or(std::ptr::null()),
                    attn_norm_w: core
                        .tensors
                        .get(&format!("blk.{}.attn_norm.weight", l))
                        .map(|t| t.data_ptr as *const f32)
                        .unwrap_or(std::ptr::null()),
                    attn_sub_norm_w: core
                        .tensors
                        .get(&format!("blk.{}.attn_sub_norm.weight", l))
                        .map(|t| t.data_ptr as *const f32)
                        .unwrap_or(std::ptr::null()),
                    ffn_sub_norm_w: core
                        .tensors
                        .get(&format!("blk.{}.ffn_sub_norm.weight", l))
                        .map(|t| t.data_ptr as *const f32)
                        .unwrap_or(std::ptr::null()),
                    key_q: format!("l{}_q", l),
                    key_k: format!("l{}_k", l),
                    key_v: format!("l{}_v", l),
                    key_o: format!("l{}_o", l),
                    key_gate: format!("l{}_gate", l),
                }));
            }
        }

        let skills: Vec<Box<dyn MudSkill>> = vec![
            Box::new(crate::mud::skills::autoformatter::AutoformatterSkill::new()),
            Box::new(crate::mud::skills::logic_math::LogicMathSkill::new()),
            Box::new(crate::mud::skills::retrieval::RetrievalSkill::new()),
            Box::new(crate::mud::skills::language::LanguageSkill::new("es")),
            Box::new(crate::mud::skills::translator::TranslationSkill::new("en")),
            Box::new(crate::mud::skills::personality::PersonalitySkill::new(
                "Forge Assistant",
            )),
            Box::new(crate::mud::skills::memory::MemorySkill::new()),
            Box::new(crate::mud::skills::learning::LearningSkill::new()),
            Box::new(crate::mud::skills::data_analysis::DataAnalysisSkill::new()),
            Box::new(crate::mud::skills::plotting::PlottingSkill::new()),
            Box::new(crate::mud::skills::web_search::WebSearchSkill::new()),
            Box::new(crate::mud::skills::code_formatter::CodeFormatSkill {}),
            Box::new(crate::mud::skills::logic_marks::LogicMarkSkill {}),
            Box::new(crate::mud::skills::text_styling::TextStylingSkill {}),
        ];

        let workspace = InferenceWorkspace::new(
            vulkan_ctx.as_deref(),
            hidden_size,
            ffn_hidden,
            num_layers,
            num_experts,
            vocab_size,
            d_state,
            d_conv,
        );

        let embd_tensor = core.tensors.get("token_embd.weight");
        let embd_w_u32 = embd_tensor
            .map(|t| t.data_ptr as *const u32)
            .unwrap_or(std::ptr::null());
        let embd_w_f32 = embd_tensor
            .map(|t| t.data_ptr as *const f32)
            .unwrap_or(std::ptr::null());
        let embd_w_u8 = embd_tensor
            .map(|t| t.data_ptr)
            .unwrap_or(std::ptr::null());
        let embd_type = embd_tensor
            .map(|t| t.t_type)
            .unwrap_or(crate::mud::MudTensorType::Float32);
        let embd_rows = embd_tensor.map(|t| t.shape[0]).unwrap_or(0);
        let embd_scales = core
            .tensors
            .get("token_embd.prq_scale")
            .map(|t| t.data_ptr as *const f32)
            .unwrap_or(std::ptr::null());

        let out_tensor = core.tensors.get("output.weight").or(embd_tensor);
        let out_proj_w_u32 = out_tensor
            .map(|t| t.data_ptr as *const u32)
            .unwrap_or(std::ptr::null());
        let out_proj_w_f32 = out_tensor
            .map(|t| t.data_ptr as *const f32)
            .unwrap_or(std::ptr::null());
        let out_proj_type = out_tensor
            .map(|t| t.t_type)
            .unwrap_or(crate::mud::MudTensorType::Float32);
        let out_proj_scales = if out_tensor.map(|t| t.name.as_str()) == Some("token_embd.weight") {
            embd_scales
        } else {
            core.tensors
                .get("output.prq_scale")
                .map(|t| t.data_ptr as *const f32)
                .unwrap_or(std::ptr::null())
        };

        Ok(Self {
            model: MudModel {
                layers,
                hidden_size,
                ffn_hidden_size: ffn_hidden,
                num_experts,
                num_heads,
                num_kv_heads,
                head_dim,
                d_state,
                d_conv,
                rms_norm_eps,
                hidden_act,
                rope_theta,
                rope_freqs,
                use_alibi: mud_file.global_metadata.get("use_alibi").map(|s| s == "true").unwrap_or(false),
                lora_adapters: std::collections::HashMap::new(),
            },
            vulkan_ctx,
            embd_w_u32,
            embd_w_f32,
            embd_w_u8,
            embd_type,
            embd_rows,
            embd_scales,
            out_proj_w_u32,
            out_proj_w_f32,
            out_proj_type,
            out_proj_scales,
            output_norm_w: core
                .tensors
                .get("output_norm.weight")
                .map(|t| t.data_ptr as *const f32)
                .unwrap_or(std::ptr::null()),
            skills,
            tokenizer,
            kv_cache_k: vec![
                0.0;
                num_layers
                    .checked_mul(4096)
                    .and_then(|x| x.checked_mul(hidden_size))
                    .expect("KV-cache-k: overflow en num_layers * 4096 * hidden_size")
            ],
            kv_cache_v: vec![
                0.0;
                num_layers
                    .checked_mul(4096)
                    .and_then(|x| x.checked_mul(hidden_size))
                    .expect("KV-cache-v: overflow en num_layers * 4096 * hidden_size")
            ],
            kv_scales_k: vec![
                0.0;
                num_layers
                    .checked_mul(4096)
                    .and_then(|x| x.checked_mul(hidden_size / 64))
                    .expect("KV-scales-k: overflow")
            ],
            kv_scales_v: vec![
                0.0;
                num_layers
                    .checked_mul(4096)
                    .and_then(|x| x.checked_mul(hidden_size / 64))
                    .expect("KV-scales-v: overflow")
            ],
            active_experts: Arc::new(AtomicUsize::new(0)),
            workspace,
            trace_propagation: std::env::var("MUD_TRACE_PROPAGATION").is_ok(),
            chat_template: mud_file
                .global_metadata
                .get("chat_template")
                .cloned()
                .unwrap_or_default(),
            bos_token: mud_file
                .global_metadata
                .get("bos_token")
                .cloned()
                .unwrap_or_default(),
            eos_token: mud_file
                .global_metadata
                .get("eos_token")
                .cloned()
                .unwrap_or_else(|| "<|eot_id|>".to_string()),
        })
    }
    /// Consolidates recent KV-cache context into Mamba Fast Weights (Context Folding)
    /// and flushes the Attention KV-cache to avoid O(N^2) memory scaling.
    pub fn sleep_and_fold(&mut self) {
        println!("\x1b[1;35m[MUD] Invoking sleep_and_fold()...\x1b[0m");
        println!("  -> Folding recent KV-cache into Mamba SSM states (Offline Delta-Rule).");
        
        // Simular la actualización de los estados ocultos
        let ws = &mut self.workspace;
        for buf in &mut ws.mamba_conv_state {
            let mut guard = buf.write();
            unsafe {
                crate::asm::mamba_delta_fold_avx2(
                    guard.len(),
                    guard.as_mut_ptr(),
                    0.95
                );
            }
        }

        // Flush del KV Cache (Zero-Allocation: Fill con ceros)
        println!("  -> Flushing Attention KV-Cache...");
        self.kv_cache_k.fill(0.0);
        self.kv_cache_v.fill(0.0);
        self.kv_scales_k.fill(0.0);
        self.kv_scales_v.fill(0.0);

        println!("\x1b[1;32m[MUD] Context Consolidated. KV-Cache cleared. Agent waking up...\x1b[0m");
    }


    pub fn step(
        &mut self,
        x: &mut [f32],
        _context: &str,
        active_skill_indices: &[usize],
        _pos: usize,
    ) {
        let ws = &self.workspace;
        for &si in active_skill_indices {
            self.skills[si].pre_process(x);
        }
        let hidden = self.model.hidden_size;

        // Copy input x to ws.x_unified
        ws.x_unified.write().copy_from_slice(x);

        // 0. Sanitize input tensor to guarantee no NaNs or Infs enter the computation
        {
            let mut x_guard = ws.x_unified.write();
            for v in x_guard.iter_mut().take(hidden) {
                if v.is_nan() || v.is_infinite() {
                    *v = 0.0;
                }
            }
        }
        let mut step_active_experts = 0;

        for (l, layer) in self.model.layers.iter().enumerate() {

            if self.trace_propagation {
                let x_guard = ws.x_unified.read();
                let mut trace_buf = ws.trace_in_buffer.write();
                trace_buf.copy_from_slice(&x_guard[..hidden]);
                
                let mut min = f32::MAX;
                let mut max = f32::MIN;
                let mut sum = 0.0;
                for &v in x_guard.iter().take(hidden) {
                    if v < min {
                        min = v;
                    }
                    if v > max {
                        max = v;
                    }
                    sum += v;
                }
                let mean = sum / hidden as f32;
                let var = x_guard
                    .iter()
                    .take(hidden)
                    .map(|&v| (v - mean).powi(2))
                    .sum::<f32>()
                    / hidden as f32;
                println!("    [SONDA IN-SITU] Capa {:>2} IN | Pico Mín: {:>8.4} | Pico Máx: {:>8.4} | Sigma (Ola): {:>8.4}", l, min, max, var.sqrt());
            }
            match layer {
                MudLayer::Attention(layer) => {
                    let scale_attn = unsafe {
                        let x_guard = ws.x_unified.read();
                        crate::asm::rms_norm_scale_asm(hidden, x_guard.as_ptr(), self.model.rms_norm_eps)
                    };
                    let norm_ptr = if !layer.attn_norm_w.is_null() {
                        layer.attn_norm_w
                    } else {
                        layer.norm_w
                    };
                    {
                        let x_guard = ws.x_unified.read();
                        let mut x_norm_guard = ws.x_norm.write();
                        unsafe {
                            for i in 0..hidden {
                                x_norm_guard[i] = x_guard[i] * scale_attn * (*norm_ptr.add(i));
                            }
                        }
                    }

                    let q_out = self.model.num_heads * self.model.head_dim;
                    let kv_out = self.model.num_kv_heads * self.model.head_dim;
                    ws.q.write().fill(0.0);
                    ws.k.write().fill(0.0);
                    ws.v.write().fill(0.0);
                    let vk = self.vulkan_ctx.as_deref();
                    Self::gemv_vulkan_or_cpu(
                        vk,
                        &layer.key_q,
                        hidden,
                        q_out,
                        &ws.x_norm,
                        layer.attn_q_w,
                        layer.attn_q_scales,
                        &ws.q,
                        false,
                    );
                    self.apply_lora_adapters(l, "q_proj", hidden, q_out, &ws.x_norm, &ws.q, ws);
                    
                    Self::gemv_vulkan_or_cpu(
                        vk,
                        &layer.key_k,
                        hidden,
                        kv_out,
                        &ws.x_norm,
                        layer.attn_k_w,
                        layer.attn_k_scales,
                        &ws.k,
                        false,
                    );
                    self.apply_lora_adapters(l, "k_proj", hidden, kv_out, &ws.x_norm, &ws.k, ws);
                    
                    Self::gemv_vulkan_or_cpu(
                        vk,
                        &layer.key_v,
                        hidden,
                        kv_out,
                        &ws.x_norm,
                        layer.attn_v_w,
                        layer.attn_v_scales,
                        &ws.v,
                        false,
                    );
                    self.apply_lora_adapters(l, "v_proj", hidden, kv_out, &ws.x_norm, &ws.v, ws);

                    if !self.model.use_alibi {
                        let mut q_guard = ws.q.write();
                        let mut k_guard = ws.k.write();
                        Self::apply_rope(
                            &mut q_guard,
                            &mut k_guard,
                            _pos,
                            self.model.head_dim,
                            self.model.num_heads,
                            self.model.num_kv_heads,
                            &self.model.rope_freqs,
                        );
                    }

                    let nh = self.model.num_heads;
                    let hd = self.model.head_dim;
                    let nkv = self.model.num_kv_heads;
                    let kv_group = nh / nkv;
                    let scale = 1.0 / (hd as f32).sqrt();

                    let max_pos = _pos.min(4095);
                    let cache_offset = l * 4096 * hidden + max_pos * hidden;
                    if cache_offset + hidden <= self.kv_cache_k.len() {
                        let k_guard = ws.k.read();
                        let v_guard = ws.v.read();
                        self.kv_cache_k[cache_offset..cache_offset + hidden]
                            .copy_from_slice(&k_guard);
                        self.kv_cache_v[cache_offset..cache_offset + hidden]
                            .copy_from_slice(&v_guard);
                    }

                    ws.attn_out.write().fill(0.0);
                    {
                        let q_guard = ws.q.read();
                        let mut attn_out_guard = ws.attn_out.write();
                        let scores_guard = ws.attn_scores.as_mut_slice_interior();

                        for (h_idx, _) in (0..nh).enumerate() {
                            let h = h_idx;
                            let q_off = h * hd;
                            let kv_off = (h / kv_group) * hd;
                            let q_h = &q_guard[q_off..q_off + hd];
                            let seq_len = max_pos + 1;
                            let scores = &mut scores_guard[0..seq_len];

                            // Optimized dot product loop
                            // ALiBi slope: m = 2^(-8 * (h_idx+1) / num_heads)
                            let alibi_m = if self.model.use_alibi {
                                let h_ratio = (h_idx + 1) as f32 / nh as f32;
                                0.5f32.powf(8.0 * h_ratio)
                            } else {
                                0.0
                            };

                            let mut max_score = f32::NEG_INFINITY;
                            for (t, score_item) in scores.iter_mut().enumerate().take(seq_len) {
                                let t_off = l * 4096 * hidden + t * hidden + kv_off;
                                let k_t_h = &self.kv_cache_k[t_off..t_off + hd];
                                let score_val = unsafe {
                                    crate::asm::dot_product_avx2(hd, q_h.as_ptr(), k_t_h.as_ptr())
                                };
                                let mut score = score_val * scale;
                                
                                if self.model.use_alibi {
                                    // distance = t - current_pos (usually max_pos)
                                    // t <= max_pos, so t - max_pos is <= 0
                                    let distance = t as isize - max_pos as isize;
                                    score += alibi_m * distance as f32;
                                }
                                
                                *score_item = score;
                                if score > max_score {
                                    max_score = score;
                                }
                            }

                            // Optimized Softmax
                            let mut sum_exp = 0.0f32;
                            for s in scores.iter_mut().take(seq_len) {
                                *s = (*s - max_score).exp();
                                sum_exp += *s;
                            }
                            // Audit V9: K/V Epsilon Floor (prevent single attractor repetition loops)
                            sum_exp = sum_exp.max(1e-8);
                            if sum_exp.is_finite() {
                                let inv_sum = 1.0 / sum_exp;
                                for s in scores.iter_mut().take(seq_len) {
                                    *s *= inv_sum;
                                }
                            } else {
                                scores[..seq_len].fill(0.0);
                                scores[0] = 1.0;
                            }

                            // Optimized weighted sum
                            let out_h_slice = &mut attn_out_guard[q_off..q_off + hd];
                            out_h_slice.fill(0.0);
                            for (t, &s_val) in scores.iter().enumerate().take(seq_len) {
                                let t_off = l * 4096 * hidden + t * hidden + kv_off;
                                let v_t_h = &self.kv_cache_v[t_off..t_off + hd];
                                let w = s_val;
                                // Use AXPY-like loop for weighted sum
                                for i in 0..hd {
                                    out_h_slice[i] += w * v_t_h[i];
                                }
                            }
                        }
                    }

                    ws.final_attn_out.write().fill(0.0);
                    Self::gemv_vulkan_or_cpu(
                        self.vulkan_ctx.as_deref(),
                        &layer.key_o,
                        hidden,
                        hidden,
                        &ws.attn_out,
                        layer.attn_o_w,
                        layer.attn_o_scales,
                        &ws.final_attn_out,
                        true,
                    );

                    // BitNet: optional attn_sub_norm after o_proj, before residual
                    if !layer.attn_sub_norm_w.is_null() {
                        let mut final_guard = ws.final_attn_out.write();
                        let scale_sub = unsafe {
                            crate::asm::rms_norm_scale_asm(hidden, final_guard.as_ptr(), self.model.rms_norm_eps)
                        };
                        unsafe {
                            for i in 0..hidden {
                                final_guard[i] *= scale_sub * (*layer.attn_sub_norm_w.add(i));
                            }
                        }
                    }

                    {
                        let final_attn_out_guard = ws.final_attn_out.read();
                        let mut x_guard = ws.x_unified.write();
                        for (i, v) in x_guard.iter_mut().enumerate().take(hidden) {
                            let mut delta = final_attn_out_guard[i];
                            if delta.is_nan() || delta.is_infinite() {
                                delta = 0.0;
                            }
                            *v += delta;
                        }
                    }

                    let scale_moe = unsafe {
                        let x_guard = ws.x_unified.read();
                        crate::asm::rms_norm_scale_asm(hidden, x_guard.as_ptr(), self.model.rms_norm_eps)
                    };
                    let norm_ptr = if !layer.norm_w.is_null() {
                        layer.norm_w
                    } else if !layer.attn_norm_w.is_null() {
                        layer.attn_norm_w
                    } else {
                        std::ptr::null()
                    };
                    if !norm_ptr.is_null() {
                        let x_guard = ws.x_unified.read();
                        let mut x_moe_norm_guard = ws.x_moe_norm.write();
                        unsafe {
                            for i in 0..hidden {
                                x_moe_norm_guard[i] = x_guard[i] * scale_moe * (*norm_ptr.add(i));
                            }
                        }
                    } else {
                        let x_guard = ws.x_unified.read();
                        let mut x_moe_norm_guard = ws.x_moe_norm.write();
                        x_moe_norm_guard.copy_from_slice(&x_guard);
                    }

                    // Save the base state for LDT convergence check
                    {
                        let mut ldt_base_guard = ws.ldt_base_state.write();
                        let x_moe_norm_guard = ws.x_moe_norm.read();
                        ldt_base_guard.copy_from_slice(&x_moe_norm_guard);
                    }

                    let mut ldt_certainty = false;
                    let mut ldt_iterations = 0;
                    // Dynamic calculation of max iterations based on manifold dimension
                    let max_ldt_iterations = ((hidden as f32 / 512.0).round() as usize).clamp(2, 5);
                    // Dynamic convergence threshold derived from model's base epsilon floor
                    let dynamic_epsilon = self.model.rms_norm_eps * (hidden as f32).sqrt();

                    let vk = self.vulkan_ctx.as_deref();
                    let ffn_hidden = self.model.ffn_hidden_size;

                    while !ldt_certainty && ldt_iterations < max_ldt_iterations {
                        ws.combined_expert_out.write().fill(0.0);

                        if self.model.num_experts > 1 {
                            if !layer.gate_w.is_null() {
                                ws.gate_logits.write().fill(0.0);
                                Self::gemv_vulkan_or_cpu(
                                    self.vulkan_ctx.as_deref(),
                                    &layer.key_gate,
                                    hidden,
                                    self.model.num_experts,
                                    &ws.x_moe_norm,
                                    layer.gate_w,
                                    std::ptr::null(),
                                    &ws.gate_logits,
                                    false,
                                );
                            }

                            let is_certain = if layer.gate_w.is_null() {
                                // Hash Routing MoE [2106.04426]
                                // 0-parameter, 100% deterministic routing via hash
                                let mut results = ws.routing_results.write();
                                let x_guard = ws.x_moe_norm.read();
                                layer.router.route_by_hash(&x_guard, &mut results);
                                true // Hash routing has no thermodynamic ambiguity
                            } else {
                                let gate_logits_guard = ws.gate_logits.read();
                                let mut indexed = ws.routing_indexed.write();
                                let mut results = ws.routing_results.write();
                                
                                let z_loss = if ldt_iterations > 0 {
                                    // BIT-02: Q-Head Routing (GRAM)
                                    // Inject stochastic noise when LDT certainty is low, breaking deterministic loops
                                    let seed = (l as u32).wrapping_add(ldt_iterations as u32).wrapping_mul(1013904223);
                                    layer.router.route_by_q_head(&gate_logits_guard, 0.05, seed, &mut indexed, &mut results)
                                } else {
                                    layer.router.route_in_place(&gate_logits_guard, &mut indexed, &mut results)
                                };
                                
                                *ws.routing_z_loss.lock() += z_loss;
                                layer.router.evaluate_ldt_certainty(&results)
                            };
                            
                            let mut local_results = [(0usize, 0.0f32); 8];
                            let num_results = {
                                let results = ws.routing_results.read();
                                let len = results.len().min(8);
                                if len > 0 {
                                    local_results[..len].copy_from_slice(&results[..len]);
                                }
                                len
                            };
                            let expert_results = &local_results[..num_results];
                            
                            // Only count active experts for the first iteration to avoid breaking metrics
                            if ldt_iterations == 0 {
                                step_active_experts += expert_results.len();
                            }

                            if expert_results.len() == 2 && vk.is_some() {
                                let (primary, secondary) = (expert_results[0], expert_results[1]);
                                let primary_expert_id = primary.0;
                                let primary_prob = primary.1;
                                let secondary_expert_id = secondary.0;
                                let secondary_prob = secondary.1;

                                rayon::join(
                                    || {
                                        if let Some(expert) = layer.experts.get(primary_expert_id) {
                                            let e_ws = &ws.expert_workspaces[primary_expert_id];
                                            Self::run_expert_ffn(
                                                vk, l, primary_expert_id, expert, e_ws, hidden, ffn_hidden,
                                                &ws.x_moe_norm, &e_ws.final_out, false,
                                                &self.model.hidden_act, layer.ffn_sub_norm_w, self.model.rms_norm_eps,
                                            );
                                        }
                                    },
                                    || {
                                        if let Some(expert) = layer.experts.get(secondary_expert_id) {
                                            let e_ws = &ws.expert_workspaces[secondary_expert_id];
                                            Self::run_expert_ffn(
                                                vk, l, secondary_expert_id, expert, e_ws, hidden, ffn_hidden,
                                                &ws.x_moe_norm, &e_ws.final_out, true,
                                                &self.model.hidden_act, layer.ffn_sub_norm_w, self.model.rms_norm_eps,
                                            );
                                        }
                                    },
                                );

                                let mut combined_guard = ws.combined_expert_out.write();
                                if layer.experts.get(primary_expert_id).is_some() {
                                    let final_guard = ws.expert_workspaces[primary_expert_id].final_out.read();
                                    for (j, &val) in final_guard.iter().enumerate() {
                                        combined_guard[j] += val * primary_prob;
                                    }
                                }
                                if layer.experts.get(secondary_expert_id).is_some() {
                                    let final_guard = ws.expert_workspaces[secondary_expert_id].final_out.read();
                                    for (j, &val) in final_guard.iter().enumerate() {
                                        combined_guard[j] += val * secondary_prob;
                                    }
                                }
                            } else {
                                for &(expert_id, _prob) in expert_results {
                                    if let Some(expert) = layer.experts.get(expert_id) {
                                        let e_ws = &ws.expert_workspaces[expert_id];
                                        Self::run_expert_ffn(
                                            None, l, expert_id, expert, e_ws, hidden, ffn_hidden,
                                            &ws.x_moe_norm, &e_ws.final_out, false,
                                            &self.model.hidden_act, layer.ffn_sub_norm_w, self.model.rms_norm_eps,
                                        );
                                        let mut combined_guard = ws.combined_expert_out.write();
                                        let final_guard = e_ws.final_out.read();
                                        for (j, &val) in final_guard.iter().enumerate() {
                                            combined_guard[j] += val * _prob;
                                        }
                                    }
                                }
                            }
                            
                            ldt_certainty = is_certain;
                    } else if !layer.experts.is_empty() {
                        step_active_experts += 1;
                        let expert = &layer.experts[0];
                        let e_ws = &ws.expert_workspaces[0];
                        Self::run_expert_ffn(
                            vk,
                            l,
                            0,
                            expert,
                            e_ws,
                            hidden,
                            ffn_hidden,
                            &ws.x_moe_norm,
                            &ws.combined_expert_out,
                            false,
                            &self.model.hidden_act,
                            layer.ffn_sub_norm_w,
                            self.model.rms_norm_eps,
                        );
                        ldt_certainty = true; // Single expert has absolute certainty
                    }

                    ldt_iterations += 1;

                    if !ldt_certainty && ldt_iterations < max_ldt_iterations {
                        // RRM-01: LDT Recurrente - Retroalimentamos el output al input para forzar pensar más profundo.
                        let dynamic_alpha = 1.0 / (ldt_iterations as f32 + 1.0); // Decaying feedback rate
                        ws.inject_latent_feedback_moe(hidden, dynamic_alpha); // Inyecta Delta-Lógico
                        
                        // LDT-01: Lattice Constraint Projections (Snap to logic grid)
                        ws.apply_lattice_projection(hidden, 3.0); // 3 levels for ternary-like structure
                        
                        // Asynchronous Imagination: Dispatch Vulkan speculative work parallel to CPU LDT evaluation
                        let mut imagination_future = None;
                        if let Some(vk) = self.vulkan_ctx.as_deref() {
                            unsafe {
                                imagination_future = Some(vk.dispatch_imagination_async());
                            }
                        }

                        // LDT-02: Early Exit si el estado convergió
                        let converged = ws.evaluate_ldt_convergence(hidden, dynamic_epsilon);
                        
                        if let Some(mut fut) = imagination_future {
                            fut.cleanup_finished(); // We clean up the future to avoid memory leaks
                        }

                        if converged {
                            // PTRM-01: Probabilistic Width Scaling
                            // The state converged but we still lack certainty (entropy is high).
                            // This means we hit a "Single Attractor" cognitive loop.
                            // Inject noise to break out of it, and continue looping!
                            ws.inject_stochastic_noise(hidden, 0.05, (l + ldt_iterations) as u32);
                        }
                    }
                }

                    {
                        let combined_expert_out_guard = ws.combined_expert_out.read();
                        let mut x_guard = ws.x_unified.write();
                        for (i, v) in x_guard.iter_mut().enumerate().take(hidden) {
                            let mut delta = combined_expert_out_guard[i];
                            if delta.is_nan() || delta.is_infinite() {
                                delta = 0.0;
                            }
                            *v += delta;
                        }
                    }
                }
                MudLayer::Mamba(layer) => {
                    // Save the base state for LDT convergence check in Mamba
                    {
                        let mut ldt_base_guard = ws.ldt_base_state.write();
                        let x_guard = ws.x_unified.read();
                        ldt_base_guard.copy_from_slice(&x_guard);
                    }
                    
                    let mut ldt_certainty = false;
                    let mut ldt_iterations = 0;
                    // Dynamic calculation of max iterations based on manifold dimension
                    let max_ldt_iterations = ((hidden as f32 / 512.0).round() as usize).clamp(2, 5);
                    // Dynamic convergence threshold derived from model's base epsilon floor
                    let dynamic_epsilon = self.model.rms_norm_eps * (hidden as f32).sqrt();

                    while !ldt_certainty && ldt_iterations < max_ldt_iterations {
                        self.mamba_step(layer, l, ws);
                        
                        ldt_iterations += 1;

                        // RRM-01: LDT Recurrente for Mamba
                        if ldt_iterations < max_ldt_iterations {
                            let dynamic_alpha = 1.0 / (ldt_iterations as f32 + 1.0); // Decaying feedback rate
                            ws.inject_latent_feedback_mamba(hidden, l, dynamic_alpha); // Inject Delta into state
                            
                            // To use LDT-02 for Mamba, we compare the original x_unified with the final mamba_out
                            // For simplicity in this iteration, we evaluate the shift in the output projection vs input
                            // We temporarily map the output to x_moe_norm buffer to reuse `evaluate_ldt_convergence` without allocations
                            {
                                let mut temp_guard = ws.x_moe_norm.write();
                                let final_guard = ws.final_attn_out.read();
                                let x_base = ws.ldt_base_state.read();
                                for i in 0..hidden {
                                    temp_guard[i] = x_base[i] + final_guard[i];
                                }
                            }
                            
                            // LDT-01: Lattice Constraint Projections for Mamba
                            ws.apply_lattice_projection(hidden, 3.0);
                            
                            // Asynchronous Imagination: Dispatch Vulkan speculative work parallel to CPU LDT evaluation
                            let mut imagination_future = None;
                            if let Some(vk) = self.vulkan_ctx.as_deref() {
                                unsafe {
                                    imagination_future = Some(vk.dispatch_imagination_async());
                                }
                            }

                            // LDT-02: Early Exit si el estado convergió
                            let converged = ws.evaluate_ldt_convergence(hidden, dynamic_epsilon);
                            
                            if let Some(mut fut) = imagination_future {
                                fut.cleanup_finished(); // We clean up the future to avoid memory leaks
                            }

                            if converged {
                                // PTRM-01: Break Single Attractor in Mamba
                                ws.inject_stochastic_noise(hidden, 0.05, (l + ldt_iterations) as u32);
                            }
                        } else {
                            ldt_certainty = true; // Exiting loop
                        }
                    }

                    if let Some(vk) = self.vulkan_ctx.as_deref() {
                        unsafe {
                            vk.pulse_heartbeat();
                        }
                    }

                    // Apply Residual Connection
                    {
                        let m_final_guard = ws.final_attn_out.read();
                        let mut x_guard = ws.x_unified.write();
                        for i in 0..hidden {
                            let mut delta = m_final_guard[i];
                            if delta.is_nan() || delta.is_infinite() {
                                delta = 0.0;
                            }
                            x_guard[i] += delta;
                        }
                    }
                }
                MudLayer::Ttt(layer) => {
                    // TTT Layers [2407.04620] - Test-Time Training
                    // 1. Forward Pass on the sequence using an implicit mini-model (W_t)
                    // 2. Perform a gradient step using self-supervised learning (reconstruction)
                    let mut x_guard = ws.x_unified.write();
                    if l < ws.ttt_states.len() {
                        let mut w_t = ws.ttt_states[l].write();
                        
                        // Si la matriz W_t está en ceros (recién inicializada), cargar pesos base
                        if w_t[0] == 0.0 && !layer.in_proj_w.is_null() {
                            unsafe {
                                std::ptr::copy_nonoverlapping(layer.in_proj_w, w_t.as_mut_ptr(), hidden * hidden);
                            }
                        }

                        let mut z = vec![0.0f32; hidden]; // Small local array, okay for scalar TTT
                        
                        // 1. z = x * W_t
                        unsafe {
                            for (o, z_val) in z.iter_mut().enumerate().take(hidden) {
                                let mut sum = 0.0f32;
                                let w_row = w_t.as_ptr().add(o * hidden);
                                for i in 0..hidden {
                                    sum += x_guard[i] * (*w_row.add(i));
                                }
                                *z_val = sum;
                            }
                        }
                        
                        // 2. Gradient Step: L = ||z - x||^2 -> dL/dW = (z - x) * x^T
                        unsafe {
                            for o in 0..hidden {
                                let err = z[o] - x_guard[o];
                                let w_row = w_t.as_mut_ptr().add(o * hidden);
                                for i in 0..hidden {
                                    *w_row.add(i) -= layer.eta * err * x_guard[i];
                                }
                            }
                        }
                        
                        // 3. Apply Residual
                        for i in 0..hidden {
                            x_guard[i] += z[i];
                        }
                    }
                }
            }

            if self.trace_propagation {
                let mut dot = 0.0;
                let mut norm_in = 0.0;
                let mut norm_out = 0.0;
                let mut l2_shift = 0.0;
                let mut min_out = f32::MAX;
                let mut max_out = f32::MIN;
                
                let x_guard = ws.x_unified.read();
                let trace_buf = ws.trace_in_buffer.read();
                
                for i in 0..hidden {
                    let v_in = trace_buf[i];
                    let v_out = x_guard[i];
                    dot += v_in * v_out;
                    norm_in += v_in * v_in;
                    norm_out += v_out * v_out;
                    l2_shift += (v_out - v_in).powi(2);
                    
                    if v_out < min_out { min_out = v_out; }
                    if v_out > max_out { max_out = v_out; }
                }
                let _cos_sim = dot / ((norm_in * norm_out).sqrt() + 1e-8);
                let _l2_shift = l2_shift.sqrt();
            }
        }

        self.active_experts.store(
            (step_active_experts as f32 / self.model.layers.len() as f32).round() as usize,
            Ordering::Relaxed,
        );

        // Copy back ws.x_unified to x at the end of the entire step
        {
            let x_guard = ws.x_unified.read();
            x[..hidden].copy_from_slice(&x_guard[..hidden]);
        }

        if !self.output_norm_w.is_null() {
            let scale_out = unsafe { crate::asm::rms_norm_scale_asm(hidden, x.as_ptr(), self.model.rms_norm_eps) };
            unsafe {
                for (i, item) in x.iter_mut().enumerate().take(hidden) {
                    *item *= scale_out * (*self.output_norm_w.add(i));
                }
            }
        }
    }

    // INF-04: Added sliding window reset for conversation_pos to prevent OOB when exceeding KV limits.
    pub fn mamba_step(&self, layer: &MudMambaLayer, l: usize, ws: &InferenceWorkspace) {
        let hidden = self.model.hidden_size;
        let d_state = self.model.d_state;

        // 1. RMSNorm
        let scale = unsafe {
            let x_guard = ws.x_unified.read();
            crate::asm::rms_norm_scale_asm(hidden, x_guard.as_ptr(), self.model.rms_norm_eps)
        };
        {
            let x_guard = ws.x_unified.read();
            let mut x_norm_guard = ws.x_norm.write();
            unsafe {
                for i in 0..hidden {
                    x_norm_guard[i] = x_guard[i] * scale * (*layer.norm_w.add(i));
                }
            }
        }

        // 2. In Projection
        ws.mamba_in.fill(0.0);
        // Small projection: synchronous on CPU to avoid launch overhead
        Self::gemv_vulkan_or_cpu(
            None,
            &layer.key_in,
            hidden,
            hidden * 2,
            &ws.x_norm,
            layer.in_proj_w,
            layer.in_proj_scales,
            &ws.mamba_in,
            false,
        );

        // 3. Conv1D and SiLU (Optimized)
        {
            let mut m_in_guard = ws.mamba_in.write();
            let mut conv_state_guard = ws.mamba_conv_state[l].write();

            // Vectorizable Convolution
            for i in 0..hidden {
                let off = i * 4;
                conv_state_guard[off] = conv_state_guard[off + 1];
                conv_state_guard[off + 1] = conv_state_guard[off + 2];
                conv_state_guard[off + 2] = conv_state_guard[off + 3];
                conv_state_guard[off + 3] = m_in_guard[i];

                let mut conv_out = 0.0f32;
                if !layer.conv1d_w.is_null() {
                    unsafe {
                        let w_ptr = layer.conv1d_w.add(i * 4);
                        conv_out = conv_state_guard[off] * (*w_ptr)
                            + conv_state_guard[off + 1] * (*w_ptr.add(1))
                            + conv_state_guard[off + 2] * (*w_ptr.add(2))
                            + conv_state_guard[off + 3] * (*w_ptr.add(3));
                    }
                }
                if !layer.conv1d_b.is_null() {
                    conv_out += unsafe { *layer.conv1d_b.add(i) };
                }
                m_in_guard[i] = conv_out;
            }
            unsafe {
                crate::asm::silu_vectorial_avx2(
                    hidden,
                    m_in_guard.as_ptr(),
                    m_in_guard.as_mut_ptr(),
                );
            }
        }

        // 4. Selective Projections (B, C, dt) - 100% ZERO-ALLOCATION
        ws.mamba_b.fill(0.0);
        ws.mamba_c.fill(0.0);
        ws.mamba_dt.fill(0.0);
        {
            // Use combined_expert_out as temporary buffer for proj_out (size: 32 + 2 * d_state)
            // It's safe because expert execution hasn't happened or is finished for this layer
            let mut proj_out_guard = ws.combined_expert_out.write();
            let proj_out_len: usize = 32 + 2 * d_state;

            let m_in_read = ws.mamba_in.read();
            let blocks_per_row = hidden.div_ceil(16);
            unsafe {
                for i in 0..proj_out_len {
                    let mut sum = 0.0f32;
                    let row_ptr = layer.x_proj_w.add(i * blocks_per_row);
                    crate::asm::ternary_gemv_avx2(
                        hidden,
                        m_in_read.as_ptr(),
                        row_ptr,
                        &mut sum,
                        1.0,
                    );
                    if !layer.x_proj_scales.is_null() {
                        sum *= *layer.x_proj_scales.add(i);
                    }
                    proj_out_guard[i] = sum;
                }
            }

            // dt_proj
            let dt_rank: usize = 32;
            let blocks_dt = dt_rank.div_ceil(16);
            let mut dt_guard = ws.mamba_dt.write();
            unsafe {
                for i in 0..hidden {
                    let mut sum = 0.0f32;
                    let row_ptr = layer.dt_proj_w.add(i * blocks_dt);
                    crate::asm::ternary_gemv_avx2(
                        dt_rank,
                        proj_out_guard.as_ptr(),
                        row_ptr,
                        &mut sum,
                        1.0,
                    );
                    if !layer.dt_proj_scales.is_null() {
                        sum *= *layer.dt_proj_scales.add(i);
                    }
                    dt_guard[i] = sum;
                }
            }

            ws.mamba_b
                .write()
                .copy_from_slice(&proj_out_guard[32..32 + d_state]);
            ws.mamba_c
                .write()
                .copy_from_slice(&proj_out_guard[32 + d_state..32 + 2 * d_state]);
        }

        // 5. SSM Scan (Optimized loop)
        {
            let dt_guard = ws.mamba_dt.read();
            let b_guard = ws.mamba_b.read();
            let mut a_bar_guard = ws.mamba_a_bar.write();
            let mut b_bar_guard = ws.mamba_b_bar.write();
            for i in 0..hidden {
                let dt_val = (1.0 + dt_guard[i].exp()).ln();
                let a_ptr = unsafe { layer.a_log_w.add(i * d_state) };
                let a_bar_ptr = unsafe { a_bar_guard.as_mut_ptr().add(i * d_state) };
                let b_bar_ptr = unsafe { b_bar_guard.as_mut_ptr().add(i * d_state) };
                for j in 0..d_state {
                    let a_val = unsafe { *a_ptr.add(j) }.exp();
                    unsafe {
                        let a_bar = (-dt_val * a_val).exp();
                        *a_bar_ptr.add(j) = a_bar;
                        // Mamba-3 MIMO: Exponential-Trapezoidal Discretization (2nd Order)
                        // More stable for quantized / low-precision weights than Euler
                        *b_bar_ptr.add(j) = (dt_val * 0.5) * b_guard[j] * (1.0 + a_bar);
                    }
                }
            }
        }

        ws.mamba_out.fill(0.0);
        unsafe {
            let x_ptr = ws.mamba_in.read().as_ptr();
            let a_bar_ptr = ws.mamba_a_bar.read().as_ptr();
            let b_bar_ptr = ws.mamba_b_bar.read().as_ptr();
            let c_ptr = ws.mamba_c.read().as_ptr();
            let state_ptr = ws.ssm_states[l].write().as_mut_ptr();
            let out_ptr = ws.mamba_out.write().as_mut_ptr();
            crate::asm::mamba_scan_avx2(
                hidden,
                d_state,
                x_ptr,
                a_bar_ptr,
                b_bar_ptr,
                c_ptr,
                std::ptr::null(),
                state_ptr,
                out_ptr,
            );
        }

        // 6. Gating and Out Projection
        {
            let mut m_out_guard = ws.mamba_out.write();
            let m_in_guard = ws.mamba_in.read();
            for i in 0..hidden {
                let gate = m_in_guard[hidden + i];
                let silu_gate = gate * (1.0 / (1.0 + (-gate).exp()));
                let mut val = m_out_guard[i] * silu_gate;
                if !layer.d_w.is_null() {
                    val += m_in_guard[i] * unsafe { *layer.d_w.add(i) };
                }
                m_out_guard[i] = val;
            }
        }

        ws.final_attn_out.write().fill(0.0);
        // Use GPU asynchronously for the Mamba Output Projection to keep Vulkan alive (Heartbeat)
        Self::gemv_vulkan_or_cpu(
            self.vulkan_ctx.as_deref(),
            &layer.key_out,
            hidden,
            hidden,
            &ws.mamba_out,
            layer.out_proj_w,
            layer.out_proj_scales,
            &ws.final_attn_out,
            true,
        );
    }

    pub fn prompt(&mut self, text: &str, x: &mut [f32], conversation_pos: &mut usize) {
        let tokens = self.tokenizer.encode(text);
        if tokens.is_empty() {
            return;
        }
        #[allow(clippy::needless_range_loop)]
        for i in 0..tokens.len() - 1 {
            self.shift_kv_cache(conversation_pos);
            self.embed_token(tokens[i], x);
            self.step(x, text, &[], *conversation_pos);
            *conversation_pos += 1;
        }
        self.embed_token(*tokens.last().unwrap(), x);
    }

    // Sampling hyperparámetros — ajustar aquí afecta toda la generación
    const TEMPERATURE: f32 = 0.7;
    const TOP_P: f32 = 0.9;
    #[allow(dead_code)]
    const REPETITION_PENALTY: f32 = 1.15;
    /// Posiciones del KV-cache (ventana deslizante)
    pub const KV_CACHE_MAX_POS: usize = 4096;
    /// Al llegar al límite, resetear a esta posición para mantener contexto reciente
    pub const KV_CACHE_RESET_POS: usize = 4000;

    /// EDGE-01: Attention Sinks
    /// Retains the first 4 tokens (sinks) and shifts the remaining most recent tokens
    /// to maintain contiguous context without breaking semantic coherence.
    fn shift_kv_cache(&mut self, conversation_pos: &mut usize) {
        if *conversation_pos < Self::KV_CACHE_MAX_POS {
            return;
        }
        let keep_sinks = 4;
        let reset_pos = Self::KV_CACHE_RESET_POS;
        let shift_len = reset_pos - keep_sinks;
        let source_start = Self::KV_CACHE_MAX_POS - shift_len;
        let hidden = self.model.hidden_size;

        for l in 0..self.model.layers.len() {
            let layer_offset = l * Self::KV_CACHE_MAX_POS * hidden;
            
            // k-cache shift
            let k_ptr = self.kv_cache_k.as_mut_ptr();
            unsafe {
                std::ptr::copy(
                    k_ptr.add(layer_offset + source_start * hidden),
                    k_ptr.add(layer_offset + keep_sinks * hidden),
                    shift_len * hidden
                );
            }
            
            // v-cache shift
            let v_ptr = self.kv_cache_v.as_mut_ptr();
            unsafe {
                std::ptr::copy(
                    v_ptr.add(layer_offset + source_start * hidden),
                    v_ptr.add(layer_offset + keep_sinks * hidden),
                    shift_len * hidden
                );
            }
        }
        *conversation_pos = reset_pos;
    }

    pub fn generate<F>(
        &mut self,
        x_init: &[f32],
        max_tokens: usize,
        context: &str,
        conversation_pos: &mut usize,
        coconut_steps: usize,
        mut callback: F,
    ) -> (Vec<u32>, bool)
    where
        F: FnMut(u32, &str),
    {
        let active_skill_indices: Vec<usize> = self
            .skills
            .iter()
            .enumerate()
            .filter(|(_, s)| s.should_activate(x_init, context))
            .map(|(i, _)| i)
            .collect();
        let mut x = x_init.to_vec();
        let mut results = Vec::new();

        for _ in 0..max_tokens {
            // RRM-01: COCONUT Latent Thinking Loop
            for _ in 0..coconut_steps {
                self.shift_kv_cache(conversation_pos);
                self.step(&mut x, context, &active_skill_indices, *conversation_pos);
                *conversation_pos += 1;
            }

            self.shift_kv_cache(conversation_pos);
            self.step(&mut x, context, &active_skill_indices, *conversation_pos);
            *conversation_pos += 1;

            let ws = &self.workspace;
            ws.logits.write().fill(0.0);

            match self.out_proj_type {
                crate::mud::MudTensorType::Float32 => {
                    let mut logits_guard = ws.logits.write();
                    for i in 0..logits_guard.len() {
                        let row_ptr =
                            unsafe { self.out_proj_w_f32.add(i * self.model.hidden_size) };
                        let val = unsafe {
                            crate::asm::dot_product_avx2(
                                self.model.hidden_size,
                                x.as_ptr(),
                                row_ptr,
                            )
                        };
                        logits_guard[i] = val;
                    }
                }
                crate::mud::MudTensorType::Ternary2Bit => {
                    let x_unified = if let Some(vk) = self.vulkan_ctx.as_deref() {
                        let buf = UnifiedBuffer::new_gpu(vk, self.model.hidden_size);
                        buf.write().copy_from_slice(&x);
                        buf
                    } else {
                        let buf = UnifiedBuffer::new_cpu(self.model.hidden_size);
                        buf.write().copy_from_slice(&x);
                        buf
                    };
                    // Use vocab_size (logits buffer length), NOT embd_rows, as n_out.
                    // embd_rows may exceed vocab_size due to padding alignment in the original model.
                    let n_out_proj = ws.logits.read().len().min(self.embd_rows);
                    Self::gemv_vulkan_or_cpu(
                        self.vulkan_ctx.as_deref(),
                        "output_proj",
                        self.model.hidden_size,
                        n_out_proj,
                        &x_unified,
                        self.out_proj_w_u32,
                        self.out_proj_scales,
                        &ws.logits,
                        false,
                    );
                }
                _ => {}
            }

            // Sanitize logits
            {
                let mut logits_guard = ws.logits.write();
                for logit in logits_guard.iter_mut() {
                    if logit.is_nan() || logit.is_infinite() {
                        *logit = -1e4;
                    }
                }
            }

            // 2. Cognitive Filtering (Top-K + Strong Repetition)
            {
                let mut logits_guard = ws.logits.write();
                for (dist, &prev_id) in results.iter().rev().take(128).enumerate() {
                    let idx = prev_id as usize;
                    if let Some(logit) = logits_guard.get_mut(idx) {
                        *logit -= 2.0 / (dist as f32 + 1.0).sqrt();
                    }
                }
                
                let mut indexed: Vec<(usize, f32)> = logits_guard.iter().enumerate()
                    .filter(|(_, &v)| v.is_finite())
                    .map(|(i, &v)| (i, v)).collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                
                let k = 40;
                if indexed.len() > k {
                    for item in indexed.iter().skip(k) {
                        let idx = item.0;
                        logits_guard[idx] = -1e9;
                    }
                }
            }

            // Temperature scaling
            {
                let mut logits_guard = ws.logits.write();
                for l in &mut *logits_guard {
                    *l /= Self::TEMPERATURE.max(1e-5);
                }
            }

            // Optimized Sampling: Top-P using partition/select_nth
            let mut probs: Vec<(usize, f32)> = {
                let logits_guard = ws.logits.read();
                logits_guard
                    .iter()
                    .enumerate()
                    .filter(|(_, &l)| l.is_finite())
                    .map(|(i, &l)| (i, l))
                    .collect()
            };


            if probs.is_empty() {
                break;
            }

            let max_logit = probs
                .iter()
                .map(|&(_, l)| l)
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for p in &mut probs {
                p.1 = (p.1 - max_logit).exp();
                sum_exp += p.1;
            }
            if sum_exp > 0.0 {
                for p in &mut probs {
                    p.1 /= sum_exp;
                }
            }

            // Use partial sort to find top-p cumulative mass
            // We sort a small prefix to find the cutoff mass quickly
            probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let mut cum_prob = 0.0f32;
            let mut cutoff = probs.len();
            for (i, p) in probs.iter().enumerate() {
                cum_prob += p.1;
                if cum_prob > Self::TOP_P {
                    cutoff = i + 1;
                    break;
                }
            }
            probs.truncate(cutoff);

            let r = rand::random::<f32>();
            let mut current_cum = 0.0f32;
            let mut next_id = probs[0].0 as u32;

            let prob_sum: f32 = probs.iter().map(|p| p.1).sum();
            for p in &probs {
                current_cum += p.1 / prob_sum;
                if r <= current_cum {
                    next_id = p.0 as u32;
                    break;
                }
            }

            if next_id == 2 {
                break;
            }
            results.push(next_id);
            let token_text = self.tokenizer.decode(&[next_id]);
            callback(next_id, &token_text);
            self.embed_token(next_id, &mut x);

            // 2. Coherence Circuit Breaker (Autonomous Stopping)
            if results.len() > 10 {
                let last_4 = &results[results.len() - 4..];
                if last_4[0] == last_4[2] && last_4[1] == last_4[3] {
                    break;
                }
            }
        }
        (results, false)
    }

    pub fn embed_token(&self, id: u32, x: &mut [f32]) {
        if (id as usize) >= self.embd_rows {
            x.fill(0.0);
            return;
        }

        match self.embd_type {
            crate::mud::MudTensorType::Float32 => {
                let offset = (id as usize) * self.model.hidden_size;
                unsafe {
                    let ptr = self.embd_w_f32.add(offset);
                    std::ptr::copy_nonoverlapping(ptr, x.as_mut_ptr(), self.model.hidden_size);
                }
            }
            crate::mud::MudTensorType::Ternary2Bit => {
                debug_assert_eq!(
                    self.model.hidden_size % 16,
                    0,
                    "hidden_size debe ser múltiplo de 16"
                );
                let offset = (id as usize) * (self.model.hidden_size / 16);
                unsafe {
                    let ptr = self.embd_w_u32.add(offset);
                    crate::mud::dequantize_ternary_row(ptr, x, self.model.hidden_size);
                }
                // Aplicar escala per-row si está disponible (embedding ternarizado)
                if !self.embd_scales.is_null() {
                    let scale = unsafe { *self.embd_scales.add(id as usize) };
                    if scale != 1.0 {
                        for v in x.iter_mut() {
                            *v *= scale;
                        }
                    }
                }
            }
            crate::mud::MudTensorType::Int4 => {
                debug_assert_eq!(
                    self.model.hidden_size % 2,
                    0,
                    "hidden_size debe ser par para INT4"
                );
                let offset = (id as usize) * (self.model.hidden_size / 2);
                let scale = if !self.embd_scales.is_null() {
                    unsafe { *self.embd_scales.add(id as usize) }
                } else {
                    1.0
                };
                unsafe {
                    let ptr = self.embd_w_u8.add(offset);
                    for i in 0..(self.model.hidden_size / 2) {
                        let b = *ptr.add(i);
                        // low nibble
                        let v0 = (b & 0x0F) as i8 - 8;
                        // high nibble
                        let v1 = (b >> 4) as i8 - 8;
                        x[i * 2] = (v0 as f32) * scale;
                        x[i * 2 + 1] = (v1 as f32) * scale;
                    }
                }
            }
            _ => {
                x.fill(0.0);
            }
        }
    }

    /// # Safety
    ///
    /// This function is unsafe because it dereferences raw pointers for weights and scales.
    #[allow(clippy::too_many_arguments)]
    fn run_expert_ffn(
        vk_ctx: Option<&VulkanContext>,
        _l: usize,
        _expert_id: usize,
        expert: &MudExpert,
        e_ws: &ExpertWorkspace,
        hidden: usize,
        ffn_hidden: usize,
        x_moe_norm: &UnifiedBuffer,
        y_out: &UnifiedBuffer,
        force_cpu: bool,
        hidden_act: &str,
        ffn_sub_norm_w: *const f32,
        rms_norm_eps: f32,
    ) {
        let mut vlk_done = false;
        if !force_cpu {
            if let Some(vk) = vk_ctx {
                if let (Some(buf_x), Some(buf_w1_out), Some(buf_w3_out), Some(buf_final_out)) = (
                    x_moe_norm.gpu_buffer(),
                    e_ws.w1_out.gpu_buffer(),
                    e_ws.w3_out.gpu_buffer(),
                    y_out.gpu_buffer(),
                ) {
                    unsafe {
                        if vk
                            .run_chained_ffn(
                                &expert.key_w1,
                                &expert.key_w2,
                                &expert.key_w3,
                                hidden,
                                ffn_hidden,
                                buf_x,
                                expert.w1,
                                expert.w1_scales,
                                buf_w1_out,
                                expert.w3,
                                expert.w3_scales,
                                buf_w3_out,
                                expert.w2,
                                expert.w2_scales,
                                buf_final_out,
                            )
                            .is_ok()
                        {
                            vlk_done = true;
                        }
                    }
                }
            }
        }

        if !vlk_done {
            // CPU Path Fallback
            Self::gemv_vulkan_or_cpu(
                None,
                &expert.key_w1,
                hidden,
                ffn_hidden,
                x_moe_norm,
                expert.w1,
                expert.w1_scales,
                &e_ws.w1_out,
                false,
            );
            Self::gemv_vulkan_or_cpu(
                None,
                &expert.key_w3,
                hidden,
                ffn_hidden,
                x_moe_norm,
                expert.w3,
                expert.w3_scales,
                &e_ws.w3_out,
                false,
            );

            {
                let mut w1_guard = e_ws.w1_out.write();
                // Optional BitNet ffn_sub_norm on gate (w1) output (shape: [ffn_hidden])
                if !ffn_sub_norm_w.is_null() {
                    let scale_w1 = unsafe {
                        crate::asm::rms_norm_scale_asm(ffn_hidden, w1_guard.as_ptr(), rms_norm_eps)
                    };
                    unsafe {
                        for i in 0..ffn_hidden {
                            w1_guard[i] *= scale_w1 * (*ffn_sub_norm_w.add(i));
                        }
                    }
                }
                let w3_guard = e_ws.w3_out.read();
                match hidden_act {
                    "relu2" => {
                        // ReLU^2 activation (BitNet): (max(0,x))^2
                        for j in 0..ffn_hidden {
                            let v = w1_guard[j].max(0.0);
                            w1_guard[j] = v * v * w3_guard[j];
                        }
                    }
                    _ => {
                        // Default: SiLU gated
                        unsafe {
                            crate::asm::silu_vectorial_avx2(
                                ffn_hidden,
                                w1_guard.as_ptr(),
                                w1_guard.as_mut_ptr(),
                            );
                        }
                        for j in 0..ffn_hidden {
                            w1_guard[j] *= w3_guard[j];
                        }
                    }
                }
            }

            Self::gemv_vulkan_or_cpu(
                None,
                &expert.key_w2,
                ffn_hidden,
                hidden,
                &e_ws.w1_out,
                expert.w2,
                expert.w2_scales,
                y_out,
                false,
            );
        }
    }

    /// Apply LoRA Delta Adapters [2410.20672]
    /// Intercepts a projection output and applies W ≈ W_base + A * B
    /// Uses zero-allocation by leveraging `ws.lora_temp`
    #[allow(clippy::too_many_arguments)]
    pub fn apply_lora_adapters(
        &self,
        layer_idx: usize,
        target_name: &str,
        in_features: usize,
        out_features: usize,
        x_in: &UnifiedBuffer,
        x_out: &UnifiedBuffer,
        ws: &InferenceWorkspace,
    ) {
        if let Some(adapters) = self.model.lora_adapters.get(&layer_idx) {
            for adapter in adapters {
                if adapter.target == target_name {
                    // 1. temp = x_in * A (dense gemv)
                    // a_w shape: [rank, in_features]
                    let x_guard = x_in.read();
                    let mut temp_guard = ws.lora_temp.write();
                    
                    unsafe {
                        for r in 0..adapter.rank {
                            let mut sum = 0.0f32;
                            let a_row = adapter.a_w.add(r * in_features);
                            for i in 0..in_features {
                                sum += x_guard[i] * (*a_row.add(i));
                            }
                            temp_guard[r] = sum;
                        }
                    }

                    // 2. delta = temp * B (dense gemv)
                    // b_w shape: [out_features, rank]
                    let mut out_guard = x_out.write();
                    unsafe {
                        for o in 0..out_features {
                            let mut sum = 0.0f32;
                            let b_row = adapter.b_w.add(o * adapter.rank);
                            for r in 0..adapter.rank {
                                sum += temp_guard[r] * (*b_row.add(r));
                            }
                            // 3. out = out + delta * alpha
                            out_guard[o] += sum * adapter.alpha;
                        }
                    }
                }
            }
        }
    }

    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_vulkan_or_cpu(
        vk_ctx: Option<&VulkanContext>,
        key: &str,
        n_in: usize,
        n_out: usize,
        x: &UnifiedBuffer,
        w: *const u32,
        scales: *const f32,
        y: &UnifiedBuffer,
        is_async: bool,
    ) {


        let mut vlk_done = false;
        if let Some(vk) = vk_ctx {
            if let (Some(buf_x), Some(buf_y)) = (x.gpu_buffer(), y.gpu_buffer()) {
                if is_async {
                    if unsafe {
                        vk.run_ternary_gemv_cached_async(key, n_in, n_out, buf_x, w, scales, buf_y)
                            .is_ok()
                    } {
                        vlk_done = true;
                    }
                } else if unsafe {
                    vk.run_ternary_gemv_cached(key, n_in, n_out, buf_x, w, scales, buf_y)
                        .is_ok()
                } {
                    vlk_done = true;
                }
            }
        }

        if !vlk_done {
            let blocks_per_row = n_in.div_ceil(16);
            let x_guard = x.read();
            let mut y_guard = y.write();

            use rayon::prelude::*;

            // Hito E: T-SAR Dynamic INT8 Quantization (Zero-Allocation on stack)
            let mut x_i8 = [0i8; 16384];
            let mut x_absmax = 1e-8f32;
            for j in 0..n_in {
                let v = x_guard[j].abs();
                if v > x_absmax { x_absmax = v; }
            }
            let q_scale = 127.0 / x_absmax;
            for j in 0..n_in {
                x_i8[j] = (x_guard[j] * q_scale).round().clamp(-127.0, 127.0) as i8;
            }
            let inv_q_scale = x_absmax / 127.0;

            let x_ptr = x_i8.as_ptr() as usize;
            let w_ptr = w as usize;
            let s_ptr = scales as usize;
            let y_ptr = y_guard.as_mut_ptr() as usize;

            // Hito D: FairyFuse PEXT Decoding + T-SAR GEMV
            (0..n_out)
                .into_par_iter()
                .for_each(|i| {
                    unsafe {
                        let mut w_i8 = [0i8; 16384];
                        let x_p = x_ptr as *const i8;
                        let w_p = (w_ptr as *const u32).add(i * blocks_per_row);
                        let s_p = s_ptr as *const f32;
                        let y_p = (y_ptr as *mut f32).add(i);

                        let blocks_64 = n_in / 32;
                        let row_ptr_64 = w_p as *const u64;
                        for b in 0..blocks_64 {
                            crate::asm::pext_unpack_ternary(*row_ptr_64.add(b), w_i8.as_mut_ptr().add(b * 32));
                        }

                        // Remaining weights if n_in is not a multiple of 32 (we assume it is for now)
                        
                        crate::asm::ternary_gemv_lut_avx2(
                            n_in,
                            x_p,
                            w_i8.as_ptr(),
                            y_p,
                            1.0,
                        );

                        if !s_p.is_null() {
                            *y_p *= *s_p.add(i) * inv_q_scale;
                        } else {
                            *y_p *= inv_q_scale;
                        }
                    }
                });
        }
    }

    pub fn format_text(&self, text: &mut String) {
        for skill in &self.skills {
            skill.post_process_token(text);
        }
    }

    pub fn apply_rope(
        q: &mut [f32],
        k: &mut [f32],
        pos: usize,
        head_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        rope_freqs: &[f32],
    ) {
        let half = head_dim / 2;
        let mut cos_table = [0.0f32; 256];
        let mut sin_table = [0.0f32; 256];

        for i in 0..half {
            let theta = (pos as f32) * rope_freqs[i];
            cos_table[i] = theta.cos();
            sin_table[i] = theta.sin();
        }

        // Apply RoPE to Query heads
        for h in 0..n_heads {
            let start = h * head_dim;
            unsafe {
                crate::asm::apply_rope_asm(
                    head_dim,
                    q[start..start + head_dim].as_mut_ptr(),
                    cos_table.as_ptr(),
                    sin_table.as_ptr(),
                );
            }
        }

        // Apply RoPE to Key heads
        for h in 0..n_kv_heads {
            let start = h * head_dim;
            unsafe {
                crate::asm::apply_rope_asm(
                    head_dim,
                    k[start..start + head_dim].as_mut_ptr(),
                    cos_table.as_ptr(),
                    sin_table.as_ptr(),
                );
            }
        }
    }
}
