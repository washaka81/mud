use crate::vulkan::VulkanContext;
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::sync::atomic::AtomicU32;
use vulkano::buffer::Subbuffer;

pub const EPSILON_FLOOR: f32 = 1e-8;

/// H1-01: Standard sampling hyperparameters with mathematically justified defaults.
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub repetition_window: usize,
}

pub struct TokenizerBuffer {
    pub final_tokens: Vec<u32>,
    pub new_parts: Vec<String>,
    pub tokens: Vec<String>,
    pub decoded_bytes: Vec<u8>,
    pub decode_out: String,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.1,
            repetition_window: 64,
        }
    }
}

pub struct AlignedBuffer {
    pub ptr: *mut f32,
    layout: std::alloc::Layout,
    pub len: usize,
}

impl AlignedBuffer {
    pub fn new(size: usize) -> Self {
        let byte_size = size.checked_mul(4).expect("AlignedBuffer size overflow");
        let layout = std::alloc::Layout::from_size_align(byte_size, 64)
            .expect("AlignedBuffer: invalid layout");
        // SAFETY: layout was computed from valid size/align; alloc_zeroed returns null on OOM (checked below)
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) as *mut f32 };
        assert!(
            !ptr.is_null(),
            "AlignedBuffer: alloc_zeroed returned null (OOM, {} bytes)",
            size * 4
        );
        Self {
            ptr,
            layout,
            len: size,
        }
    }
    pub fn as_slice(&self) -> &[f32] {
        // SAFETY: self.ptr is valid, non-null, aligned for f32, and points to `self.len` elements
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        // SAFETY: self.ptr is valid and non-null; no other reference aliases this region because &mut self is exclusive
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
        // SAFETY: ptr was allocated by alloc_zeroed with the same layout in new(); this is the unique Drop call
        unsafe {
            std::alloc::dealloc(self.ptr as *mut u8, self.layout);
        }
    }
}

// SAFETY: AlignedBuffer owns its heap allocation; Send/Sync are safe because mutation goes through &mut
unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

/// # Soundness
/// `Cpu(RwLock<AlignedBuffer>)` provides safe interior mutability:
/// `read()` and `write()` both take `&self`, but the RwLock enforces
/// mutual exclusion at runtime, preventing aliased mutable references.
pub enum UnifiedBuffer {
    Cpu(RwLock<AlignedBuffer>),
    Gpu(Subbuffer<[f32]>),
}

pub enum UnifiedReadGuard<'a> {
    Cpu(RwLockReadGuard<'a, AlignedBuffer>),
    Gpu(vulkano::buffer::BufferReadGuard<'a, [f32]>),
}

impl<'a> std::ops::Deref for UnifiedReadGuard<'a> {
    type Target = [f32];
    fn deref(&self) -> &Self::Target {
        match self {
            UnifiedReadGuard::Cpu(g) => g.deref(),
            UnifiedReadGuard::Gpu(g) => g.deref(),
        }
    }
}

pub enum UnifiedWriteGuard<'a> {
    Cpu(RwLockWriteGuard<'a, AlignedBuffer>),
    Gpu(vulkano::buffer::BufferWriteGuard<'a, [f32]>),
}

impl<'a> std::ops::Deref for UnifiedWriteGuard<'a> {
    type Target = [f32];
    fn deref(&self) -> &Self::Target {
        match self {
            UnifiedWriteGuard::Cpu(g) => g.deref(),
            UnifiedWriteGuard::Gpu(g) => g.deref(),
        }
    }
}

impl<'a> std::ops::DerefMut for UnifiedWriteGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            UnifiedWriteGuard::Cpu(g) => g.deref_mut(),
            UnifiedWriteGuard::Gpu(g) => g.deref_mut(),
        }
    }
}

impl UnifiedBuffer {
    pub fn new_cpu(size: usize) -> Self {
        crate::mud::memory_profiler::GLOBAL_PROFILER.register_allocation(size * 4); // f32
        UnifiedBuffer::Cpu(RwLock::new(AlignedBuffer::new(size)))
    }

    pub fn new_cpu_from_slice(slice: &[f32]) -> Self {
        crate::mud::memory_profiler::GLOBAL_PROFILER.register_allocation(slice.len() * 4); // f32
        let mut buf = AlignedBuffer::new(slice.len());
        buf.as_mut_slice().copy_from_slice(slice);
        UnifiedBuffer::Cpu(RwLock::new(buf))
    }

    pub fn new_gpu(vk: &VulkanContext, size: usize) -> Self {
        crate::mud::memory_profiler::GLOBAL_PROFILER.register_allocation(size * 4); // f32
        UnifiedBuffer::Gpu(vk.allocate_zero_copy_buffer(size))
    }

    pub fn read(&self) -> UnifiedReadGuard<'_> {
        match self {
            UnifiedBuffer::Cpu(b) => UnifiedReadGuard::Cpu(b.read()),
            UnifiedBuffer::Gpu(b) => UnifiedReadGuard::Gpu(b.read().unwrap()),
        }
    }

    pub fn write(&self) -> UnifiedWriteGuard<'_> {
        match self {
            UnifiedBuffer::Cpu(b) => UnifiedWriteGuard::Cpu(b.write()),
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
            UnifiedBuffer::Cpu(b) => b.read().len,
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

// SAFETY: RwLock<AlignedBuffer> is Send+Sync because AlignedBuffer is Send+Sync.
// GPU Subbuffer maintains its own thread-safety guarantees.
unsafe impl Send for UnifiedBuffer {}
unsafe impl Sync for UnifiedBuffer {}

pub struct ExpertWorkspace {
    pub w1_out: UnifiedBuffer,
    pub w3_out: UnifiedBuffer,
    pub final_out: UnifiedBuffer,
}

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
    pub attn_scores: UnifiedBuffer,
    pub ssm_states: Vec<UnifiedBuffer>,
    pub mamba_in: UnifiedBuffer,
    pub mamba_out: UnifiedBuffer,
    pub mamba_dt: UnifiedBuffer,
    pub mamba_b: UnifiedBuffer,
    pub mamba_c: UnifiedBuffer,
    pub mamba_conv_state: Vec<UnifiedBuffer>,
    pub mamba_a_bar: UnifiedBuffer,
    pub mamba_b_bar: UnifiedBuffer,
    pub mamba_cos_table: UnifiedBuffer,
    pub mamba_sin_table: UnifiedBuffer,
    pub ttt_states: Vec<UnifiedBuffer>,
    pub ttt_z: UnifiedBuffer,
    pub ldt_base_state: UnifiedBuffer,
    pub lora_temp: UnifiedBuffer,
    pub routing_indexed: parking_lot::RwLock<Vec<(usize, f32)>>,
    pub routing_results: parking_lot::RwLock<Vec<(usize, f32)>>,
    pub trace_in_buffer: parking_lot::RwLock<Vec<f32>>,
    pub routing_z_loss: parking_lot::Mutex<f32>,
    pub lop_temp: parking_lot::Mutex<Vec<(usize, f32)>>,
    pub lop_active: parking_lot::Mutex<Vec<bool>>,
    pub sample_candidates: parking_lot::Mutex<Vec<(usize, f32)>>,
    pub ttt_initialized: parking_lot::RwLock<Vec<bool>>,
    // *** NEW BUFFERS FOR DIFFUSION (BLOCK EVALUATION) ***
    pub diffusion_canvas: UnifiedBuffer,      // Holds N * hidden
    pub diffusion_prev_canvas: UnifiedBuffer, // Tracks state for Early Exit Entropy Calculation
    pub diffusion_q_block: UnifiedBuffer,     // N * hidden
    pub diffusion_k_block: UnifiedBuffer,     // N * hidden
    pub diffusion_v_block: UnifiedBuffer,     // N * hidden
    // *** PRIORITY 6: MCTS TEST-TIME COMPUTE BRANCHES ***
    pub mcts_branches: Vec<UnifiedBuffer>,    // G branches for Monte Carlo Tree Search
    pub diffusion_attn_scores: UnifiedBuffer, // N * N * num_heads
    pub diffusion_mask: parking_lot::Mutex<Vec<bool>>, // To track mask vs unmasked tokens
    pub diffusion_x_norm_block: UnifiedBuffer, // N * hidden
    pub diffusion_gate_block: UnifiedBuffer,  // N * ffn_hidden
    pub diffusion_up_block: UnifiedBuffer,    // N * ffn_hidden
    pub diffusion_ffn_out_block: UnifiedBuffer, // N * hidden
    // *** REUSABLE TEMP BUFFER FOR DIFFUSION ROPE (avoids per-token vec! alloc) ***
    pub diffusion_rope_q_temp: UnifiedBuffer, // size: hidden (max q_out)
    pub diffusion_rope_k_temp: UnifiedBuffer, // size: hidden (max kv_out)
    // *** REUSABLE LOGITS BLOCK FOR DIFFUSION (avoids per-step vec! alloc) ***
    pub diffusion_logits_block: parking_lot::Mutex<Vec<f32>>,
    // *** NEW BUFFER FOR TOKENIZER ***
    pub tokenizer_buf: TokenizerBuffer,
    // *** NEW BUFFERS FOR LDT-03 (Micro-Intelligence GRPO) ***
    pub ldt_parallel_waves: Vec<UnifiedBuffer>, // G parallel states for GRPO
    pub ldt_reference_lattice: UnifiedBuffer,   // Pre-loaded constraint matrices
    pub ldt_micro: crate::mud::ldt_micro::LdtMicroModel, // GRPO Inference Engine
    pub x_draft: UnifiedBuffer, // Intermediate hidden state checkpoint for self-speculation
    pub draft_logits: UnifiedBuffer, // Pre-allocated logits buffer for draft distribution comparison
    // --- INT8 Activation Quantization (4× bandwidth reduction on CPU path) ---
    /// Pre-allocated INT8 buffer for quantized attention-norm activations (x_norm → i8).
    /// Filled once per token per layer before Q/K/V projections on the CPU path.
    pub x_norm_i8: RwLock<Vec<i8>>,
    /// Activation quantization scale for x_norm: `absmax(x_norm) / 127.0`.
    /// Stored as f32 bits in AtomicU32 for lock-free access from the forward pass.
    pub x_norm_act_scale: AtomicU32,
    /// Pre-allocated INT8 buffer for quantized FFN-norm activations (x_moe_norm → i8).
    /// Filled before W1/W3 FFN projections on the CPU path.
    pub x_moe_norm_i8: RwLock<Vec<i8>>,
    /// Activation quantization scale for x_moe_norm.
    pub x_moe_norm_act_scale: AtomicU32,
}

impl InferenceWorkspace {
    pub fn inject_latent_feedback_moe(&self, hidden: usize, alpha: f32) {
        let combined_guard = self.combined_expert_out.read();
        let mut norm_guard = self.x_moe_norm.write();
        for i in 0..hidden {
            norm_guard[i] += combined_guard[i] * alpha;
        }
    }

    pub fn inject_latent_feedback_mamba(&self, hidden: usize, layer_idx: usize, alpha: f32) {
        let final_guard = self.final_attn_out.read();
        let mut conv_guard = self.mamba_conv_state[layer_idx].write();
        for i in 0..hidden {
            conv_guard[i * 4 + 3] += final_guard[i] * alpha;
        }
    }

    pub fn evaluate_ldt_convergence(&self, hidden: usize, epsilon: f32) -> bool {
        let base_guard = self.ldt_base_state.read();
        let current_guard = self.x_moe_norm.read();
        let mut sum_sq = 0.0f32;
        for i in 0..hidden {
            let diff = current_guard[i] - base_guard[i];
            sum_sq += diff * diff;
        }
        let l2_shift = sum_sq.sqrt();
        l2_shift < epsilon
    }

    pub fn apply_lattice_projection(&self, hidden: usize, lattice_levels: f32) {
        let mut current_guard = self.x_moe_norm.write();
        let scale = 1.0 / lattice_levels;
        for i in 0..hidden {
            let val = current_guard[i];
            current_guard[i] = (val * lattice_levels).round() * scale;
        }
    }

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
        num_heads: usize,
        num_experts: usize,
        vocab_size: usize,
        d_state: usize,
        d_conv: usize,
        max_pos: usize,
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
            attn_scores: init_buf(max_pos * num_heads),
            ssm_states,
            mamba_in: init_buf(hidden * 2),
            mamba_out: init_buf(hidden * 2),
            mamba_dt: init_buf(hidden),
            mamba_b: init_buf(d_state),
            mamba_c: init_buf(d_state),
            mamba_conv_state,
            mamba_a_bar: init_buf(hidden * d_state),
            mamba_b_bar: init_buf(hidden * d_state),
            mamba_cos_table: init_buf(d_state / 2),
            mamba_sin_table: init_buf(d_state / 2),
            ttt_states: Vec::with_capacity(num_layers),
            ttt_z: init_buf(hidden),
            ldt_base_state: init_buf(hidden),
            lora_temp: init_buf(max_pos),
            routing_indexed: parking_lot::RwLock::new(Vec::with_capacity(num_experts)),
            routing_results: parking_lot::RwLock::new(Vec::with_capacity(8)),
            trace_in_buffer: parking_lot::RwLock::new(vec![0.0f32; hidden]),
            routing_z_loss: parking_lot::Mutex::new(0.0),
            lop_temp: parking_lot::Mutex::new(Vec::with_capacity(max_pos)),
            lop_active: parking_lot::Mutex::new(vec![false; max_pos]),
            sample_candidates: parking_lot::Mutex::new(Vec::with_capacity(vocab_size)),
            ttt_initialized: parking_lot::RwLock::new(vec![false; num_layers]),
            diffusion_canvas: init_buf(max_pos * hidden),
            diffusion_prev_canvas: init_buf(max_pos * hidden),
            diffusion_q_block: init_buf(max_pos * hidden),
            diffusion_k_block: init_buf(max_pos * hidden),
            diffusion_v_block: init_buf(max_pos * hidden),
            mcts_branches: {
                let mut branches = Vec::with_capacity(8); // G=8 branches for MCTS
                for _ in 0..8 {
                    branches.push(init_buf(max_pos * hidden));
                }
                branches
            },
            diffusion_attn_scores: init_buf(max_pos * max_pos * num_heads),
            diffusion_mask: parking_lot::Mutex::new(vec![false; max_pos]),
            diffusion_x_norm_block: init_buf(max_pos * hidden),
            diffusion_gate_block: init_buf(max_pos * ffn_hidden),
            diffusion_up_block: init_buf(max_pos * ffn_hidden),
            diffusion_ffn_out_block: init_buf(max_pos * hidden),
            diffusion_rope_q_temp: init_buf(hidden),
            diffusion_rope_k_temp: init_buf(hidden),
            diffusion_logits_block: parking_lot::Mutex::new(Vec::new()),
            tokenizer_buf: TokenizerBuffer {
                final_tokens: Vec::with_capacity(max_pos),
                new_parts: Vec::with_capacity(64),
                tokens: Vec::with_capacity(max_pos),
                decoded_bytes: Vec::with_capacity(max_pos),
                decode_out: String::with_capacity(max_pos),
            },
            ldt_parallel_waves: {
                let mut waves = Vec::with_capacity(8); // G=8 parallel reflections for GRPO
                for _ in 0..8 {
                    waves.push(init_buf(max_pos * hidden));
                }
                waves
            },
            ldt_reference_lattice: init_buf(max_pos * hidden),
            ldt_micro: crate::mud::ldt_micro::LdtMicroModel::new(hidden, 1),
            x_draft: init_buf(hidden),
            draft_logits: init_buf(vocab_size),
            x_norm_i8: RwLock::new(vec![0i8; hidden]),
            x_norm_act_scale: AtomicU32::new(1.0f32.to_bits()),
            x_moe_norm_i8: RwLock::new(vec![0i8; hidden]),
            x_moe_norm_act_scale: AtomicU32::new(1.0f32.to_bits()),
        }
    }
}
