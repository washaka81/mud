use crate::mud::MudFile;
use crate::mud::routing::MudRouter;
use std::sync::{Arc, RwLock};
use crate::vulkan::VulkanContext;
use crate::mud::skills::MudSkill;
use crate::model::tokenizer::Tokenizer;
use crate::mud::graph::MudKnowledgeGraph;
use crate::mud::store::MudStore;

pub struct MudExpert {
    pub w1: *const u32, pub w2: *const u32, pub w3: *const u32,
    pub w1_scale: f32, pub w2_scale: f32, pub w3_scale: f32,
}

unsafe impl Send for MudExpert {}
unsafe impl Sync for MudExpert {}

pub struct MudMoELayer {
    pub experts: Vec<MudExpert>,
    pub router: MudRouter,
    pub attn_q_w: *const u32, pub attn_k_w: *const u32, pub attn_v_w: *const u32, pub attn_o_w: *const u32,
    pub attn_q_scale: f32, pub attn_k_scale: f32, pub attn_v_scale: f32, pub attn_o_scale: f32,
    pub gate_w: *const u32, pub norm_w: *const f32, pub attn_norm_w: *const f32,
}

unsafe impl Send for MudMoELayer {}
unsafe impl Sync for MudMoELayer {}

pub struct MudModel {
    pub layers: Vec<MudMoELayer>,
    pub knowledge_graph: Arc<RwLock<MudKnowledgeGraph>>,
    pub hidden_size: usize,
    pub ffn_hidden_size: usize,
    pub num_experts: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

unsafe impl Send for MudModel {}
unsafe impl Sync for MudModel {}

pub struct InferenceWorkspace {
    pub x_norm: AlignedBuffer,
    pub q: AlignedBuffer,
    pub k: AlignedBuffer,
    pub v: AlignedBuffer,
    pub attn_out: AlignedBuffer,
    pub final_attn_out: AlignedBuffer,
    pub x_moe_norm: AlignedBuffer,
    pub gate_logits: AlignedBuffer,
    pub combined_expert_out: AlignedBuffer,
    pub expert_workspaces: Vec<ExpertWorkspace>,
    pub logits: Vec<f32>,
    pub attn_scores: AlignedBuffer,
}

pub struct ExpertWorkspace {
    pub w1_out: AlignedBuffer,
    pub w3_out: AlignedBuffer,
    pub final_out: AlignedBuffer,
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
        Self { ptr, layout, len: size }
    }
    pub fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl std::ops::Deref for AlignedBuffer {
    type Target = [f32];
    fn deref(&self) -> &Self::Target { self.as_slice() }
}

impl std::ops::DerefMut for AlignedBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target { self.as_mut_slice() }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr as *mut u8, self.layout); }
    }
}

unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

impl InferenceWorkspace {
    pub fn new(hidden: usize, ffn_hidden: usize, num_experts: usize, vocab_size: usize) -> Self {
        let mut expert_workspaces = Vec::with_capacity(num_experts);
        for _ in 0..num_experts {
            expert_workspaces.push(ExpertWorkspace {
                w1_out: AlignedBuffer::new(ffn_hidden),
                w3_out: AlignedBuffer::new(ffn_hidden),
                final_out: AlignedBuffer::new(hidden),
            });
        }
        Self {
            x_norm: AlignedBuffer::new(hidden),
            q: AlignedBuffer::new(hidden), k: AlignedBuffer::new(hidden), v: AlignedBuffer::new(hidden),
            attn_out: AlignedBuffer::new(hidden),
            final_attn_out: AlignedBuffer::new(hidden),
            x_moe_norm: AlignedBuffer::new(hidden),
            gate_logits: AlignedBuffer::new(num_experts),
            combined_expert_out: AlignedBuffer::new(hidden),
            expert_workspaces,
            logits: vec![0.0; vocab_size],
            attn_scores: AlignedBuffer::new(4096),
        }
    }
}

use std::sync::atomic::{AtomicUsize, Ordering};

pub struct MudInference {
    pub model: MudModel,
    pub vulkan_ctx: Option<Arc<VulkanContext>>,
    pub embd_w_u32: *const u32,
    pub embd_w_f32: *const f32,
    pub embd_type: crate::mud::MudTensorType,
    pub embd_rows: usize,
    pub embd_scales: *const f32,
    pub output_norm_w: *const f32,
    pub skills: Vec<Box<dyn MudSkill>>,
    pub tokenizer: Tokenizer,
    pub store: Arc<MudStore>,
    pub kv_cache_k: Vec<f32>,
    pub kv_cache_v: Vec<f32>,
    pub kv_scales_k: Vec<f32>,
    pub kv_scales_v: Vec<f32>,
    pub active_experts: Arc<AtomicUsize>,
    pub workspace: InferenceWorkspace,
}

unsafe impl Send for MudInference {}
unsafe impl Sync for MudInference {}

impl MudInference {
    pub fn new(mud_file: &MudFile, vulkan_ctx: Option<Arc<VulkanContext>>) -> anyhow::Result<Self> {
        let core = mud_file.skills.get("core").ok_or_else(|| anyhow::anyhow!("No core skill found"))?;
        let store = Arc::new(MudStore::open("models/knowledge.db")?);
        let tokens_str = mud_file.global_metadata.get("tokenizer.tokens").ok_or_else(|| anyhow::anyhow!("No tokenizer tokens"))?;
        let merges_str = mud_file.global_metadata.get("tokenizer.merges").map(|s| s.as_str()).unwrap_or("");
        let tokenizer = Tokenizer::from_mud_metadata(tokens_str, merges_str);

        let hidden_size = mud_file.global_metadata.get("hidden_size").and_then(|s| s.parse::<usize>().ok()).unwrap_or(512);
        let num_layers = mud_file.global_metadata.get("num_layers").and_then(|s| s.parse::<usize>().ok()).unwrap_or(12);
        let num_experts = mud_file.global_metadata.get("num_experts").and_then(|s| s.parse::<usize>().ok()).unwrap_or(4);
        let top_k = mud_file.global_metadata.get("top_k").and_then(|s| s.parse::<usize>().ok()).unwrap_or(2);
        let ffn_hidden = mud_file.global_metadata.get("ffn_hidden").and_then(|s| s.parse::<usize>().ok()).unwrap_or(1024);
        let num_heads = mud_file.global_metadata.get("num_heads").and_then(|s| s.parse::<usize>().ok()).unwrap_or(4);
        let num_kv_heads = mud_file.global_metadata.get("num_kv_heads").and_then(|s| s.parse::<usize>().ok()).unwrap_or(num_heads);
        let head_dim = mud_file.global_metadata.get("head_dim").and_then(|s| s.parse::<usize>().ok()).unwrap_or(64);
        let vocab_size = tokenizer.id_to_token.len();

        let mut layers = Vec::with_capacity(num_layers);
        for l in 0..num_layers {
            let mut experts = Vec::with_capacity(num_experts);
            for e in 0..num_experts {
                experts.push(MudExpert {
                    w1: core.tensors.get(&format!("blk.{}.expert.{}.w1.weight", l, e)).map(|t| t.data_ptr as *const u32).unwrap_or(std::ptr::null()),
                    w2: core.tensors.get(&format!("blk.{}.expert.{}.w2.weight", l, e)).map(|t| t.data_ptr as *const u32).unwrap_or(std::ptr::null()),
                    w3: core.tensors.get(&format!("blk.{}.expert.{}.w3.weight", l, e)).map(|t| t.data_ptr as *const u32).unwrap_or(std::ptr::null()),
                    w1_scale: core.tensors.get(&format!("blk.{}.expert.{}.w1.scale", l, e)).map(|t| unsafe { *(t.data_ptr as *const f32) }).unwrap_or(1.0),
                    w2_scale: core.tensors.get(&format!("blk.{}.expert.{}.w2.scale", l, e)).map(|t| unsafe { *(t.data_ptr as *const f32) }).unwrap_or(1.0),
                    w3_scale: core.tensors.get(&format!("blk.{}.expert.{}.w3.scale", l, e)).map(|t| unsafe { *(t.data_ptr as *const f32) }).unwrap_or(1.0),
                });
            }
            layers.push(MudMoELayer {
                experts, router: MudRouter::new(num_experts, top_k),
                attn_q_w: core.tensors.get(&format!("blk.{}.attn_q.weight", l)).map(|t| t.data_ptr as *const u32).unwrap_or(std::ptr::null()),
                attn_k_w: core.tensors.get(&format!("blk.{}.attn_k.weight", l)).map(|t| t.data_ptr as *const u32).unwrap_or(std::ptr::null()),
                attn_v_w: core.tensors.get(&format!("blk.{}.attn_v.weight", l)).map(|t| t.data_ptr as *const u32).unwrap_or(std::ptr::null()),
                attn_o_w: core.tensors.get(&format!("blk.{}.attn_output.weight", l)).map(|t| t.data_ptr as *const u32).unwrap_or(std::ptr::null()),
                attn_q_scale: core.tensors.get(&format!("blk.{}.attn_q.scale", l)).map(|t| unsafe { *(t.data_ptr as *const f32) }).unwrap_or(1.0),
                attn_k_scale: core.tensors.get(&format!("blk.{}.attn_k.scale", l)).map(|t| unsafe { *(t.data_ptr as *const f32) }).unwrap_or(1.0),
                attn_v_scale: core.tensors.get(&format!("blk.{}.attn_v.scale", l)).map(|t| unsafe { *(t.data_ptr as *const f32) }).unwrap_or(1.0),
                attn_o_scale: core.tensors.get(&format!("blk.{}.attn_output.scale", l)).map(|t| unsafe { *(t.data_ptr as *const f32) }).unwrap_or(1.0),
                gate_w: core.tensors.get(&format!("blk.{}.gate.weight", l)).map(|t| t.data_ptr as *const u32).unwrap_or(std::ptr::null()),
                norm_w: core.tensors.get(&format!("blk.{}.norm.weight", l)).map(|t| t.data_ptr as *const f32).unwrap_or(std::ptr::null()),
                attn_norm_w: core.tensors.get(&format!("blk.{}.attn_norm.weight", l)).map(|t| t.data_ptr as *const f32).unwrap_or(std::ptr::null()),
            });
        }

        let skills: Vec<Box<dyn MudSkill>> = vec![
            Box::new(crate::mud::skills::autoformatter::AutoformatterSkill::new()),
            Box::new(crate::mud::skills::logic_math::LogicMathSkill::new()),
            Box::new(crate::mud::skills::retrieval::RetrievalSkill::new()),
            Box::new(crate::mud::skills::language::LanguageSkill::new("es")),
            Box::new(crate::mud::skills::translator::TranslationSkill::new("en")),
            Box::new(crate::mud::skills::personality::PersonalitySkill::new("Forge Assistant")),
            Box::new(crate::mud::skills::memory::MemorySkill::new()),
            Box::new(crate::mud::skills::learning::LearningSkill::new()),
            Box::new(crate::mud::skills::data_analysis::DataAnalysisSkill::new()),
            Box::new(crate::mud::skills::plotting::PlottingSkill::new()),
            Box::new(crate::mud::skills::web_search::WebSearchSkill::new()),
            Box::new(crate::mud::skills::code_formatter::CodeFormatSkill {}),
            Box::new(crate::mud::skills::logic_marks::LogicMarkSkill {}),
            Box::new(crate::mud::skills::text_styling::TextStylingSkill {}),
        ];

        let mut graph = MudKnowledgeGraph::new();
        if let Ok(hubs) = store.get_top_hubs(200) {
            for (content, emb, rank) in hubs {
                graph.add_node(content, emb);
                if let Some(&idx) = graph.content_to_index.get(&graph.nodes.last().unwrap().content) {
                    graph.nodes[idx].rank = rank;
                }
            }
        }
        let knowledge_graph = Arc::new(RwLock::new(graph));
        let workspace = InferenceWorkspace::new(hidden_size, ffn_hidden, num_experts, vocab_size);

        let embd_tensor = core.tensors.get("token_embd.weight");
        let embd_w_u32 = embd_tensor.map(|t| t.data_ptr as *const u32).unwrap_or(std::ptr::null());
        let embd_w_f32 = embd_tensor.map(|t| t.data_ptr as *const f32).unwrap_or(std::ptr::null());
        let embd_type = embd_tensor.map(|t| t.t_type).unwrap_or(crate::mud::MudTensorType::Float32);
        let embd_rows = embd_tensor.map(|t| t.shape[0]).unwrap_or(0);
        let embd_scales = core.tensors.get("embed_scales").map(|t| t.data_ptr as *const f32).unwrap_or(std::ptr::null());

        Ok(Self {
            model: MudModel { layers, knowledge_graph, hidden_size, ffn_hidden_size: ffn_hidden, num_experts, num_heads, num_kv_heads, head_dim },
            vulkan_ctx,
            embd_w_u32,
            embd_w_f32,
            embd_type,
            embd_rows,
            embd_scales,
            output_norm_w: core.tensors.get("output_norm.weight").map(|t| t.data_ptr as *const f32).unwrap_or(std::ptr::null()),
            skills, tokenizer, store,
            kv_cache_k: vec![0.0; num_layers.checked_mul(4096).and_then(|x| x.checked_mul(hidden_size)).expect("KV-cache-k: overflow en num_layers * 4096 * hidden_size")],
            kv_cache_v: vec![0.0; num_layers.checked_mul(4096).and_then(|x| x.checked_mul(hidden_size)).expect("KV-cache-v: overflow en num_layers * 4096 * hidden_size")],
            kv_scales_k: vec![0.0; num_layers.checked_mul(4096).and_then(|x| x.checked_mul(hidden_size / 64)).expect("KV-scales-k: overflow")],
            kv_scales_v: vec![0.0; num_layers.checked_mul(4096).and_then(|x| x.checked_mul(hidden_size / 64)).expect("KV-scales-v: overflow")],
            active_experts: Arc::new(AtomicUsize::new(0)),
            workspace,
        })
    }

    pub fn step(&mut self, x: &mut [f32], _context: &str, active_skill_indices: &[usize], _pos: usize) {
        let ws = &mut self.workspace;
        for &si in active_skill_indices { self.skills[si].pre_process(x); }
        let hidden = self.model.hidden_size;

        // 0. Sanitize input tensor to guarantee no NaNs or Infs enter the computation
        for v in x.iter_mut().take(hidden) {
            if v.is_nan() || v.is_infinite() { *v = 0.0; }
        }
        let ffn_hidden = self.model.ffn_hidden_size;
        let mut step_active_experts = 0;

        for (l, layer) in self.model.layers.iter().enumerate() {
            let scale_attn = unsafe { crate::asm::rms_norm_scale_asm(hidden, x.as_ptr(), 1e-6) };
            let norm_ptr = if !layer.attn_norm_w.is_null() { layer.attn_norm_w } else { layer.norm_w };
            unsafe { for i in 0..hidden { ws.x_norm[i] = x[i] * scale_attn * (*norm_ptr.add(i)); } }

            let q_out = self.model.num_heads * self.model.head_dim;
            let kv_out = self.model.num_kv_heads * self.model.head_dim;
            ws.q.fill(0.0); ws.k.fill(0.0); ws.v.fill(0.0);
            let key_q = format!("l{}_q", l); let key_k = format!("l{}_k", l); let key_v = format!("l{}_v", l);
            Self::gemv_vulkan_or_cpu(self.vulkan_ctx.as_deref(), &key_q, hidden, q_out, &ws.x_norm, layer.attn_q_w, layer.attn_q_scale, &mut ws.q);
            Self::gemv_vulkan_or_cpu(self.vulkan_ctx.as_deref(), &key_k, hidden, kv_out, &ws.x_norm, layer.attn_k_w, layer.attn_k_scale, &mut ws.k);
            Self::gemv_vulkan_or_cpu(self.vulkan_ctx.as_deref(), &key_v, hidden, kv_out, &ws.x_norm, layer.attn_v_w, layer.attn_v_scale, &mut ws.v);

            // INF-09: Apply RoPE (Rotary Position Embeddings)
            Self::apply_rope(&mut ws.q, &mut ws.k, _pos, self.model.head_dim, self.model.num_heads, self.model.num_kv_heads);

            // ── Multi-Head Attention (MHA) with GQA + KV Cache ──────────────────────
            let nh = self.model.num_heads;
            let hd = self.model.head_dim;
            let nkv = self.model.num_kv_heads;
            let kv_group = nh / nkv;
            let scale = 1.0 / (hd as f32).sqrt();

            // Store current key and value in KV Cache
            // INF-01: cache_offset puede exceder len si _pos >= 4096. Usar min(_pos, 4095) y bounds check.
            let max_pos = _pos.min(4095);
            let cache_offset = l * 4096 * hidden + max_pos * hidden;
            if cache_offset + hidden <= self.kv_cache_k.len() {
                self.kv_cache_k[cache_offset..cache_offset + hidden].copy_from_slice(&ws.k);
                self.kv_cache_v[cache_offset..cache_offset + hidden].copy_from_slice(&ws.v);
            }

            ws.attn_out.fill(0.0);

            for h in 0..nh {
                let q_off = h * hd;
                let kv_off = (h / kv_group) * hd;
                let q_h = &ws.q[q_off .. q_off + hd];

                // Use pre-allocated workspace buffer for attention scores
                let seq_len = max_pos + 1;
                let scores = &mut ws.attn_scores[0..seq_len];
                let mut max_score = f32::NEG_INFINITY;

                for t in 0..seq_len {
                    let t_off = l * 4096 * hidden + t * hidden + kv_off;
                    let k_t_h = &self.kv_cache_k[t_off .. t_off + hd];
                    let mut score = 0.0f32;
                    for i in 0..hd {
                        score += q_h[i] * k_t_h[i];
                    }
                    score *= scale;
                    scores[t] = score;
                    if score > max_score { max_score = score; }
                }

                let mut sum_exp = 0.0f32;
                for i in 0..seq_len {
                    let exp = (scores[i] - max_score).exp();
                    scores[i] = exp;
                    sum_exp += exp;
                }
                if sum_exp > 0.0 && sum_exp.is_finite() {
                    for i in 0..seq_len { scores[i] /= sum_exp; }
                } else {
                    scores.fill(0.0);
                    scores[0] = 1.0;
                }

                let out_h_slice = &mut ws.attn_out[q_off .. q_off + hd];
                out_h_slice.fill(0.0);
                for t in 0..seq_len {
                    let t_off = l * 4096 * hidden + t * hidden + kv_off;
                    let v_t_h = &self.kv_cache_v[t_off .. t_off + hd];
                    let w = scores[t];
                    for i in 0..hd { out_h_slice[i] += w * v_t_h[i]; }
                }
            }

            ws.final_attn_out.fill(0.0);
            let key_o = format!("l{}_o", l);
            Self::gemv_vulkan_or_cpu(self.vulkan_ctx.as_deref(), &key_o, hidden, hidden, &ws.attn_out, layer.attn_o_w, layer.attn_o_scale, &mut ws.final_attn_out);
            
            // Apply Delta Clipping and Sanitize
            for (i, v) in x.iter_mut().enumerate().take(hidden) {
                let mut delta = ws.final_attn_out[i];
                if delta.is_nan() || delta.is_infinite() { delta = 0.0; }
                *v += delta.clamp(-5.0, 5.0); 
            }

            // MoE Block / FFN
            let scale_moe = unsafe { crate::asm::rms_norm_scale_asm(hidden, x.as_ptr(), 1e-6) };
            let norm_ptr = if !layer.norm_w.is_null() { layer.norm_w } else if !layer.attn_norm_w.is_null() { layer.attn_norm_w } else { std::ptr::null() };
            if !norm_ptr.is_null() {
                unsafe { for i in 0..hidden { ws.x_moe_norm[i] = x[i] * scale_moe * (*norm_ptr.add(i)); } }
            } else {
                ws.x_moe_norm.copy_from_slice(&x);
            }

            ws.combined_expert_out.fill(0.0);
            let vk = self.vulkan_ctx.as_deref();
            let x_moe_norm = &ws.x_moe_norm;

            if self.model.num_experts > 1 {
                ws.gate_logits.fill(0.0);
                let key_gate = format!("l{}_gate", l);
                Self::gemv_vulkan_or_cpu(self.vulkan_ctx.as_deref(), &key_gate, hidden, self.model.num_experts, &ws.x_moe_norm, layer.gate_w, 1.0, &mut ws.gate_logits);

                let routing = layer.router.route(&ws.gate_logits);
                step_active_experts += routing.len();

                // Parallel execution of experts using rayon
                let expert_results: Vec<(usize, f32)> = routing.iter().cloned().collect();
                
                // Use workspace to avoid allocations
                let layer_experts = &layer.experts;
                let expert_ws = &mut ws.expert_workspaces;
                
                for (expert_id, prob) in expert_results {
                    if let Some(expert) = layer_experts.get(expert_id) {
                        let e_ws = &mut expert_ws[expert_id];
                        
                        let key_w1 = format!("l{}_e{}_w1", l, expert_id);
                        let key_w2 = format!("l{}_e{}_w2", l, expert_id);
                        let key_w3 = format!("l{}_e{}_w3", l, expert_id);
                        
                        Self::gemv_vulkan_or_cpu(vk, &key_w1, hidden, ffn_hidden, x_moe_norm, expert.w1, expert.w1_scale, &mut e_ws.w1_out);
                        Self::gemv_vulkan_or_cpu(vk, &key_w3, hidden, ffn_hidden, x_moe_norm, expert.w3, expert.w3_scale, &mut e_ws.w3_out);
                        
                        for j in 0..ffn_hidden {
                            let val = e_ws.w1_out[j];
                            let silu = val / (1.0 + (-val).exp());
                            e_ws.w1_out[j] = silu * e_ws.w3_out[j];
                        }

                        Self::gemv_vulkan_or_cpu(vk, &key_w2, ffn_hidden, hidden, &e_ws.w1_out, expert.w2, expert.w2_scale, &mut e_ws.final_out);
                        
                        for (j, &val) in e_ws.final_out.iter().enumerate() {
                            ws.combined_expert_out[j] += val * prob;
                        }
                    }
                }
            } else if !layer.experts.is_empty() {
                // MoE Bypass for single-expert models (dense models)
                step_active_experts += 1;
                let expert = &layer.experts[0];
                let e_ws = &mut ws.expert_workspaces[0];
                
                let key_w1 = format!("l{}_e0_w1", l);
                let key_w2 = format!("l{}_e0_w2", l);
                let key_w3 = format!("l{}_e0_w3", l);
                
                Self::gemv_vulkan_or_cpu(vk, &key_w1, hidden, ffn_hidden, x_moe_norm, expert.w1, expert.w1_scale, &mut e_ws.w1_out);
                Self::gemv_vulkan_or_cpu(vk, &key_w3, hidden, ffn_hidden, x_moe_norm, expert.w3, expert.w3_scale, &mut e_ws.w3_out);
                
                for j in 0..ffn_hidden {
                    let val = e_ws.w1_out[j];
                    let silu = val / (1.0 + (-val).exp());
                    e_ws.w1_out[j] = silu * e_ws.w3_out[j];
                }

                Self::gemv_vulkan_or_cpu(vk, &key_w2, ffn_hidden, hidden, &e_ws.w1_out, expert.w2, expert.w2_scale, &mut ws.combined_expert_out);
            }
            
            // Apply Delta Clipping and Sanitize
            for (i, v) in x.iter_mut().enumerate().take(hidden) {
                let mut delta = ws.combined_expert_out[i];
                if delta.is_nan() || delta.is_infinite() { delta = 0.0; }
                *v += delta.clamp(-5.0, 5.0); 
            }
        }
        
        self.active_experts.store((step_active_experts as f32 / self.model.layers.len() as f32).round() as usize, Ordering::Relaxed);
        
        if !self.output_norm_w.is_null() {
            let scale_out = unsafe { crate::asm::rms_norm_scale_asm(hidden, x.as_ptr(), 1e-6) };
            unsafe { for i in 0..hidden { x[i] *= scale_out * (*self.output_norm_w.add(i)); } }
        }
    }

    // INF-04: Added sliding window reset for conversation_pos to prevent OOB when exceeding 4096.
    pub fn prompt(&mut self, text: &str, x: &mut [f32], conversation_pos: &mut usize) {
        let tokens = self.tokenizer.encode(text);
        for &t in &tokens {
            if *conversation_pos >= 4096 { *conversation_pos = 4000; } // Keep some context
            self.embed_token(t, x);
            self.step(x, text, &[], *conversation_pos);
            *conversation_pos += 1;
        }
    }

    pub fn generate(&mut self, x_init: &[f32], max_tokens: usize, context: &str, conversation_pos: &mut usize) -> (Vec<u32>, bool) {
        let active_skill_indices: Vec<usize> = self.skills.iter().enumerate().filter(|(_, s)| s.should_activate(x_init, context)).map(|(i, _)| i).collect();
        let mut x = x_init.to_vec();
        let mut results = Vec::new();
        let mut used_knowledge = false;
        
        let temperature = 0.7f32;
        let top_p = 0.9f32;
        let repetition_penalty = 1.15f32;

        for step_idx in 0..max_tokens {

            // 1. Autonomous Synapse Injection (Reduced frequency)
            if step_idx % 50 == 0 { // Every 50 tokens instead of 20
                let retrieved_facts = self.model.knowledge_graph.write().unwrap()
                    .autonomous_jump_search(&x, &self.store, 1);
                
                if !retrieved_facts.is_empty() {
                    used_knowledge = true;

                    for fact in retrieved_facts.iter().take(2) {
                        let fact_tokens = self.tokenizer.encode(fact);
                        for &ft in fact_tokens.iter().take(3) { 
                            let mut fact_vec = vec![0.0; self.model.hidden_size];
                            self.embed_token(ft, &mut fact_vec);
                            for i in 0..self.model.hidden_size { x[i] = x[i] * 0.95 + fact_vec[i] * 0.05; }
                        }
                    }
                }
            }

            if *conversation_pos >= 4096 { *conversation_pos = 4000; }
            self.step(&mut x, context, &active_skill_indices, *conversation_pos);
            *conversation_pos += 1;
            
            let ws = &mut self.workspace;
            ws.logits.fill(0.0);
            
            match self.embd_type {
                crate::mud::MudTensorType::Float32 => {
                    for i in 0..ws.logits.len() {
                        let row_ptr = unsafe { self.embd_w_f32.add(i * self.model.hidden_size) };
                        ws.logits[i] = unsafe { crate::asm::dot_product_avx2(self.model.hidden_size, x.as_ptr(), row_ptr) };
                    }
                }
                crate::mud::MudTensorType::Ternary2Bit => {
                    Self::gemv_vulkan_or_cpu(self.vulkan_ctx.as_deref(), "output_proj", self.model.hidden_size, self.embd_rows, &x, self.embd_w_u32, 1.0, &mut ws.logits);
                    
                    // Aplicar escalas per-row si el embedding está ternarizado
                    if !self.embd_scales.is_null() {
                        for i in 0..ws.logits.len() {
                            ws.logits[i] *= unsafe { *self.embd_scales.add(i) };
                        }
                    }
                }
                _ => {}
            }
            
            // Debug: track logit magnitude
            if step_idx == 0 {
                let max_l = ws.logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let min_l = ws.logits.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let mean_abs = ws.logits.iter().map(|v| v.abs()).sum::<f32>() / ws.logits.len() as f32;
                eprintln!("[DEBUG] Step 0 Logits | Max: {:.4} | Min: {:.4} | MeanAbs: {:.4}", max_l, min_l, mean_abs);
            }
            
            // Sanitize logits
            for logit in ws.logits.iter_mut() {
                if logit.is_nan() || logit.is_infinite() { *logit = -1e4; }
            }

            // Apply repetition penalty (bounds-safe: prev_id podría ser mayor que vocab_size si hay corrupción)
            for &prev_id in results.iter().rev().take(64) {
                let idx = prev_id as usize;
                if let Some(logit) = ws.logits.get_mut(idx) {
                    if *logit > 0.0 { *logit /= repetition_penalty; } else { *logit *= repetition_penalty; }
                }
            }

            // Temperature scaling
            for l in &mut ws.logits { *l /= temperature.max(1e-5); }

            // Sampling: Top-P
            let mut probs: Vec<(usize, f32)> = ws.logits.iter().enumerate()
                .filter(|(i, &l)| *i > 3 && l.is_finite()) // Skip special tokens like <unk>, <s>, </s>
                .map(|(i, &l)| (i, l))
                .collect();
            
            if probs.is_empty() { break; }

            let max_logit = probs.iter().map(|&(_, l)| l).fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for p in &mut probs {
                p.1 = (p.1 - max_logit).exp();
                sum_exp += p.1;
            }
            for p in &mut probs { p.1 /= sum_exp; }

            probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            let mut cum_prob = 0.0f32;
            let mut cutoff = probs.len();
            for (i, p) in probs.iter().enumerate() {
                cum_prob += p.1;
                if cum_prob > top_p { cutoff = i + 1; break; }
            }
            probs.truncate(cutoff);

            let r = rand::random::<f32>();
            let mut current_cum = 0.0f32;
            let mut next_id = probs[0].0 as u32;
            let prob_sum: f32 = probs.iter().map(|p| p.1).sum();
            for p in &probs {
                current_cum += p.1 / prob_sum;
                if r <= current_cum { next_id = p.0 as u32; break; }
            }

            if next_id == 2 { break; }
            results.push(next_id);
            self.embed_token(next_id, &mut x);

            // 2. Coherence Circuit Breaker (Autonomous Stopping)
            if results.len() > 10 {
                let last_4 = &results[results.len()-4..];
                if last_4[0] == last_4[2] && last_4[1] == last_4[3] {
                    break;
                }
            }
        }
        (results, used_knowledge)
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
                debug_assert_eq!(self.model.hidden_size % 16, 0, "hidden_size debe ser múltiplo de 16");
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
            _ => {
                x.fill(0.0);
            }
        }
    }

    pub fn gemv_vulkan_or_cpu(vk_ctx: Option<&VulkanContext>, key: &str, n_in: usize, n_out: usize, x: &[f32], w: *const u32, scale: f32, y: &mut [f32]) {
        if w.is_null() { return; }
        let mut vlk_done = false;
        if let Some(vk) = vk_ctx {
            if unsafe { vk.run_ternary_gemv_cached(key, n_in, n_out, x, w, scale, y).is_ok() } {
                vlk_done = true;
            }
        }
        
        if !vlk_done {
            let blocks_per_row = n_in / 16;
            unsafe {
                let mut i = 0;
                while i + 3 < n_out {
                    let row_ptr = w.add(i * blocks_per_row);
                    crate::asm::ternary_gemv_4rows_avx2(n_in, x.as_ptr(), row_ptr, y.as_mut_ptr().add(i), scale, blocks_per_row);
                    i += 4;
                }
                while i < n_out {
                    let row_ptr = w.add(i * blocks_per_row);
                    crate::asm::ternary_gemv_avx2(n_in, x.as_ptr(), row_ptr, &mut y[i], scale);
                    i += 1;
                }
            }
        }
    }

    pub fn format_text(&self, text: &mut String) {
        for skill in &self.skills { skill.post_process_token(text); }
    }

    pub fn apply_rope(q: &mut [f32], k: &mut [f32], pos: usize, head_dim: usize, n_heads: usize, n_kv_heads: usize) {
        let half = head_dim / 2;
        for h in 0..n_heads {
            let start = h * head_dim;
            for i in 0..half {
                let freq = 1.0 / 10000.0f32.powf((i * 2) as f32 / head_dim as f32);
                let theta = (pos as f32) * freq;
                let cos = theta.cos(); let sin = theta.sin();
                
                let q_i = q[start + i];
                let q_half = q[start + i + half];
                
                q[start + i] = q_i * cos - q_half * sin;
                q[start + i + half] = q_i * sin + q_half * cos;
            }
        }
        for h in 0..n_kv_heads {
            let start = h * head_dim;
            for i in 0..half {
                let freq = 1.0 / 10000.0f32.powf((i * 2) as f32 / head_dim as f32);
                let theta = (pos as f32) * freq;
                let cos = theta.cos(); let sin = theta.sin();
                
                let k_i = k[start + i];
                let k_half = k[start + i + half];
                
                k[start + i] = k_i * cos - k_half * sin;
                k[start + i + half] = k_i * sin + k_half * cos;
            }
        }
    }
}
