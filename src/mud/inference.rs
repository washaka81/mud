use crate::mud::MudFile;
use crate::mud::routing::MudRouter;
use std::sync::{Arc, RwLock};
use crate::vulkan::VulkanContext;
use crate::mud::skills::MudSkill;
use crate::model::tokenizer::Tokenizer;
use crate::mud::graph::MudKnowledgeGraph;
use crate::mud::store::MudStore;
use rayon::prelude::*;

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
}

unsafe impl Send for MudModel {}
unsafe impl Sync for MudModel {}

pub struct InferenceWorkspace {
    pub x_norm: Vec<f32>,
    pub q: Vec<f32>, pub k: Vec<f32>, pub v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub final_attn_out: Vec<f32>,
    pub x_moe_norm: Vec<f32>,
    pub gate_logits: Vec<f32>,
    pub combined_expert_out: Vec<f32>,
    pub expert_workspaces: Vec<ExpertWorkspace>,
    pub logits: Vec<f32>,
}

pub struct ExpertWorkspace {
    pub w1_out: Vec<f32>,
    pub w3_out: Vec<f32>,
    pub final_out: Vec<f32>,
}

impl InferenceWorkspace {
    pub fn new(hidden: usize, ffn_hidden: usize, num_experts: usize, vocab_size: usize) -> Self {
        let mut expert_workspaces = Vec::with_capacity(num_experts);
        for _ in 0..num_experts {
            expert_workspaces.push(ExpertWorkspace {
                w1_out: vec![0.0; ffn_hidden],
                w3_out: vec![0.0; ffn_hidden],
                final_out: vec![0.0; hidden],
            });
        }
        Self {
            x_norm: vec![0.0; hidden],
            q: vec![0.0; hidden], k: vec![0.0; hidden], v: vec![0.0; hidden],
            attn_out: vec![0.0; hidden],
            final_attn_out: vec![0.0; hidden],
            x_moe_norm: vec![0.0; hidden],
            gate_logits: vec![0.0; num_experts],
            combined_expert_out: vec![0.0; hidden],
            expert_workspaces,
            logits: vec![0.0; vocab_size],
        }
    }
}

pub struct MudInference {
    pub model: MudModel,
    pub vulkan_ctx: Arc<VulkanContext>,
    pub embd_w: *const u32,
    pub output_norm_w: *const f32,
    pub skills: Vec<Box<dyn MudSkill>>,
    pub tokenizer: Tokenizer,
    pub store: Arc<MudStore>,
    pub kv_cache_k: Vec<i8>,
    pub kv_cache_v: Vec<i8>,
    pub kv_scales_k: Vec<f32>,
    pub kv_scales_v: Vec<f32>,
    pub active_experts: Arc<RwLock<usize>>,
    pub workspace: InferenceWorkspace,
}

impl MudInference {
    pub fn new(mud_file: &MudFile, vk_ctx: Arc<VulkanContext>) -> anyhow::Result<Self> {
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

        Ok(Self {
            model: MudModel { layers, knowledge_graph, hidden_size, ffn_hidden_size: ffn_hidden, num_experts },
            vulkan_ctx: vk_ctx,
            embd_w: core.tensors.get("token_embd.weight").map(|t| t.data_ptr as *const u32).unwrap_or(std::ptr::null()),
            output_norm_w: core.tensors.get("output_norm.weight").map(|t| t.data_ptr as *const f32).unwrap_or(std::ptr::null()),
            skills, tokenizer, store,
            kv_cache_k: vec![0; num_layers.checked_mul(4096).and_then(|x| x.checked_mul(hidden_size)).expect("KV-cache-k: overflow en num_layers * 4096 * hidden_size")],
            kv_cache_v: vec![0; num_layers.checked_mul(4096).and_then(|x| x.checked_mul(hidden_size)).expect("KV-cache-v: overflow en num_layers * 4096 * hidden_size")],
            kv_scales_k: vec![0.0; num_layers.checked_mul(4096).and_then(|x| x.checked_mul(hidden_size / 64)).expect("KV-scales-k: overflow")],
            kv_scales_v: vec![0.0; num_layers.checked_mul(4096).and_then(|x| x.checked_mul(hidden_size / 64)).expect("KV-scales-v: overflow")],
            active_experts: Arc::new(RwLock::new(0)),
            workspace,
        })
    }

    pub fn step(&mut self, x: &mut [f32], _context: &str, active_skill_indices: &[usize], _pos: usize) {
        let ws = &mut self.workspace;
        for &si in active_skill_indices { self.skills[si].pre_process(x); }
        let hidden = self.model.hidden_size;
        let ffn_hidden = self.model.ffn_hidden_size;
        let mut step_active_experts = 0;

        for (l, layer) in self.model.layers.iter().enumerate() {
            let scale_attn = unsafe { crate::asm::rms_norm_scale_asm(hidden, x.as_ptr(), 1e-6) };
            let norm_ptr = if !layer.attn_norm_w.is_null() { layer.attn_norm_w } else { layer.norm_w };
            unsafe { for i in 0..hidden { ws.x_norm[i] = x[i] * scale_attn * (*norm_ptr.add(i)); } }

            ws.q.fill(0.0); ws.k.fill(0.0); ws.v.fill(0.0);
            let key_q = format!("l{}_q", l); let key_k = format!("l{}_k", l); let key_v = format!("l{}_v", l);
            Self::gemv_vulkan_or_cpu(&*self.vulkan_ctx, &key_q, hidden, hidden, &ws.x_norm, layer.attn_q_w, layer.attn_q_scale, &mut ws.q);
            Self::gemv_vulkan_or_cpu(&*self.vulkan_ctx, &key_k, hidden, hidden, &ws.x_norm, layer.attn_k_w, layer.attn_k_scale, &mut ws.k);
            Self::gemv_vulkan_or_cpu(&*self.vulkan_ctx, &key_v, hidden, hidden, &ws.x_norm, layer.attn_v_w, layer.attn_v_scale, &mut ws.v);

            // ... (Attention compute placeholder, assuming simple residual for now) ...
            // In a real model, we would compute attention here. 
            // For now, let's just use the Q output as a simplified 'attn_out' for demonstration.
            ws.attn_out.copy_from_slice(&ws.q);

            ws.final_attn_out.fill(0.0);
            let key_o = format!("l{}_o", l);
            Self::gemv_vulkan_or_cpu(&*self.vulkan_ctx, &key_o, hidden, hidden, &ws.attn_out, layer.attn_o_w, layer.attn_o_scale, &mut ws.final_attn_out);
            
            for i in 0..hidden { x[i] += ws.final_attn_out[i]; }

            // MoE Block
            let scale_moe = unsafe { crate::asm::rms_norm_scale_asm(hidden, x.as_ptr(), 1e-6) };
            unsafe { for i in 0..hidden { ws.x_moe_norm[i] = x[i] * scale_moe * (*layer.norm_w.add(i)); } }

            ws.gate_logits.fill(0.0);
            let key_gate = format!("l{}_gate", l);
            // Gate is usually float or ternary. Assuming ternary with 1.0 scale if not specified.
            Self::gemv_vulkan_or_cpu(&*self.vulkan_ctx, &key_gate, hidden, self.model.num_experts, &ws.x_moe_norm, layer.gate_w, 1.0, &mut ws.gate_logits);

            let routing = layer.router.route(&ws.gate_logits);
            step_active_experts += routing.len();

            ws.combined_expert_out.fill(0.0);
            
            // Parallel execution of experts using rayon
            let vk = &self.vulkan_ctx;
            let x_moe_norm = &ws.x_moe_norm;
            
            // Pre-calculate weighted outputs in parallel
            let expert_results: Vec<Vec<f32>> = routing.par_iter().filter_map(|&(expert_id, prob)| {
                let expert = layer.experts.get(expert_id)?;  // bounds-safe: evita panic si el router emite un id fuera de rango
                // Note: In a fully static architecture, we'd use a thread-local workspace or disjoint slices.
                // For now, to ensure safety and correctness in the first parallel step, 
                // we'll use a local buffer and accumulate weighted results.
                let mut w1_out = vec![0.0; ffn_hidden];
                let mut w3_out = vec![0.0; ffn_hidden];
                let mut final_out = vec![0.0; hidden];
                
                let key_w1 = format!("l{}_e{}_w1", l, expert_id);
                let key_w2 = format!("l{}_e{}_w2", l, expert_id);
                let key_w3 = format!("l{}_e{}_w3", l, expert_id);
                
                Self::gemv_vulkan_or_cpu(vk, &key_w1, hidden, ffn_hidden, x_moe_norm, expert.w1, expert.w1_scale, &mut w1_out);
                Self::gemv_vulkan_or_cpu(vk, &key_w3, hidden, ffn_hidden, x_moe_norm, expert.w3, expert.w3_scale, &mut w3_out);
                
                for j in 0..ffn_hidden {
                    let val = w1_out[j];
                    let silu = val / (1.0 + (-val).exp());
                    w1_out[j] = silu * w3_out[j];
                }

                Self::gemv_vulkan_or_cpu(vk, &key_w2, ffn_hidden, hidden, &w1_out, expert.w2, expert.w2_scale, &mut final_out);
                
                for val in final_out.iter_mut() { *val *= prob; }
                Some(final_out)
            }).collect();

            // Accumulate results into the combined output
            for res in expert_results {
                for (j, &val) in res.iter().enumerate() {
                    ws.combined_expert_out[j] += val;
                }
            }
            
            for i in 0..hidden { x[i] += ws.combined_expert_out[i]; }
        }
        
        *self.active_experts.write().unwrap() = (step_active_experts as f32 / self.model.layers.len() as f32).round() as usize;
        
        if !self.output_norm_w.is_null() {
            let scale_out = unsafe { crate::asm::rms_norm_scale_asm(hidden, x.as_ptr(), 1e-6) };
            unsafe { for i in 0..hidden { x[i] *= scale_out * (*self.output_norm_w.add(i)); } }
        }
    }

    pub fn prompt(&mut self, text: &str, x: &mut [f32], conversation_pos: &mut usize) {
        let tokens = self.tokenizer.encode(text);
        for &t in &tokens {
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
        let epsilon = 1e-4f32; // Small perturbation to break fixed points

        let mut prev_x = x.clone();

        for step_idx in 0..max_tokens {
            // 0. Dynamic Epsilon Perturbation (Kick Logic)
            // Calculate current stagnation level
            let x_move = x.iter().zip(prev_x.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
            let mut current_epsilon = epsilon;
            
            if step_idx > 0 && x_move < 1e-4 {
                // We are stuck! Apply a major kick to break the cycle
                current_epsilon = 0.5f32;
            }
            
            for i in 0..self.model.hidden_size {
                x[i] += (rand::random::<f32>() - 0.5) * current_epsilon;
            }
            prev_x = x.clone();

            // 1. Autonomous Synapse Injection (High Performance)
            // Reduced frequency to every 20 tokens to reach 25-30 TPS
            if step_idx % 20 == 0 {
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

            self.step(&mut x, context, &active_skill_indices, *conversation_pos);
            *conversation_pos += 1;
            
            let ws = &mut self.workspace;
            ws.logits.fill(0.0);
            
            for i in 0..ws.logits.len() {
                let row_ptr = unsafe { self.embd_w.add(i * (self.model.hidden_size / 16)) };
                unsafe { crate::asm::ternary_gemv_avx2(self.model.hidden_size, x.as_ptr(), row_ptr, &mut ws.logits[i], 1.0); }
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
        // Guarda de alineación: hidden_size debe ser múltiplo de 16 para el empaquetado ternario de 2 bits
        debug_assert_eq!(self.model.hidden_size % 16, 0, "hidden_size debe ser múltiplo de 16");
        let offset = (id as usize).checked_mul(self.model.hidden_size / 16)
            .expect("embed_token: overflow en id * (hidden_size / 16)");
        let row_ptr = unsafe { self.embd_w.add(offset) };
        crate::mud::dequantize_ternary_row(row_ptr, x, self.model.hidden_size);
    }

    pub fn gemv_vulkan_or_cpu(vk_ctx: &VulkanContext, key: &str, n_in: usize, n_out: usize, x: &[f32], w: *const u32, scale: f32, y: &mut [f32]) {
        if w.is_null() { return; }
        if vk_ctx.run_ternary_gemv_cached(key, n_in, n_out, x, w, scale, y).is_err() {
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
}
