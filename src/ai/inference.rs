use crate::ai::MudFile;
use crate::ai::routing::MudRouter;
use std::sync::{Arc, RwLock};
use crate::vulkan::VulkanContext;
use crate::ai::skills::MudSkill;
use crate::model::tokenizer::Tokenizer;
use crate::ai::graph::MudKnowledgeGraph;
use crate::ai::store::MudStore;
use rand::RngExt;

/// A Ternary MoE Expert using SwiGLU activation.
pub struct MudExpert {
    pub w1: *const u32, // Gate projection
    pub w2: *const u32, // Down projection
    pub w3: *const u32, // Up projection
}

/// A Ternary MoE Layer.
pub struct MudMoELayer {
    pub experts: Vec<MudExpert>,
    pub router: MudRouter,
    pub attn_q_w: *const u32,
    pub attn_k_w: *const u32,
    pub attn_v_w: *const u32,
    pub attn_o_w: *const u32,
    pub gate_w: *const u32,
    pub norm_w: *const f32,
    pub attn_norm_w: *const f32,
}

pub struct MudModel {
    pub layers: Vec<MudMoELayer>,
    pub knowledge_graph: Arc<RwLock<MudKnowledgeGraph>>,
    pub hidden_size: usize,
    pub ffn_hidden_size: usize,
    pub num_experts: usize,
}

pub struct MudInference {
    pub model: MudModel,
    pub vulkan_ctx: Arc<VulkanContext>,
    pub embd_w: *const u32,
    pub output_norm_w: *const f32,
    pub skills: Vec<Box<dyn MudSkill>>,
    pub tokenizer: Tokenizer,
    pub store: Arc<MudStore>,
    /// Working Memory: Key Cache
    pub kv_cache_k: Vec<f32>,
    /// Working Memory: Value Cache
    pub kv_cache_v: Vec<f32>,
}

impl MudInference {
    pub fn new(mud_file: &MudFile, vk_ctx: Arc<VulkanContext>) -> anyhow::Result<Self> {
        let core = mud_file.skills.get("core").ok_or_else(|| anyhow::anyhow!("No core skill found"))?;
        
        let store = Arc::new(MudStore::open("models/knowledge.db")?);

        // Standalone MUD Tokenizer
        let tokens_str = mud_file.global_metadata.get("tokenizer.tokens")
            .ok_or_else(|| anyhow::anyhow!("No tokenizer tokens in MUD metadata"))?;
        let merges_str = mud_file.global_metadata.get("tokenizer.merges").map(|s| s.as_str()).unwrap_or("");
        let tokenizer = crate::model::tokenizer::Tokenizer::from_mud_metadata(tokens_str, merges_str);

        let hidden_size = mud_file.global_metadata.get("hidden_size")
            .and_then(|s| s.parse::<usize>().ok()).unwrap_or(512);
        let num_layers = mud_file.global_metadata.get("num_layers")
            .and_then(|s| s.parse::<usize>().ok()).unwrap_or(12);
        let num_experts = mud_file.global_metadata.get("num_experts")
            .and_then(|s| s.parse::<usize>().ok()).unwrap_or(4);
        let ffn_hidden = mud_file.global_metadata.get("ffn_hidden")
            .and_then(|s| s.parse::<usize>().ok()).unwrap_or(1024);

        // Allocate KV Cache (2048 tokens context window)
        let max_seq_len = 2048;
        let kv_size = num_layers * max_seq_len * hidden_size;
        let kv_cache_k = vec![0.0f32; kv_size];
        let kv_cache_v = vec![0.0f32; kv_size];

        let mut layers = Vec::with_capacity(num_layers);
        for l in 0..num_layers {
            let mut experts = Vec::with_capacity(num_experts);
            for e in 0..num_experts {
                let w1 = core.tensors.get(&format!("blk.{}.expert.{}.w1.weight", l, e)).map(|t| t.data_ptr as *const u32).unwrap_or(std::ptr::null());
                let w2 = core.tensors.get(&format!("blk.{}.expert.{}.w2.weight", l, e)).map(|t| t.data_ptr as *const u32).unwrap_or(std::ptr::null());
                let w3 = core.tensors.get(&format!("blk.{}.expert.{}.w3.weight", l, e)).map(|t| t.data_ptr as *const u32).unwrap_or(std::ptr::null());
                experts.push(MudExpert { w1, w2, w3 });
            }

            layers.push(MudMoELayer {
                experts,
                router: MudRouter::new(num_experts, 2),
                attn_q_w: core.tensors.get(&format!("blk.{}.attn_q.weight", l)).map(|t| t.data_ptr as *const u32).unwrap_or(std::ptr::null()),
                attn_k_w: core.tensors.get(&format!("blk.{}.attn_k.weight", l)).map(|t| t.data_ptr as *const u32).unwrap_or(std::ptr::null()),
                attn_v_w: core.tensors.get(&format!("blk.{}.attn_v.weight", l)).map(|t| t.data_ptr as *const u32).unwrap_or(std::ptr::null()),
                attn_o_w: core.tensors.get(&format!("blk.{}.attn_output.weight", l)).map(|t| t.data_ptr as *const u32).unwrap_or(std::ptr::null()),
                gate_w: core.tensors.get(&format!("blk.{}.gate.weight", l)).map(|t| t.data_ptr as *const u32).unwrap_or(std::ptr::null()),
                norm_w: core.tensors.get(&format!("blk.{}.norm.weight", l)).map(|t| t.data_ptr as *const f32).unwrap_or(std::ptr::null()),
                attn_norm_w: core.tensors.get(&format!("blk.{}.attn_norm.weight", l)).map(|t| t.data_ptr as *const f32).unwrap_or(std::ptr::null()),
            });
        }

        let skills: Vec<Box<dyn MudSkill>> = vec![
            Box::new(crate::ai::skills::autoformatter::AutoformatterSkill::new()),
            Box::new(crate::ai::skills::logic_math::LogicMathSkill::new()),
            Box::new(crate::ai::skills::retrieval::RetrievalSkill::new()),
            Box::new(crate::ai::skills::language::LanguageSkill::new("es")),
            Box::new(crate::ai::skills::translator::TranslationSkill::new("en")),
            Box::new(crate::ai::skills::personality::PersonalitySkill::new("Forge Assistant")),
            Box::new(crate::ai::skills::memory::MemorySkill::new()),
            Box::new(crate::ai::skills::learning::LearningSkill::new()),
            Box::new(crate::ai::skills::data_analysis::DataAnalysisSkill::new()),
            Box::new(crate::ai::skills::plotting::PlottingSkill::new()),
            Box::new(crate::ai::skills::web_search::WebSearchSkill::new()),
        ];

        let mut graph = MudKnowledgeGraph::new();
        // Dynamic Loading Strategy (The Google Algorithm): 
        // Only load the top 200 most important 'hubs' into RAM to avoid memory collapse.
        if let Ok(hubs) = store.get_top_hubs(200) {
            for (content, emb, rank) in hubs {
                graph.add_node(content, emb);
                if let Some(&idx) = graph.content_to_index.get(&graph.nodes.last().unwrap().content) {
                    graph.nodes[idx].rank = rank;
                }
            }
            if !graph.nodes.is_empty() {
                println!("  [MUD] Loaded {} top hubs from knowledge store.", graph.nodes.len());
            }
        }
        // Fallback for cold start
        if graph.nodes.is_empty() {
            graph.add_node("MUD stands for Modular Understanding Dynamics.".to_string(), vec![0.1; hidden_size]);
            graph.add_node("The Forge engine uses Ternary 1.58-bit weights.".to_string(), vec![0.2; hidden_size]);
        }
        let knowledge_graph = Arc::new(RwLock::new(graph));

        let output_norm_w = core.tensors.get("output_norm.weight").map(|t| t.data_ptr as *const f32).unwrap_or(std::ptr::null());

        Ok(Self {
            model: MudModel { layers, knowledge_graph, hidden_size, ffn_hidden_size: ffn_hidden, num_experts },
            vulkan_ctx: vk_ctx,
            embd_w: core.tensors.get("token_embd.weight").map(|t| t.data_ptr as *const u32).unwrap_or(std::ptr::null()),
            output_norm_w,
            skills,
            tokenizer,
            store,
            kv_cache_k,
            kv_cache_v,
        })
    }

    /// Performs forward step with KV Cache (Attention).
    pub fn step(&self, x: &mut [f32], _context: &str, active_skills: &[&Box<dyn MudSkill>], pos: usize) {
        for skill in active_skills { skill.pre_process(x); }

        let hidden = self.model.hidden_size;
        let kv_layer_offset = 2048 * hidden;

        for (l, layer) in self.model.layers.iter().enumerate() {
            let scale_attn = unsafe { crate::asm::rms_norm_scale_asm(hidden, x.as_ptr(), 1e-6) };
            let mut x_norm = vec![0.0f32; hidden];
            if !layer.attn_norm_w.is_null() {
                unsafe { for i in 0..hidden { x_norm[i] = x[i] * scale_attn * (*layer.attn_norm_w.add(i)); } }
            } else {
                unsafe { for i in 0..hidden { x_norm[i] = x[i] * scale_attn * (*layer.norm_w.add(i)); } }
            }

            // --- 1. ATTENTION (Working Memory) ---
            let mut q = vec![0.0f32; hidden];
            let mut k = vec![0.0f32; hidden];
            let mut v = vec![0.0f32; hidden];

            if !layer.attn_q_w.is_null() {
                self.vulkan_ctx.run_ternary_gemv(hidden, hidden, &x_norm, layer.attn_q_w, 1.0, &mut q).unwrap();
                self.vulkan_ctx.run_ternary_gemv(hidden, hidden, &x_norm, layer.attn_k_w, 1.0, &mut k).unwrap();
                self.vulkan_ctx.run_ternary_gemv(hidden, hidden, &x_norm, layer.attn_v_w, 1.0, &mut v).unwrap();
            } else {
                q.copy_from_slice(&x_norm);
                k.copy_from_slice(&x_norm);
                v.copy_from_slice(&x_norm);
            }

            let head_size = 64;
            let n_heads = hidden / head_size;
            let max_seq_len = 2048;
            let cache_idx = pos % max_seq_len;
            let num_tokens_in_cache = std::cmp::min(pos + 1, max_seq_len);

            // Apply RoPE
            for h in 0..n_heads {
                let h_start = h * head_size;
                for i in (0..head_size).step_by(2) {
                    let freq = 1.0 / 10000.0f32.powf(i as f32 / head_size as f32);
                    let val = pos as f32 * freq;
                    let cos_val = val.cos();
                    let sin_val = val.sin();

                    let q0 = q[h_start + i];
                    let q1 = q[h_start + i + 1];
                    q[h_start + i] = q0 * cos_val - q1 * sin_val;
                    q[h_start + i + 1] = q0 * sin_val + q1 * cos_val;

                    let k0 = k[h_start + i];
                    let k1 = k[h_start + i + 1];
                    k[h_start + i] = k0 * cos_val - k1 * sin_val;
                    k[h_start + i + 1] = k0 * sin_val + k1 * cos_val;
                }
            }

            let offset = (l * kv_layer_offset) + (cache_idx * hidden);
            unsafe {
                let k_ptr = self.kv_cache_k.as_ptr() as *mut f32;
                let v_ptr = self.kv_cache_v.as_ptr() as *mut f32;
                std::ptr::copy_nonoverlapping(k.as_ptr(), k_ptr.add(offset), hidden);
                std::ptr::copy_nonoverlapping(v.as_ptr(), v_ptr.add(offset), hidden);
            }

            let mut attn_out = vec![0.0f32; hidden];
            let att_scale = 1.0 / (head_size as f32).sqrt();

            for h in 0..n_heads {
                let h_start = h * head_size;
                let q_h = &q[h_start..h_start + head_size];
                let mut scores = vec![0.0f32; num_tokens_in_cache];

                for p in 0..num_tokens_in_cache {
                    let k_idx = (l * kv_layer_offset) + (p * hidden) + h_start;
                    let k_h = &self.kv_cache_k[k_idx..k_idx + head_size];
                    let mut score = 0.0f32;
                    for i in 0..head_size { score += q_h[i] * k_h[i]; }
                    scores[p] = score * att_scale;
                }

                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                for s in scores.iter_mut() { *s = (*s - max_s).exp(); sum_exp += *s; }
                for s in scores.iter_mut() { *s /= sum_exp; }

                for p in 0..num_tokens_in_cache {
                    let v_idx = (l * kv_layer_offset) + (p * hidden) + h_start;
                    let v_h = &self.kv_cache_v[v_idx..v_idx + head_size];
                    for i in 0..head_size { attn_out[h_start + i] += scores[p] * v_h[i]; }
                }
            }
            
            let mut final_attn_out = vec![0.0f32; hidden];
            if !layer.attn_o_w.is_null() {
                self.vulkan_ctx.run_ternary_gemv(hidden, hidden, &attn_out, layer.attn_o_w, 1.0, &mut final_attn_out).unwrap();
            } else {
                final_attn_out = attn_out;
            }
            
            for i in 0..hidden { x[i] += final_attn_out[i]; }

            // --- 2. MOE EXPERTS (with fresh pre-norm after attention residual) ---
            let scale_moe = unsafe { crate::asm::rms_norm_scale_asm(hidden, x.as_ptr(), 1e-6) };
            let mut x_moe_norm = vec![0.0f32; hidden];
            unsafe { for i in 0..hidden { x_moe_norm[i] = x[i] * scale_moe * (*layer.norm_w.add(i)); } }
            let mut gate_logits = vec![0.0f32; self.model.num_experts];
            self.vulkan_ctx.run_ternary_gemv(hidden, self.model.num_experts, &x_moe_norm, layer.gate_w as *const u32, 1.0, &mut gate_logits).unwrap();

            for skill in active_skills { skill.route_bias(&mut gate_logits); }
            let routing = layer.router.route(&gate_logits);
            
            let mut combined_expert_out = vec![0.0f32; hidden];
            for (expert_id, prob) in routing {
                let expert = &layer.experts[expert_id];
                let mut expert_w1 = vec![0.0f32; self.model.ffn_hidden_size];
                let mut expert_w3 = vec![0.0f32; self.model.ffn_hidden_size];
                self.vulkan_ctx.run_ternary_gemv(hidden, self.model.ffn_hidden_size, &x_moe_norm, expert.w1 as *const u32, 1.0, &mut expert_w1).unwrap();
                self.vulkan_ctx.run_ternary_gemv(hidden, self.model.ffn_hidden_size, &x_moe_norm, expert.w3 as *const u32, 1.0, &mut expert_w3).unwrap();

                for i in 0..self.model.ffn_hidden_size {
                    let silu = expert_w1[i] * (1.0 / (1.0 + (-expert_w1[i]).exp()));
                    expert_w1[i] = silu * expert_w3[i];
                }
                let mut expert_final = vec![0.0f32; hidden];
                self.vulkan_ctx.run_ternary_gemv(self.model.ffn_hidden_size, hidden, &expert_w1, expert.w2 as *const u32, 1.0, &mut expert_final).unwrap();

                for i in 0..hidden { combined_expert_out[i] += expert_final[i] * prob; }
            }
            let damping = 1.0 / (self.model.layers.len() as f32).sqrt();
            for i in 0..hidden { x[i] += combined_expert_out[i] * damping; }
        }

        // Final output RMSNorm
        if !self.output_norm_w.is_null() {
            let scale_out = unsafe { crate::asm::rms_norm_scale_asm(hidden, x.as_ptr(), 1e-6) };
            unsafe { for i in 0..hidden { x[i] = x[i] * scale_out * (*self.output_norm_w.add(i)); } }
        }
    }

    pub fn embed_token(&self, id: u32, x: &mut [f32]) {
        let row_ptr = unsafe { self.embd_w.add(id as usize * (self.model.hidden_size / 16)) };
        crate::ai::dequantize_ternary_row(row_ptr, x, self.model.hidden_size);
    }

    pub fn generate(&self, x_init: &[f32], max_tokens: usize, context: &str, conversation_pos: &mut usize) -> Vec<u32> {
        let active_skills: Vec<&Box<dyn MudSkill>> = self.skills.iter()
            .filter(|s| s.should_activate(x_init, context))
            .collect();

        // 1. Autonomous Knowledge Retrieval (The Google & Bridge Algorithm)
        let retrieved_facts = self.model.knowledge_graph.write().unwrap()
            .autonomous_jump_search(x_init, &self.store, 2);
        
        for fact in &retrieved_facts { println!("  [MKG Autonomous Retrieval] Found: {}", fact); }
        for skill in &active_skills { skill.execute_autonomous_action(context, self); }

        let mut x = x_init.to_vec();
        let mut results = Vec::new();
        let vocab_size = self.tokenizer.id_to_token.len();
        // SEARCH OPTIMIZATION: Removed arbitrary limit, searching full vocabulary
        let search_limit = vocab_size;

        for _ in 0..max_tokens {
            self.step(&mut x, context, &active_skills, *conversation_pos);
            *conversation_pos += 1;
            let mut logits = vec![f32::NEG_INFINITY; vocab_size];
            
            let mut row_buf = vec![0.0f32; self.model.hidden_size];
            for i in 0..search_limit {
                let row_ptr = unsafe { self.embd_w.add(i * (self.model.hidden_size / 16)) };
                crate::ai::dequantize_ternary_row(row_ptr, &mut row_buf, self.model.hidden_size);
                let mut dot = 0.0f32;
                for j in 0..self.model.hidden_size { dot += x[j] * row_buf[j]; }
                logits[i] = dot;
            }
            
            // DYNAMIC REPETITION PENALTY (Exponential)
            for (idx, &prev_id) in results.iter().rev().enumerate() {
                if (prev_id as usize) < logits.len() {
                    let decay = 1.0 / (idx as f32 + 1.0);
                    logits[prev_id as usize] -= 20.0 * decay; 
                }
            }
            
            let temperature = 0.7;
            let top_k = 40;
            let top_p = 0.9;
            
            let mut probs: Vec<(usize, f32)> = logits.iter().enumerate()
                .filter(|(_, &v)| !v.is_nan())
                .map(|(i, &l)| (i, l / temperature))
                .collect();
                
            // Softmax
            let max_logit = probs.iter().fold(f32::NEG_INFINITY, |m, &(_, v)| m.max(v));
            let mut sum_exp = 0.0;
            for p in &mut probs {
                p.1 = (p.1 - max_logit).exp();
                sum_exp += p.1;
            }
            for p in &mut probs {
                p.1 /= sum_exp;
            }
            
            // Top-K
            probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            if probs.len() > top_k {
                probs.truncate(top_k);
            }
            
            // Top-P
            let mut cum_prob = 0.0;
            let mut last_idx = probs.len();
            for (i, p) in probs.iter().enumerate() {
                cum_prob += p.1;
                if cum_prob > top_p {
                    last_idx = i + 1;
                    break;
                }
            }
            probs.truncate(last_idx);
            
            // Re-normalize after Top-P / Top-K
            let sum_prob: f32 = probs.iter().map(|p| p.1).sum();
            for p in &mut probs {
                p.1 /= sum_prob;
            }
            
            // Sample
            let mut rng = rand::rng();
            let r: f32 = rng.random();
            let mut cum = 0.0;
            let mut next_id = probs.first().map(|p| p.0 as u32).unwrap_or(0);
            for p in probs {
                cum += p.1;
                if r <= cum {
                    next_id = p.0 as u32;
                    break;
                }
            }
            
            if next_id == 0 { break; } 
            results.push(next_id);
            self.embed_token(next_id, &mut x);
        }
        results
    }

    pub fn format_text(&self, text: &mut String) {
        for skill in &self.skills { skill.post_process_token(text); }
    }
}
