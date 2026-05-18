use crate::gguf::{GGUFModel, MetadataValue};
use crate::model::tokenizer::Tokenizer;
use crate::model::transformer::{ForgeModel, TransformerLayer};
use crate::asm::{BlockQ4_0, dequantize_q4_0_row};
use crate::vulkan::VulkanContext;
use std::sync::Arc;

/// Engine for managing the high-level inference process.
/// Handles tokenizer interaction, KV cache management, and the generation loop.
pub struct ForgeInference {
    /// The core Transformer model.
    pub model: ForgeModel,
    /// Tokenizer for encoding/decoding text.
    pub tokenizer: Tokenizer,
    /// Key-value cache for the K vectors.
    pub kv_cache_k: Vec<f32>,
    /// Key-value cache for the V vectors.
    pub kv_cache_v: Vec<f32>,
    /// Pointer to the embedding weights.
    pub embd_w: *const BlockQ4_0,
    /// Pointer to the output projection weights (un-embedding).
    pub output_w: *const BlockQ4_0,
    /// Pointer to the final normalization weights.
    pub output_norm_w: *const f32,
    /// Maximum sequence length supported by the cache.
    pub max_seq_len: usize,
}

impl ForgeInference {
    /// Initializes a new inference engine from a loaded GGUF model.
    /// Allocates the KV cache and maps all necessary tensors.
    pub fn new(gguf: &GGUFModel, vk_ctx: Arc<VulkanContext>) -> anyhow::Result<Self> {
        let tokenizer = Tokenizer::from_gguf(gguf)?;
        
        let hidden_size = extract_u32(gguf, "MUD2.embedding_length")? as usize;
        let n_layers = extract_u32(gguf, "MUD2.block_count")? as usize;
        let n_heads = extract_u32(gguf, "MUD2.attention.head_count")? as usize;
        let n_kv_heads = extract_u32(gguf, "MUD2.attention.head_count_kv")? as usize;
        let head_size = hidden_size / n_heads;
        
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            layers.push(TransformerLayer {
                attn_q_w: gguf.get_tensor_q4_0(&format!("blk.{}.attn_q.weight", i)).unwrap_or(std::ptr::null()),
                attn_q_b: gguf.tensors.get(&format!("blk.{}.attn_q.bias", i)).map(|t| t.data_ptr as *const f32).unwrap_or(std::ptr::null()),
                attn_k_w: gguf.get_tensor_q4_0(&format!("blk.{}.attn_k.weight", i)).unwrap_or(std::ptr::null()),
                attn_k_b: gguf.tensors.get(&format!("blk.{}.attn_k.bias", i)).map(|t| t.data_ptr as *const f32).unwrap_or(std::ptr::null()),
                attn_v_w: gguf.get_tensor_q4_0(&format!("blk.{}.attn_v.weight", i)).unwrap_or(std::ptr::null()),
                attn_v_b: gguf.tensors.get(&format!("blk.{}.attn_v.bias", i)).map(|t| t.data_ptr as *const f32).unwrap_or(std::ptr::null()),
                attn_o_w: gguf.get_tensor_q4_0(&format!("blk.{}.attn_output.weight", i)).unwrap_or(std::ptr::null()),
                attn_o_b: gguf.tensors.get(&format!("blk.{}.attn_output.bias", i)).map(|t| t.data_ptr as *const f32).unwrap_or(std::ptr::null()),
                
                ffn_gate_w: gguf.get_tensor_q4_0(&format!("blk.{}.ffn_gate.weight", i)).unwrap_or(std::ptr::null()),
                ffn_up_w: gguf.get_tensor_q4_0(&format!("blk.{}.ffn_up.weight", i)).unwrap_or(std::ptr::null()),
                ffn_down_w: gguf.get_tensor_q4_0(&format!("blk.{}.ffn_down.weight", i)).unwrap_or(std::ptr::null()),
                
                attn_norm_w: gguf.tensors.get(&format!("blk.{}.attn_norm.weight", i)).map(|t| t.data_ptr as *const f32).unwrap_or(std::ptr::null()),
                ffn_norm_w: gguf.tensors.get(&format!("blk.{}.ffn_norm.weight", i)).map(|t| t.data_ptr as *const f32).unwrap_or(std::ptr::null()),
                rms_norm_eps: 1e-6,
            });
        }
        
        let model = ForgeModel {
            layers,
            vulkan_ctx: vk_ctx,
            hidden_size,
            ffn_hidden_size: extract_u32(gguf, "MUD2.feed_forward_length").unwrap_or((hidden_size * 4) as u32) as usize,
            n_heads,
            n_kv_heads,
            head_size,
        };
        
        let max_seq_len = 2048;
        let kv_cache_size = n_layers * max_seq_len * n_kv_heads * head_size;
        
        Ok(Self {
            model,
            tokenizer,
            kv_cache_k: vec![0.0; kv_cache_size],
            kv_cache_v: vec![0.0; kv_cache_size],
            embd_w: gguf.get_tensor_q4_0("token_embd.weight").unwrap_or(std::ptr::null()),
            output_w: gguf.get_tensor_q4_0("output.weight").unwrap_or(std::ptr::null()),
            output_norm_w: gguf.tensors.get("output_norm.weight").map(|t| t.data_ptr as *const f32).unwrap_or(std::ptr::null()),
            max_seq_len,
        })
    }

    /// Generates text from a given prompt.
    /// Implements autoregressive decoding with Top-P sampling and repetition penalty.
    pub fn generate(&mut self, prompt: &str, max_new_tokens: usize) -> anyhow::Result<()> {
        let prompt_tokens = self.tokenizer.encode(prompt);
        let mut tokens = Vec::new();
        
        let mut x = vec![0.0f32; self.model.hidden_size];
        let vocab_size = self.tokenizer.id_to_token.len();
        
        println!("Procesando prompt ({} tokens)...", prompt_tokens.len());
        
        // 1. PROCESS PROMPT (Except the last token)
        for (pos, &token) in prompt_tokens.iter().enumerate().take(prompt_tokens.len() - 1) {
            let row_ptr = unsafe { self.embd_w.add(token as usize * (self.model.hidden_size / 32)) };
            dequantize_q4_0_row(row_ptr, &mut x, self.model.hidden_size);
            self.model.decode_step(&mut x, pos, &mut self.kv_cache_k, &mut self.kv_cache_v);
            tokens.push(token);
        }

        // 2. PROCESS LAST PROMPT TOKEN (Prepares state 'x' for the first prediction)
        let last_prompt_token = *prompt_tokens.last().unwrap();
        let row_ptr = unsafe { self.embd_w.add(last_prompt_token as usize * (self.model.hidden_size / 32)) };
        dequantize_q4_0_row(row_ptr, &mut x, self.model.hidden_size);
        let last_pos = prompt_tokens.len() - 1;
        self.model.decode_step(&mut x, last_pos, &mut self.kv_cache_k, &mut self.kv_cache_v);
        tokens.push(last_prompt_token);
        
        println!("Iniciando generación...");
        
        for pos_offset in 1..=max_new_tokens {
            let pos = prompt_tokens.len() + pos_offset - 1;
            
            // 3. LOGITS & SAMPLING (Using the 'x' from previous step)
            let mut logits = vec![0.0f32; vocab_size];
            unsafe {
                let scale = crate::asm::rms_norm_scale_asm(self.model.hidden_size, x.as_ptr(), 1e-6);
                let mut x_final = vec![0.0f32; self.model.hidden_size];
                for i in 0..self.model.hidden_size {
                    x_final[i] = x[i] * scale * (*self.output_norm_w.add(i));
                }

                if !self.output_w.is_null() {
                    let mut row_f32 = vec![0.0f32; self.model.hidden_size];
                    for i in 0..vocab_size {
                        let weight_row_ptr = self.output_w.add(i * (self.model.hidden_size / 32));
                        dequantize_q4_0_row(weight_row_ptr, &mut row_f32, self.model.hidden_size);
                        let mut sum = 0.0f32;
                        for j in 0..self.model.hidden_size { sum += x_final[j] * row_f32[j]; }
                        logits[i] = sum;
                    }
                }
            }
            
            // 4. SAMPLING STRATEGY
            let temperature = 0.7;
            let top_p = 0.9;
            let repetition_penalty = 1.1;

            // Apply Repetition Penalty
            for &prev_token in &tokens {
                let idx = prev_token as usize;
                if logits[idx] > 0.0 { logits[idx] /= repetition_penalty; } else { logits[idx] *= repetition_penalty; }
            }

            // Apply Temperature
            if temperature > 0.0 {
                for l in logits.iter_mut() { *l /= temperature; }
            }

            // Softmax conversion
            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, |a, b| if a > b { a } else { b });
            let mut sum_exp = 0.0;
            let mut probs: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &l)| {
                let p = (l - max_l).exp();
                sum_exp += p;
                (i, p)
            }).collect();
            
            for p in probs.iter_mut() { p.1 /= sum_exp; }

            // Nucleus (Top-P) Filtering
            probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let mut cum_sum = 0.0;
            let mut last_idx = probs.len();
            for i in 0..probs.len() {
                cum_sum += probs[i].1;
                if cum_sum > top_p { last_idx = i + 1; break; }
            }
            let top_probs = &probs[..last_idx];

            // Stochastic Selection
            let mut r = rand::random::<f32>();
            let mut sampled_idx = top_probs[0].0;
            let mut norm_sum = 0.0;
            for p in top_probs { norm_sum += p.1; }
            r *= norm_sum;

            let mut current_cum = 0.0;
            for p in top_probs {
                current_cum += p.1;
                if r <= current_cum { sampled_idx = p.0; break; }
            }

            let next_token = sampled_idx as u32;
            
            // Stop if End-Of-Sequence token is hit
            if next_token == 151645 { break; }
            tokens.push(next_token);
            
            let text = self.tokenizer.decode(&[next_token]);
            print!("{}", text);
            std::io::Write::flush(&mut std::io::stdout())?;
            
            // 5. PREPARE 'x' FOR THE NEXT ITERATION
            let row_ptr = unsafe { self.embd_w.add(next_token as usize * (self.model.hidden_size / 32)) };
            dequantize_q4_0_row(row_ptr, &mut x, self.model.hidden_size);
            self.model.decode_step(&mut x, pos, &mut self.kv_cache_k, &mut self.kv_cache_v);
        }
        
        println!("\nGeneración completada.");
        Ok(())
    }
}

/// Internal helper to extract a u32 from GGUF metadata, handling different integer types.
fn extract_u32(gguf: &GGUFModel, key: &str) -> anyhow::Result<u32> {
    match gguf.metadata.get(key) {
        Some(MetadataValue::Uint32(v)) => Ok(*v),
        Some(MetadataValue::Uint64(v)) => Ok(*v as u32),
        _ => anyhow::bail!("Key {} no encontrada o no es u32", key),
    }
}
