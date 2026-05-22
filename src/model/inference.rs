use crate::gguf::{GGUFModel, MetadataValue};
use crate::model::tokenizer::Tokenizer;
use crate::model::transformer::{ForgeModel, TransformerLayer, InferenceWorkspace};
use crate::asm::{BlockQ4_0, dequantize_q4_0_row};
use crate::vulkan::VulkanContext;
use std::sync::Arc;

pub struct ForgeInference {
    pub model: ForgeModel,
    pub tokenizer: Tokenizer,
    pub kv_cache_k: Vec<i8>,
    pub kv_cache_v: Vec<i8>,
    pub kv_scales_k: Vec<f32>,
    pub kv_scales_v: Vec<f32>,
    pub embd_w: *const BlockQ4_0,
    pub output_w: *const BlockQ4_0,
    pub output_norm_w: *const f32,
    pub max_seq_len: usize,
    pub ws: InferenceWorkspace,
}

impl ForgeInference {
    pub fn new(gguf: &GGUFModel, vk_ctx: Arc<VulkanContext>) -> anyhow::Result<Self> {
        let tokenizer = Tokenizer::from_gguf(gguf)?;
        let hidden_size = extract_u32(gguf, "MUD2.embedding_length")? as usize;
        let n_layers = extract_u32(gguf, "MUD2.block_count")? as usize;
        let n_heads = extract_u32(gguf, "MUD2.attention.head_count")? as usize;
        let n_kv_heads = extract_u32(gguf, "MUD2.attention.head_count_kv")? as usize;
        let head_size = hidden_size / n_heads;
        let ffn_hidden = extract_u32(gguf, "MUD2.feed_forward_length").unwrap_or((hidden_size * 4) as u32) as usize;
        
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
        
        let model = ForgeModel { layers, vulkan_ctx: vk_ctx, hidden_size, ffn_hidden_size: ffn_hidden, n_heads, n_kv_heads, head_size };
        let max_seq_len = 2048;
        let kv_cache_size = n_layers * max_seq_len * n_kv_heads * head_size;
        let scales_size = n_layers * max_seq_len * n_kv_heads;
        let ws = InferenceWorkspace::new(hidden_size, n_kv_heads * head_size, ffn_hidden);
        
        Ok(Self { model, tokenizer, kv_cache_k: vec![0; kv_cache_size], kv_cache_v: vec![0; kv_cache_size], 
                  kv_scales_k: vec![0.0; scales_size], kv_scales_v: vec![0.0; scales_size],
                  embd_w: gguf.get_tensor_q4_0("token_embd.weight").unwrap_or(std::ptr::null()),
                  output_w: gguf.get_tensor_q4_0("output.weight").unwrap_or(std::ptr::null()),
                  output_norm_w: gguf.tensors.get("output_norm.weight").map(|t| t.data_ptr as *const f32).unwrap_or(std::ptr::null()),
                  max_seq_len, ws })
    }

    pub fn generate(&mut self, prompt: &str, max_new_tokens: usize) -> anyhow::Result<()> {
        let prompt_tokens = self.tokenizer.encode(prompt);
        let mut tokens = Vec::new();
        let mut x = vec![0.0f32; self.model.hidden_size];
        
        for (pos, &token) in prompt_tokens.iter().enumerate() {
            let row_ptr = unsafe { self.embd_w.add(token as usize * (self.model.hidden_size / 32)) };
            unsafe { dequantize_q4_0_row(row_ptr, &mut x, self.model.hidden_size); }
            self.model.decode_step(&mut x, pos, &mut self.kv_cache_k, &mut self.kv_cache_v, &mut self.kv_scales_k, &mut self.kv_scales_v, &mut self.ws);
            tokens.push(token);
        }
        
        for pos_offset in 1..=max_new_tokens {
            let pos = prompt_tokens.len() + pos_offset - 1;
            
            unsafe {
                let scale = crate::asm::rms_norm_scale_asm(self.model.hidden_size, x.as_ptr(), 1e-6);
                for i in 0..self.model.hidden_size {
                    self.ws.row_f32[i] = x[i] * scale * (if !self.output_norm_w.is_null() { *self.output_norm_w.add(i) } else { 1.0 });
                }

                if !self.output_w.is_null() {
                    for i in 0..self.tokenizer.id_to_token.len() {
                        let row_ptr = self.output_w.add(i * (self.model.hidden_size / 32));
                        let mut row = vec![0.0f32; self.model.hidden_size];
                        dequantize_q4_0_row(row_ptr, &mut row, self.model.hidden_size);
                        let mut sum = 0.0f32;
                        for j in 0..self.model.hidden_size { sum += self.ws.row_f32[j] * row[j]; }
                        self.ws.attn_proj[i] = sum; // reusing buffer
                    }
                }
            }
            
            let sampled_idx = self.ws.attn_proj.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            
            if sampled_idx as u32 == 2 { break; }
            tokens.push(sampled_idx as u32);
            print!("{}", self.tokenizer.decode(&[sampled_idx as u32]));
            std::io::Write::flush(&mut std::io::stdout())?;
            
            let row_ptr = unsafe { self.embd_w.add(sampled_idx * (self.model.hidden_size / 32)) };
            unsafe { dequantize_q4_0_row(row_ptr, &mut x, self.model.hidden_size); }
            self.model.decode_step(&mut x, pos + 1, &mut self.kv_cache_k, &mut self.kv_cache_v, &mut self.kv_scales_k, &mut self.kv_scales_v, &mut self.ws);
        }
        Ok(())
    }
}

fn extract_u32(gguf: &GGUFModel, key: &str) -> anyhow::Result<u32> {
    match gguf.metadata.get(key) {
        Some(MetadataValue::Uint32(v)) => Ok(*v),
        Some(MetadataValue::Uint64(v)) => Ok(*v as u32),
        _ => anyhow::bail!("Key {} no encontrada o no es u32", key),
    }
}
