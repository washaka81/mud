use crate::asm::{BlockQ4_0, dequantize_q4_0_row};
use crate::vulkan::VulkanContext;
use std::sync::Arc;

/// Represents a single Transformer layer with its associated weights and biases.
pub struct TransformerLayer {
    /// Query projection weight (Q)
    pub attn_q_w: *const BlockQ4_0,
    /// Query projection bias
    pub attn_q_b: *const f32,
    /// Key projection weight (K)
    pub attn_k_w: *const BlockQ4_0,
    /// Key projection bias
    pub attn_k_b: *const f32,
    /// Value projection weight (V)
    pub attn_v_w: *const BlockQ4_0,
    /// Value projection bias
    pub attn_v_b: *const f32,
    /// Attention output projection weight (Wo)
    pub attn_o_w: *const BlockQ4_0,
    /// Attention output projection bias
    pub attn_o_b: *const f32,
    /// FFN Gate projection weight
    pub ffn_gate_w: *const BlockQ4_0,
    /// FFN Up projection weight
    pub ffn_up_w: *const BlockQ4_0,
    /// FFN Down projection weight
    pub ffn_down_w: *const BlockQ4_0,
    /// Weights for the attention layer normalization (RMSNorm)
    pub attn_norm_w: *const f32, 
    /// Weights for the FFN layer normalization (RMSNorm)
    pub ffn_norm_w: *const f32,
    /// Small constant for numerical stability in RMSNorm
    pub rms_norm_eps: f32,
}

/// The core Transformer model architecture.
/// Orchestrates the execution of layers and manages global parameters.
pub struct ForgeModel {
    /// List of transformer layers
    pub layers: Vec<TransformerLayer>,
    /// Context for Vulkan computation (used for offloading)
    pub vulkan_ctx: Arc<VulkanContext>,
    /// Dimension of the embedding/hidden space
    pub hidden_size: usize,
    /// Dimension of the internal FFN hidden space
    pub ffn_hidden_size: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of KV heads (for GQA/MQA)
    pub n_kv_heads: usize,
    /// Dimension of each individual head
    pub head_size: usize,
}

impl ForgeModel {
    /// Performs a single forward pass step for one token.
    /// Updates 'x' in-place and appends K,V vectors to the provided caches.
    pub fn decode_step(
        &self, 
        x: &mut [f32], 
        pos: usize,
        kv_cache_k: &mut [f32], 
        kv_cache_v: &mut [f32],
    ) {
        let kv_dim = self.n_kv_heads * self.head_size;
        let kv_cache_layer_offset = self.n_kv_heads * self.head_size * 2048;
        
        for (l, layer) in self.layers.iter().enumerate() {
            let mut q = vec![0.0f32; self.hidden_size];
            let mut k = vec![0.0f32; kv_dim];
            let mut v = vec![0.0f32; kv_dim];
            
            // 1. Attention Projections (Q, K, V)
            self.gemv_pure_rust(self.hidden_size, self.hidden_size, x, layer.attn_q_w, layer.attn_norm_w, &mut q, layer.rms_norm_eps);
            self.gemv_pure_rust(self.hidden_size, kv_dim, x, layer.attn_k_w, layer.attn_norm_w, &mut k, layer.rms_norm_eps);
            self.gemv_pure_rust(self.hidden_size, kv_dim, x, layer.attn_v_w, layer.attn_norm_w, &mut v, layer.rms_norm_eps);
            
            // Apply Biases (Crucial for Qwen 2.5)
            unsafe {
                if !layer.attn_q_b.is_null() { for i in 0..self.hidden_size { q[i] += *layer.attn_q_b.add(i); } }
                if !layer.attn_k_b.is_null() { for i in 0..kv_dim { k[i] += *layer.attn_k_b.add(i); } }
                if !layer.attn_v_b.is_null() { for i in 0..kv_dim { v[i] += *layer.attn_v_b.add(i); } }
            }
            
            // 2. Rotary Position Embeddings (RoPE)
            self.apply_rope(&mut q, &mut k, pos);
            
            // 3. KV Cache Update
            let layer_offset = l * kv_cache_layer_offset;
            let final_offset = layer_offset + pos * kv_dim;
            kv_cache_k[final_offset..final_offset + kv_dim].copy_from_slice(&k);
            kv_cache_v[final_offset..final_offset + kv_dim].copy_from_slice(&v);
            
            // 4. Attention Mechanism (CPU-based for now)
            let mut attn_out = vec![0.0f32; self.hidden_size];
            self.compute_attention_cpu(&q, &kv_cache_k[layer_offset..], &kv_cache_v[layer_offset..], &mut attn_out, pos);
            
            // 5. Output Projection (Wo)
            let mut attn_proj = vec![0.0f32; self.hidden_size];
            self.gemv_pure_rust_no_norm(self.hidden_size, self.hidden_size, &attn_out, layer.attn_o_w, &mut attn_proj);
            unsafe { if !layer.attn_o_b.is_null() { for i in 0..self.hidden_size { attn_proj[i] += *layer.attn_o_b.add(i); } } }
            
            // Residual Connection
            for i in 0..self.hidden_size { x[i] += attn_proj[i]; }
            
            let mag_attn = x.iter().map(|v| v*v).sum::<f32>().sqrt();
            if pos == 0 { println!("  Layer {} Attn Magnitude: {:.4}", l, mag_attn); }
            
            // 6. Feed-Forward Network (SwiGLU)
            let ffn_hidden = self.ffn_hidden_size;
            let mut ffn_gate = vec![0.0f32; ffn_hidden];
            let mut ffn_up = vec![0.0f32; ffn_hidden];
            
            // Gate and Up projections
            self.gemv_pure_rust(self.hidden_size, ffn_hidden, x, layer.ffn_gate_w, layer.ffn_norm_w, &mut ffn_gate, layer.rms_norm_eps);
            self.gemv_pure_rust(self.hidden_size, ffn_hidden, x, layer.ffn_up_w, layer.ffn_norm_w, &mut ffn_up, layer.rms_norm_eps);
            
            // SwiGLU activation: (SiLU(gate) * up)
            for i in 0..ffn_hidden {
                let g = ffn_gate[i];
                let silu = g * (1.0 / (1.0 + (-g).exp()));
                ffn_gate[i] = silu * ffn_up[i];
            }
            
            // Down projection
            let mut ffn_down = vec![0.0f32; self.hidden_size];
            self.gemv_pure_rust_no_norm(ffn_hidden, self.hidden_size, &ffn_gate, layer.ffn_down_w, &mut ffn_down);
            
            // Residual Connection
            for i in 0..self.hidden_size { x[i] += ffn_down[i]; }
            
            let mag_ffn = x.iter().map(|v| v*v).sum::<f32>().sqrt();
            if pos == 0 { println!("  Layer {} FFN Magnitude:  {:.4}", l, mag_ffn); }
        }
    }

    /// Performs Matrix-Vector multiplication (GEMV) with integrated RMSNorm.
    /// input -> RMSNorm -> GEMV(weights) -> out
    pub fn gemv_pure_rust(&self, n_in: usize, n_out: usize, x: &[f32], weights: *const BlockQ4_0, norm_w: *const f32, out: &mut [f32], eps: f32) {
        let mut ss = 0.0f32;
        for i in 0..n_in { ss += x[i] * x[i]; }
        let scale = 1.0 / ((ss / n_in as f32) + eps).sqrt();
        
        let mut x_norm = vec![0.0f32; n_in];
        for i in 0..n_in {
            unsafe { x_norm[i] = x[i] * scale * (*norm_w.add(i)); }
        }
        self.gemv_pure_rust_no_norm(n_in, n_out, &x_norm, weights, out);
    }

    /// Performs Matrix-Vector multiplication (GEMV) without normalization.
    /// Uses on-the-fly dequantization for Q4_0 weights.
    pub fn gemv_pure_rust_no_norm(&self, n_in: usize, n_out: usize, input: &[f32], weights: *const BlockQ4_0, out: &mut [f32]) {
        let row_size_blocks = n_in / 32;
        for i in 0..n_out {
            let weight_row_ptr = unsafe { weights.add(i * row_size_blocks) };
            let mut row_f32 = vec![0.0f32; n_in];
            dequantize_q4_0_row(weight_row_ptr, &mut row_f32, n_in);
            let mut sum = 0.0f32;
            for j in 0..n_in { sum += input[j] * row_f32[j]; }
            out[i] = sum;
        }
    }

    /// Applies Rotary Position Embeddings (RoPE) to the Q and K vectors.
    /// This implementation follows the standard LLaMA/Qwen approach with half-dimension rotation.
    pub fn apply_rope(&self, q: &mut [f32], k: &mut [f32], pos: usize) {
        let head_size = self.head_size;
        let n_heads = q.len() / head_size;
        let n_kv_heads = k.len() / head_size;
        let half = head_size / 2;
        let freq_base = 1000000.0f32; // Standard for Qwen 2.5
        
        for h in 0..n_heads {
            let start = h * head_size;
            for i in 0..half {
                let freq = 1.0 / freq_base.powf((2 * i) as f32 / head_size as f32);
                let theta = (pos as f32) * freq;
                let cos = theta.cos();
                let sin = theta.sin();
                let q0 = q[start + i];
                let q1 = q[start + i + half];
                q[start + i]        = q0 * cos - q1 * sin;
                q[start + i + half] = q0 * sin + q1 * cos;
            }
        }
        for h in 0..n_kv_heads {
            let start = h * head_size;
            for i in 0..half {
                let freq = 1.0 / freq_base.powf((2 * i) as f32 / head_size as f32);
                let theta = (pos as f32) * freq;
                let cos = theta.cos();
                let sin = theta.sin();
                let k0 = k[start + i];
                let k1 = k[start + i + half];
                k[start + i]        = k0 * cos - k1 * sin;
                k[start + i + half] = k0 * sin + k1 * cos;
            }
        }
    }

    /// Computes multi-head attention (CPU reference implementation).
    /// Supports Grouped-Query Attention (GQA).
    fn compute_attention_cpu(&self, q: &[f32], k_cache: &[f32], v_cache: &[f32], out: &mut [f32], pos: usize) {
        let head_size = self.head_size;
        let n_heads = self.n_heads;
        let n_kv_heads = self.n_kv_heads;
        let group_size = n_heads / n_kv_heads;
        let kv_dim = n_kv_heads * head_size;
        let scale = 1.0 / (head_size as f32).sqrt();

        for h in 0..n_heads {
            let q_h = &q[h * head_size..(h + 1) * head_size];
            let kv_h = h / group_size;
            let mut scores = vec![0.0f32; pos + 1];

            for p in 0..=pos {
                let k_start = p * kv_dim + kv_h * head_size;
                let k_hp = &k_cache[k_start .. k_start + head_size];
                let mut score = 0.0f32;
                for i in 0..head_size { score += q_h[i] * k_hp[i]; }
                scores[p] = score * scale;
            }

            // Softmax
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_score).exp();
                sum += *s;
            }
            for s in scores.iter_mut() { *s /= sum; }

            // Weight sum of V
            let out_h_start = h * head_size;
            for p in 0..=pos {
                let v_start = p * kv_dim + kv_h * head_size;
                let v_hp = &v_cache[v_start .. v_start + head_size];
                for i in 0..head_size { out[out_h_start + i] += scores[p] * v_hp[i]; }
            }
        }
    }
}
