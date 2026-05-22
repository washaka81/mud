use crate::asm::BlockQ4_0;
use crate::vulkan::VulkanContext;
use std::sync::Arc;

pub struct TransformerLayer {
    pub attn_q_w: *const BlockQ4_0, pub attn_q_b: *const f32,
    pub attn_k_w: *const BlockQ4_0, pub attn_k_b: *const f32,
    pub attn_v_w: *const BlockQ4_0, pub attn_v_b: *const f32,
    pub attn_o_w: *const BlockQ4_0, pub attn_o_b: *const f32,
    pub ffn_gate_w: *const BlockQ4_0,
    pub ffn_up_w: *const BlockQ4_0,
    pub ffn_down_w: *const BlockQ4_0,
    pub attn_norm_w: *const f32, 
    pub ffn_norm_w: *const f32,
    pub rms_norm_eps: f32,
}

pub struct InferenceWorkspace {
    pub q: Vec<f32>, pub k: Vec<f32>, pub v: Vec<f32>,
    pub attn_out: Vec<f32>, pub attn_proj: Vec<f32>,
    pub ffn_gate: Vec<f32>, pub ffn_up: Vec<f32>, pub ffn_down: Vec<f32>,
    pub row_f32: Vec<f32>,
}

impl InferenceWorkspace {
    pub fn new(hidden_size: usize, kv_dim: usize, ffn_hidden: usize) -> Self {
        Self {
            q: vec![0.0f32; hidden_size],
            k: vec![0.0f32; kv_dim],
            v: vec![0.0f32; kv_dim],
            attn_out: vec![0.0f32; hidden_size],
            attn_proj: vec![0.0f32; hidden_size],
            ffn_gate: vec![0.0f32; ffn_hidden],
            ffn_up: vec![0.0f32; ffn_hidden],
            ffn_down: vec![0.0f32; hidden_size],
            row_f32: vec![0.0f32; hidden_size],
        }
    }
}

pub struct ForgeModel {
    pub layers: Vec<TransformerLayer>,
    pub vulkan_ctx: Arc<VulkanContext>,
    pub hidden_size: usize,
    pub ffn_hidden_size: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_size: usize,
}

impl ForgeModel {
    pub fn decode_step(
        &self, 
        x: &mut [f32], 
        pos: usize,
        kv_cache_k: &mut [i8], 
        kv_cache_v: &mut [i8],
        kv_scales_k: &mut [f32],
        kv_scales_v: &mut [f32],
        ws: &mut InferenceWorkspace,
    ) {
        let kv_dim = self.n_kv_heads * self.head_size;
        let kv_cache_layer_offset = self.n_kv_heads * self.head_size * 2048;
        let scales_layer_offset = self.n_kv_heads * 2048;
        
        for (l, layer) in self.layers.iter().enumerate() {
            ws.q.fill(0.0); ws.k.fill(0.0); ws.v.fill(0.0);
            
            unsafe {
                crate::asm::q4_0_gemv_fused(self.hidden_size, self.hidden_size, x, layer.attn_q_w, layer.attn_norm_w, &mut ws.q, layer.rms_norm_eps);
                crate::asm::q4_0_gemv_fused(self.hidden_size, kv_dim, x, layer.attn_k_w, layer.attn_norm_w, &mut ws.k, layer.rms_norm_eps);
                crate::asm::q4_0_gemv_fused(self.hidden_size, kv_dim, x, layer.attn_v_w, layer.attn_norm_w, &mut ws.v, layer.rms_norm_eps);
            }
            
            self.apply_rope(&mut ws.q, &mut ws.k, pos);
            
            let layer_offset = l * kv_cache_layer_offset;
            let scale_offset = l * scales_layer_offset + pos * self.n_kv_heads;
            
            for h in 0..self.n_kv_heads {
                let h_start = h * self.head_size;
                let k_h = &ws.k[h_start..h_start + self.head_size];
                let v_h = &ws.v[h_start..h_start + self.head_size];
                
                let k_max = k_h.iter().map(|&x| x.abs()).fold(0.0f32, f32::max).max(1e-5);
                let v_max = v_h.iter().map(|&x| x.abs()).fold(0.0f32, f32::max).max(1e-5);
                
                kv_scales_k[scale_offset + h] = k_max / 127.0;
                kv_scales_v[scale_offset + h] = v_max / 127.0;
                
                let k_scale_inv = 127.0 / k_max;
                let v_scale_inv = 127.0 / v_max;
                
                let final_offset = layer_offset + pos * kv_dim;
                for i in 0..self.head_size {
                    kv_cache_k[final_offset + h_start + i] = (k_h[i] * k_scale_inv).round() as i8;
                    kv_cache_v[final_offset + h_start + i] = (v_h[i] * v_scale_inv).round() as i8;
                }
            }
            
            ws.attn_out.fill(0.0);
            self.compute_attention_quantized(&ws.q, &kv_cache_k[layer_offset..], &kv_cache_v[layer_offset..], 
                                           &kv_scales_k[l * scales_layer_offset..], &kv_scales_v[l * scales_layer_offset..], 
                                           &mut ws.attn_out, pos, &mut ws.attn_proj);
            
            ws.attn_proj.fill(0.0);
            unsafe { self.gemv_pure_rust_no_norm(self.hidden_size, self.hidden_size, &ws.attn_out, layer.attn_o_w, &mut ws.attn_proj); }
            
            for (i, item) in x.iter_mut().enumerate().take(self.hidden_size) { *item += ws.attn_proj[i]; }
            
            ws.ffn_gate.fill(0.0); ws.ffn_up.fill(0.0);
            unsafe {
                crate::asm::q4_0_gemv_fused(self.hidden_size, self.ffn_hidden_size, x, layer.ffn_gate_w, layer.ffn_norm_w, &mut ws.ffn_gate, layer.rms_norm_eps);
                crate::asm::q4_0_gemv_fused(self.hidden_size, self.ffn_hidden_size, x, layer.ffn_up_w, layer.ffn_norm_w, &mut ws.ffn_up, layer.rms_norm_eps);
            }
            
            for i in 0..self.ffn_hidden_size {
                let g = ws.ffn_gate[i];
                ws.ffn_gate[i] = (g * (1.0 / (1.0 + (-g).exp()))) * ws.ffn_up[i];
            }
            
            ws.ffn_down.fill(0.0);
            unsafe { self.gemv_pure_rust_no_norm(self.ffn_hidden_size, self.hidden_size, &ws.ffn_gate, layer.ffn_down_w, &mut ws.ffn_down); }
            for (i, item) in x.iter_mut().enumerate().take(self.hidden_size) { *item += ws.ffn_down[i]; }
        }
    }

    fn compute_attention_quantized(&self, q: &[f32], k_cache: &[i8], v_cache: &[i8], 
                                 k_scales: &[f32], v_scales: &[f32], 
                                 out: &mut [f32], pos: usize, scores: &mut [f32]) {
        let head_size = self.head_size;
        let n_heads = self.n_heads;
        let n_kv_heads = self.n_kv_heads;
        let group_size = n_heads / n_kv_heads;
        let kv_dim = n_kv_heads * head_size;
        let scale = 1.0 / (head_size as f32).sqrt();

        for h in 0..n_heads {
            let q_h = &q[h * head_size..(h + 1) * head_size];
            let kv_h = h / group_size;

            for p in 0..=pos {
                let k_start = p * kv_dim + kv_h * head_size;
                let k_hp_i8 = &k_cache[k_start .. k_start + head_size];
                let k_scale = k_scales[p * self.n_kv_heads + kv_h];
                let mut score = 0.0f32;
                for i in 0..head_size { score += q_h[i] * (k_hp_i8[i] as f32 * k_scale); }
                scores[p] = score * scale;
            }

            let max_score = scores[0..=pos].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for p in 0..=pos {
                scores[p] = (scores[p] - max_score).exp();
                sum += scores[p];
            }
            let inv_sum = 1.0 / (sum + 1e-30);
            for p in 0..=pos { scores[p] *= inv_sum; }

            let out_h_start = h * head_size;
            for p in 0..=pos {
                let v_start = p * kv_dim + kv_h * head_size;
                let v_hp_i8 = &v_cache[v_start .. v_start + head_size];
                let v_scale = v_scales[p * self.n_kv_heads + kv_h];
                for i in 0..head_size { out[out_h_start + i] += scores[p] * (v_hp_i8[i] as f32 * v_scale); }
            }
        }
    }

    pub unsafe fn gemv_pure_rust_no_norm(&self, n_in: usize, n_out: usize, input: &[f32], weights: *const BlockQ4_0, out: &mut [f32]) {
        let row_size_blocks = n_in / 32;
        for i in 0..n_out {
            let weight_row_ptr = weights.add(i * row_size_blocks);
            let mut row_f32 = vec![0.0f32; n_in];
            crate::asm::dequantize_q4_0_row(weight_row_ptr, &mut row_f32, n_in);
            let mut sum = 0.0f32;
            for j in 0..n_in { sum += input[j] * row_f32[j]; }
            out[i] = sum;
        }
    }

    pub fn apply_rope(&self, q: &mut [f32], k: &mut [f32], pos: usize) {
        let head_size = self.head_size;
        let n_heads = q.len() / head_size;
        let n_kv_heads = k.len() / head_size;
        for h in 0..n_heads {
            let start = h * head_size;
            for i in (0..head_size).step_by(2) {
                let freq = 1.0 / 10000.0f32.powf(i as f32 / head_size as f32);
                let theta = (pos as f32) * freq;
                let cos = theta.cos(); let sin = theta.sin();
                let (q0, q1) = (q[start+i], q[start+i+1]);
                q[start+i] = q0 * cos - q1 * sin; q[start+i+1] = q0 * sin + q1 * cos;
            }
        }
        for h in 0..n_kv_heads {
            let start = h * head_size;
            for i in (0..head_size).step_by(2) {
                let freq = 1.0 / 10000.0f32.powf(i as f32 / head_size as f32);
                let theta = (pos as f32) * freq;
                let cos = theta.cos(); let sin = theta.sin();
                let (k0, k1) = (k[start+i], k[start+i+1]);
                k[start+i] = k0 * cos - k1 * sin; k[start+i+1] = k0 * sin + k1 * cos;
            }
        }
    }
}
