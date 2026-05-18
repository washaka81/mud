use forge_llm::gguf::GGUFModel;
use forge_llm::model::inference::ForgeInference;
use forge_llm::vulkan::VulkanContext;
use forge_llm::asm::dequantize_q4_0_row;
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    let model_path = "models/MUD2.5-coder-1.5b-instruct-q4_0.gguf";
    if !std::path::Path::new(model_path).exists() { return Ok(()); }

    println!("=== Forge LLM Deep Statistical Trace ===");
    let vk = Arc::new(VulkanContext::new()?);
    let model_data = GGUFModel::load(model_path)?;
    let inf = ForgeInference::new(&model_data, vk)?;

    let prompt = "def";
    let tokens = inf.tokenizer.encode(prompt);
    let mut x = vec![0.0f32; inf.model.hidden_size];
    
    // 1. Embedding
    let row_ptr = unsafe { inf.embd_w.add(tokens[0] as usize * (inf.model.hidden_size / 32)) };
    dequantize_q4_0_row(row_ptr, &mut x, inf.model.hidden_size);
    print_stats("Embedding", &x);

    // 2. Trace Layer 0
    let layer = &inf.model.layers[0];
    let n = inf.model.hidden_size;
    let n_ff = inf.model.ffn_hidden_size;

    // A. RMSNorm
    let mut ss = 0.0f32;
    for i in 0..n { ss += x[i] * x[i]; }
    let scale = 1.0 / ((ss / n as f32) + 1e-6).sqrt();
    let mut x_norm = vec![0.0f32; n];
    for i in 0..n { unsafe { x_norm[i] = x[i] * scale * (*layer.attn_norm_w.add(i)); } }
    print_stats("After RMSNorm", &x_norm);

    // B. Q Projection
    let mut q = vec![0.0f32; n];
    inf.model.gemv_pure_rust_no_norm(n, n, &x_norm, layer.attn_q_w, &mut q);
    print_stats("Q Projection", &q);
    
    // C. Attention Output (Simulated identity for trace)
    let attn_out = q.clone(); // Just for trace
    let mut attn_proj = vec![0.0f32; n];
    inf.model.gemv_pure_rust_no_norm(n, n, &attn_out, layer.attn_o_w, &mut attn_proj);
    print_stats("Wo Projection", &attn_proj);

    // D. FFN
    let mut ffn_gate = vec![0.0f32; n_ff];
    let mut ffn_up = vec![0.0f32; n_ff];
    inf.model.gemv_pure_rust_no_norm(n, n_ff, &x_norm, layer.ffn_gate_w, &mut ffn_gate);
    inf.model.gemv_pure_rust_no_norm(n, n_ff, &x_norm, layer.ffn_up_w, &mut ffn_up);
    print_stats("FFN Gate", &ffn_gate);
    print_stats("FFN Up", &ffn_up);

    for i in 0..n_ff {
        let g = ffn_gate[i];
        let silu = g * (1.0 / (1.0 + (-g).exp()));
        ffn_gate[i] = silu * ffn_up[i];
    }
    print_stats("FFN SwiGLU", &ffn_gate);

    let mut ffn_down = vec![0.0f32; n];
    inf.model.gemv_pure_rust_no_norm(n_ff, n, &ffn_gate, layer.ffn_down_w, &mut ffn_down);
    print_stats("FFN Down", &ffn_down);

    Ok(())
}

fn print_stats(name: &str, data: &[f32]) {
    let n = data.len() as f32;
    let mean = data.iter().sum::<f32>() / n;
    let var = data.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
    let mag = data.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
    println!("  {:<20} | Sigma: {:>8.4} | Mean: {:>8.4} | Mag: {:>8.4}", 
             name, var.sqrt(), mean, mag);
}
