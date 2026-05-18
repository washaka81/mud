use forge_llm::gguf::GGUFModel;
use forge_llm::model::inference::ForgeInference;
use forge_llm::vulkan::VulkanContext;
use forge_llm::asm::{BlockQ4_0, q4_0_gemv_asm};
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    let model_path = "models/MUD2.5-coder-1.5b-instruct-q4_0.gguf";
    if !std::path::Path::new(model_path).exists() { return Ok(()); }

    println!("=== Forge LLM Structural & Transposition Audit ===");
    let vk = Arc::new(VulkanContext::new()?);
    let model_data = GGUFModel::load(model_path)?;
    let inf = ForgeInference::new(&model_data, vk)?;

    println!("\n[1. Layer Mapping Verification]");
    for i in 0..inf.model.layers.len() {
        let l = &inf.model.layers[i];
        if i < 2 || i == inf.model.layers.len() - 1 {
            println!("  Layer {:>2}: Q_Ptr={:?}, K_Ptr={:?}, FFN_Gate={:?}", 
                     i, l.attn_q_w, l.attn_k_w, l.ffn_gate_w);
        }
    }

    println!("\n[2. Transposition Probe (GEMV Orientation)]");
    probe_transposition(&inf);

    println!("\n[3. FFN Block Analysis]");
    analyze_ffn(&inf);

    Ok(())
}

fn probe_transposition(inf: &ForgeInference) {
    let layer = &inf.model.layers[0];
    let n_in = inf.model.hidden_size;
    let n_out = inf.model.hidden_size; // attn_q is square
    
    // Create a structured input [1, 2, 3, ...]
    let x: Vec<f32> = (0..n_in).map(|i| (i % 10) as f32).collect();
    
    // Scenario A: Current (Row-major)
    let mut out_a = vec![0.0f32; n_out];
    let row_size_blocks = n_in / 32;
    for i in 0..n_out {
        let mut sum = 0.0f32;
        unsafe {
            let weight_ptr = layer.attn_q_w.add(i * row_size_blocks);
            q4_0_gemv_asm(n_in, x.as_ptr(), weight_ptr, &mut sum);
        }
        out_a[i] = sum;
    }
    
    let stats_a = calc_stats(&out_a);
    println!("  Orientation A (Row-major dot): Sigma={:.4}, Mag={:.4}", stats_a.std, stats_a.mag);

    // Scenario B: Column-major dot (Strided)
    // Note: This is hard to do directly with Q4_0 blocks, so we dequantize a block and probe
    // If the model is Column-major, the "features" are stored across different blocks.
}

fn analyze_ffn(inf: &ForgeInference) {
    let layer = &inf.model.layers[0];
    let n_in = inf.model.hidden_size;
    let n_ff = inf.model.ffn_hidden_size;
    
    println!("  Layer 0 FFN Stats:");
    print_tensor_stats("Gate", layer.ffn_gate_w, n_ff, n_in);
    print_tensor_stats("Up  ", layer.ffn_up_w, n_ff, n_in);
    print_tensor_stats("Down", layer.ffn_down_w, n_in, n_ff);
}

fn print_tensor_stats(name: &str, ptr: *const BlockQ4_0, rows: usize, cols: usize) {
    let mut sum_d = 0.0f64;
    let mut max_d = 0.0f32;
    for i in 0..rows {
        unsafe {
            let block = &*ptr.add(i * (cols / 32));
            let d = block.d.to_f32();
            sum_d += d as f64;
            if d > max_d { max_d = d; }
        }
    }
    println!("    {}: AvgScale={:.6}, MaxScale={:.6}", name, sum_d / rows as f64, max_d);
}

struct Stats {
    std: f32,
    mag: f64,
}

fn calc_stats(data: &[f32]) -> Stats {
    let n = data.len() as f32;
    let mean = data.iter().sum::<f32>() / n;
    let var = data.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
    let mag = data.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
    Stats { std: var.sqrt(), mag }
}
