use forge_llm::gguf::GGUFModel;
use forge_llm::model::inference::ForgeInference;
use forge_llm::vulkan::VulkanContext;
use forge_llm::asm::dequantize_q4_0_row;
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    let model_data = GGUFModel::load("models/qwen2.5-coder-1.5b-instruct-q4_0.gguf")?;
    let inf = ForgeInference::new(&model_data, Arc::new(VulkanContext::new()?))?;

    let tokens = inf.tokenizer.encode("def");
    let mut x = vec![0.0f32; inf.model.hidden_size];
    let row_ptr = unsafe { inf.embd_w.add(tokens[0] as usize * (inf.model.hidden_size / 32)) };
    dequantize_q4_0_row(row_ptr, &mut x, inf.model.hidden_size);

    println!("=== Layer-by-Layer Sigma Trace ===");
    for l in 0..inf.model.layers.len() {
        let layer = &inf.model.layers[l];
        
        // --- ATENCIÓN ---
        let mut q = vec![0.0f32; inf.model.hidden_size];
        inf.model.gemv_pure_rust(inf.model.hidden_size, inf.model.hidden_size, &x, layer.attn_q_w, layer.attn_norm_w, &mut q, 1e-6);
        
        // Simulación simplificada: sumamos la proyección Q directamente al residual para ver la deriva
        let mut attn_proj = vec![0.0f32; inf.model.hidden_size];
        inf.model.gemv_pure_rust_no_norm(inf.model.hidden_size, inf.model.hidden_size, &q, layer.attn_o_w, &mut attn_proj);
        for i in 0..inf.model.hidden_size { x[i] += attn_proj[i]; }

        // --- FFN ---
        let mut ffn_gate = vec![0.0f32; inf.model.ffn_hidden_size];
        inf.model.gemv_pure_rust(inf.model.hidden_size, inf.model.ffn_hidden_size, &x, layer.ffn_gate_w, layer.ffn_norm_w, &mut ffn_gate, 1e-6);
        
        let mut ffn_down = vec![0.0f32; inf.model.hidden_size];
        inf.model.gemv_pure_rust_no_norm(inf.model.ffn_hidden_size, inf.model.hidden_size, &ffn_gate, layer.ffn_down_w, &mut ffn_down);
        for i in 0..inf.model.hidden_size { x[i] += ffn_down[i]; }

        let sigma = calc_sigma(&x);
        println!("  Layer {:>2} | Sigma: {:.4}", l, sigma);
        if sigma > 100.0 { 
            println!("    ❌ EXPLOSIÓN DETECTADA EN CAPA {}!", l);
            break;
        }
    }

    Ok(())
}

fn calc_sigma(data: &[f32]) -> f32 {
    let n = data.len() as f32;
    let mean = data.iter().sum::<f32>() / n;
    let var = data.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
    var.sqrt()
}
