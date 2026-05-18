use forge_llm::gguf::GGUFModel;
use forge_llm::model::inference::ForgeInference;
use forge_llm::vulkan::VulkanContext;
use forge_llm::asm::{q4_0_gemv_asm, dequantize_q4_0_row};
use std::sync::Arc;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let model_path = "models/MUD2.5-coder-1.5b-instruct-q4_0.gguf";
    if !std::path::Path::new(model_path).exists() {
        println!("Error: Modelo no encontrado en {}. Descárgalo primero.", model_path);
        return Ok(());
    }

    println!("=== Forge LLM Full Transformer Audit & Benchmark ===");
    let vk = Arc::new(VulkanContext::new()?);
    let model_data = GGUFModel::load(model_path)?;
    let mut inference = ForgeInference::new(&model_data, vk)?;

    println!("\n[1. Statistical Audit - Single Layer Trace]");
    run_statistical_audit(&mut inference);

    println!("\n[2. Computation Benchmark - CPU ASM Kernels]");
    run_cpu_benchmark(&inference);

    println!("\n[3. Computation Benchmark - Vulkan iGPU]");
    run_vulkan_benchmark(&inference);

    Ok(())
}

fn run_statistical_audit(inf: &mut ForgeInference) {
    let prompt = "def fast_fibonacci(n):";
    let tokens = inf.tokenizer.encode(prompt);
    let mut x = vec![0.0f32; inf.model.hidden_size];
    
    // Embedding
    let row_ptr = unsafe { inf.embd_w.add(tokens[0] as usize * (inf.model.hidden_size / 32)) };
    dequantize_q4_0_row(row_ptr, &mut x, inf.model.hidden_size);
    print_stats("Embedding (Token 0)", &x);

    let layer = &inf.model.layers[0];
    let mut q = vec![0.0f32; inf.model.hidden_size];
    
    // Q Projection
    inf.model.gemv_pure_rust(inf.model.hidden_size, inf.model.hidden_size, &x, layer.attn_q_w, layer.attn_norm_w, &mut q, layer.rms_norm_eps);
    print_stats("Q Projection (Layer 0)", &q);

    // RoPE
    let mut k_dummy = vec![0.0f32; inf.model.n_kv_heads * inf.model.head_size];
    inf.model.apply_rope(&mut q, &mut k_dummy, 0);
    print_stats("Q after RoPE", &q);

    // FFN
    let mut ffn_gate = vec![0.0f32; inf.model.ffn_hidden_size];
    let mut ffn_up = vec![0.0f32; inf.model.ffn_hidden_size];
    inf.model.gemv_pure_rust(inf.model.hidden_size, inf.model.ffn_hidden_size, &x, layer.ffn_gate_w, layer.ffn_norm_w, &mut ffn_gate, layer.rms_norm_eps);
    inf.model.gemv_pure_rust(inf.model.hidden_size, inf.model.ffn_hidden_size, &x, layer.ffn_up_w, layer.ffn_norm_w, &mut ffn_up, layer.rms_norm_eps);
    
    for i in 0..ffn_gate.len() {
        let g = ffn_gate[i];
        let silu = g * (1.0 / (1.0 + (-g).exp()));
        ffn_gate[i] = silu * ffn_up[i];
    }
    print_stats("FFN after SwiGLU", &ffn_gate);
}

fn run_cpu_benchmark(inf: &ForgeInference) {
    let n_in = inf.model.hidden_size;
    let n_out = inf.model.hidden_size;
    let x = vec![1.0f32; n_in];
    let layer = &inf.model.layers[0];
    
    println!("  GEMV ASM ({} x {}):", n_in, n_out);
    benchmark_op("GEMV ASM Core", || {
        let mut out = [0.0f32; 1];
        unsafe { q4_0_gemv_asm(n_in, x.as_ptr(), layer.attn_q_w, out.as_mut_ptr()); }
    }, 1000);
}

fn run_vulkan_benchmark(inf: &ForgeInference) {
    println!("  Vulkan Dispatch Latency (iGPU Iris Xe):");
    benchmark_op("Vulkan Simple Mul", || {
        inf.model.vulkan_ctx.run_test_compute().unwrap();
    }, 50);
}

fn print_stats(name: &str, data: &[f32]) {
    let mut sum = 0.0;
    let mut sq_sum = 0.0;
    for &v in data {
        sum += v;
        sq_sum += v * v;
    }
    let mean = sum / data.len() as f32;
    let std = ((sq_sum / data.len() as f32) - (mean * mean)).sqrt();
    println!("    {:<25} | Mean: {:>8.4} | Std: {:>8.4} | Mag: {:>8.4}", 
             name, mean, std, (sq_sum as f64).sqrt());
}

fn benchmark_op<F: FnMut()>(name: &str, mut f: F, iters: usize) {
    for _ in 0..10 { f(); } // Warmup
    let start = Instant::now();
    for _ in 0..iters { f(); }
    let duration = start.elapsed();
    println!("    {:<25}: {:?} per iter", name, duration / iters as u32);
}
