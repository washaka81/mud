use forge_llm::mud::MudFile;
use forge_llm::mud::inference::MudInference;
use forge_llm::vulkan::VulkanContext;
use std::sync::Arc;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    println!("===============================================");
    println!(" MUD ENGINE: MEMORY AUDIT & BENCHMARK");
    println!(" Architecture: Ternary 1.58-bit MoE");
    println!(" Target: Intel i7-1260p & Iris Xe");
    println!("===============================================");

    let mud_path = "models/core_skills.mud";
    let vk = Arc::new(VulkanContext::new().unwrap());
    
    // 1. MEMORY AUDIT
    let start_mem = Instant::now();
    let mud_file = MudFile::load(mud_path)?;
    let mut engine = MudInference::new(&mud_file, Some(vk))?;
    let load_duration = start_mem.elapsed();

    let mmap_size = mud_file.mmap.as_ref().unwrap().len() as f64 / 1024.0 / 1024.0;
    let num_experts = engine.model.num_experts;
    let hidden_size = engine.model.hidden_size;

    println!("\n[1. RESOURCE UTILIZATION]");
    println!("  Model Mmap Size:      {:.2} MB", mmap_size);
    println!("  Active Experts:       {}", num_experts);
    println!("  Hidden Dimension:     {}", hidden_size);
    println!("  Total Model Loading:  {:?}", load_duration);

    // 2. TERNARY SIMD BENCHMARK
    println!("\n[2. KERNEL PERFORMANCE]");
    let mut x = vec![1.0f32; hidden_size];
    let iters = 1000;
    
    let start_bench = Instant::now();
    for i in 0..iters {
        engine.step(&mut x, "benchmark", &[], i % 2048);
    }
    let bench_duration = start_bench.elapsed();
    let avg_step = bench_duration / iters as u32;
    
    println!("  Average Inference Step: {:?}", avg_step);
    println!("  Theoretical Throughput: {:.2} steps/sec", 1.0 / avg_step.as_secs_f64());

    // 3. SCALABILITY PROJECTION
    println!("\n[3. SCALABILITY PROJECTION (10+ Experts)]");
    let base_expert_size = (hidden_size * hidden_size * 3 * 2) as f64 / 8.0 / 1024.0 / 1024.0; // 2-bit packing
    println!("  Memory per Expert:    {:.4} MB", base_expert_size);
    println!("  Estimated RAM for 16 Experts: {:.2} MB", mmap_size + (base_expert_size * 8.0));
    println!("  Max Experts (on 16GB RAM):   ~{}", (1024.0 * 8.0 / base_expert_size) as usize);

    println!("\nBenchmark Complete.");
    Ok(())
}
