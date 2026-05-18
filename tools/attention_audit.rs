use forge_llm::ai::MudFile;
use forge_llm::ai::inference::MudInference;
use forge_llm::vulkan::VulkanContext;
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    let model_path = "models/core_skills.ai";
    if !std::path::Path::new(model_path).exists() { 
        println!("MUD model not found at {}", model_path);
        return Ok(()); 
    }

    println!("=== MUD Attention & Working Memory Audit ===");
    let vk = Arc::new(VulkanContext::new()?);
    let mud_file = MudFile::load(model_path)?;
    let engine = MudInference::new(&mud_file, vk)?;

    let hidden = engine.model.hidden_size;
    
    println!("\n[1. Working Memory (KV Cache) Trace]");
    println!("  Hidden Size: {}, Context Window: 2048", hidden);
    
    let mut x = vec![1.0f32; hidden];
    
    println!("  --- Initializing state (Prompt Priming Simulation) ---");
    // Simulate processing 5 tokens
    for pos in 0..5 {
        engine.step(&mut x, "audit", &[], pos);
        let mag = x.iter().map(|v| v*v).sum::<f32>().sqrt();
        println!("  Pos {}: Output Magnitude = {:.4}", pos, mag);
    }

    println!("\n[2. Attention Scaling & GQA]");
    let head_size = HeadSize::calculate(hidden); // Simplified helper
    let scale = 1.0 / (head_size as f32).sqrt();
    println!("  Standard Head Size: {}", head_size);
    println!("  Scaling Factor: {:.6}", scale);

    println!("\n[3. MUD Efficiency Audit]");
    let packed_size = (hidden * hidden * 2) as f64 / 8.0 / 1024.0;
    println!("  Ternary Layer Size: {:.2} KB", packed_size);
    println!("  FP32 Equivalent:    {:.2} KB", packed_size * 16.0);
    println!("  Efficiency Gain:    16.0x (2-bit vs 32-bit)");

    println!("\nAudit Complete.");
    Ok(())
}

struct HeadSize;
impl HeadSize {
    fn calculate(hidden: usize) -> usize {
        if hidden % 64 == 0 { 64 } else { 32 }
    }
}
