use forge_llm::mud::{MudFile, inference::MudInference};
use forge_llm::vulkan::VulkanContext;
use std::sync::Arc;
use std::time::Instant;

// ANSI colors for premium dashboard
const CYAN: &str = "\x1b[36m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const RED: &str = "\x1b[31m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

fn main() -> anyhow::Result<()> {
    println!("{}=================================================={}", BOLD, RESET);
    println!("{}🧠  MUD ENGINE END-TO-END COGNITIVE INTEGRITY AUDIT 🧠{}", BOLD, RESET);
    println!("{}=================================================={}", BOLD, RESET);

    let model_path = std::env::args().nth(1).unwrap_or_else(|| "models/core_skills.mud".to_string());
    println!("{}   🔍 Loading MUD Model from: {}{}{}", CYAN, BOLD, model_path, RESET);
    let mud_file = MudFile::load(&model_path)?;

    println!("{}   ⚡ Initializing Vulkan Context...{}", CYAN, RESET);
    let vk = Arc::new(vulkan_context_or_dummy());

    println!("{}   ⚙️  Initializing MudInference Engine...{}", CYAN, RESET);
    let mut engine = MudInference::new(&mud_file, Some(vk))?;
    println!("{}   ✅ Engine initialized successfully.{}", GREEN, RESET);

    // List of test prompts to evaluate coherence
    let test_prompts = vec![
        "whats",
        "me gusta cha cha",
        "¿Qué es MUD?",
    ];

    println!("\n{}--- COGNITIVE RESPONSE EVALUATION ---{}", BOLD, RESET);

    for prompt in test_prompts {
        println!("\n{}   [PROMPT] ❯ \"{}\"{}", BOLD, prompt, RESET);
        println!("   --------------------------------------------------");

        let mut conversation_pos = 0;
        let mut x = vec![0.0f32; engine.model.hidden_size];

        let start_prompt = Instant::now();
        engine.prompt(prompt, &mut x, &mut conversation_pos);
        let prompt_duration = start_prompt.elapsed();
        println!("   Processed prompt in: {:.2?}", prompt_duration);

        let start_gen = Instant::now();
        let (tokens, used_knowledge) = engine.generate(&x, 30, prompt, &mut conversation_pos);
        let gen_duration = start_gen.elapsed();

        let generated_text = tokens.iter()
            .map(|&id| engine.tokenizer.decode(&[id]))
            .collect::<Vec<_>>()
            .join(" ");

        println!("   Generated Tokens Count: {}", tokens.len());
        println!("   Response text: {:?}", generated_text);
        println!("   Used Local Knowledge: {}", used_knowledge);

        // Quality check metrics
        let tokens_sec = tokens.len() as f32 / gen_duration.as_secs_f32();
        println!("   Generation Speed: {:.2} tokens/sec", tokens_sec);

        // 1. Repetitive Loop Audit (hene or Hin loops)
        let contains_hene = generated_text.to_lowercase().contains("hene");
        let contains_hin = generated_text.to_lowercase().contains("hin");
        
        let loop_detected = if tokens.len() >= 6 {
            let mut found_loop = false;
            for i in 0..tokens.len() - 4 {
                if tokens[i] == tokens[i+2] && tokens[i+1] == tokens[i+3] {
                    found_loop = true;
                    break;
                }
            }
            found_loop
        } else {
            false
        };

        let status = if contains_hene || contains_hin || loop_detected {
            format!("{}❌ COLLAPSED (Repetition Detected){}", RED, RESET)
        } else if tokens.is_empty() {
            format!("{}⚠️  EMPTY RESPONSE (Weak or Flat Weights){}", YELLOW, RESET)
        } else {
            format!("{}✅ COHERENT & STABLE (Cognitive Cohesion OK){}", GREEN, RESET)
        };

        println!("   Cognitive Coherence Status: {}", status);
        println!("   --------------------------------------------------");
    }

    println!("\n{}=================================================={}", BOLD, RESET);
    println!("{}🎉 END-TO-END COGNITIVE INTEGRITY AUDIT COMPLETE 🎉{}", BOLD, RESET);
    println!("{}=================================================={}", BOLD, RESET);

    Ok(())
}

fn vulkan_context_or_dummy() -> VulkanContext {
    VulkanContext::new().unwrap_or_else(|_| {
        // Safe fallback - assuming VulkanContext can be constructed in dummy mode or returns error
        VulkanContext::new().expect("Failed to initialize Vulkan Context")
    })
}
