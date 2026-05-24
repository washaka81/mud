use forge_llm::mud::MudFile;
use forge_llm::mud::inference::MudInference;
use forge_llm::vulkan::VulkanContext;
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    let model_path = "models/core_skills.mud";
    println!("=== MUD ABSOLUTE TRUTH AUDITOR ===");
    
    let vk_ctx = Arc::new(VulkanContext::new()?);
    let mud_file = MudFile::load(model_path)?;
    let mut engine = MudInference::new(&mud_file, vk_ctx)?;
    
    // Execute a series of "Ground Truth" prompts
    let tests = vec![
        "¿Qué es MUD?",
        "Explica el Teorema de Bayes.",
        "¿Cuál es la importancia del Teorema del Límite Central?",
    ];

    for prompt in tests {
        println!("\nPrompt: {}", prompt);
        let mut x = vec![0.0f32; engine.model.hidden_size];
        let mut conversation_pos = 0;
        engine.prompt(prompt, &mut x, &mut conversation_pos);
        let (response_tokens, _) = engine.generate(&x, 32, prompt, &mut conversation_pos);
        let response_text = engine.tokenizer.decode(&response_tokens);
        
        println!("Response: {}", response_text);
        
        // Truth Validation via RAG Search
        let conn = rusqlite::Connection::open("models/knowledge.db")?;
        let mut stmt = conn.prepare("SELECT content FROM facts WHERE content LIKE ?1 LIMIT 1")?;
        let search_pattern = format!("%{}%", prompt.split_whitespace().next().unwrap_or(""));
        let mut rows = stmt.query(rusqlite::params![search_pattern])?;
        
        if let Some(row) = rows.next()? {
            let fact: String = row.get(0)?;
            println!("  [Ground Truth] Database Fact: {}", fact);
            // Calculate pseudo-veracity
            if response_text.contains(&fact[..20].trim()) {
                println!("  [Veracity] ✅ HIGH (Response matches DB Ground Truth)");
            } else {
                println!("  [Veracity] ⚠️ LOW (Response deviates from DB Ground Truth)");
            }
        }
    }

    Ok(())
}
