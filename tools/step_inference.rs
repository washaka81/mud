use forge_llm::vulkan::VulkanContext;
use forge_llm::mud::MudFile;
use forge_llm::mud::inference::MudInference;
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    let mud_path = "models/core_skills.mud";
    let vk = Arc::new(VulkanContext::new()?);
    let mud_file = MudFile::load(mud_path)?;
    let mut engine = MudInference::new(&mud_file, vk)?;

    let prompt = "La verdad absoluta";
    let mut x = vec![0.0f32; engine.model.hidden_size];
    let mut conversation_pos = 0;
    
    println!("--- EJERCICIO DE INFERENCIA DESGLOSADA (PROBABILIDADES) ---");
    println!("Prompt Inicial: {}\n", prompt);

    let prompt_tokens = engine.tokenizer.encode(prompt);
    let mut tokens = Vec::new();

    // 1. Carga inicial de 3 palabras (tokens)
    for &t in prompt_tokens.iter().take(3) {
        engine.embed_token(t, &mut x);
        engine.step(&mut x, prompt, &[], conversation_pos);
        conversation_pos += 1;
        tokens.push(t);
        print!("{} ", engine.tokenizer.decode(&[t]));
    }
    println!("\n[3 tokens iniciales cargados]");

    // 2. Bucle de predicción paso a paso hasta 48 palabras
    for step in 4..=48 {
        let ws = &mut engine.workspace;
        ws.logits.fill(0.0);
        
        // Predicción de la siguiente palabra mediante producto de matrices (Hidden state * Embeddings)
        for i in 0..ws.logits.len() {
            let row_ptr = unsafe { engine.embd_w.add(i * (engine.model.hidden_size / 16)) };
            unsafe { forge_llm::asm::ternary_gemv_avx2(engine.model.hidden_size, x.as_ptr(), row_ptr, &mut ws.logits[i], 1.0); }
        }

        // Obtener las Top 3 probabilidades para este paso
        let mut probs: Vec<(usize, f32)> = ws.logits.iter().enumerate()
            .filter(|(i, &l)| *i < engine.tokenizer.id_to_token.len() && l.is_finite())
            .map(|(i, &l)| (i, l))
            .collect();
        
        if probs.is_empty() {
            println!("  ⚠️ No valid probabilities found for step {}. Using fallback token.", step);
            probs.push((0, 0.0));
        }

        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let (next_id, score) = probs[0];
        let next_token = next_id as u32;
        let word = engine.tokenizer.decode(&[next_token]);

        println!("\n[Paso {}] Prediciendo palabra #{}...", step, step);
        println!("  Candidatos:");
        for (i, p) in probs.iter().take(3).enumerate() {
            println!("    {}. '{}' (Logit: {:.2})", i+1, engine.tokenizer.decode(&[p.0 as u32]), p.1);
        }
        println!("  Elegida: '{}' con confianza {:.2}", word, score);

        tokens.push(next_token);
        engine.embed_token(next_token, &mut x);
        engine.step(&mut x, "", &[], conversation_pos);
        conversation_pos += 1;

        if next_token == 2 { 
            println!("\n[FIN] Token EOS detectado.");
            break; 
        }
    }

    println!("\n--- CUENTO FINAL GENERADO POR MUD ---");
    println!("{}", engine.tokenizer.decode(&tokens));
    
    Ok(())
}

#[test]
fn dummy_test_to_check_compilation() {}
