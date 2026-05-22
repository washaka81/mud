use forge_llm::vulkan::VulkanContext;
use forge_llm::mud::MudFile;
use forge_llm::mud::inference::MudInference;
use std::sync::Arc;

fn calculate_variance(data: &[f32]) -> f32 {
    let n = data.len() as f32;
    if n == 0.0 { return 0.0; }
    let mean = data.iter().sum::<f32>() / n;
    let var = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
    var
}

fn calculate_logit_entropy(logits: &[f32]) -> f32 {
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
    let sum_exp: f32 = exps.iter().sum();
    
    let mut entropy = 0.0f32;
    for e in exps {
        let p = e / (sum_exp + 1e-9);
        if p > 1e-9 {
            entropy -= p * p.log2();
        }
    }
    entropy
}

fn main() -> anyhow::Result<()> {
    let mud_path = "models/core_skills.mud";
    let vk = Arc::new(VulkanContext::new()?);
    let mud_file = MudFile::load(mud_path)?;
    let mut engine = MudInference::new(&mud_file, vk)?;

    let prompt = "La verdad absoluta";
    let mut x = vec![0.0f32; engine.model.hidden_size];
    let mut prev_x = x.clone();
    let mut conversation_pos = 0;

    println!("{:<5} | {:<15} | {:<8} | {:<8} | {:<8} | {:<8}", 
             "Paso", "Token", "LogitVar", "Entropy", "X-Move", "X-Sigma");
    println!("{}", "-".repeat(70));

    let prompt_tokens = engine.tokenizer.encode(prompt);
    for &t in &prompt_tokens {
        engine.embed_token(t, &mut x);
        engine.step(&mut x, prompt, &[], conversation_pos);
        conversation_pos += 1;
        prev_x = x.clone();
    }

    for step in 1..=48 {
        let ws = &mut engine.workspace;
        ws.logits.fill(0.0);
        
        // 1. Logit Calculation
        for i in 0..ws.logits.len() {
            let row_ptr = unsafe { engine.embd_w.add(i * (engine.model.hidden_size / 16)) };
            unsafe { forge_llm::asm::ternary_gemv_avx2(engine.model.hidden_size, x.as_ptr(), row_ptr, &mut ws.logits[i], 1.0); }
        }

        // 2. Statistics Calculation
        let l_var = calculate_variance(&ws.logits);
        let l_entropy = calculate_logit_entropy(&ws.logits);
        
        // Euclidean distance to previous state (X-Move)
        let x_move = x.iter().zip(prev_x.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
        let _x_sigma = calculate_variance(&x).sqrt();

        // 3. Selection
        let next_id = ws.logits.iter().enumerate()
            .filter(|(i, _)| *i > 3)
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap().0 as u32;

        let active_experts = *engine.active_experts.read().unwrap();

        let word = engine.tokenizer.decode(&[next_id]);
        
        println!("{:<5} | {:<15} | {:<8.2} | {:<8.2} | {:<8.2} | Exp: {}", 
                 step, word, l_var, l_entropy, x_move, active_experts);

        // 4. Update
        prev_x = x.clone();
        engine.embed_token(next_id, &mut x);
        engine.step(&mut x, "", &[], conversation_pos);
        conversation_pos += 1;

        if next_id == 2 { break; }
    }

    println!("\n--- DIAGNÓSTICO MATEMÁTICO ---");
    println!("1. Si 'X-Move' tiende a 0: El estado interno se ha congelado (Stagnation).");
    println!("2. Si 'Entropy' es < 1.0: El modelo ha perdido la duda y repite mecánicamente.");
    println!("3. Si 'LogitVar' explota: Los pesos están saturando el un-embedding.");

    Ok(())
}
