use forge_llm::gguf::GGUFModel;
use forge_llm::model::inference::ForgeInference;
use forge_llm::vulkan::VulkanContext;
use forge_llm::asm::dequantize_q4_0_row;
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    let model_data = GGUFModel::load("models/MUD2.5-coder-1.5b-instruct-q4_0.gguf")?;
    let mut inf = ForgeInference::new(&model_data, Arc::new(VulkanContext::new()?))?;

    let prompt = "The capital of France is";
    let tokens = inf.tokenizer.encode(prompt);
    println!("Prompt tokens: {:?}", tokens);
    
    let mut x = vec![0.0f32; inf.model.hidden_size];
    
    // Procesar Token 0 y predecir Token 1
    let t0 = tokens[0];
    let row_ptr = unsafe { inf.embd_w.add(t0 as usize * (inf.model.hidden_size / 32)) };
    forge_llm::asm::dequantize_q4_0_row(row_ptr, &mut x, inf.model.hidden_size);
    inf.model.decode_step(&mut x, 0, &mut inf.kv_cache_k, &mut inf.kv_cache_v);
    
    // Logits
    let vocab_size = inf.tokenizer.id_to_token.len();
    let mut logits = vec![0.0f32; vocab_size];
    unsafe {
        let scale = forge_llm::asm::rms_norm_scale_asm(inf.model.hidden_size, x.as_ptr(), 1e-6);
        for i in 0..inf.model.hidden_size { x[i] = x[i] * scale * (*inf.output_norm_w.add(i)); }
        // --- PRUEBA DE TRANSPOSICIÓN (LOGITS) ---
        let mut row_f32 = vec![0.0f32; inf.model.hidden_size];
        
        println!("Testing Row-major vs Column-major for Logits:");
        
        // Scenario A: Row-major (Current)
        // ... already calculated in the loop below ...

        // Scenario B: Column-major
        // (x * W_transposed)
        // Nota: Q4_0 no permite transpuesto fácil, así que esto es solo conceptual.
        // Pero si el modelo fuera Column-major, el GEMV actual daría basura (como ahora).

        for i in 0..vocab_size {
            let weight_row_ptr = inf.output_w.add(i * (inf.model.hidden_size / 32));
            dequantize_q4_0_row(weight_row_ptr, &mut row_f32, inf.model.hidden_size);
            let mut sum = 0.0f32;
            for j in 0..inf.model.hidden_size { sum += x[j] * row_f32[j]; }
            logits[i] = sum;
        }
    }
    
    let mut indexed: Vec<_> = logits.into_iter().enumerate()
        .filter(|(_, v)| !v.is_nan())
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    println!("\nPredicción para el token 2 (esperado ID {}):", tokens[1]);
    for i in 0..5 {
        let (id, _score) = indexed[i];
        println!("  {:>2}. ID: {:>6} | Token: {:?}", i+1, id, &inf.tokenizer.id_to_token[id]);
    }

    Ok(())
}
