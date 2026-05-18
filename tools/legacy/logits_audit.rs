use forge_llm::gguf::GGUFModel;
use forge_llm::model::inference::ForgeInference;
use forge_llm::vulkan::VulkanContext;
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    let model_data = GGUFModel::load("models/MUD2.5-coder-1.5b-instruct-q4_0.gguf")?;
    let mut inf = ForgeInference::new(&model_data, Arc::new(VulkanContext::new()?))?;

    let prompt = "def fast_fibonacci(n):";
    println!("Prompt: {}", prompt);
    
    let tokens = inf.tokenizer.encode(prompt);
    let mut x = vec![0.0f32; inf.model.hidden_size];
    
    for (pos, &token) in tokens.iter().enumerate() {
        let row_ptr = unsafe { inf.embd_w.add(token as usize * (inf.model.hidden_size / 32)) };
        forge_llm::asm::dequantize_q4_0_row(row_ptr, &mut x, inf.model.hidden_size);
        inf.model.decode_step(&mut x, pos, &mut inf.kv_cache_k, &mut inf.kv_cache_v);
    }

    // Calcular logits para el siguiente token
    let vocab_size = inf.tokenizer.id_to_token.len();
    let mut logits = vec![0.0f32; vocab_size];
    
    unsafe {
        let scale = forge_llm::asm::rms_norm_scale_asm(inf.model.hidden_size, x.as_ptr(), 1e-6);
        for i in 0..inf.model.hidden_size { x[i] = x[i] * scale * (*inf.output_norm_w.add(i)); }
        
        let row_size_blocks = inf.model.hidden_size / 32;

        if inf.output_w.is_null() {
            println!("  ❌ ERROR: inf.output_w es NULL!");
            return Ok(());
        }

        // --- VERIFICACIÓN DE REFERENCIA (Logit 0) ---
        let mut row0_f32 = vec![0.0f32; inf.model.hidden_size];
        forge_llm::asm::dequantize_q4_0_row(inf.output_w, &mut row0_f32, inf.model.hidden_size);
        
        let weight_nan = row0_f32.iter().any(|v| v.is_nan());
        if weight_nan { println!("  ❌ ERROR: Pesos de output.weight contienen NaNs!"); }
        
        let x_nan = x.iter().any(|v| v.is_nan());
        if x_nan { println!("  ❌ ERROR: El vector de activación 'x' contiene NaNs antes del ASM!"); }

        let mut manual_sum = 0.0f32;
        for j in 0..inf.model.hidden_size { manual_sum += x[j] * row0_f32[j]; }
        
        let mut asm_val = 0.0f32;
        forge_llm::asm::q4_0_gemv_asm(inf.model.hidden_size, x.as_ptr(), inf.output_w, &mut asm_val);
        
        println!("  Logit 0 Reference -> Rust: {:.4}, ASM: {:.4} (Delta: {:.6})", 
                 manual_sum, asm_val, (manual_sum - asm_val).abs());

        for i in 0..vocab_size {
            let mut val = 0.0f32;
            forge_llm::asm::q4_0_gemv_asm(inf.model.hidden_size, x.as_ptr(), inf.output_w.add(i * row_size_blocks), &mut val);
            logits[i] = val;
        }
    }

    let mut nan_count = 0;
    let mut indexed: Vec<_> = logits.into_iter().enumerate()
        .filter(|(_, v)| {
            if v.is_nan() { nan_count += 1; false } else { true }
        })
        .collect();
    
    if nan_count > 0 { println!("  ⚠️ ADVERTENCIA: Se detectaron {} NaNs en los logits!", nan_count); }
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\nTop 10 tokens para la continuación:");
    for i in 0..10 {
        let (id, score) = indexed[i];
        let token = &inf.tokenizer.id_to_token[id];
        println!("  {:>2}. ID: {:>6} | Score: {:>8.4} | Token: {:?}", i+1, id, score, token);
    }

    Ok(())
}
