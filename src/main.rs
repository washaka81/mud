use std::env;
use std::io::{self, Write};
use forge_llm::mud::MudFile;
use forge_llm::model::tokenizer::Tokenizer;
use forge_llm::mud::slime::SlimeWorkspace;
use forge_llm::mud::slime_forward::{evaluate_slime_block, SlimeLayer, apply_output_norm};
use rand::RngExt;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: forge_llm <model.mud>");
        return Ok(());
    }
    let model_path = &args[1];
    println!("Loading model: {}", model_path);
    let mud = MudFile::load(model_path)?;
    
    let core = mud.skills.get("core").ok_or_else(|| anyhow::anyhow!("Missing core skill"))?;
    
    let hidden_size = mud.global_metadata.get("hidden_size").and_then(|v| v.parse().ok()).expect("Missing hidden_size in metadata");
    let num_layers = mud.global_metadata.get("num_hidden_layers").or_else(|| mud.global_metadata.get("num_layers")).and_then(|v| v.parse().ok()).expect("Missing num_layers in metadata");
    let n_heads = mud.global_metadata.get("num_attention_heads").or_else(|| mud.global_metadata.get("num_heads")).and_then(|v| v.parse().ok()).expect("Missing num_heads in metadata");
    let n_kv_heads = mud.global_metadata.get("num_key_value_heads").or_else(|| mud.global_metadata.get("num_kv_heads")).and_then(|v| v.parse().ok()).expect("Missing num_kv_heads in metadata");
    let ffn_mid = mud.global_metadata.get("intermediate_size").or_else(|| mud.global_metadata.get("ffn_hidden")).and_then(|v| v.parse().ok()).expect("Missing ffn_mid in metadata");
    let vocab_size = mud.global_metadata.get("vocab_size").and_then(|v| v.parse().ok()).expect("Missing vocab_size in metadata");
    let rms_norm_eps = mud.global_metadata.get("rms_norm_eps").and_then(|v| v.parse().ok()).unwrap_or(1e-5);
    let max_pos = mud.global_metadata.get("max_position_embeddings").and_then(|v| v.parse().ok()).expect("Missing max_position_embeddings in metadata");
    let rope_theta = mud.global_metadata.get("rope.freq_base").or_else(|| mud.global_metadata.get("rope_theta")).and_then(|v| v.parse().ok()).unwrap_or(10000.0);

    let mut layers: Vec<SlimeLayer> = Vec::new();
    for blk in 0..num_layers {
        let prefix = format!("blk.{}.", blk);
        let t = |name: &str| -> *const u8 { core.tensors.get(&format!("{}{}.weight", prefix, name)).map(|t| t.data_ptr).unwrap_or(std::ptr::null()) };
        let ts = |name: &str| -> *const f32 { core.tensors.get(&format!("{}{}.prq_scale", prefix, name)).map(|t| t.data_ptr as *const f32).unwrap_or(std::ptr::null()) };
        let tn = |name: &str| -> *const f32 { core.tensors.get(&format!("{}{}.weight", prefix, name)).map(|t| t.data_ptr as *const f32).unwrap_or(std::ptr::null()) };
        let (ffn_up_name, ffn_gate_name) = if core.tensors.contains_key(&format!("{}expert.0.up.weight", prefix)) {
            ("expert.0.up", "expert.0.gate")
        } else { ("expert.0.w1", "expert.0.w3") };
        layers.push(SlimeLayer {
            q_w: t("attn_q"), k_w: t("attn_k"), v_w: t("attn_v"), o_w: t("attn_output"),
            q_scales: ts("attn_q"), k_scales: ts("attn_k"), v_scales: ts("attn_v"), o_scales: ts("attn_output"),
            ffn_up_w: t(ffn_up_name), ffn_gate_w: t(ffn_gate_name), ffn_down_w: t("expert.0.w2"),
            ffn_up_scales: ts(ffn_up_name), ffn_gate_scales: ts(ffn_gate_name), ffn_down_scales: ts("expert.0.w2"),
            attn_norm_w: tn("attn_norm"), ffn_norm_w: tn("norm"),
            attn_sub_norm_w: tn("attn_sub_norm"), ffn_sub_norm_w: tn("ffn_sub_norm"),
            mhc_alpha_w: tn("mhc_alpha"), mhc_beta_w: tn("mhc_beta"), mhc_radius_w: tn("mhc_radius"),
            n_kv_heads, ffn_mid, rope_theta,
        });
    }
    
    let tokens_str = mud.global_metadata.get("tokenizer.tokens").map(|s| s.as_str()).unwrap_or("");
    let merges_str = mud.global_metadata.get("tokenizer.merges").map(|s| s.as_str()).unwrap_or("");
    let tokenizer = Tokenizer::from_mud_metadata(tokens_str, merges_str);
    
    // Setup embedding and output weights
    let emb_tensor = core.tensors.get("token_embd.weight").ok_or_else(|| anyhow::anyhow!("Missing token_embd.weight"))?;
    let emb_ptr = emb_tensor.owned_data.as_ref().map(|d| d.as_ptr()).unwrap_or(emb_tensor.data_ptr);
    let emb_slice = unsafe { std::slice::from_raw_parts(emb_ptr as *const f32, vocab_size * hidden_size) };
    
    let out_tensor = core.tensors.get("output.weight").unwrap_or(emb_tensor);
    let out_ptr = out_tensor.owned_data.as_ref().map(|d| d.as_ptr()).unwrap_or(out_tensor.data_ptr);
    let out_slice = unsafe { std::slice::from_raw_parts(out_ptr as *const f32, vocab_size * hidden_size) };
    
    // Ternarize out_slice to match corpus_trainer.rs (removes adversarial continuous noise / DC bias)
    let mut out_slice_ternary = out_slice.to_vec();
    for v_idx in 0..vocab_size {
        let start = v_idx * hidden_size;
        let slice = &mut out_slice_ternary[start..start + hidden_size];
        let absmean = slice.iter().map(|v| v.abs()).sum::<f32>() / (hidden_size as f32);
        let scale = (absmean * 0.707).max(1e-8);
        for v in slice { *v = (*v / scale).round().clamp(-1.0, 1.0) * scale; }
    }
    
    // Ternarize out_slice to match corpus_trainer.rs (removes adversarial continuous noise / DC bias)
    let mut out_slice_ternary = out_slice.to_vec();
    for v_idx in 0..vocab_size {
        let start = v_idx * hidden_size;
        let slice = &mut out_slice_ternary[start..start + hidden_size];
        let absmean = slice.iter().map(|v| v.abs()).sum::<f32>() / (hidden_size as f32);
        let scale = (absmean * 0.707).max(1e-8);
        for v in slice { *v = (*v / scale).round().clamp(-1.0, 1.0) * scale; }
    }
    
    let out_norm_tensor = core.tensors.get("output_norm.weight").ok_or_else(|| anyhow::anyhow!("Missing output_norm.weight"))?;
    let out_norm_ptr = out_norm_tensor.owned_data.as_ref().map(|d| d.as_ptr()).unwrap_or(out_norm_tensor.data_ptr);
    let out_norm_slice = unsafe { std::slice::from_raw_parts(out_norm_ptr as *const f32, hidden_size) };

    let head_dim = hidden_size / n_heads;
    
    // Dynamically compute max_emb if missing to prevent mHC geometric prison
    let computed_max_emb = emb_slice.iter().map(|v| v.abs()).fold(0.0f32, |a, b| a.max(b));
    let max_emb = mud.global_metadata.get("max_emb").and_then(|v| v.parse().ok()).unwrap_or(computed_max_emb);
    
    let mut ws = SlimeWorkspace::new(hidden_size, max_pos, n_heads, n_kv_heads, head_dim, ffn_mid, num_layers, max_emb);

    println!("Model loaded successfully. Starting MUD Inference CLI...");
    println!("Type 'quit' or 'exit' to stop.");

    let stdin = io::stdin();
    
    loop {
        print!("\n> ");
        io::stdout().flush()?;
        let mut input = String::new();
        let bytes_read = stdin.read_line(&mut input)?;
        if bytes_read == 0 { break; } // EOF reached
        let input = input.trim();
        if input.is_empty() { continue; }
        if input == "quit" || input == "exit" { break; }

        let prompt = format!("<|user|>\n{}<|end|>\n<|assistant|>\n", input);
        let mut tokens = tokenizer.encode(&prompt);
        
        ws.kv_cache.fill(0.0);
        ws.v_cache.fill(0.0);
        ws.jepa_mu.fill(0.0);
        ws.jepa_inv_sigma.fill(0.0);
        ws.jepa_var_ema.fill(0.0);
        
        print!("AI: ");
        io::stdout().flush()?;
        
        let max_gen = 256;
        let mut generated = 0;
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut current_pos = 0;
        
        // Context processing
        while current_pos < tokens.len() {
            ws.clear_registers();
            let tid = tokens[current_pos] as usize;
            let emb_start = tid * hidden_size;
            
            for (i, v) in emb_slice[emb_start..emb_start + hidden_size].iter().enumerate() {
                forge_llm::mud::slime::SlimeRegister::init_from_embed(
                    &mut ws.registers[i],
                    &mut ws.jepa_z,
                    i,
                    hidden_size,
                    num_layers,
                    *v,
                    current_pos == 0
                );
            }
            
            for (l_idx, layer) in layers.iter().enumerate() {
                evaluate_slime_block(layer, l_idx, &mut ws, current_pos, rms_norm_eps, None);
            }
            
            current_pos += 1;
        }

        // Auto-regressive generation loop
        while generated < max_gen {
            apply_output_norm(&mut ws, out_norm_slice.as_ptr(), rms_norm_eps);
            
            let mut logits = vec![0.0f32; vocab_size];
            let mut reg_f32: Vec<f32> = ws.registers.iter().map(|r| r.read_accum()).collect();
            // Center activations to remove DC bias from hidden state
            // The partially-converged model has a mean offset in reg_f32 that
            // projects maximally onto specific token embeddings (e.g. "outheastern").
            let reg_mean = reg_f32.iter().sum::<f32>() / hidden_size as f32;
            for v in reg_f32.iter_mut() { *v -= reg_mean; }
            let scale_up = 24.0;
            for (v_idx, logit) in logits.iter_mut().enumerate().take(vocab_size) {
                let start = v_idx * hidden_size;
                let mut dot = 0.0;
                for i in 0..hidden_size {
                    dot += reg_f32[i] * out_slice_ternary[start + i];
                }
                *logit = dot * scale_up;
            }

            // DC Bias Removal: subtract mean logit to eliminate structural bias
            // A partially-converged model accumulates a DC offset on all logits.
            // Subtracting the mean is mathematically neutral (softmax-invariant)
            // but removes the structural attractor that causes token dominance.
            let logit_mean = logits.iter().sum::<f32>() / vocab_size as f32;
            for l in logits.iter_mut() {
                *l -= logit_mean;
            }

            // Strong Repetition Penalty
            for &prev_token in &generated_tokens {
                if logits[prev_token as usize] > 0.0 {
                    logits[prev_token as usize] /= 10.0;
                } else {
                    logits[prev_token as usize] *= 10.0;
                }
            }

            // Doppler-Shift Temperature
            let entropy = forge_llm::mud::self_play::calculate_shannon_entropy(&logits);
            let temp = if entropy < 1.5 { 1.5 } else { 0.8 };
            
            for l in logits.iter_mut() {
                *l /= temp;
            }

            // Softmax & Top-P (0.95)
            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = logits.iter().map(|&l| (l - max_l).exp()).sum();
            
            let mut probs: Vec<(usize, f32)> = logits.iter().enumerate()
                .map(|(i, &l)| (i, (l - max_l).exp() / sum_exp))
                .collect();
                
            probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            
            let mut cumsum = 0.0;
            let mut top_p_probs = Vec::new();
            for p in probs {
                cumsum += p.1;
                top_p_probs.push(p);
                if cumsum >= 0.95 { break; }
            }

            let top_p_sum: f32 = top_p_probs.iter().map(|(_, p)| p).sum();
            let mut rng = rand::rng();
            let mut r: f32 = rng.random();
            
            let mut best_idx = top_p_probs.last().unwrap().0;
            for &(idx, p) in &top_p_probs {
                let norm_p = p / top_p_sum;
                if r < norm_p {
                    best_idx = idx;
                    break;
                }
                r -= norm_p;
            }
            
            let next_token = best_idx as u32;
            tokens.push(next_token);
            generated_tokens.push(next_token);
            
            let decoded_piece = tokenizer.decode(&[next_token]);
            print!("{}", decoded_piece);
            io::stdout().flush()?;
            
            if decoded_piece.contains("<|end|>") || next_token == 128001 { // End token fallback
                break; 
            }
            
            ws.clear_registers();
            let emb_start = best_idx * hidden_size;
            for (i, v) in emb_slice[emb_start..emb_start + hidden_size].iter().enumerate() {
                forge_llm::mud::slime::SlimeRegister::init_from_embed(
                    &mut ws.registers[i],
                    &mut ws.jepa_z,
                    i,
                    hidden_size,
                    num_layers,
                    *v,
                    false
                );
            }
            
            for (l_idx, layer) in layers.iter().enumerate() {
                evaluate_slime_block(layer, l_idx, &mut ws, current_pos, rms_norm_eps, None);
            }
            


            current_pos += 1;
            generated += 1;
        }
        println!();
    }

    Ok(())
}
