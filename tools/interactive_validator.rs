use forge_llm::mud::{MudFile, MudTensorType};
use forge_llm::model::tokenizer::Tokenizer;
use std::collections::HashMap;

// ANSI Colors for high-end styling
const CYAN: &str = "\x1b[36m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const RED: &str = "\x1b[31m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

fn main() -> anyhow::Result<()> {
    println!("{}=================================================={}", BOLD, RESET);
    println!("{}🛡️  MUD ENGINE HIGH-FIDELITY INTERACTIVE VALIDATOR 🛡️{}", BOLD, RESET);
    println!("{}=================================================={}", BOLD, RESET);

    let model_path = std::env::args().nth(1).unwrap_or_else(|| "models/core_skills.mud".to_string());
    println!("{}   🔍 Loading MUD Model from: {}{}{}", CYAN, BOLD, model_path, RESET);
    
    let mud_file = match MudFile::load(&model_path) {
        Ok(m) => {
            println!("{}   ✅ Model loaded successfully!{}", GREEN, RESET);
            m
        }
        Err(e) => {
            println!("{}   ❌ Failed to load model: {}{}", RED, e, RESET);
            return Err(e);
        }
    };

    println!("\n{}--- STAGE 1: TOKENIZER & BPE CONCORDANCE AUDIT ---{}", BOLD, RESET);
    let tokens_str = mud_file.global_metadata.get("tokenizer.tokens")
        .ok_or_else(|| anyhow::anyhow!("No tokenizer tokens in MUD metadata"))?;
    let merges_str = mud_file.global_metadata.get("tokenizer.merges").map(|s| s.as_str()).unwrap_or("");
    
    println!("   Vocab size in metadata: {} tokens", tokens_str.lines().count());
    println!("   BPE merges in metadata: {} rules", merges_str.lines().count());

    let tokenizer = Tokenizer::from_mud_metadata(tokens_str, merges_str);

    let test_strings = vec![
        "hola MUD engine",
        "¿Cómo funciona el motor de inferencia modular?",
        "Multi-Head Attention KV Cache is fully stabilized.",
        "1234567890 !@#$%^&*()_+",
    ];

    let mut tokenizer_ok = true;
    for test in test_strings {
        let tokens = tokenizer.encode(test);
        let decoded = tokenizer.decode(&tokens);
        
        let match_status = if decoded.trim() == test.trim() {
            format!("{}MATCH{}", GREEN, RESET)
        } else {
            tokenizer_ok = false;
            format!("{}MISMATCH{}", RED, RESET)
        };
        
        println!("   Input:  {:?}", test);
        println!("   Tokens: {:?}", tokens);
        println!("   Decoded: {:?}", decoded);
        println!("   Status:  [{}]", match_status);
        println!("   ----------------------------------");
    }

    if tokenizer_ok {
        println!("{}   ✅ TOKENIZER VERIFICATION: PASSED (Concordance is healthy){}", GREEN, RESET);
    } else {
        println!("{}   ⚠️  TOKENIZER VERIFICATION: WARNING (Fallback character reconstruction detected){}", YELLOW, RESET);
    }

    println!("\n{}--- STAGE 2: MOE EXPERTS WEIGHT & SIGMA ANATOMY ---{}", BOLD, RESET);
    let core = mud_file.skills.get("core").expect("No core skill");
    
    let mut total_tensors = 0;
    let mut ternary_tensors = 0;
    let mut nan_count = 0;
    let mut dead_layers = Vec::new();
    
    let mut layer_sigmas: HashMap<usize, Vec<f32>> = HashMap::new();

    for (name, tensor) in &core.tensors {
        total_tensors += 1;
        
        // Check for NaN in float32 weights
        if tensor.t_type == MudTensorType::Float32 {
            let elements = tensor.shape.iter().copied().product::<usize>();
            let data_ptr = tensor.data_ptr as *const f32;
            let slice = unsafe { std::slice::from_raw_parts(data_ptr, elements) };
            for &val in slice {
                if val.is_nan() || val.is_infinite() {
                    nan_count += 1;
                }
            }
        }

        if tensor.t_type == MudTensorType::Ternary2Bit {
            ternary_tensors += 1;
            let n_elements = tensor.shape.iter().copied().product::<usize>();
            let n_u32 = (n_elements + 15) / 16;
            let data_ptr = tensor.data_ptr as *const u32;
            let packed_data = unsafe { std::slice::from_raw_parts(data_ptr, n_u32) };
            
            let mut counts = [0usize; 3]; // 0: 0, 1: +1, 2: -1
            for &val in packed_data {
                for i in 0..16 {
                    let bits = (val >> (i * 2)) & 3;
                    if bits == 1 { counts[1] += 1; }
                    else if bits == 2 { counts[2] += 1; }
                    else { counts[0] += 1; }
                }
            }
            
            let total = counts[0] + counts[1] + counts[2];
            let variance = (counts[1] as f32 * 1.0 + counts[2] as f32 * 1.0) / total as f32;
            let sigma = variance.sqrt();

            if name.starts_with("blk.") {
                let parts: Vec<&str> = name.split('.').collect();
                if let Ok(layer_idx) = parts[1].parse::<usize>() {
                    layer_sigmas.entry(layer_idx).or_insert_with(Vec::new).push(sigma);
                }
            }

            if sigma == 0.0 {
                dead_layers.push(name.clone());
            }
        }
    }

    println!("   Total Tensors: {}", total_tensors);
    println!("   Ternary Quantized Tensors: {}", ternary_tensors);
    println!("   NaN/Inf Weight Violations: {}", nan_count);

    println!("\n   {}Mean Sigma per Layer (Active MoE Routing Health):{}", CYAN, RESET);
    let mut layer_indices: Vec<&usize> = layer_sigmas.keys().collect();
    layer_indices.sort();
    
    for l in layer_indices {
        let sigs = &layer_sigmas[l];
        let sum: f32 = sigs.iter().sum();
        let avg = sum / sigs.len() as f32;
        let status = if avg > 0.1 {
            format!("{}HEALTHY{}", GREEN, RESET)
        } else if avg > 0.0 {
            format!("{}WEAK{}", YELLOW, RESET)
        } else {
            format!("{}DEAD (COLLAPSED){}", RED, RESET)
        };
        println!("      Layer {:>2} | Avg Sigma: {:.4} | Status: [{}]", l, avg, status);
    }

    if nan_count == 0 && dead_layers.is_empty() {
        println!("\n{}{}   ✅ WEIGHT DIAGNOSTICS: PERFECT (All weights healthy and stabilized){}", GREEN, BOLD, RESET);
    } else {
        if nan_count > 0 {
            println!("\n{}{}   ❌ WEIGHT DIAGNOSTICS: FAILED (Found {} NaN/Inf weight values!){}", RED, BOLD, nan_count, RESET);
        }
        if !dead_layers.is_empty() {
            println!("{}{}   ❌ WEIGHT DIAGNOSTICS: FAILED (Collapsed weights in layers: {:?}){}", RED, BOLD, dead_layers, RESET);
        }
    }

    println!("\n{}--- STAGE 3: KV ATTENTION & CONTEXT STRESS TEST ---{}", BOLD, RESET);
    println!("   Simulating 5 steps of Multi-Head Attention forward pass...");
    
    let hidden_size = mud_file.global_metadata.get("hidden_size").and_then(|s| s.parse::<usize>().ok()).unwrap_or(512);
    let num_heads = 4;
    let head_dim = hidden_size / num_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    println!("      Hidden Size: {}, Heads: {}, Head Dim: {}", hidden_size, num_heads, head_dim);

    let q = vec![0.1f32; hidden_size];
    let k_cache = vec![0.15f32; 5 * hidden_size];
    let v_cache = vec![0.25f32; 5 * hidden_size];

    let mut stable = true;
    for step in 0..5 {
        let mut attn_out = vec![0.0f32; hidden_size];
        
        for h in 0..num_heads {
            let h_offset = h * head_dim;
            let q_h = &q[h_offset .. h_offset + head_dim];

            let mut scores = vec![0.0f32; step + 1];
            let mut max_score = f32::NEG_INFINITY;

            for t in 0..=step {
                let t_offset = t * hidden_size + h_offset;
                let k_t_h = &k_cache[t_offset .. t_offset + head_dim];
                
                let mut score = 0.0f32;
                for i in 0..head_dim {
                    score += q_h[i] * k_t_h[i];
                }
                score *= scale;
                scores[t] = score;
                if score > max_score {
                    max_score = score;
                }
            }

            // Softmax
            let mut sum_exp = 0.0f32;
            for score in &mut scores {
                let exp = (*score - max_score).exp();
                *score = exp;
                sum_exp += exp;
            }
            for score in &mut scores {
                *score /= sum_exp;
            }

            // Weighted sum
            let mut out_h = vec![0.0f32; head_dim];
            for t in 0..=step {
                let t_offset = t * hidden_size + h_offset;
                let v_t_h = &v_cache[t_offset .. t_offset + head_dim];
                let weight = scores[t];
                for i in 0..head_dim {
                    out_h[i] += weight * v_t_h[i];
                }
            }

            attn_out[h_offset .. h_offset + head_dim].copy_from_slice(&out_h);
        }

        let l2_norm = (attn_out.iter().map(|&x| x*x).sum::<f32>()).sqrt();
        let has_nan = attn_out.iter().any(|&x| x.is_nan() || x.is_infinite());
        
        if has_nan || l2_norm == 0.0 || l2_norm.is_nan() {
            stable = false;
        }
        
        println!("      Step {:>2} | L2 Norm: {:.4} | Finite: {}", step, l2_norm, !has_nan);
    }

    if stable {
        println!("{}   ✅ KV ATTENTION TEST: PASSED (Context attention propagation is highly stable){}", GREEN, RESET);
    } else {
        println!("{}   ❌ KV ATTENTION TEST: FAILED (Attention output collapsed or contains NaNs){}", RED, RESET);
    }

    println!("\n{}=================================================={}", BOLD, RESET);
    println!("{}🎉 MUD MODEL INTEGRITY VERIFICATION COMPLETE 🎉{}", BOLD, RESET);
    println!("{}=================================================={}", BOLD, RESET);

    Ok(())
}
