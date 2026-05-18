use forge_llm::ai::MudFile;
use forge_llm::ai::inference::MudInference;
use forge_llm::vulkan::VulkanContext;
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    println!("=== MUD DEEP STATISTICAL & INFORMATION THEORY AUDIT ===");
    let model_path = "models/core_skills.ai";
    let vk = Arc::new(VulkanContext::new()?);
    let mud_file = MudFile::load(model_path)?;
    let engine = MudInference::new(&mud_file, vk)?;

    let hidden = engine.model.hidden_size;
    let mut x = vec![0.0f32; hidden];

    // 1. Initial Signal Integrity (Logarithmic Variance)
    println!("\n[1. Input Signal Dispersion]");
    engine.embed_token(3, &mut x); // "hola"
    let (mean, var, sigma) = calc_stats(&x);
    println!("  Embedding ('hola') -> Mean: {:.4}, Var: {:.4}, Sigma: {:.4}", mean, var, sigma);

    // 2. Step-by-Step Trace (Explosion Point Identification)
    println!("\n[2. Activation Flow Audit]");
    let mut x_trace = x.clone();
    
    // We manually trace parts of engine.step logic to find the explosion
    // Trace 1: RMSNorm effect
    let scale = calc_rms_norm_scale(&x_trace);
    println!("  RMSNorm Scale Factor: {:.8}", scale);
    for i in 0..hidden { x_trace[i] *= scale; }
    let (_, _, s1) = calc_stats(&x_trace);
    println!("  Post-RMSNorm Sigma: {:.4}", s1);

    // Trace 2: Expert w1 projection (The potential exploder)
    let expert = &engine.model.layers[0].experts[0];
    let mut w1_out = vec![0.0f32; engine.model.ffn_hidden_size];
    unsafe {
        for i in 0..engine.model.ffn_hidden_size {
            let row_ptr = expert.w1.add(i * (hidden / 16));
            let mut val = 0.0f32;
            forge_llm::asm::ternary_gemv_avx2(hidden, x_trace.as_ptr(), row_ptr, &mut val, 1.0);
            w1_out[i] = val;
        }
    }
    let (_, v_w1, s_w1) = calc_stats(&w1_out);
    println!("  Expert W1 Projection (Linear) -> Var: {:.2e}, Sigma: {:.2e}", v_w1, s_w1);

    // Trace 3: SiLU Gating effect
    for i in 0..w1_out.len() {
        let silu = w1_out[i] * (1.0 / (1.0 + (-w1_out[i]).exp()));
        w1_out[i] = silu;
    }
    let (_, v_silu, s_silu) = calc_stats(&w1_out);
    println!("  Post-SiLU Gating -> Var: {:.2e}, Sigma: {:.2e}", v_silu, s_silu);

    // 3. Information Theory: Confidence & Entropy
    println!("\n[3. Logits Confidence & Entropy]");
    let mut logits = vec![0.0f32; 100];
    let mut row_buf = vec![0.0f32; hidden];
    for i in 0..100 {
        let row_ptr = unsafe { engine.embd_w.add(i * (hidden / 16)) };
        dequantize_ternary_ref(row_ptr, &mut row_buf, hidden);
        let mut dot = 0.0f32;
        for j in 0..hidden { dot += x_trace[j] * row_buf[j]; }
        logits[i] = dot;
    }

    // Softmax and Entropy
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0f32;
    let exps: Vec<f32> = logits.iter().map(|&l| {
        let e = (l - max_l).exp();
        sum_exp += e;
        e
    }).collect();
    
    let probs: Vec<f32> = exps.iter().map(|&e| e / sum_exp).collect();

    let entropy = -probs.iter().map(|&p| if p > 0.0 { p * p.ln() } else { 0.0 }).sum::<f32>();
    let max_entropy = (probs.len() as f32).ln();
    let confidence = 1.0 - (entropy / max_entropy);
    
    println!("  Distribution Entropy: {:.4} (Max possible: {:.4})", entropy, max_entropy);
    println!("  Model Confidence: {:.2}%", confidence * 100.0);

    // 4. Covariance: Input-Output Alignment
    let covariance = calc_covariance(&x, &x_trace);
    println!("\n[4. Signal Preservation]");
    println!("  Input-Output Covariance: {:.4e}", covariance);
    
    if confidence < 0.05 {
        println!("  ❌ DIAGNOSIS: TOTAL SIGNAL COLLAPSE. The model is guessing randomly due to logit saturation.");
    } else if s_w1 > 1e6 {
        println!("  ❌ DIAGNOSIS: NUMERICAL EXPLOSION. Expert weights are amplifying signals exponentially.");
    }

    Ok(())
}

fn calc_stats(data: &[f32]) -> (f32, f32, f32) {
    let n = data.len() as f32;
    let mean = data.iter().sum::<f32>() / n;
    let var = data.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
    (mean, var, var.sqrt())
}

fn calc_rms_norm_scale(x: &[f32]) -> f32 {
    let n = x.len() as f32;
    let sum_sq = x.iter().map(|&v| v * v).sum::<f32>();
    1.0 / (sum_sq / n + 1e-6).sqrt()
}

fn calc_covariance(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let (mean_a, _, _) = calc_stats(a);
    let (mean_b, _, _) = calc_stats(b);
    a.iter().zip(b.iter()).map(|(&va, &vb)| (va - mean_a) * (vb - mean_b)).sum::<f32>() / n
}

fn dequantize_ternary_ref(packed: *const u32, out: &mut [f32], n: usize) {
    let u32_count = n / 16;
    unsafe {
        for i in 0..u32_count {
            let val = *packed.add(i);
            for j in 0..16 {
                let bits = (val >> (j * 2)) & 3;
                out[i * 16 + j] = match bits { 1 => 1.0, 2 => -1.0, _ => 0.0 };
            }
        }
    }
}
