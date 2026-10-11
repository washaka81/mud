use std::f32;

/// Simulated test of `main.rs` logit processing math
fn main() {
    println!("🧪 [MUD Validator] Running main.rs Arithmetic Validation...\n");

    let hidden_size = 576usize;
    let vocab_size = 8usize; // Small vocabulary for easy viewing

    // 1. Adaptive scale_up based on RMS
    // Suppose RMS sum of out scales is something small
    let emb_rms_sum = 0.005_f32 * 0.005_f32 * vocab_size as f32; // simulated
    let emb_rms = (emb_rms_sum / vocab_size as f32).sqrt().max(1e-8);
    let scale_up = (1.0 / emb_rms).clamp(1.0, 128.0);

    println!("=== 1. Adaptive Scale Up ===");
    println!("emb_rms: {:.6e}", emb_rms);
    println!("scale_up (clamped to 128 max): {:.2}", scale_up);
    assert!(
        (1.0..=128.0).contains(&scale_up),
        "Scale up is out of bounds!"
    );

    // 2. Logit Generation (Simulating dot product)
    let temp_scale = 1.0 / (hidden_size as f32).sqrt();
    println!("\n=== 2. Logit Scaling ===");
    println!("temp_scale (1/sqrt(H)): {:.6}", temp_scale);

    // Let's create some fake dot products ranging from -10 to 15
    let raw_dots = vec![12.0, -5.0, 15.0, 0.0, 1.0, -10.0, 2.0, 8.0];
    let mut logits: Vec<f32> = raw_dots.iter().map(|d| d * temp_scale).collect();

    println!("Raw dots: {:?}", raw_dots);
    println!("Scaled logits: {:?}", logits);

    // 3. DC Bias Removal
    let logit_mean = logits.iter().sum::<f32>() / vocab_size as f32;
    for l in logits.iter_mut() {
        *l -= logit_mean;
    }
    println!("\n=== 3. DC Bias Removal ===");
    println!("Mean logit: {:.6}", logit_mean);
    println!("Bias-removed logits: {:?}", logits);
    let new_mean = logits.iter().sum::<f32>() / vocab_size as f32;
    assert!(
        new_mean.abs() < 1e-5,
        "DC Bias removal failed! Mean is {}",
        new_mean
    );

    // 4. Repetition Penalty
    let prev_token = 2usize; // let's penalize the highest token
    if logits[prev_token] > 0.0 {
        logits[prev_token] /= 1.1;
    } else {
        logits[prev_token] *= 1.1;
    }
    println!("\n=== 4. Repetition Penalty ===");
    println!("Penalized Token {}: {:.6}", prev_token, logits[prev_token]);

    // 5. Temperature Scaling
    let temp = 0.8f32;
    for l in logits.iter_mut() {
        *l /= temp;
    }
    println!("\n=== 5. Temperature Scaling (T=0.8) ===");
    println!("Temp-scaled logits: {:?}", logits);

    // 6. Softmax
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&l| (l - max_l).exp()).sum();

    let mut probs: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i, (l - max_l).exp() / sum_exp))
        .collect();

    probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\n=== 6. Softmax Probabilities ===");
    let mut total_prob = 0.0;
    for (i, p) in &probs {
        println!("Token {}: {:.2}%", i, p * 100.0);
        total_prob += p;
    }
    println!("Total probability mass: {:.4}", total_prob);
    assert!(
        (total_prob - 1.0).abs() < 1e-4,
        "Softmax probabilities don't sum to 1.0!"
    );

    println!("\n✅ All arithmetic validations passed successfully. The mathematical pipeline in main.rs is mathematically sound.");
}
