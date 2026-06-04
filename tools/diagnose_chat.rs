use forge_llm::mud::inference::MudInference;
/// Diagnostic chat: runs inference with instrumentation to identify where signal breaks down
use forge_llm::mud::MudFile;
use forge_llm::vulkan::VulkanContext;
use std::io::Write;
use std::sync::Arc;

fn stats(name: &str, v: &[f32]) {
    let n = v.len();
    if n == 0 {
        println!("  {}: EMPTY", name);
        return;
    }
    let sum: f32 = v.iter().sum();
    let mean = sum / n as f32;
    let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let variance: f32 = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
    let std = variance.sqrt();
    let nonzero = v.iter().filter(|&&x| x != 0.0).count();
    let nan_count = v.iter().filter(|x| x.is_nan()).count();
    let inf_count = v.iter().filter(|x| x.is_infinite()).count();
    println!(
        "  {}: mean={:.6}, std={:.6}, min={:.4}, max={:.4}, nonzero={}/{}, nan={}, inf={}",
        name, mean, std, min, max, nonzero, n, nan_count, inf_count
    );
}

fn main() -> anyhow::Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/qwen2_0.5b.mud".to_string());
    println!("=== Diagnostic Chat ===");
    println!("Loading: {}", model_path);

    let mud_file = MudFile::load(&model_path)?;
    let vk = VulkanContext::new().map(Arc::new).ok();
    let mut inference = MudInference::new(&mud_file, vk)?;

    let hidden = inference.model.hidden_size;
    println!(
        "hidden_size={}, num_layers={}, vocab_size={}",
        hidden,
        inference.model.layers.len(),
        inference.tokenizer.id_to_token.len()
    );

    // Test embedding a single token
    let test_token = 9707u32; // "Hola" or similar
    let mut x = vec![0.0f32; hidden];
    inference.embed_token(test_token, &mut x);
    stats("embed(token=9707)", &x);

    // Test embedding token 0
    let mut x0 = vec![0.0f32; hidden];
    inference.embed_token(0, &mut x0);
    stats("embed(token=0)", &x0);

    // Test a simple prompt - just 1 token
    let prompt = "Hola";
    let tokens = inference.tokenizer.encode(prompt);
    println!(
        "\nTokenized '{}': {:?} ({} tokens)",
        prompt,
        &tokens[..tokens.len().min(10)],
        tokens.len()
    );

    // Process first token manually and observe step output
    if let Some(&first_tok) = tokens.first() {
        let mut x = vec![0.0f32; hidden];
        inference.embed_token(first_tok, &mut x);
        stats(&format!("embed(token={})", first_tok), &x);

        // Run the step and observe output
        inference.step(&mut x, prompt, &[], 0);
        stats("after step(pos=0)", &x);
    }

    // Now do the full prompt processing
    let mut x = vec![0.0f32; hidden];
    let mut conversation_pos = 0;
    inference.prompt(prompt, &mut x, &mut conversation_pos);
    stats("after full prompt", &x);

    // Generate one token to see logits
    print!("\nGenerating response: ");
    std::io::stdout().flush()?;

    let (tokens_out, _) = inference.generate(&x, 20, prompt, &mut conversation_pos, 0, |_id, text| {
        print!("{}", text);
        let _ = std::io::stdout().flush();
    });
    println!();

    println!("\nGenerated {} tokens: {:?}", tokens_out.len(), tokens_out);

    // Check logits distribution after the last generate
    {
        let logits = inference.workspace.logits.read();
        stats("final logits", &logits);

        // Show top 10 logits
        let mut indexed: Vec<(usize, f32)> =
            logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        println!("\nTop 10 logits:");
        for (i, (idx, val)) in indexed.iter().take(10).enumerate() {
            let token_str = inference
                .tokenizer
                .id_to_token
                .get(*idx)
                .cloned()
                .unwrap_or_else(|| format!("<unk:{}>", idx));
            println!(
                "  #{}: token_id={} '{}' logit={:.4}",
                i + 1,
                idx,
                token_str,
                val
            );
        }

        println!("\nBottom 10 logits:");
        for (i, (idx, val)) in indexed.iter().rev().take(10).enumerate() {
            let token_str = inference
                .tokenizer
                .id_to_token
                .get(*idx)
                .cloned()
                .unwrap_or_else(|| format!("<unk:{}>", idx));
            println!(
                "  #{}: token_id={} '{}' logit={:.4}",
                i + 1,
                idx,
                token_str,
                val
            );
        }
    }

    println!("\n=== Diagnostic Complete ===");
    Ok(())
}
