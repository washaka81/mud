use forge_llm::mud::inference::MudInference;
/// Deep layer-by-layer diagnostic: trace signal magnitudes through each layer
use forge_llm::mud::MudFile;
use forge_llm::vulkan::VulkanContext;
use std::sync::Arc;

fn stats_brief(v: &[f32]) -> String {
    let n = v.len();
    if n == 0 {
        return "EMPTY".to_string();
    }
    let sum: f32 = v.iter().sum();
    let mean = sum / n as f32;
    let variance: f32 = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
    let std = variance.sqrt();
    let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    format!("mean={:.6} std={:.4} [{:.4}, {:.4}]", mean, std, min, max)
}

fn main() -> anyhow::Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/qwen2_0.5b.mud".to_string());
    let mud_file = MudFile::load(&model_path)?;
    let vk = VulkanContext::new().map(Arc::new).ok();
    let mut inference = MudInference::new(&mud_file, vk)?;

    let hidden = inference.model.hidden_size;
    println!("=== Layer-by-Layer Signal Trace ===");
    println!("hidden={}, layers={}", hidden, inference.model.layers.len());

    // Embed a token
    let mut x = vec![0.0f32; hidden];
    let token = 68012u32; // "Hola"
    inference.embed_token(token, &mut x);
    println!("embed: {}", stats_brief(&x));

    // Run RMS norm manually to see what happens to the signal
    let scale = unsafe { forge_llm::asm::rms_norm_scale_asm(hidden, x.as_ptr(), 1e-6) };
    println!("rms_norm_scale of embedding: {:.6}", scale);

    // What does the normalized x look like?
    if let forge_llm::mud::inference::MudLayer::Attention(layer) = &inference.model.layers[0] {
        let norm_ptr = layer.attn_norm_w;
        let mut x_norm = vec![0.0f32; hidden];
        if !norm_ptr.is_null() {
            for i in 0..hidden {
                x_norm[i] = x[i] * scale * unsafe { *norm_ptr.add(i) };
            }
            println!("x_norm (layer 0 attn): {}", stats_brief(&x_norm));
        }

        // Check what norm weights look like
        if !norm_ptr.is_null() {
            let norm_w: Vec<f32> = (0..hidden).map(|i| unsafe { *norm_ptr.add(i) }).collect();
            println!("attn_norm_w (layer 0): {}", stats_brief(&norm_w));
        }

        // Check the Q projection weight scale
        println!(
            "Layer 0 Scale Pointers: Q={:?}, K={:?}, V={:?}, O={:?}",
            layer.attn_q_scales, layer.attn_k_scales, layer.attn_v_scales, layer.attn_o_scales
        );
        if !layer.experts.is_empty() {
            println!(
                "Layer 0 Expert 0 Scales: W1={:?}, W2={:?}, W3={:?}",
                layer.experts[0].w1_scales, layer.experts[0].w2_scales, layer.experts[0].w3_scales
            );
        }

        // Run Q GEMV manually on the workspace
        {
            let ws = &inference.workspace;
            ws.x_norm.write().copy_from_slice(&x_norm);
            ws.q.write().fill(0.0);

            let q_out = inference.model.num_heads * inference.model.head_dim;
            MudInference::gemv_vulkan_or_cpu(
                None,
                "test_q",
                hidden,
                q_out,
                &ws.x_norm,
                layer.attn_q_w,
                layer.attn_q_scales,
                &ws.q,
                false,
            );

            let q_guard = ws.q.read();
            println!("Q projection (layer 0): {}", stats_brief(&q_guard));
        }
    } else if let forge_llm::mud::inference::MudLayer::Mamba(_layer) = &inference.model.layers[0] {
        println!("Layer 0 is MAMBA");
        // Similar diagnostics for Mamba could be added here
    }

    // Now run the full step
    let mut x2 = x.clone();
    inference.step(&mut x2, "test", &[], 0);
    println!("\nafter step(pos=0): {}", stats_brief(&x2));

    // Check what happens after step with larger position (more KV cache)
    let prompt_tokens = [68012u32, 11, 108385, 99185]; // some tokens
    let mut x_seq = vec![0.0f32; hidden];
    for (pos, &tok) in prompt_tokens.iter().enumerate() {
        inference.embed_token(tok, &mut x_seq);
        println!("  token={}: embed {}", tok, stats_brief(&x_seq));
        inference.step(&mut x_seq, "test", &[], pos);
        println!("  token={}: after_step {}", tok, stats_brief(&x_seq));
    }

    // Now check logits computation
    {
        let ws = &inference.workspace;
        ws.logits.write().fill(0.0);

        // Output projection (ternary embedding as output projection)
        let embd_type = inference.embd_type;
        println!("\nembd_type: {:?}", embd_type);

        if embd_type == forge_llm::mud::MudTensorType::Ternary2Bit {
            let x_unified = forge_llm::mud::inference::UnifiedBuffer::new_cpu(hidden);
            x_unified.write().copy_from_slice(&x_seq);

            let n_out_proj = ws.logits.read().len();
            MudInference::gemv_vulkan_or_cpu(
                None,
                "output_proj_diag",
                hidden,
                n_out_proj.min(1000),
                &x_unified,
                inference.embd_w_u32,
                inference.embd_scales,
                &ws.logits,
                false,
            );

            let logits = ws.logits.read();
            println!(
                "logits (first 1000, no embed scale): {}",
                stats_brief(&logits[..1000])
            );
        }

        // Apply embed_scales
        if !inference.embd_scales.is_null() {
            let mut logits_guard = ws.logits.write();
            for i in 0..1000 {
                logits_guard[i] *= unsafe { *inference.embd_scales.add(i) };
            }
        }

        {
            let logits = ws.logits.read();
            println!(
                "logits (first 1000, with embed scale): {}",
                stats_brief(&logits[..1000])
            );
        }
    }

    println!("\n=== Trace Complete ===");
    Ok(())
}
