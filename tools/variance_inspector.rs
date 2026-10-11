use forge_llm::model::tokenizer::Tokenizer;
use forge_llm::mud::slime::SlimeWorkspace;
use forge_llm::mud::slime_forward::{evaluate_slime_block, SlimeLayer};
use forge_llm::mud::MudFile;
use std::env;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: variance_inspector <model.mud>");
        return Ok(());
    }
    let model_path = &args[1];
    println!("Loading model: {}", model_path);
    let mud = MudFile::load(model_path)?;

    let core = mud
        .skills
        .get("core")
        .ok_or_else(|| anyhow::anyhow!("Missing core skill"))?;

    let hidden_size: usize = mud
        .global_metadata
        .get("hidden_size")
        .and_then(|v| v.parse().ok())
        .unwrap();
    let num_layers: usize = mud
        .global_metadata
        .get("num_hidden_layers")
        .or_else(|| mud.global_metadata.get("num_layers"))
        .and_then(|v| v.parse().ok())
        .unwrap();
    let n_heads: usize = mud
        .global_metadata
        .get("num_attention_heads")
        .or_else(|| mud.global_metadata.get("num_heads"))
        .and_then(|v| v.parse().ok())
        .unwrap();
    let n_kv_heads: usize = mud
        .global_metadata
        .get("num_key_value_heads")
        .or_else(|| mud.global_metadata.get("num_kv_heads"))
        .and_then(|v| v.parse().ok())
        .unwrap();
    let ffn_mid: usize = mud
        .global_metadata
        .get("intermediate_size")
        .or_else(|| mud.global_metadata.get("ffn_hidden"))
        .and_then(|v| v.parse().ok())
        .unwrap();
    let vocab_size: usize = mud
        .global_metadata
        .get("vocab_size")
        .and_then(|v| v.parse().ok())
        .unwrap();
    let rms_norm_eps: f32 = mud
        .global_metadata
        .get("rms_norm_eps")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1e-5);
    let max_pos: usize = mud
        .global_metadata
        .get("max_position_embeddings")
        .and_then(|v| v.parse().ok())
        .unwrap();
    let rope_theta: f32 = mud
        .global_metadata
        .get("rope.freq_base")
        .and_then(|v| v.parse().ok())
        .unwrap_or(10000.0);

    let mut layers: Vec<SlimeLayer> = Vec::new();
    for blk in 0..num_layers {
        let prefix = format!("blk.{}.", blk);
        let t = |name: &str| -> *const u8 {
            core.tensors
                .get(&format!("{}{}.weight", prefix, name))
                .map(|t| t.data_ptr)
                .unwrap_or(std::ptr::null())
        };
        let ts = |name: &str| -> *const f32 {
            core.tensors
                .get(&format!("{}{}.prq_scale", prefix, name))
                .map(|t| t.data_ptr as *const f32)
                .unwrap_or(std::ptr::null())
        };
        let tn = |name: &str| -> *const f32 {
            core.tensors
                .get(&format!("{}{}.weight", prefix, name))
                .map(|t| t.data_ptr as *const f32)
                .unwrap_or(std::ptr::null())
        };
        let ffn = forge_llm::mud::moe_load::dense_ffn_names_for_train(&core.tensors, blk);
        layers.push(SlimeLayer {
            q_w: t("attn_q"),
            k_w: t("attn_k"),
            v_w: t("attn_v"),
            o_w: t("attn_output"),
            q_scales: ts("attn_q"),
            k_scales: ts("attn_k"),
            v_scales: ts("attn_v"),
            o_scales: ts("attn_output"),
            ffn_up_w: t(&ffn.up),
            ffn_gate_w: t(&ffn.gate),
            ffn_down_w: t(&ffn.down),
            ffn_up_scales: ts(&ffn.up),
            ffn_gate_scales: ts(&ffn.gate),
            ffn_down_scales: ts(&ffn.down),
            attn_norm_w: tn("attn_norm"),
            ffn_norm_w: {
                let a = tn("ffn_norm");
                if !a.is_null() {
                    a
                } else {
                    tn("norm")
                }
            },
            attn_sub_norm_w: tn("attn_sub_norm"),
            ffn_sub_norm_w: tn("ffn_sub_norm"),
            q_norm_w: tn("attn_q_norm"),
            k_norm_w: tn("attn_k_norm"),
            mhc_alpha_w: tn("mhc_alpha"),
            mhc_beta_w: tn("mhc_beta"),
            mhc_radius_w: tn("mhc_radius"),
            n_kv_heads,
            ffn_mid,
            rope_theta,
        });
    }

    let tokens_str = mud
        .global_metadata
        .get("tokenizer.tokens")
        .map(|s| s.as_str())
        .unwrap_or("");
    let merges_str = mud
        .global_metadata
        .get("tokenizer.merges")
        .map(|s| s.as_str())
        .unwrap_or("");
    let tokenizer = Tokenizer::from_mud_metadata(tokens_str, merges_str);

    let emb_tensor = core.tensors.get("token_embd.weight").unwrap();
    let mut emb_f32 = vec![0.0f32; vocab_size * hidden_size];

    if emb_tensor.t_type == forge_llm::mud::MudTensorType::Float32 {
        let emb_data_ptr = emb_tensor
            .owned_data
            .as_ref()
            .map(|d| d.as_ptr() as *const f32)
            .unwrap_or(emb_tensor.data_ptr as *const f32);
        let emb_f32_slice =
            unsafe { std::slice::from_raw_parts(emb_data_ptr, vocab_size * hidden_size) };
        emb_f32.copy_from_slice(emb_f32_slice);
    } else {
        let emb_scales_tensor = core.tensors.get("token_embd.prq_scale").unwrap();
        let emb_packed_ptr = emb_tensor
            .owned_data
            .as_ref()
            .map(|d| d.as_ptr())
            .unwrap_or(emb_tensor.data_ptr);
        let emb_byte_len = vocab_size * hidden_size / 8 * 4;
        let emb_packed: &[u8] = unsafe { std::slice::from_raw_parts(emb_packed_ptr, emb_byte_len) };
        let emb_scales_ptr = emb_scales_tensor.data_ptr as *const f32;
        let emb_scales: &[f32] = unsafe { std::slice::from_raw_parts(emb_scales_ptr, vocab_size) };
        forge_llm::mud::slime_backward::unpack_ternary2bit_to_f32(
            emb_packed,
            emb_scales,
            hidden_size,
            &mut emb_f32,
        );
    }

    let head_dim = hidden_size / n_heads;
    let max_emb = 10.0;
    let mut workspace = SlimeWorkspace::new(
        hidden_size,
        max_pos,
        n_heads,
        n_kv_heads,
        head_dim,
        ffn_mid,
        num_layers,
        max_emb,
    );

    // Apply scale_up dynamically like in inference
    let mut emb_sq_sum = 0.0;
    for v in &emb_f32[0..hidden_size] {
        emb_sq_sum += v * v;
    }
    let emb_rms = (emb_sq_sum / hidden_size as f32).sqrt().max(1e-8);
    let scale_up = 1.0 / emb_rms; // Unclamped to rescue QAT decayed embeddings
    println!("Embedding RMS: {:.8}, scale_up: {:.4}", emb_rms, scale_up);
    for v in emb_f32.iter_mut() {
        *v *= scale_up;
    }
    let emb_slice: &[f32] = &emb_f32;

    let prompt =
        "The fundamental architecture of this system relies on high entropy latent states.";
    let tokens = tokenizer.encode(prompt);

    println!(
        "Running deep variance diagnostic for {} tokens...",
        tokens.len()
    );
    println!("Prompt: \"{}\"\n", prompt);

    println!(
        "{:<8} | {:<12} | {:<12} | {:<12}",
        "Layer", "VarH", "VarJ", "Var_EMA (Comb)"
    );
    println!("{:-<53}", "");

    for (pos, &token) in tokens.iter().enumerate() {
        let emb_offset = token as usize * hidden_size;
        if pos == 0 {
            println!(
                "First 5 values of embedding for token {}: {:?}",
                token,
                &emb_slice[emb_offset..emb_offset + 5]
            );
        }
        for (i, v) in emb_slice[emb_offset..emb_offset + hidden_size]
            .iter()
            .enumerate()
        {
            forge_llm::mud::slime::SlimeRegister::init_from_embed(
                &mut workspace.registers[i],
                &mut workspace.jepa_z,
                i,
                hidden_size,
                num_layers,
                *v,
                pos == 0,
            );
        }

        for (layer_idx, layer) in layers.iter().enumerate() {
            if pos == 0 {
                let alpha_w = layer.mhc_alpha_w;
                let beta_w = layer.mhc_beta_w;
                if !alpha_w.is_null() && !beta_w.is_null() {
                    let mut sum_alpha = 0.0;
                    for i in 0..hidden_size {
                        sum_alpha += unsafe { *alpha_w.add(i) }.abs();
                    }
                    let mean_alpha = sum_alpha / hidden_size as f32;
                    if mean_alpha < 0.1 {
                        println!(
                            "Layer {} mHC collapsed! mean(|alpha|) = {:.4}",
                            layer_idx, mean_alpha
                        );
                    }
                }
            }
            evaluate_slime_block(layer, layer_idx, &mut workspace, pos, rms_norm_eps, None);

            // We only print telemetry for the final token to see the accumulated context.
            if pos == tokens.len() - 1 {
                let var_h = workspace.jepa_var_ema[layer_idx * 2];
                let var_j = workspace.jepa_var_ema[layer_idx * 2 + 1];
                let var_ema_avg = (var_h + var_j) / 2.0;

                let alert = if var_h < 0.001 || var_j < 0.001 {
                    "⚠️ COLLAPSED"
                } else {
                    "✅ OK"
                };

                println!(
                    "{:<8} | {:<12.6} | {:<12.6} | {:<12.6} {}",
                    layer_idx, var_h, var_j, var_ema_avg, alert
                );
            }
        }
    }

    Ok(())
}
