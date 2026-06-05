use memmap2::Mmap;
use safetensors::SafeTensors;
use std::fs::File;
// unused imports removed

pub fn mmap_file(path: &str) -> anyhow::Result<Mmap> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    Ok(mmap)
}

pub fn parse_safetensors<'a>(mmap: &'a Mmap) -> anyhow::Result<SafeTensors<'a>> {
    Ok(SafeTensors::deserialize(mmap)?)
}

pub fn map_llama_to_mud(t_name: &str) -> Option<(String, bool)> {
    if t_name.ends_with(".weight_scale") {
        return None;
    }
    // Already MUD/GGUF natively formatted keys
    if t_name == "token_embd.weight" {
        return Some((t_name.to_string(), false));
    }
    if t_name == "output_norm.weight" || t_name == "output.weight" {
        return Some((t_name.to_string(), false));
    }
    if t_name.starts_with("blk.") {
        let is_norm = t_name.contains("norm");
        let is_scale = t_name.ends_with(".scale");
        let ternarize = !is_norm && !is_scale;
        return Some((t_name.to_string(), ternarize));
    }

    // Skip bias tensors except for Mamba's conv1d which is critical
    if t_name.ends_with(".bias") && !t_name.contains("conv1d") {
        return None;
    }
    if t_name == "model.embed_tokens.weight" {
        return Some(("token_embd.weight".to_string(), false));
    }
    if t_name == "model.norm.weight" {
        return Some(("output_norm.weight".to_string(), false));
    }
    if t_name == "lm_head.weight" {
        return Some(("output.weight".to_string(), false)); // NEVER ternarize final logits projection
    }

    // Layer mapping
    if t_name.starts_with("model.layers.") {
        let parts: Vec<&str> = t_name.split('.').collect();
        if parts.len() < 4 {
            return None;
        }
        let layer_idx = parts[2];
        let sub = parts[3];

        let prefix = format!("blk.{}", layer_idx);

        if sub == "input_layernorm" {
            return Some((format!("{}.attn_norm.weight", prefix), false));
        }
        if sub == "post_attention_layernorm" {
            return Some((format!("{}.norm.weight", prefix), false));
        }

        // Mamba / SSM Support (Jamba/Mamba nomenclature)
        if sub == "mamba" || sub == "mixer" || sub == "ssm" {
            let proj = parts[4];
            let is_scale = parts.last() == Some(&"scale");
            let suffix = if is_scale {
                "scale"
            } else {
                parts.last().unwrap_or(&"weight")
            };
            let ternarize = !is_scale && (proj.contains("proj")); // Proyectamos en ternario, estados/bias en f32

            let mapped = match proj {
                "in_proj" => format!("{}.ssm_in.{}", prefix, suffix),
                "out_proj" => format!("{}.ssm_out.{}", prefix, suffix),
                "x_proj" => format!("{}.ssm_x.{}", prefix, suffix),
                "dt_proj" => format!("{}.ssm_dt.{}", prefix, suffix),
                "A_log" | "ssm_a" => format!("{}.ssm_a", prefix), // a y d no suelen tener escalas separadas
                "D" | "ssm_d" => format!("{}.ssm_d", prefix),
                "conv1d" => format!("{}.ssm_conv1d.{}", prefix, suffix),
                _ => return None,
            };
            return Some((mapped, ternarize));
        }

        if sub == "self_attn" || sub == "attention" {
            if parts.len() < 5 {
                return None;
            }
            let proj = parts[4];
            if proj == "norm" {
                return Some((format!("{}.attn_norm.weight", prefix), false));
            }
            if proj == "attn_sub_norm" {
                return Some((format!("{}.attn_sub_norm.weight", prefix), false));
            }
            let is_scale = parts.last() == Some(&"scale");
            let suffix = if is_scale { "scale" } else { "weight" };
            let ternarize = !is_scale; // we only ternarize the weights, not the scales!

            let mapped = match proj {
                "q_proj" | "wq" => format!("{}.attn_q.{}", prefix, suffix),
                "k_proj" | "wk" => format!("{}.attn_k.{}", prefix, suffix),
                "v_proj" | "wv" => format!("{}.attn_v.{}", prefix, suffix),
                "o_proj" | "wo" => format!("{}.attn_output.{}", prefix, suffix),
                "qkv_proj" => format!("{}.attn_qkv.{}", prefix, suffix),
                _ => return None,
            };
            return Some((mapped, ternarize));
        }

        // MOE & MLP (dynamic matching for LLaMA, Qwen2MoE, Mixtral, DeepSeek)
        if sub == "mlp" || sub == "block_sparse_moe" || sub == "moe" {
            if parts.len() < 5 {
                return None;
            }

            let is_scale = parts.last() == Some(&"scale");
            let suffix = if is_scale { "scale" } else { "weight" };
            let ternarize = !is_scale;

            // Check for router gate: model.layers.L.[mlp/moe/block_sparse_moe].gate.weight
            if parts[4] == "gate" && parts.len() == 6 {
                return Some((format!("{}.gate.{}", prefix, suffix), false)); // gates are always f32
            }

            // Check for expert weights
            // Case A: mlp.experts.E.gate_proj/down_proj/up_proj.weight (Qwen2MoE)
            if parts[4] == "experts" && parts.len() >= 8 {
                let expert_idx = parts[5];
                let proj = parts[6];
                let mapped_proj = match proj {
                    "gate_proj" | "w1" => "w1",
                    "down_proj" | "w2" => "w2",
                    "up_proj" | "w3" => "w3",
                    _ => return None,
                };
                return Some((
                    format!(
                        "{}.expert.{}.{}.{}",
                        prefix, expert_idx, mapped_proj, suffix
                    ),
                    ternarize,
                ));
            }

            // Case B: block_sparse_moe/moe.experts.E.w1/w2/w3.weight (Mixtral / DeepSeek)
            if parts[4] == "experts" && parts.len() >= 7 {
                let expert_idx = parts[5];
                let proj = parts[6];
                let mapped_proj = match proj {
                    "w1" | "gate_proj" => "w1",
                    "w2" | "down_proj" => "w2",
                    "w3" | "up_proj" => "w3",
                    _ => return None,
                };
                return Some((
                    format!(
                        "{}.expert.{}.{}.{}",
                        prefix, expert_idx, mapped_proj, suffix
                    ),
                    ternarize,
                ));
            }

            // Case C: standard non-MoE MLP: mlp.gate_proj/down_proj/up_proj.weight (LLaMA / Qwen / Mistral)
            if parts.len() == 6 {
                let proj = parts[4];
                if proj == "ffn_sub_norm" {
                    return Some((format!("{}.ffn_sub_norm.weight", prefix), false));
                }
                let mapped_proj = match proj {
                    "gate_proj" | "w1" => "w1",
                    "down_proj" | "w2" => "w2",
                    "up_proj" | "w3" => "w3",
                    "gate_up_proj" => "gate_up",
                    _ => return None,
                };
                return Some((
                    format!("{}.expert.0.{}.{}", prefix, mapped_proj, suffix),
                    ternarize,
                ));
            }
        }
    }

    None
}
