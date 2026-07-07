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
    if t_name.ends_with(".bias") && !t_name.contains("conv1d") && !t_name.contains("dt_bias") {
        return None;
    }

    if t_name.ends_with("embed_tokens.weight") {
        return Some(("token_embd.weight".to_string(), false));
    }
    if t_name.ends_with(".norm.weight")
        && !t_name.contains("layers.")
        && !t_name.contains("blocks.")
    {
        return Some(("output_norm.weight".to_string(), false));
    }
    if t_name.ends_with("lm_head.weight") || t_name.ends_with("output.weight") {
        return Some(("output.weight".to_string(), false)); // NEVER ternarize final logits projection
    }

    // Layer mapping (Agnostic to 'model.', 'model.language_model.', etc.)
    let layer_marker = if t_name.contains(".layers.") {
        ".layers."
    } else if t_name.contains(".blocks.") {
        ".blocks."
    } else {
        return None;
    };

    let parts: Vec<&str> = t_name.split(layer_marker).collect();
    if parts.len() < 2 {
        return None;
    }

    let sub_parts: Vec<&str> = parts[1].split('.').collect();
    if sub_parts.len() < 2 {
        return None;
    }

    let layer_idx = sub_parts[0];
    let sub = sub_parts[1];
    let prefix = format!("blk.{}", layer_idx);

    if sub == "input_layernorm" || sub == "ln_1" || sub == "attention_norm" {
        return Some((format!("{}.attn_norm.weight", prefix), false));
    }
    if sub == "post_attention_layernorm" || sub == "ln_2" || sub == "ffn_norm" {
        return Some((format!("{}.norm.weight", prefix), false));
    }

    // Mamba / SSM Support (Jamba/Mamba/Ornith nomenclature)
    if sub == "mamba" || sub == "mixer" || sub == "ssm" || sub == "linear_attn" {
        if sub_parts.len() < 3 {
            return None;
        }
        let proj = sub_parts[2];
        let is_scale = sub_parts.last() == Some(&"scale");
        let suffix = if is_scale {
            "scale"
        } else {
            sub_parts.last().unwrap_or(&"weight")
        };
        let ternarize = !is_scale && (proj.contains("proj")); // Proyectamos en ternario, estados/bias en f32

        let mapped = match proj {
            "in_proj" | "in_proj_qkv" => format!("{}.ssm_in.{}", prefix, suffix),
            "in_proj_a" => format!("{}.ssm_in_a.{}", prefix, suffix),
            "in_proj_b" => format!("{}.ssm_in_b.{}", prefix, suffix),
            "in_proj_z" => format!("{}.ssm_in_z.{}", prefix, suffix),
            "out_proj" => format!("{}.ssm_out.{}", prefix, suffix),
            "x_proj" => format!("{}.ssm_x.{}", prefix, suffix),
            "dt_proj" | "dt_bias" => format!("{}.ssm_dt.{}", prefix, suffix),
            "A_log" | "ssm_a" => format!("{}.ssm_a", prefix), // a y d no suelen tener escalas separadas
            "D" | "ssm_d" => format!("{}.ssm_d", prefix),
            "conv1d" => format!("{}.ssm_conv1d.{}", prefix, suffix),
            _ => return None,
        };
        return Some((mapped, ternarize));
    }

    // Self-Attention
    if sub == "self_attn" || sub == "attention" {
        if sub_parts.len() < 3 {
            return None;
        }
        let proj = sub_parts[2];
        if proj == "norm" {
            return Some((format!("{}.attn_norm.weight", prefix), false));
        }
        if proj == "attn_sub_norm" {
            return Some((format!("{}.attn_sub_norm.weight", prefix), false));
        }
        let is_scale =
            sub_parts.last() == Some(&"scale") || sub_parts.last() == Some(&"weight_scale");
        let suffix = if is_scale { "prq_scale" } else { "weight" };
        let ternarize = !is_scale; // we only ternarize the weights, not the scales!

        let mapped = match proj {
            "q_proj" | "wq" => format!("{}.attn_q.{}", prefix, suffix),
            "k_proj" | "wk" => format!("{}.attn_k.{}", prefix, suffix),
            "v_proj" | "wv" => format!("{}.attn_v.{}", prefix, suffix),
            "o_proj" | "wo" | "out_proj" => format!("{}.attn_output.{}", prefix, suffix),
            "qkv_proj" => format!("{}.attn_qkv.{}", prefix, suffix),
            _ => return None,
        };
        return Some((mapped, ternarize));
    }

    // MOE & MLP
    if sub == "mlp" || sub == "block_sparse_moe" || sub == "moe" {
        if sub_parts.len() < 3 {
            return None;
        }
        let is_moe = sub == "block_sparse_moe" || sub == "moe";
        let is_scale =
            sub_parts.last() == Some(&"scale") || sub_parts.last() == Some(&"weight_scale");
        let suffix = if is_scale { "prq_scale" } else { "weight" };
        let ternarize = !is_scale;

        // Gate / Router
        if sub_parts[2] == "gate" {
            return Some((format!("{}.gate.weight", prefix), false));
        }

        if is_moe && sub_parts[2] == "experts" {
            if sub_parts.len() < 5 {
                return None;
            }
            let expert_id = sub_parts[3];
            let w_name = sub_parts[4];
            let mapped = match w_name {
                "w1" | "gate_proj" => format!("{}.expert.{}.w1.{}", prefix, expert_id, suffix),
                "w2" | "down_proj" => format!("{}.expert.{}.w2.{}", prefix, expert_id, suffix),
                "w3" | "up_proj" => format!("{}.expert.{}.w3.{}", prefix, expert_id, suffix),
                _ => return None,
            };
            return Some((mapped, ternarize));
        } else {
            let w_name = sub_parts[2];
            let mapped = match w_name {
                "w1" | "gate_proj" => format!("{}.expert.0.w1.{}", prefix, suffix),
                "w2" | "down_proj" => format!("{}.expert.0.w2.{}", prefix, suffix),
                "w3" | "up_proj" => format!("{}.expert.0.w3.{}", prefix, suffix),
                "gate_up_proj" => format!("{}.expert.0.gate_up.{}", prefix, suffix),
                "ffn_sub_norm" => format!("{}.ffn_sub_norm.{}", prefix, suffix),
                _ => return None,
            };
            return Some((mapped, ternarize));
        }
    }

    None
}
