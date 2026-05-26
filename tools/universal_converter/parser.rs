use safetensors::SafeTensors;
use memmap2::Mmap;
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
    // Returns Option<(MappedName, ShouldTernarize)>
    // Skip bias tensors (Qwen, etc.) — inference doesn't use biases
    if t_name.ends_with(".bias") {
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
        if parts.len() < 4 { return None; }
        let layer_idx = parts[2];
        let sub = parts[3];
        
        let prefix = format!("blk.{}", layer_idx);
        
        if sub == "input_layernorm" {
            return Some((format!("{}.attn_norm.weight", prefix), false));
        }
        if sub == "post_attention_layernorm" {
            return Some((format!("{}.norm.weight", prefix), false));
        }
        
        if sub == "self_attn" || sub == "attention" {
            if parts.len() < 5 { return None; }
            let proj = parts[4];
            if proj == "norm" {
                return Some((format!("{}.attn_norm.weight", prefix), false));
            }
            let is_scale = parts.last() == Some(&"scale");
            let suffix = if is_scale { "scale" } else { "weight" };
            let ternarize = !is_scale; // we only ternarize the weights, not the scales!
            
            let mapped = match proj {
                "q_proj" | "wq" => format!("{}.attn_q.{}", prefix, suffix),
                "k_proj" | "wk" => format!("{}.attn_k.{}", prefix, suffix),
                "v_proj" | "wv" => format!("{}.attn_v.{}", prefix, suffix),
                "o_proj" | "wo" => format!("{}.attn_output.{}", prefix, suffix),
                _ => return None,
            };
            return Some((mapped, ternarize));
        }
        
        if sub == "mlp" {
            if parts.len() < 5 { return None; }
            let proj = parts[4];
            let is_scale = parts.last() == Some(&"scale");
            let suffix = if is_scale { "scale" } else { "weight" };
            let ternarize = !is_scale;
            
            // Map standard MLP to expert 0
            let mapped = match proj {
                "gate_proj" | "w1" => format!("{}.expert.0.w1.{}", prefix, suffix),
                "down_proj" | "w2" => format!("{}.expert.0.w2.{}", prefix, suffix),
                "up_proj" | "w3" => format!("{}.expert.0.w3.{}", prefix, suffix),
                _ => return None,
            };
            return Some((mapped, ternarize));
        }
        
        if sub == "moe" {
            if parts.len() < 5 { return None; }
            let comp = parts[4];
            if comp == "norm" {
                return Some((format!("{}.norm.weight", prefix), false));
            }
            if comp == "gate" {
                return Some((format!("{}.gate.weight", prefix), false)); // gates are f32
            }
            if comp == "experts" {
                if parts.len() < 7 { return None; }
                let expert_idx = parts[5];
                let proj = parts[6];
                let is_scale = parts.last() == Some(&"scale");
                let suffix = if is_scale { "scale" } else { "weight" };
                let ternarize = !is_scale; // scale is f32
                
                let mapped = match proj {
                    "w1" => format!("{}.expert.{}.w1.{}", prefix, expert_idx, suffix),
                    "w2" => format!("{}.expert.{}.w2.{}", prefix, expert_idx, suffix),
                    "w3" => format!("{}.expert.{}.w3.{}", prefix, expert_idx, suffix),
                    _ => return None,
                };
                return Some((mapped, ternarize));
            }
        }
    }
    
    None
}
