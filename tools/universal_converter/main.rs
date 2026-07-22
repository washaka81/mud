mod calibration;
mod parser;
mod quantizer;

use forge_llm::mud::{MudTensorType, StreamingMudWriter};
use serde_json::Value;
use std::collections::HashMap;
use std::env;
use std::fs;

// Helper to extract vocab/merges from JSON (unchanged)
fn extract_vocab_from_json(path: &str, expected_size: usize) -> Option<String> {
    let content = fs::read_to_string(path).ok()?;
    let json: Value = serde_json::from_str(&content).ok()?;
    let mut id_to_token = vec!["".to_string(); expected_size];
    let mut found_any = false;
    if let Some(vocab_obj) = json
        .get("model")
        .and_then(|m| m.get("vocab"))
        .and_then(|v| v.as_object())
    {
        for (token, id_val) in vocab_obj {
            if let Some(id) = id_val.as_u64() {
                if (id as usize) < expected_size {
                    id_to_token[id as usize] = token.clone();
                    found_any = true;
                }
            }
        }
    } else if let Some(vocab_obj) = json.get("vocab").and_then(|v| v.as_object()) {
        for (token, id_val) in vocab_obj {
            if let Some(id) = id_val.as_u64() {
                if (id as usize) < expected_size {
                    id_to_token[id as usize] = token.clone();
                    found_any = true;
                }
            }
        }
    }
    if !found_any {
        return None;
    }

    // Step 3: Handle added_tokens (critical for LLaMA-3 / BitNet)
    if let Some(added_tokens) = json.get("added_tokens").and_then(|a| a.as_array()) {
        for token_obj in added_tokens {
            if let (Some(content), Some(id)) = (
                token_obj.get("content").and_then(|v| v.as_str()),
                token_obj.get("id").and_then(|v| v.as_u64()),
            ) {
                if (id as usize) < expected_size {
                    id_to_token[id as usize] = content.to_string();
                }
            }
        }
    }

    for (i, token) in id_to_token.iter_mut().enumerate() {
        if token.is_empty() {
            *token = format!("<dummy_{}>", i);
        }
    }
    Some(id_to_token.join("\n"))
}

fn extract_merges_from_json(path: &str) -> Option<String> {
    let content = fs::read_to_string(path).ok()?;
    let json: Value = serde_json::from_str(&content).ok()?;
    let merges_arr = json.get("model")?.get("merges")?.as_array()?;
    let mut merges = Vec::new();
    for val in merges_arr {
        if let Some(s) = val.as_str() {
            merges.push(s.to_string());
        }
    }
    Some(merges.join("\n"))
}

fn extract_config_metadata(input_path: &str) -> Option<HashMap<String, String>> {
    let path = std::path::Path::new(input_path);
    let input_dir = if path.is_dir() { path } else { path.parent()? };
    let config_path = input_dir.join("config.json");
    if !config_path.exists() {
        return None;
    }
    let content = fs::read_to_string(config_path).ok()?;
    let json: Value = serde_json::from_str(&content).ok()?;
    let mut meta = HashMap::new();

    // Phase 12+: Deep Configuration Incrustation
    meta.insert("raw_config_json".to_string(), content);

    // Support nested config for Qwen3.5/Vision models
    let config = if let Some(_text_config) = json.get("text_config").and_then(|v| v.as_object()) {
        json.get("text_config").unwrap()
    } else {
        &json
    };

    if let Some(archs) = json.get("architectures").and_then(|a| a.as_array()) {
        if let Some(first_arch) = archs.first().and_then(|a| a.as_str()) {
            meta.insert("arch_original".to_string(), first_arch.to_string());
        }
    } else if let Some(model_type) = json.get("model_type").and_then(|m| m.as_str()) {
        meta.insert("arch_original".to_string(), model_type.to_string());
    }
    if let Some(layers) = config
        .get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .or_else(|| config.get("num_layers").and_then(|v| v.as_u64()))
    {
        meta.insert("num_layers".to_string(), layers.to_string());
    }
    if let Some(h) = config.get("hidden_size").and_then(|v| v.as_u64()) {
        meta.insert("hidden_size".to_string(), h.to_string());
    }
    if let Some(ffn) = config.get("intermediate_size").and_then(|v| v.as_u64()) {
        meta.insert("ffn_hidden".to_string(), ffn.to_string());
    }
    if let Some(exp) = config
        .get("num_local_experts")
        .and_then(|v| v.as_u64())
        .or_else(|| config.get("num_experts").and_then(|v| v.as_u64()))
    {
        meta.insert("num_experts".to_string(), exp.to_string());
    }
    if let Some(k) = config
        .get("num_experts_per_tok")
        .and_then(|v| v.as_u64())
        .or_else(|| config.get("num_experts_per_token").and_then(|v| v.as_u64()))
        .or_else(|| config.get("top_k").and_then(|v| v.as_u64()))
    {
        meta.insert("top_k".to_string(), k.to_string());
    }
    if let Some(heads) = config.get("num_attention_heads").and_then(|v| v.as_u64()) {
        meta.insert("num_heads".to_string(), heads.to_string());
    }
    if let Some(kv_heads) = config.get("num_key_value_heads").and_then(|v| v.as_u64()) {
        meta.insert("num_kv_heads".to_string(), kv_heads.to_string());
    }
    if let Some(d_state) = config
        .get("state_size")
        .and_then(|v| v.as_u64())
        .or_else(|| config.get("ssm_d_state").and_then(|v| v.as_u64()))
        .or_else(|| config.get("d_state").and_then(|v| v.as_u64()))
    {
        meta.insert("d_state".to_string(), d_state.to_string());
    }
    if let Some(d_conv) = config
        .get("conv_kernel")
        .and_then(|v| v.as_u64())
        .or_else(|| config.get("ssm_d_conv").and_then(|v| v.as_u64()))
        .or_else(|| config.get("d_conv").and_then(|v| v.as_u64()))
    {
        meta.insert("d_conv".to_string(), d_conv.to_string());
    }
    if let Some(eps) = config.get("rms_norm_eps").and_then(|v| v.as_f64()) {
        meta.insert("rms_norm_eps".to_string(), eps.to_string());
    }
    if let Some(act) = config.get("hidden_act").and_then(|v| v.as_str()) {
        meta.insert("hidden_act".to_string(), act.to_string());
    }
    if let Some(theta) = config
        .get("rope_theta")
        .and_then(|v| v.as_f64())
        .or_else(|| {
            config
                .get("rope_parameters")
                .and_then(|p| p.as_object())
                .and_then(|p| p.get("rope_theta"))
                .and_then(|v| v.as_f64())
        })
    {
        meta.insert("rope_theta".to_string(), format!("{:.1}", theta));
    }
    if let Some(tie) = config.get("tie_word_embeddings").and_then(|v| v.as_bool()) {
        meta.insert("tie_word_embeddings".to_string(), tie.to_string());
    }
    if let Some(max_pos) = config
        .get("max_position_embeddings")
        .and_then(|v| v.as_u64())
    {
        meta.insert("max_position_embeddings".to_string(), max_pos.to_string());
    }
    if let Some(vsize) = config.get("vocab_size").and_then(|v| v.as_u64()) {
        meta.insert("vocab_size".to_string(), vsize.to_string());
    }
    if let Some(hd) = config.get("head_dim").and_then(|v| v.as_u64()) {
        meta.insert("head_dim".to_string(), hd.to_string());
    }
    Some(meta)
}

#[derive(Debug, Clone)]
struct MudTensorPlan {
    name: String,
    source_name: Option<String>,
    shape: Vec<usize>,
    t_type: MudTensorType,
    should_ternarize: bool,
    sub_range: Option<std::ops::Range<usize>>,
    owned_data: Option<Vec<u8>>,
    source_file_idx: Option<usize>,
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <input.safetensors|dir> <output.mud> [--ternarize-emb] [--untie-emb]",
            args[0]
        );
        std::process::exit(1);
    }
    let input_path = &args[1];
    let output_path = &args[2];
    let ternarize_emb = args.iter().any(|a| a == "--ternarize-emb");

    println!("🚀 Starting Universal Streaming Ternary Converter (RAM-Efficient)");

    // Step 1: Map Safetensors
    let mut safetensors_files = vec![];
    let md = std::fs::metadata(input_path)?;
    if md.is_dir() {
        for entry in std::fs::read_dir(input_path)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                safetensors_files.push(path);
            }
        }
    } else {
        safetensors_files.push(std::path::PathBuf::from(input_path));
    }
    if safetensors_files.is_empty() {
        anyhow::bail!("No safetensors found");
    }

    let mut mapped_files = vec![];
    for file in &safetensors_files {
        mapped_files.push(parser::mmap_file(file.to_str().unwrap())?);
    }
    let mut safe_tensors_list = vec![];
    for mmap in &mapped_files {
        safe_tensors_list.push(parser::parse_safetensors(mmap)?);
    }

    // Step 2: Calibrate & Extract Metadata
    let scales_map = calibration::compute_scales(&safe_tensors_list);
    let mut config_meta = extract_config_metadata(input_path).unwrap_or_default();

    // Pass 0: Pre-collect BitNet scales
    let mut bitnet_scales = HashMap::new();
    for safe_tensors in &safe_tensors_list {
        for (name, tensor_view) in safe_tensors.tensors() {
            if name.ends_with(".weight_scale") {
                let bytes = tensor_view.data();
                let mut scale = 1.0f32;
                match tensor_view.dtype() {
                    safetensors::tensor::Dtype::F32 => {
                        if bytes.len() == 4 {
                            let mut b = [0u8; 4];
                            b.copy_from_slice(bytes);
                            scale = f32::from_le_bytes(b);
                        }
                    }
                    safetensors::tensor::Dtype::BF16 => {
                        if bytes.len() == 2 {
                            let mut b = [0u8; 2];
                            b.copy_from_slice(bytes);
                            scale = half::bf16::from_le_bytes(b).to_f32();
                        }
                    }
                    safetensors::tensor::Dtype::F16 => {
                        if bytes.len() == 2 {
                            let mut b = [0u8; 2];
                            b.copy_from_slice(bytes);
                            scale = half::f16::from_le_bytes(b).to_f32();
                        }
                    }
                    _ => {}
                }
                bitnet_scales.insert(name.replace(".weight_scale", ".weight"), scale);
                println!("🔍 Collected BitNet scale: {} for {}", scale, name);
            }
        }
    }

    // Pass 0.5: Infer critical dimensions from tensors directly
    let mut inferred_vocab_size = 0;
    let mut inferred_hidden_size = 0;
    let mut inferred_ffn_hidden = 0;
    for safe_tensors in &safe_tensors_list {
        for (name, tensor_view) in safe_tensors.tensors() {
            let shape = tensor_view.shape();
            if shape.len() == 2 {
                if name.contains("embed_tokens")
                    || name == "tok_embeddings.weight"
                    || name == "model.embed_tokens.weight"
                {
                    inferred_vocab_size = shape[0];
                    inferred_hidden_size = shape[1];
                }
                if name.contains("gate_proj") || name.contains("up_proj") {
                    inferred_ffn_hidden = shape[0];
                }
            }
        }
    }

    // Pass 1: Plan the model
    let mut plan = Vec::new();
    let mut max_layer = 0;
    let mut max_expert = 0;

    for (f_idx, safe_tensors) in safe_tensors_list.iter().enumerate() {
        for (name, tensor_view) in safe_tensors.tensors() {
            if name.ends_with(".weight_scale") {
                continue;
            }

            if let Some((mapped_name, mut should_ternarize)) = parser::map_llama_to_mud(&name) {
                if ternarize_emb
                    && (mapped_name == "token_embd.weight" || mapped_name == "output.weight")
                {
                    should_ternarize = true;
                }
                // Tracking
                if mapped_name.starts_with("blk.") {
                    let parts: Vec<&str> = mapped_name.split('.').collect();
                    if let Ok(l) = parts[1].parse::<usize>() {
                        max_layer = max_layer.max(l);
                    }
                    if parts.len() >= 4 && parts[2] == "expert" {
                        if let Ok(e) = parts[3].parse::<usize>() {
                            max_expert = max_expert.max(e);
                        }
                    }
                }

                let original_shape = if tensor_view.dtype() == safetensors::tensor::Dtype::U8 {
                    let mut s = tensor_view.shape().to_vec();
                    s[0] *= 4;
                    s
                } else {
                    tensor_view.shape().to_vec()
                };

                if mapped_name.ends_with(".attn_qkv.weight") {
                    let prefix = mapped_name.trim_end_matches(".attn_qkv.weight");
                    let num_heads = config_meta
                        .get("num_heads")
                        .and_then(|s| s.parse::<usize>().ok())
                        .ok_or_else(|| {
                            anyhow::anyhow!("num_heads missing in config.json. Cannot split QKV")
                        })?;
                    let num_kv_heads = config_meta
                        .get("num_kv_heads")
                        .and_then(|s| s.parse::<usize>().ok())
                        .unwrap_or(num_heads);
                    let hidden_size = config_meta
                        .get("hidden_size")
                        .and_then(|s| s.parse::<usize>().ok())
                        .unwrap_or(original_shape[1]);
                    let head_dim = config_meta
                        .get("head_dim")
                        .and_then(|s| s.parse::<usize>().ok())
                        .unwrap_or(hidden_size / num_heads);

                    let q_rows = num_heads * head_dim;
                    let k_rows = num_kv_heads * head_dim;
                    let v_rows = num_kv_heads * head_dim;

                    let q_name = format!("{}.attn_q.weight", prefix);
                    plan.push(MudTensorPlan {
                        name: q_name.clone(),
                        source_name: Some(name.clone()),
                        shape: vec![q_rows, hidden_size],
                        t_type: if should_ternarize {
                            MudTensorType::Ternary2Bit
                        } else {
                            MudTensorType::Float32
                        },
                        should_ternarize,
                        sub_range: Some(0..q_rows * hidden_size),
                        owned_data: None,
                        source_file_idx: Some(f_idx),
                    });
                    if should_ternarize {
                        plan.push(MudTensorPlan {
                            name: q_name.replace(".weight", ".prq_scale"),
                            source_name: None,
                            shape: vec![q_rows],
                            t_type: MudTensorType::Float32,
                            should_ternarize: false,
                            sub_range: None,
                            owned_data: None,
                            source_file_idx: None,
                        });
                    }
                    // Repeat for K and V... (omitted for brevity in thinking, implementing now)
                    let k_name = format!("{}.attn_k.weight", prefix);
                    plan.push(MudTensorPlan {
                        name: k_name.clone(),
                        source_name: Some(name.clone()),
                        shape: vec![k_rows, hidden_size],
                        t_type: if should_ternarize {
                            MudTensorType::Ternary2Bit
                        } else {
                            MudTensorType::Float32
                        },
                        should_ternarize,
                        sub_range: Some(q_rows * hidden_size..(q_rows + k_rows) * hidden_size),
                        owned_data: None,
                        source_file_idx: Some(f_idx),
                    });
                    if should_ternarize {
                        plan.push(MudTensorPlan {
                            name: k_name.replace(".weight", ".prq_scale"),
                            source_name: None,
                            shape: vec![k_rows],
                            t_type: MudTensorType::Float32,
                            should_ternarize: false,
                            sub_range: None,
                            owned_data: None,
                            source_file_idx: None,
                        });
                    }
                    let v_name = format!("{}.attn_v.weight", prefix);
                    plan.push(MudTensorPlan {
                        name: v_name.clone(),
                        source_name: Some(name.clone()),
                        shape: vec![v_rows, hidden_size],
                        t_type: if should_ternarize {
                            MudTensorType::Ternary2Bit
                        } else {
                            MudTensorType::Float32
                        },
                        should_ternarize,
                        sub_range: Some(
                            (q_rows + k_rows) * hidden_size
                                ..(q_rows + k_rows + v_rows) * hidden_size,
                        ),
                        owned_data: None,
                        source_file_idx: Some(f_idx),
                    });
                    if should_ternarize {
                        plan.push(MudTensorPlan {
                            name: v_name.replace(".weight", ".prq_scale"),
                            source_name: None,
                            shape: vec![v_rows],
                            t_type: MudTensorType::Float32,
                            should_ternarize: false,
                            sub_range: None,
                            owned_data: None,
                            source_file_idx: None,
                        });
                    }
                } else if mapped_name.ends_with(".gate_up.weight") {
                    let prefix = mapped_name.trim_end_matches(".gate_up.weight");
                    let rows = original_shape[0] / 2;
                    let cols = original_shape[1];
                    let g_name = format!("{}.gate.weight", prefix);
                    plan.push(MudTensorPlan {
                        name: g_name.clone(),
                        source_name: Some(name.clone()),
                        shape: vec![rows, cols],
                        t_type: if should_ternarize {
                            MudTensorType::Ternary2Bit
                        } else {
                            MudTensorType::Float32
                        },
                        should_ternarize,
                        sub_range: Some(0..rows * cols),
                        owned_data: None,
                        source_file_idx: Some(f_idx),
                    });
                    if should_ternarize {
                        plan.push(MudTensorPlan {
                            name: g_name.replace(".weight", ".prq_scale"),
                            source_name: None,
                            shape: vec![rows],
                            t_type: MudTensorType::Float32,
                            should_ternarize: false,
                            sub_range: None,
                            owned_data: None,
                            source_file_idx: None,
                        });
                    }
                    let u_name = format!("{}.up.weight", prefix);
                    plan.push(MudTensorPlan {
                        name: u_name.clone(),
                        source_name: Some(name.clone()),
                        shape: vec![rows, cols],
                        t_type: if should_ternarize {
                            MudTensorType::Ternary2Bit
                        } else {
                            MudTensorType::Float32
                        },
                        should_ternarize,
                        sub_range: Some(rows * cols..2 * rows * cols),
                        owned_data: None,
                        source_file_idx: Some(f_idx),
                    });
                    if should_ternarize {
                        plan.push(MudTensorPlan {
                            name: u_name.replace(".weight", ".prq_scale"),
                            source_name: None,
                            shape: vec![rows],
                            t_type: MudTensorType::Float32,
                            should_ternarize: false,
                            sub_range: None,
                            owned_data: None,
                            source_file_idx: None,
                        });
                    }
                } else {
                    plan.push(MudTensorPlan {
                        name: mapped_name.clone(),
                        source_name: Some(name.clone()),
                        shape: original_shape.clone(),
                        t_type: if should_ternarize {
                            MudTensorType::Ternary2Bit
                        } else {
                            MudTensorType::Float32
                        },
                        should_ternarize,
                        sub_range: None,
                        owned_data: None,
                        source_file_idx: Some(f_idx),
                    });
                    if should_ternarize {
                        plan.push(MudTensorPlan {
                            name: mapped_name.replace(".weight", ".prq_scale"),
                            source_name: None,
                            shape: vec![original_shape[0]],
                            t_type: MudTensorType::Float32,
                            should_ternarize: false,
                            sub_range: None,
                            owned_data: None,
                            source_file_idx: None,
                        });
                    }
                }

                // LM-head: only emit separate output.weight when embeddings are NOT tied
                // (HF tie_word_embeddings=true → share token_embd at inference).
                // Force-untie: --untie-emb  |  Force ternary head: --ternarize-emb
                if name == "model.embed_tokens.weight" {
                    let tie = config_meta
                        .get("tie_word_embeddings")
                        .map(|s| s == "true" || s == "1")
                        .unwrap_or(true);
                    let force_untie = args.iter().any(|a| a == "--untie-emb");
                    if !tie || force_untie || ternarize_emb {
                        println!(
                            "  [convert] emitting separate output.weight (tie={} force_untie={} ternarize_emb={})",
                            tie, force_untie, ternarize_emb
                        );
                        plan.push(MudTensorPlan {
                            name: "output.weight".to_string(),
                            source_name: Some(name.clone()),
                            shape: original_shape.clone(),
                            t_type: if ternarize_emb {
                                MudTensorType::Ternary2Bit
                            } else {
                                MudTensorType::Float32
                            },
                            should_ternarize: ternarize_emb,
                            sub_range: None,
                            owned_data: None,
                            source_file_idx: Some(f_idx),
                        });
                        if ternarize_emb {
                            plan.push(MudTensorPlan {
                                name: "output.prq_scale".to_string(),
                                source_name: None,
                                shape: vec![original_shape[0]],
                                t_type: MudTensorType::Float32,
                                should_ternarize: false,
                                sub_range: None,
                                owned_data: None,
                                source_file_idx: None,
                            });
                        }
                    } else {
                        println!(
                            "  [convert] tied embeddings — no separate output.weight (use token_embd)"
                        );
                    }
                }
            }
        }
    }

    // 🛡️ BITDISTILL SUBLN INJECTION
    // Synthesize Sub-LayerNorm weights (1.0) if they don't exist, to prevent Ternary Shock.
    let hidden_size = config_meta
        .get("hidden_size")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(inferred_hidden_size);
    if hidden_size == 0 {
        anyhow::bail!("Could not infer hidden_size and not found in config.json");
    }

    let ffn_hidden = config_meta
        .get("ffn_hidden")
        .and_then(|s| s.parse::<usize>().ok())
        .or_else(|| {
            config_meta
                .get("intermediate_size")
                .and_then(|s| s.parse::<usize>().ok())
        })
        .unwrap_or(inferred_ffn_hidden);
    if ffn_hidden == 0 {
        anyhow::bail!("Could not infer ffn_hidden and not found in config.json");
    }

    for l in 0..=max_layer {
        let attn_sub_name = format!("blk.{}.attn_sub_norm.weight", l);
        if !plan.iter().any(|p| p.name == attn_sub_name) {
            let mut data = Vec::with_capacity(hidden_size * 4);
            for _ in 0..hidden_size {
                data.extend_from_slice(&1.0f32.to_le_bytes());
            }
            plan.push(MudTensorPlan {
                name: attn_sub_name,
                source_name: None,
                shape: vec![hidden_size],
                t_type: MudTensorType::Float32,
                should_ternarize: false,
                sub_range: None,
                owned_data: Some(data),
                source_file_idx: None,
            });
        }

        let ffn_sub_name = format!("blk.{}.ffn_sub_norm.weight", l);
        if !plan.iter().any(|p| p.name == ffn_sub_name) {
            let mut data = Vec::with_capacity(ffn_hidden * 4);
            for _ in 0..ffn_hidden {
                data.extend_from_slice(&1.0f32.to_le_bytes());
            }
            plan.push(MudTensorPlan {
                name: ffn_sub_name,
                source_name: None,
                shape: vec![ffn_hidden],
                t_type: MudTensorType::Float32,
                should_ternarize: false,
                sub_range: None,
                owned_data: Some(data),
                source_file_idx: None,
            });
        }

        let mhc_alpha_name = format!("blk.{}.mhc_alpha.weight", l);
        if !plan.iter().any(|p| p.name == mhc_alpha_name) {
            let mut data = Vec::with_capacity(hidden_size * 4);
            for _ in 0..hidden_size {
                data.extend_from_slice(&0.85f32.to_le_bytes());
            }
            plan.push(MudTensorPlan {
                name: mhc_alpha_name,
                source_name: None,
                shape: vec![hidden_size],
                t_type: MudTensorType::Float32,
                should_ternarize: false,
                sub_range: None,
                owned_data: Some(data),
                source_file_idx: None,
            });
        }

        let mhc_beta_name = format!("blk.{}.mhc_beta.weight", l);
        if !plan.iter().any(|p| p.name == mhc_beta_name) {
            let mut data = Vec::with_capacity(hidden_size * 4);
            for _ in 0..hidden_size {
                data.extend_from_slice(&0.15f32.to_le_bytes());
            }
            plan.push(MudTensorPlan {
                name: mhc_beta_name,
                source_name: None,
                shape: vec![hidden_size],
                t_type: MudTensorType::Float32,
                should_ternarize: false,
                sub_range: None,
                owned_data: Some(data),
                source_file_idx: None,
            });
        }

        let mhc_radius_name = format!("blk.{}.mhc_radius.weight", l);
        if !plan.iter().any(|p| p.name == mhc_radius_name) {
            let mut data = Vec::with_capacity(4);
            // Dynamic limit based on sqrt(hidden) scaled arbitrarily, defaulting to 1000.0 if unknown
            let rad = (hidden_size as f32).sqrt() * 5.0; // Moderate start
            data.extend_from_slice(&rad.to_le_bytes());
            plan.push(MudTensorPlan {
                name: mhc_radius_name,
                source_name: None,
                shape: vec![1],
                t_type: MudTensorType::Float32,
                should_ternarize: false,
                sub_range: None,
                owned_data: Some(data),
                source_file_idx: None,
            });
        }
    }

    // Step 3: Finalize Metadata
    let mut global_metadata = HashMap::new();
    let has_gate = plan.iter().any(|p| p.name.contains(".gate.weight"));
    let num_layers = config_meta
        .get("num_layers")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(max_layer + 1);
    let num_experts = config_meta
        .get("num_experts")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(if has_gate || max_expert > 0 {
            max_expert + 1
        } else {
            1
        });
    let top_k = config_meta
        .get("top_k")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(if num_experts > 1 { 2 } else { 1 });

    for k in [
        "hidden_act",
        "rope_theta",
        "rms_norm_eps",
        "d_state",
        "d_conv",
        "hidden_size",
        "ffn_hidden",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "max_position_embeddings",
        "bos_token_id",
        "eos_token_id",
        "raw_config_json",
        "vocab_size",
    ] {
        if let Some(v) = config_meta.get(k) {
            global_metadata.insert(k.to_string(), v.clone());
        }
    }
    global_metadata.insert("arch".to_string(), "mud-ternary-moe-v1-master".to_string());
    global_metadata.insert("num_layers".to_string(), num_layers.to_string());
    global_metadata.insert("num_hidden_layers".to_string(), num_layers.to_string());
    global_metadata.insert("num_experts".to_string(), num_experts.to_string());
    global_metadata.insert("top_k".to_string(), top_k.to_string());
    // Stream L: mirror alternate key names from config into canonical set
    if let Some(v) = config_meta.get("intermediate_size") {
        global_metadata
            .entry("intermediate_size".into())
            .or_insert_with(|| v.clone());
        global_metadata
            .entry("ffn_hidden".into())
            .or_insert_with(|| v.clone());
    }
    if let Some(v) = config_meta.get("num_attention_heads") {
        global_metadata
            .entry("num_attention_heads".into())
            .or_insert_with(|| v.clone());
        global_metadata
            .entry("num_heads".into())
            .or_insert_with(|| v.clone());
    }
    if let Some(v) = config_meta.get("num_key_value_heads") {
        global_metadata
            .entry("num_key_value_heads".into())
            .or_insert_with(|| v.clone());
        global_metadata
            .entry("num_kv_heads".into())
            .or_insert_with(|| v.clone());
    }

    // Tokenizer logic (abbreviated, same as before but into global_metadata)
    let expected_vocab_size = config_meta
        .get("vocab_size")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(inferred_vocab_size);
    if expected_vocab_size == 0 {
        anyhow::bail!("Could not infer vocab_size and not found in config.json");
    }
    config_meta.insert("vocab_size".to_string(), expected_vocab_size.to_string());

    let path = std::path::Path::new(input_path);
    let input_dir = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or(std::path::Path::new("."))
    };
    let tokenizer_path = input_dir.join("tokenizer.json");
    if let Some(tokens_str) =
        extract_vocab_from_json(tokenizer_path.to_str().unwrap(), expected_vocab_size)
    {
        global_metadata.insert("tokenizer.tokens".to_string(), tokens_str);
        if let Some(merges_str) = extract_merges_from_json(tokenizer_path.to_str().unwrap()) {
            global_metadata.insert("tokenizer.merges".to_string(), merges_str);
        }
    }

    // Stream L: fill canonical P-13 aliases so healthcheck/auditor always resolve dims
    if let Err(e) = forge_llm::mud::p13::ensure_canonical_metadata_aliases(&mut global_metadata) {
        eprintln!("[P-13] warning: could not fully normalize metadata: {e}");
    } else if let Err(e) = forge_llm::mud::p13::validate_converter_emit(&global_metadata) {
        eprintln!("[P-13] warning: emit incomplete: {e}");
    }

    // Pass 2: Writing & Quantizing
    let tensors_meta: Vec<(String, MudTensorType, Vec<usize>)> = plan
        .iter()
        .map(|p| (p.name.clone(), p.t_type, p.shape.clone()))
        .collect();
    let mut writer = StreamingMudWriter::create(output_path, &global_metadata, &tensors_meta)?;

    let mut current_scales: HashMap<String, Vec<f32>> = HashMap::new();

    for p in plan {
        if let Some(data) = p.owned_data {
            writer.write_tensor_data(&data)?;
        } else if p.name.contains(".prq_scale") {
            let scales = current_scales
                .remove(&p.name)
                .expect("Scale missing in stream");
            let bytes: Vec<u8> = scales.iter().flat_map(|s| s.to_le_bytes()).collect();
            writer.write_tensor_data(&bytes)?;
        } else if let Some(source_name) = p.source_name {
            let f_idx = p.source_file_idx.unwrap();
            let tensor_view = safe_tensors_list[f_idx].tensor(&source_name)?;

            // Disabled to force row-wise absmean uniformity for 26% sparsity compliance
            let is_native_ternary = false;

            // H1-04: Check for direct U8 repack BEFORE consuming sub_range
            // Disabled to force row-wise absmean uniformity for 26% sparsity compliance
            let is_u8_direct = false;

            let mut f32_data = if is_u8_direct {
                // Skip float conversion entirely for direct repack path
                Vec::new()
            } else {
                quantizer::to_f32_vec(&tensor_view)
            };

            if let Some(range) = p.sub_range {
                f32_data = f32_data[range].to_vec();
            }

            if p.should_ternarize {
                let bitnet_s = bitnet_scales.get(&source_name).copied().unwrap_or(1.0);

                if is_u8_direct {
                    let shape = tensor_view.shape();
                    let rows_packed = shape[0];
                    let cols = shape[1];
                    let (packed, mut scales) = quantizer::repack_bitnet_to_mud(
                        tensor_view.data(),
                        rows_packed,
                        cols,
                        bitnet_s,
                    );
                    if !is_native_ternary {
                        if let Some(d) = scales_map.get(&source_name) {
                            for s in &mut scales {
                                *s *= d;
                            }
                        }
                    }
                    let scale_name = p.name.replace(".weight", ".prq_scale");
                    current_scales.insert(scale_name, scales.clone());

                    // DEEP AUDIT:
                    quantizer::audit_ternary_fidelity(
                        &p.name,
                        &f32_data,
                        &packed,
                        &scales,
                        rows_packed * 4,
                        cols,
                        bitnet_s,
                    );

                    writer.write_tensor_data(&packed)?;
                } else if is_native_ternary {
                    let n_rows = f32_data.len() / p.shape[1];
                    let (packed, scales) =
                        quantizer::pack_native_ternary_f32(&f32_data, n_rows, p.shape[1], bitnet_s);
                    let scale_name = p.name.replace(".weight", ".prq_scale");
                    current_scales.insert(scale_name, scales.clone());

                    // DEEP AUDIT:
                    quantizer::audit_ternary_fidelity(
                        &p.name, &f32_data, &packed, &scales, n_rows, p.shape[1], bitnet_s,
                    );

                    writer.write_tensor_data(&packed)?;
                } else {
                    let n_rows = f32_data.len() / p.shape[1];
                    // H1-08 ternary: apply 1/√H to output.weight *before* pack (was float-only).
                    // Without this, untied emb→output copy has wrong logit magnitude vs HF.
                    if p.name == "output.weight" {
                        let hidden = p.shape[1].max(1);
                        let logit_scale = 1.0 / (hidden as f32).sqrt();
                        for f in &mut f32_data {
                            *f *= logit_scale;
                        }
                        println!(
                            "  [convert] output.weight × 1/√H ({logit_scale:.5}) before ternary pack"
                        );
                    }
                    let (packed, mut scales) =
                        quantizer::ternarize_f32_and_pack(&f32_data, n_rows, p.shape[1], bitnet_s);
                    if !is_native_ternary {
                        if let Some(d) = scales_map.get(&source_name) {
                            for s in &mut scales {
                                *s *= d;
                            }
                        }
                    }
                    let scale_name = p.name.replace(".weight", ".prq_scale");
                    current_scales.insert(scale_name, scales.clone());

                    // DEEP AUDIT:
                    quantizer::audit_ternary_fidelity(
                        &p.name, &f32_data, &packed, &scales, n_rows, p.shape[1], bitnet_s,
                    );

                    writer.write_tensor_data(&packed)?;
                }
            } else {
                // H1-08: Logit Scaling for Tied Embeddings (Float32 path)
                if p.name == "output.weight" {
                    let hidden = p.shape[1];
                    let logit_scale = 1.0 / (hidden as f32).sqrt();
                    for f in &mut f32_data {
                        *f *= logit_scale;
                    }
                }
                let mut bytes = Vec::with_capacity(f32_data.len() * 4);
                for f in f32_data {
                    bytes.extend_from_slice(&f.to_le_bytes());
                }
                writer.write_tensor_data(&bytes)?;
            }
        }
    }

    writer.close(output_path)?;

    println!("✅ MUD file created successfully: {}", output_path);

    // QAT-09: Generate ECC parity bytes for all ternary tensors post-conversion.
    println!("🔧 Generating ECC parity tensors...");
    {
        let mut mud = forge_llm::mud::MudFile::load(output_path)
            .map_err(|e| anyhow::anyhow!("ECC: failed to reload MUD file: {}", e))?;
        let ecc_count = mud.ecc_generate_all();
        mud.save(output_path)
            .map_err(|e| anyhow::anyhow!("ECC: failed to save MUD file: {}", e))?;
        println!(
            "  ✅ ECC parity generated for {} ternary tensors",
            ecc_count
        );
    }

    // ── Post-conversion metadata validation ──
    // P-13: verify critical metadata exists (engine + trainer depend on it)
    if args.iter().any(|a| a == "--check" || a == "--validate") {
        println!("🔍 Validating MUD metadata...");
        if let Ok(mud) = forge_llm::mud::MudFile::load(output_path) {
            let required_keys = [
                "hidden_size",
                "num_layers",
                "num_heads",
                "num_kv_heads",
                "ffn_hidden",
                "vocab_size",
                "rms_norm_eps",
            ];
            let mut ok = true;
            for k in &required_keys {
                if !mud.global_metadata.contains_key(*k) {
                    eprintln!("  ❌ Missing metadata: {}", k);
                    ok = false;
                }
            }
            if ok {
                let h = mud
                    .global_metadata
                    .get("hidden_size")
                    .cloned()
                    .unwrap_or_default();
                let l = mud
                    .global_metadata
                    .get("num_layers")
                    .cloned()
                    .unwrap_or_default();
                let v = mud
                    .global_metadata
                    .get("vocab_size")
                    .cloned()
                    .unwrap_or_default();
                println!("  ✅ Metadata OK: hidden={}, layers={}, vocab={}", h, l, v);
                println!("  ✅ Ready for engine + trainer (P-13: no hardcoded dims)");
            }
        } else {
            eprintln!("  ❌ Failed to load MUD file for validation");
        }
    }

    Ok(())
}
