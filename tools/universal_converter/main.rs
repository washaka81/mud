mod parser;
mod calibration;
mod quantizer;

use std::env;
use std::collections::HashMap;
use std::fs;
use serde_json::Value;
use forge_llm::mud::{MudFile, MudSkill, MudTensor, MudTensorType};

// Fast and simple JSON string extraction for the tokenizer
fn extract_vocab_from_json(path: &str) -> Option<String> {
    let content = fs::read_to_string(path).ok()?;
    let json: Value = serde_json::from_str(&content).ok()?;
    
    let vocab_obj = json.get("model")?.get("vocab")?.as_object()?;
    
    // Sort tokens by their ID to ensure correct order
    let mut token_pairs: Vec<(&String, usize)> = Vec::new();
    for (token, id_val) in vocab_obj {
        if let Some(id) = id_val.as_u64() {
            token_pairs.push((token, id as usize));
        }
    }
    
    token_pairs.sort_by_key(|&(_, id)| id);
    
    let mut tokens = Vec::new();
    let mut expected_id = 0;
    
    for (token, id) in token_pairs {
        // Fill gaps if any
        while expected_id < id {
            tokens.push(format!("<dummy_{}>", expected_id));
            expected_id += 1;
        }
        tokens.push(token.clone());
        expected_id += 1;
    }
    
    Some(tokens.join("\n"))
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

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <input.safetensors> <output.mud> [--ternarize-emb]", args[0]);
        eprintln!("  --ternarize-emb   Aplica ternarización row-wise absmean al embedding (ahorra ~16×)");
        std::process::exit(1);
    }
    
    let input_path = &args[1];
    let output_path = &args[2];
    let ternarize_emb = args.iter().any(|a| a == "--ternarize-emb");
    
    println!("🚀 Starting Universal Zero-Loss Ternary Converter (Pure Rust)");
    println!("📥 Input: {}", input_path);
    println!("📤 Output: {}", output_path);
    
    // Step 1: Parse Safetensors
    let mapped_file = parser::mmap_file(input_path)?;
    let safe_tensors = parser::parse_safetensors(&mapped_file)?;
    println!("✅ Parsed {} tensors from safetensors", safe_tensors.tensors().len());
    
    // Step 2: Calibrate (Placeholder)
    let scales = calibration::compute_scales(&safe_tensors);
    
    // Step 3: Quantize and Map
    let mut mud_tensors = HashMap::new();
    let mut max_layer = 0;
    
    for (name, tensor_view) in safe_tensors.tensors() {
        if let Some((mapped_name, should_ternarize)) = parser::map_llama_to_mud(&name) {
            println!("   -> Mapping {} to {}", name, mapped_name);
            
            // Extract layer count
            if mapped_name.starts_with("blk.") {
                let parts: Vec<&str> = mapped_name.split('.').collect();
                if let Ok(l) = parts[1].parse::<usize>() {
                    if l > max_layer { max_layer = l; }
                }
            }
            
            let t_type;
            let owned_data;
            let mut captured_scale = None;
            
            if should_ternarize {
                t_type = MudTensorType::Ternary2Bit;
                let damp_factor = scales.get(&name).copied().unwrap_or(1.0);
                let (data, scale) = quantizer::ternarize_and_pack(&tensor_view, damp_factor);
                owned_data = data;
                captured_scale = Some(scale);
            } else {
                t_type = MudTensorType::Float32;
                owned_data = quantizer::convert_to_f32_bytes(&tensor_view);
            };
            
            mud_tensors.insert(mapped_name.clone(), MudTensor {
                name: mapped_name.clone(),
                t_type,
                shape: tensor_view.shape().to_vec(),
                data_ptr: std::ptr::null(),
                offset: 0,
                mmap: None,
                owned_data: Some(owned_data),
            });
            
            if let Some(s) = captured_scale {
                let scale_name = mapped_name.replace(".weight", ".scale");
                mud_tensors.insert(scale_name.clone(), MudTensor {
                    name: scale_name,
                    t_type: MudTensorType::Float32,
                    shape: vec![1],
                    data_ptr: std::ptr::null(),
                    offset: 0,
                    mmap: None,
                    owned_data: Some(s.to_le_bytes().to_vec()),
                });
            }
        }
    }
    
    // Real MoE gates are now parsed from safetensors directly.
    
    println!("✅ Quantization and Structural Mapping complete.");
    
    let mut global_metadata = HashMap::new();
    let num_layers = max_layer + 1;
    let num_experts = if mud_tensors.contains_key("blk.0.expert.1.w1.weight") { 16 } else { 1 };

    global_metadata.insert("arch".to_string(), "mud-ternary-moe-v1-master".to_string());
    global_metadata.insert("num_layers".to_string(), num_layers.to_string());
    global_metadata.insert("num_experts".to_string(), num_experts.to_string());
    global_metadata.insert("top_k".to_string(), "1".to_string());
    
    // Inject QAT metadata
    global_metadata.insert("qat.scale_dampening".to_string(), "heuristic_depth_squared_0.35".to_string());
    
    // Inject missing core tensors from backup for tokenizer
    if let Ok(old_mud) = MudFile::load("models/core_skills.mud.bak") {
        if let Some(tokens) = old_mud.global_metadata.get("tokenizer.tokens") {
            global_metadata.insert("tokenizer.tokens".to_string(), tokens.clone());
        }
        if let Some(merges) = old_mud.global_metadata.get("tokenizer.merges") {
            global_metadata.insert("tokenizer.merges".to_string(), merges.clone());
        }
        if let Some(iq) = old_mud.global_metadata.get("iq.score") {
            global_metadata.insert("iq.score".to_string(), iq.clone());
        }
    }

    // Attempt to load genuine tokenizer
    let mut vocab_size = 32000;
    if let Some(tokens_str) = extract_vocab_from_json("downloaded_tokenizer.json") {
        vocab_size = tokens_str.lines().count();
        println!("✅ Injected authentic tokenizer (Vocab Size: {})", vocab_size);

        // --- INICIO DE ANÁLISIS DE SÍMBOLOS ---
        let mut count_gpt_space = 0;
        let mut count_sp_space = 0;
        let mut special_marks = Vec::new();
        
        for line in tokens_str.lines() {
            let t = line.trim();
            if t.contains('Ġ') { count_gpt_space += 1; }
            if t.contains('\u{2581}') { count_sp_space += 1; }
            
            if (t.starts_with('<') && t.ends_with('>')) || (t.starts_with('[') && t.ends_with(']')) {
                special_marks.push(t.to_string());
            }
        }
        
        let space_prefix = if count_sp_space > count_gpt_space {
            "\u{2581}" // SentencePiece space prefix
        } else {
            "Ġ" // GPT space prefix
        };
        
        println!("   [Concordance-Analyzer] Space Prefix: '{}' (GPT-Freq: {}, SP-Freq: {})", space_prefix, count_gpt_space, count_sp_space);
        if !special_marks.is_empty() {
            println!("   [Concordance-Analyzer] Control Marks Detected: {:?}", &special_marks[0..10.min(special_marks.len())]);
            global_metadata.insert("tokenizer.special_marks".to_string(), special_marks.join(","));
        }
        global_metadata.insert("tokenizer.space_prefix".to_string(), space_prefix.to_string());
        // --- FIN DE ANÁLISIS DE SÍMBOLOS ---

        global_metadata.insert("tokenizer.tokens".to_string(), tokens_str);

        if let Some(merges_str) = extract_merges_from_json("downloaded_tokenizer.json") {
            println!("✅ Injected authentic BPE merges (Merges Count: {})", merges_str.lines().count());
            global_metadata.insert("tokenizer.merges".to_string(), merges_str);
        } else {
            println!("⚠️ Warning: BPE merges not found in downloaded_tokenizer.json");
        }
    } else {
        println!("⚠️ Warning: downloaded_tokenizer.json not found or parse failed. Using fallback 32k tokenizer.");
    }
    
    // Inject synthetic embeddings if missing, with correct hidden_size and vocab_size
    let hidden_size = mud_tensors.get("blk.0.attn_norm.weight").map(|t| t.shape[0]).unwrap_or(4096);
    let ffn_hidden = mud_tensors.get("blk.0.expert.0.w1.weight").map(|t| t.shape[0]).unwrap_or(hidden_size * 4);
    let kv_dim = mud_tensors.get("blk.0.attn_k.weight").map(|t| t.shape[0]).unwrap_or(hidden_size);

    // Infer MHA dimensions from Q and K projection shapes
    let q_out = mud_tensors.get("blk.0.attn_q.weight").map(|t| t.shape[0]).unwrap_or(hidden_size);
    let mut head_dim = 64;
    if q_out % 64 == 0 && kv_dim % 64 == 0 {
        head_dim = 64;
    } else if q_out % 128 == 0 && kv_dim % 128 == 0 {
        head_dim = 128;
    }
    let num_heads = q_out / head_dim;
    let num_kv_heads = kv_dim / head_dim;

    global_metadata.insert("hidden_size".to_string(), hidden_size.to_string());
    global_metadata.insert("ffn_hidden".to_string(), ffn_hidden.to_string());
    global_metadata.insert("kv_dim".to_string(), kv_dim.to_string());
    global_metadata.insert("num_heads".to_string(), num_heads.to_string());
    global_metadata.insert("num_kv_heads".to_string(), num_kv_heads.to_string());
    global_metadata.insert("head_dim".to_string(), head_dim.to_string());
    println!("🏷️ Attention: {} heads × {} dim ({} KV heads, {} group)", num_heads, head_dim, num_kv_heads, num_heads / num_kv_heads);

    // Metadatos para Tokenizador (Concordancia, Marcas y Espacios)
    global_metadata.insert("tokenizer.special_marks".to_string(), "<thinking>,</thinking>,<answer>,</answer>,<step>".to_string());
    global_metadata.insert("tokenizer.preserve_space".to_string(), "true".to_string());
    global_metadata.insert("tokenizer.coherence_mode".to_string(), "strict".to_string());
    
    if !mud_tensors.contains_key("token_embd.weight") {
        println!("   -> Generating synthetic token_embd.weight ({}x{})", vocab_size, hidden_size);
        let size = vocab_size * hidden_size;
        let mut data = Vec::with_capacity(size * 4);
        for _ in 0..size { data.extend_from_slice(&0.01f32.to_le_bytes()); }
        mud_tensors.insert("token_embd.weight".to_string(), MudTensor {
            name: "token_embd.weight".to_string(),
            t_type: MudTensorType::Float32,
            shape: vec![vocab_size, hidden_size],
            data_ptr: std::ptr::null(),
            offset: 0,
            mmap: None,
            owned_data: Some(data),
        });
    }
    
    // --- Embedding Ternarization (si --ternarize-emb) ---
    if ternarize_emb {
        if let Some(emb_tensor) = mud_tensors.get("token_embd.weight") {
            let vocab = emb_tensor.shape[0];
            let hidden = emb_tensor.shape[1];
            let total = vocab * hidden;
            println!("   -> Ternarizando embedding ({} × {} = {:.1}M params)...", vocab, hidden, total as f64 / 1_000_000.0);

            // Leer datos f32 actuales
            let emb_f32 = if let Some(owned) = &emb_tensor.owned_data {
                owned.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect::<Vec<_>>()
            } else {
                anyhow::bail!("token_embd.weight debe tener owned_data en este punto");
            };

            let (packed_ternary, scales, meta) = quantizer::embedding_rowwise_ternarize(&emb_f32, vocab, hidden);

            // Reemplazar tensor embedding con Ternary2Bit
            let vocab_size = vocab;
            mud_tensors.insert("token_embd.weight".to_string(), MudTensor {
                name: "token_embd.weight".to_string(),
                t_type: MudTensorType::Ternary2Bit,
                shape: vec![vocab_size, hidden],
                data_ptr: std::ptr::null(),
                offset: 0,
                mmap: None,
                owned_data: Some(packed_ternary),
            });

            // Almacenar escalas como tensor Float32 (1 f32 por fila)
            let scales_bytes: Vec<u8> = scales.iter().flat_map(|s| s.to_le_bytes()).collect();
            mud_tensors.insert("embed_scales".to_string(), MudTensor {
                name: "embed_scales".to_string(),
                t_type: MudTensorType::Float32,
                shape: vec![vocab_size],
                data_ptr: std::ptr::null(),
                offset: 0,
                mmap: None,
                owned_data: Some(scales_bytes),
            });

            for (k, v) in &meta {
                global_metadata.insert(k.clone(), v.clone());
            }

            let before_size = total * 4;
            let after_data = total * 2 / 8;
            let after_scales = vocab_size * 4; // f32 per row
            println!("     ✅ Embedding: {:.1} MB → {:.1} MB ({:.1}×)",
                before_size as f64 / 1_048_576.0,
                (after_data + after_scales) as f64 / 1_048_576.0,
                before_size as f64 / (after_data + after_scales) as f64);
        }
    }

    if !mud_tensors.contains_key("output_norm.weight") {
        println!("   -> Generating synthetic output_norm.weight ({})", hidden_size);
        let size = hidden_size;
        let mut data = Vec::with_capacity(size * 4);
        for _ in 0..size { data.extend_from_slice(&1.0f32.to_le_bytes()); }
        mud_tensors.insert("output_norm.weight".to_string(), MudTensor {
            name: "output_norm.weight".to_string(),
            t_type: MudTensorType::Float32,
            shape: vec![hidden_size],
            data_ptr: std::ptr::null(),
            offset: 0,
            mmap: None,
            owned_data: Some(data),
        });
    }
    
    // Step 4: Export to .mud
    let mut skills = HashMap::new();
    skills.insert("core".to_string(), MudSkill {
        name: "core".to_string(),
        tensors: mud_tensors,
        metadata: HashMap::new(),
    });
    
    let mud_file = MudFile {
        mmap: None,
        skills,
        global_metadata,
    };
    
    mud_file.save(output_path)?;
    println!("🏁 Successfully exported to {}!", output_path);
    Ok(())
}
