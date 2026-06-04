mod calibration;
mod parser;
mod quantizer;

use forge_llm::mud::{MudFile, MudSkill, MudTensor, MudTensorType};
use serde_json::Value;
use std::collections::HashMap;
use std::env;
use std::fs;

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

fn extract_config_metadata(input_path: &str) -> Option<HashMap<String, String>> {
    let path = std::path::Path::new(input_path);
    let parent = path.parent()?;
    let config_path = parent.join("config.json");
    if !config_path.exists() {
        return None;
    }

    let content = fs::read_to_string(config_path).ok()?;
    let json: Value = serde_json::from_str(&content).ok()?;
    let mut meta = HashMap::new();

    if let Some(archs) = json.get("architectures").and_then(|a| a.as_array()) {
        if let Some(first_arch) = archs.first().and_then(|a| a.as_str()) {
            meta.insert("arch_original".to_string(), first_arch.to_string());
        }
    } else if let Some(model_type) = json.get("model_type").and_then(|m| m.as_str()) {
        meta.insert("arch_original".to_string(), model_type.to_string());
    }

    if let Some(layers) = json
        .get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .or_else(|| json.get("num_layers").and_then(|v| v.as_u64()))
    {
        meta.insert("num_layers".to_string(), layers.to_string());
    }

    if let Some(h) = json.get("hidden_size").and_then(|v| v.as_u64()) {
        meta.insert("hidden_size".to_string(), h.to_string());
    }

    if let Some(ffn) = json.get("intermediate_size").and_then(|v| v.as_u64()) {
        meta.insert("ffn_hidden".to_string(), ffn.to_string());
    }

    if let Some(exp) = json
        .get("num_local_experts")
        .and_then(|v| v.as_u64())
        .or_else(|| json.get("num_experts").and_then(|v| v.as_u64()))
    {
        meta.insert("num_experts".to_string(), exp.to_string());
    }

    if let Some(k) = json
        .get("num_experts_per_tok")
        .and_then(|v| v.as_u64())
        .or_else(|| json.get("num_experts_per_token").and_then(|v| v.as_u64()))
        .or_else(|| json.get("top_k").and_then(|v| v.as_u64()))
    {
        meta.insert("top_k".to_string(), k.to_string());
    }

    if let Some(heads) = json.get("num_attention_heads").and_then(|v| v.as_u64()) {
        meta.insert("num_heads".to_string(), heads.to_string());
    }

    if let Some(kv_heads) = json.get("num_key_value_heads").and_then(|v| v.as_u64()) {
        meta.insert("num_kv_heads".to_string(), kv_heads.to_string());
    }

    // Mamba / Jamba specific parameters
    if let Some(d_state) = json
        .get("state_size")
        .and_then(|v| v.as_u64())
        .or_else(|| json.get("ssm_d_state").and_then(|v| v.as_u64()))
        .or_else(|| json.get("d_state").and_then(|v| v.as_u64()))
    {
        meta.insert("d_state".to_string(), d_state.to_string());
    }

    if let Some(d_conv) = json
        .get("conv_kernel")
        .and_then(|v| v.as_u64())
        .or_else(|| json.get("ssm_d_conv").and_then(|v| v.as_u64()))
        .or_else(|| json.get("d_conv").and_then(|v| v.as_u64()))
    {
        meta.insert("d_conv".to_string(), d_conv.to_string());
    }

    if let Some(eps) = json.get("rms_norm_eps").and_then(|v| v.as_f64()) {
        meta.insert("rms_norm_eps".to_string(), eps.to_string());
    }

    if let Some(act) = json.get("hidden_act").and_then(|v| v.as_str()) {
        meta.insert("hidden_act".to_string(), act.to_string());
    }

    Some(meta)
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <input.safetensors> <output.mud> [--ternarize-emb]",
            args[0]
        );
        eprintln!(
            "  --ternarize-emb   Aplica ternarización row-wise absmean al embedding (ahorra ~16×)"
        );
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];
    let ternarize_emb = args.iter().any(|a| a == "--ternarize-emb");

    println!("🚀 Starting Universal Zero-Loss Ternary Converter (Pure Rust)");
    println!("📥 Input: {}", input_path);
    println!("📤 Output: {}", output_path);

    // Step 1: Parse Safetensors
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
        eprintln!("❌ No safetensors files found in {}", input_path);
        std::process::exit(1);
    }

    let mut mapped_files = vec![];
    for file in &safetensors_files {
        mapped_files.push(parser::mmap_file(file.to_str().unwrap())?);
    }

    let mut safe_tensors_list = vec![];
    let mut total_tensors = 0;
    for mapped_file in &mapped_files {
        let safe_tensors = parser::parse_safetensors(mapped_file)?;
        total_tensors += safe_tensors.tensors().len();
        safe_tensors_list.push(safe_tensors);
    }
    
    println!(
        "✅ Parsed {} tensors from {} safetensors files",
        total_tensors, safetensors_files.len()
    );

    // Step 2: Calibrate and compute depth-based dampening factors
    let scales_map = calibration::compute_scales(&safe_tensors_list);

    // Step 3: Quantize and Map
    let mut mud_tensors = HashMap::new();
    let mut max_layer = 0;
    let mut max_expert = 0;

    // Extract BitNet weight scales
    let mut bitnet_scales = HashMap::new();
    for safe_tensors in &safe_tensors_list {
        for (name, tensor_view) in safe_tensors.tensors() {
            if name.ends_with(".weight_scale") {
                let f32_bytes = quantizer::convert_to_f32_bytes(&tensor_view);
                if f32_bytes.len() >= 4 {
                    let scale_val = f32::from_le_bytes([f32_bytes[0], f32_bytes[1], f32_bytes[2], f32_bytes[3]]);
                    bitnet_scales.insert(name.replace(".weight_scale", ".weight"), scale_val);
                }
            }
        }
    }

    for safe_tensors in &safe_tensors_list {
        for (name, tensor_view) in safe_tensors.tensors() {
        if let Some((mapped_name, should_ternarize)) = parser::map_llama_to_mud(&name) {
            println!("   -> Mapping {} to {}", name, mapped_name);

            // Extract layer and expert counts
            if mapped_name.starts_with("blk.") {
                let parts: Vec<&str> = mapped_name.split('.').collect();
                if let Ok(l) = parts[1].parse::<usize>() {
                    if l > max_layer {
                        max_layer = l;
                    }
                }
                if parts.len() >= 4 && parts[2] == "expert" {
                    if let Ok(e) = parts[3].parse::<usize>() {
                        if e > max_expert {
                            max_expert = e;
                        }
                    }
                }
            }

            let t_type;
            let owned_data;
            let mut captured_scales = None;

            if should_ternarize {
                t_type = MudTensorType::Ternary2Bit;
                let bitnet_s = bitnet_scales.get(&name).copied().unwrap_or(1.0);
                let (data, mut scales) = quantizer::ternarize_and_pack(&tensor_view, bitnet_s);
                if let Some(dampening) = scales_map.get(&name) {
                    for s in &mut scales {
                        *s *= dampening;
                    }
                }
                owned_data = data;
                captured_scales = Some(scales);
            } else {
                t_type = MudTensorType::Float32;
                owned_data = quantizer::convert_to_f32_bytes(&tensor_view);
            };

            mud_tensors.insert(
                mapped_name.clone(),
                MudTensor {
                    name: mapped_name.clone(),
                    t_type,
                    shape: if tensor_view.dtype() == safetensors::tensor::Dtype::U8 {
                        let mut s = tensor_view.shape().to_vec();
                        s[0] *= 4;
                        s
                    } else {
                        tensor_view.shape().to_vec()
                    },
                    data_ptr: std::ptr::null(),
                    offset: 0,
                    mmap: None,
                    owned_data: Some(owned_data),
                },
            );

            if let Some(scales) = captured_scales {
                let scale_name = mapped_name.replace(".weight", ".prq_scale");
                let n_rows = scales.len();
                let scales_bytes: Vec<u8> = scales.iter().flat_map(|s| s.to_le_bytes()).collect();

                mud_tensors.insert(
                    scale_name.clone(),
                    MudTensor {
                        name: scale_name,
                        t_type: MudTensorType::Float32,
                        shape: vec![n_rows],
                        data_ptr: std::ptr::null(),
                        offset: 0,
                        mmap: None,
                        owned_data: Some(scales_bytes),
                    },
                );
            }

            // UNTIE EMBEDDINGS: If this is the embedding layer, also create the output projection in FP32
            // Since tie_word_embeddings is true in Qwen, we manually untie them to preserve both 
            // semantic locality (Ternary without diffusion) and logits precision (FP32).
            if name == "model.embed_tokens.weight" {
                println!("   -> [UNTIE] Duplicating model.embed_tokens.weight to output.weight (FP32)");
                let fp32_data = quantizer::convert_to_f32_bytes(&tensor_view);
                mud_tensors.insert(
                    "output.weight".to_string(),
                    MudTensor {
                        name: "output.weight".to_string(),
                        t_type: MudTensorType::Float32,
                        shape: if tensor_view.dtype() == safetensors::tensor::Dtype::U8 {
                            let mut s = tensor_view.shape().to_vec();
                            s[0] *= 4;
                            s
                        } else {
                            tensor_view.shape().to_vec()
                        },
                        data_ptr: std::ptr::null(),
                        offset: 0,
                        mmap: None,
                        owned_data: Some(fp32_data),
                    },
                );
            }
        }
    }
    }

    // Real MoE gates are now parsed from safetensors directly.

    println!("✅ Quantization and Structural Mapping complete.");

    let mut global_metadata = HashMap::new();
    let has_gate = mud_tensors.keys().any(|k| k.contains(".gate.weight"));
    let config_meta = extract_config_metadata(input_path).unwrap_or_default();

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

    global_metadata.insert("arch".to_string(), "mud-ternary-moe-v1-master".to_string());
    global_metadata.insert("num_layers".to_string(), num_layers.to_string());
    global_metadata.insert("num_experts".to_string(), num_experts.to_string());
    global_metadata.insert("top_k".to_string(), top_k.to_string());

    if let Some(d_state) = config_meta.get("d_state") {
        global_metadata.insert("d_state".to_string(), d_state.clone());
    }
    if let Some(d_conv) = config_meta.get("d_conv") {
        global_metadata.insert("d_conv".to_string(), d_conv.clone());
    }
    if let Some(act) = config_meta.get("hidden_act") {
        global_metadata.insert("hidden_act".to_string(), act.clone());
    }

    // Inject QAT metadata
    global_metadata.insert(
        "qat.scale_dampening".to_string(),
        "heuristic_depth_squared_0.35".to_string(),
    );

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

    // Locate tokenizer.json dynamically next to the safetensors file
    let input_dir = std::path::Path::new(input_path)
        .parent()
        .unwrap_or(std::path::Path::new("."));
    let tokenizer_file = input_dir.join("tokenizer.json");
    let tokenizer_path_str = tokenizer_file.to_string_lossy().to_string();
    let tokenizer_path = if tokenizer_file.exists() {
        &tokenizer_path_str
    } else {
        "models/qwen2_0.5b/tokenizer.json"
    };

    // Load tokenizer_config.json for chat_template / bos / eos
    let tokenizer_config_file = input_dir.join("tokenizer_config.json");
    if tokenizer_config_file.exists() {
        if let Ok(raw) = std::fs::read_to_string(&tokenizer_config_file) {
            if let Ok(cfg) = serde_json::from_str::<serde_json::Value>(&raw) {
                if let Some(tmpl) = cfg.get("chat_template").and_then(|v| v.as_str()) {
                    global_metadata.insert("chat_template".to_string(), tmpl.to_string());
                    println!("✅ Injected chat_template from tokenizer_config.json");
                }
                if let Some(bos) = cfg.get("bos_token").and_then(|v| v.as_str()) {
                    global_metadata.insert("bos_token".to_string(), bos.to_string());
                }
                if let Some(eos) = cfg.get("eos_token").and_then(|v| v.as_str()) {
                    global_metadata.insert("eos_token".to_string(), eos.to_string());
                }
                // Check if the model is natively ternary (skip PRQ re-quantization)
                if cfg.get("model_type").and_then(|v| v.as_str()) == Some("bitnet") {
                    global_metadata.insert("native_ternary".to_string(), "true".to_string());
                    println!("🔢 Detected native ternary model (BitNet) — PRQ passthrough mode");
                }
            }
        }
    }

    if let Some(tokens_str) = extract_vocab_from_json(tokenizer_path) {
        vocab_size = tokens_str.lines().count();
        println!(
            "✅ Injected authentic tokenizer from {} (Vocab Size: {})",
            tokenizer_path, vocab_size
        );

        // --- INICIO DE ANÁLISIS DE SÍMBOLOS ---
        let mut count_gpt_space = 0;
        let mut count_sp_space = 0;
        let mut special_marks = Vec::new();

        for line in tokens_str.lines() {
            let t = line.trim();
            // Qwen typically uses GPT-style spaces (Ġ)
            if t.contains('Ġ') {
                count_gpt_space += 1;
            }
            if t.contains('\u{2581}') {
                count_sp_space += 1;
            }

            if (t.starts_with('<') && t.ends_with('>')) || (t.starts_with('[') && t.ends_with(']'))
            {
                special_marks.push(t.to_string());
            }
        }

        let space_prefix = if count_sp_space > count_gpt_space {
            "\u{2581}" // SentencePiece space prefix
        } else {
            "Ġ" // GPT space prefix
        };

        println!(
            "   [Concordance-Analyzer] Space Prefix: '{}' (GPT-Freq: {}, SP-Freq: {})",
            space_prefix, count_gpt_space, count_sp_space
        );
        if !special_marks.is_empty() {
            println!(
                "   [Concordance-Analyzer] Control Marks Detected: {:?}",
                &special_marks[0..10.min(special_marks.len())]
            );
            global_metadata.insert(
                "tokenizer.special_marks".to_string(),
                special_marks.join(","),
            );
        }
        global_metadata.insert(
            "tokenizer.space_prefix".to_string(),
            space_prefix.to_string(),
        );
        // --- FIN DE ANÁLISIS DE SÍMBOLOS ---

        global_metadata.insert("tokenizer.tokens".to_string(), tokens_str);

        if let Some(merges_str) = extract_merges_from_json(tokenizer_path) {
            println!(
                "✅ Injected authentic BPE merges (Merges Count: {})",
                merges_str.lines().count()
            );
            global_metadata.insert("tokenizer.merges".to_string(), merges_str);
        } else {
            println!("⚠️ Warning: BPE merges not found in {}", tokenizer_path);
        }
    } else {
        println!(
            "⚠️ Warning: {} not found or parse failed. Using fallback 32k tokenizer.",
            tokenizer_path
        );
    }

    // Inject synthetic embeddings if missing, with correct hidden_size and vocab_size
    let hidden_size = config_meta
        .get("hidden_size")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(
            mud_tensors
                .get("blk.0.attn_norm.weight")
                .or_else(|| mud_tensors.get("blk.0.norm.weight"))
                .map(|t| t.shape[0])
                .unwrap_or(4096),
        );

    let ffn_hidden = config_meta
        .get("ffn_hidden")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(
            mud_tensors
                .keys()
                .find(|k| k.starts_with("blk.0.expert.") && k.ends_with(".w1.weight"))
                .and_then(|k| mud_tensors.get(k))
                .map(|t| t.shape[0])
                .unwrap_or(hidden_size * 4),
        );

    let kv_dim = config_meta
        .get("num_kv_heads")
        .and_then(|s| s.parse::<usize>().ok())
        .zip(
            config_meta
                .get("head_dim")
                .and_then(|s| s.parse::<usize>().ok()),
        )
        .map(|(kv_heads, h_dim)| kv_heads * h_dim)
        .unwrap_or(
            mud_tensors
                .get("blk.0.attn_k.weight")
                .map(|t| t.shape[0])
                .unwrap_or(hidden_size),
        );

    // Infer MHA dimensions from Q and K projection shapes
    let q_out = mud_tensors
        .get("blk.0.attn_q.weight")
        .map(|t| t.shape[0])
        .unwrap_or(hidden_size);
    let mut head_dim = config_meta
        .get("head_dim")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(64);

    if !config_meta.contains_key("head_dim") {
        if q_out % 64 == 0 && kv_dim.is_multiple_of(64) {
            head_dim = 64;
        } else if q_out % 128 == 0 && kv_dim.is_multiple_of(128) {
            head_dim = 128;
        }
    }

    let num_heads = config_meta
        .get("num_heads")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(q_out / head_dim);

    let num_kv_heads = config_meta
        .get("num_kv_heads")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(kv_dim / head_dim);

    global_metadata.insert("hidden_size".to_string(), hidden_size.to_string());
    global_metadata.insert("ffn_hidden".to_string(), ffn_hidden.to_string());
    global_metadata.insert("kv_dim".to_string(), kv_dim.to_string());
    global_metadata.insert("num_heads".to_string(), num_heads.to_string());
    global_metadata.insert("num_kv_heads".to_string(), num_kv_heads.to_string());
    global_metadata.insert("head_dim".to_string(), head_dim.to_string());
    println!(
        "🏷️ Attention: {} heads × {} dim ({} KV heads, {} group)",
        num_heads,
        head_dim,
        num_kv_heads,
        num_heads / num_kv_heads
    );

    // Metadatos para Tokenizador (Concordancia, Marcas y Espacios)
    global_metadata.insert(
        "tokenizer.special_marks".to_string(),
        "<thinking>,</thinking>,<answer>,</answer>,<step>".to_string(),
    );
    global_metadata.insert("tokenizer.preserve_space".to_string(), "true".to_string());
    global_metadata.insert("tokenizer.coherence_mode".to_string(), "strict".to_string());

    if !mud_tensors.contains_key("token_embd.weight") {
        println!(
            "   -> Generating synthetic token_embd.weight ({}x{})",
            vocab_size, hidden_size
        );
        let size = vocab_size * hidden_size;
        let mut data = Vec::with_capacity(size * 4);
        for _ in 0..size {
            data.extend_from_slice(&0.01f32.to_le_bytes());
        }
        mud_tensors.insert(
            "token_embd.weight".to_string(),
            MudTensor {
                name: "token_embd.weight".to_string(),
                t_type: MudTensorType::Float32,
                shape: vec![vocab_size, hidden_size],
                data_ptr: std::ptr::null(),
                offset: 0,
                mmap: None,
                owned_data: Some(data),
            },
        );
    }

    // --- Embedding Ternarization (si --ternarize-emb) ---
    if ternarize_emb {
        if let Some(emb_tensor) = mud_tensors.get("token_embd.weight") {
            let vocab = emb_tensor.shape[0];
            let hidden = emb_tensor.shape[1];
            let total = vocab * hidden;
            println!(
                "   -> Ternarizando embedding ({} × {} = {:.1}M params)...",
                vocab,
                hidden,
                total as f64 / 1_000_000.0
            );

            // Leer datos f32 actuales
            let emb_f32 = if let Some(owned) = &emb_tensor.owned_data {
                owned
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect::<Vec<_>>()
            } else {
                anyhow::bail!("token_embd.weight debe tener owned_data en este punto");
            };

            let (packed_ternary, scales, meta) =
                quantizer::embedding_rowwise_ternarize(&emb_f32, vocab, hidden);

            // Reemplazar tensor embedding con Ternary2Bit
            let vocab_size = vocab;
            mud_tensors.insert(
                "token_embd.weight".to_string(),
                MudTensor {
                    name: "token_embd.weight".to_string(),
                    t_type: MudTensorType::Ternary2Bit,
                    shape: vec![vocab_size, hidden],
                    data_ptr: std::ptr::null(),
                    offset: 0,
                    mmap: None,
                    owned_data: Some(packed_ternary),
                },
            );

            // Almacenar escalas como tensor Float32 (1 f32 por fila)
            let scales_bytes: Vec<u8> = scales.iter().flat_map(|s| s.to_le_bytes()).collect();
            mud_tensors.insert(
                "embed_scales".to_string(),
                MudTensor {
                    name: "embed_scales".to_string(),
                    t_type: MudTensorType::Float32,
                    shape: vec![vocab_size],
                    data_ptr: std::ptr::null(),
                    offset: 0,
                    mmap: None,
                    owned_data: Some(scales_bytes),
                },
            );

            for (k, v) in &meta {
                global_metadata.insert(k.clone(), v.clone());
            }

            let before_size = total * 4;
            let after_data = total * 2 / 8;
            let after_scales = vocab_size * 4; // f32 per row
            println!(
                "     ✅ Embedding: {:.1} MB → {:.1} MB ({:.1}×)",
                before_size as f64 / 1_048_576.0,
                (after_data + after_scales) as f64 / 1_048_576.0,
                before_size as f64 / (after_data + after_scales) as f64
            );
        }
    }

    if !mud_tensors.contains_key("output_norm.weight") {
        println!(
            "   -> Generating synthetic output_norm.weight ({})",
            hidden_size
        );
        let size = hidden_size;
        let mut data = Vec::with_capacity(size * 4);
        for _ in 0..size {
            data.extend_from_slice(&1.0f32.to_le_bytes());
        }
        mud_tensors.insert(
            "output_norm.weight".to_string(),
            MudTensor {
                name: "output_norm.weight".to_string(),
                t_type: MudTensorType::Float32,
                shape: vec![hidden_size],
                data_ptr: std::ptr::null(),
                offset: 0,
                mmap: None,
                owned_data: Some(data),
            },
        );
    }

    // Step 4: Export to .mud
    let mut skills = HashMap::new();
    skills.insert(
        "core".to_string(),
        MudSkill {
            name: "core".to_string(),
            tensors: mud_tensors,
            metadata: HashMap::new(),
        },
    );

    let mud_file = MudFile {
        mmap: None,
        skills,
        global_metadata,
    };

    mud_file.save(output_path)?;
    println!("🏁 Successfully exported to {}!", output_path);
    Ok(())
}
