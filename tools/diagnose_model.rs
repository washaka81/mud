/// Diagnostic tool: dumps tensor inventory and statistics from a .mud file
/// to help identify why inference produces gibberish.
use forge_llm::mud::MudFile;

fn main() -> anyhow::Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/smollm2.mud".to_string());
    println!("=== MUD Model Diagnostic ===");
    println!("Loading: {}", model_path);

    let mud_file = MudFile::load(&model_path)?;

    // 1. Global metadata
    println!("\n--- Global Metadata ---");
    let mut meta_keys: Vec<_> = mud_file.global_metadata.keys().collect();
    meta_keys.sort();
    for key in &meta_keys {
        let val = &mud_file.global_metadata[*key];
        if key.starts_with("tokenizer.") {
            println!(
                "  {} = [{}...] (len={})",
                key,
                &val[..val.len().min(80)],
                val.len()
            );
        } else {
            println!("  {} = {}", key, val);
        }
    }

    // 2. Skill inventory
    println!("\n--- Skills ---");
    for (name, skill) in &mud_file.skills {
        println!("  Skill '{}': {} tensors", name, skill.tensors.len());
    }

    // 3. Core tensor inventory
    let core = mud_file.skills.get("core").expect("No core skill");
    println!("\n--- Core Tensors (sorted) ---");
    let mut tensor_names: Vec<_> = core.tensors.keys().collect();
    tensor_names.sort();

    let hidden_size = mud_file
        .global_metadata
        .get("hidden_size")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    let num_layers = mud_file
        .global_metadata
        .get("num_layers")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    let num_experts = mud_file
        .global_metadata
        .get("num_experts")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);

    // Track what we found
    let mut found_attn_norm = vec![false; num_layers];
    let mut found_ffn_norm = vec![false; num_layers];
    let mut found_output_norm = false;
    let mut null_weight_count = 0;

    for name in &tensor_names {
        let tensor = &core.tensors[*name];
        let elements: usize = tensor.shape.iter().product();
        let type_name = format!("{:?}", tensor.t_type);
        let data_null = tensor.data_ptr.is_null();
        if data_null {
            null_weight_count += 1;
        }

        // Sample a few values if Float32
        let sample = if !data_null
            && tensor.t_type == forge_llm::mud::MudTensorType::Float32
            && elements > 0
        {
            let n = elements.min(8);
            let mut vals = vec![0.0f32; n];
            unsafe {
                std::ptr::copy_nonoverlapping(tensor.data_ptr as *const f32, vals.as_mut_ptr(), n);
            }
            format!(" sample={:?}", vals)
        } else if !data_null
            && tensor.t_type == forge_llm::mud::MudTensorType::Ternary2Bit
            && elements > 0
        {
            let n = elements.min(32);
            let mut vals = vec![0.0f32; n];
            unsafe {
                forge_llm::mud::dequantize_ternary_row(tensor.data_ptr as *const u32, &mut vals, n);
            }
            let nonzero = vals.iter().filter(|&&v| v != 0.0).count();
            format!(" ternary_sample: {}/{} nonzero", nonzero, n)
        } else {
            String::new()
        };

        // Track norm layers
        if name.contains("attn_norm.weight") {
            if let Some(l) = name
                .strip_prefix("blk.")
                .and_then(|s| s.split('.').next())
                .and_then(|s| s.parse::<usize>().ok())
            {
                if l < num_layers {
                    found_attn_norm[l] = true;
                }
            }
        }
        if name.contains("norm.weight")
            && !name.contains("attn_norm")
            && !name.contains("output_norm")
        {
            if let Some(l) = name
                .strip_prefix("blk.")
                .and_then(|s| s.split('.').next())
                .and_then(|s| s.parse::<usize>().ok())
            {
                if l < num_layers {
                    found_ffn_norm[l] = true;
                }
            }
        }
        if *name == "output_norm.weight" {
            found_output_norm = true;
        }

        println!(
            "  {} | shape={:?} | type={} | null={}{}",
            name, tensor.shape, type_name, data_null, sample
        );
    }

    // 4. Structural analysis
    println!("\n--- Structural Analysis ---");
    println!(
        "  hidden_size={}, num_layers={}, num_experts={}",
        hidden_size, num_layers, num_experts
    );
    println!("  total tensors: {}", tensor_names.len());
    println!("  null data pointers: {}", null_weight_count);

    // Check expected tensors exist
    let _expected_per_layer: Vec<String> = {
        let mut v = vec![
            "attn_q.weight".to_string(),
            "attn_k.weight".to_string(),
            "attn_v.weight".to_string(),
            "attn_output.weight".to_string(),
        ];
        for e in 0..num_experts {
            v.push(format!("expert.{}.w1.weight", e));
            v.push(format!("expert.{}.w2.weight", e));
            v.push(format!("expert.{}.w3.weight", e));
        }
        v
    };

    let mut missing = Vec::new();
    for l in 0..num_layers {
        let is_mamba = core.tensors.contains_key(&format!("blk.{}.ssm_a", l));

        let expected_this_layer: Vec<String> = if is_mamba {
            vec![
                "ssm_in.weight".to_string(),
                "ssm_out.weight".to_string(),
                "ssm_x.weight".to_string(),
                "ssm_dt.weight".to_string(),
                "ssm_a".to_string(),
                "ssm_d".to_string(),
                "ssm_conv1d.weight".to_string(),
            ]
        } else {
            let mut v = vec![
                "attn_q.weight".to_string(),
                "attn_k.weight".to_string(),
                "attn_v.weight".to_string(),
                "attn_output.weight".to_string(),
            ];
            for e in 0..num_experts {
                v.push(format!("expert.{}.w1.weight", e));
                v.push(format!("expert.{}.w2.weight", e));
                v.push(format!("expert.{}.w3.weight", e));
            }
            v
        };

        for suffix in &expected_this_layer {
            let key = format!("blk.{}.{}", l, suffix);
            if !core.tensors.contains_key(&key) {
                missing.push(key);
            }
        }

        if !found_attn_norm[l] && !found_ffn_norm[l] {
            println!("  ⚠️  Layer {} has NO norm weights at all!", l);
        } else if !is_mamba {
            if !found_attn_norm[l] {
                println!("  ⚠️  Layer {} missing attn_norm.weight", l);
            }
            if !found_ffn_norm[l] {
                println!(
                    "  ⚠️  Layer {} missing ffn_norm.weight (blk.{l}.norm.weight)",
                    l
                );
            }
        }
    }

    if !found_output_norm {
        println!("  ⚠️  Missing output_norm.weight!");
    }

    if !missing.is_empty() {
        println!("\n  MISSING TENSORS ({}):", missing.len());
        for m in &missing[..missing.len().min(20)] {
            println!("    ❌ {}", m);
        }
        if missing.len() > 20 {
            println!("    ... and {} more", missing.len() - 20);
        }
    } else {
        println!("  ✅ All expected layer tensors present.");
    }

    // 5. Check Qwen2 specifics: the model should have 1 expert (dense), not 4
    if num_experts > 1 {
        println!("\n  ℹ️  num_experts={} — this is a MoE model", num_experts);
    } else {
        println!(
            "\n  ℹ️  num_experts={} — this is a dense model",
            num_experts
        );
    }

    // 6. Quick inference pipeline check
    println!("\n--- Embedding Check ---");
    if let Some(embd) = core.tensors.get("token_embd.weight") {
        println!(
            "  token_embd.weight: shape={:?}, type={:?}",
            embd.shape, embd.t_type
        );
        let token_id = 1u32; // check token 1
        if embd.t_type == forge_llm::mud::MudTensorType::Ternary2Bit {
            let offset = (token_id as usize) * (hidden_size / 8);
            let mut row = vec![0.0f32; hidden_size];
            unsafe {
                forge_llm::mud::dequantize_ternary_row(
                    (embd.data_ptr as *const u32).add(offset),
                    &mut row,
                    hidden_size,
                );
            }
            let nonzero = row.iter().filter(|&&v| v != 0.0).count();
            let sum_abs: f32 = row.iter().map(|v| v.abs()).sum();
            println!(
                "  Token {}: {}/{} nonzero, sum_abs={:.2}",
                token_id, nonzero, hidden_size, sum_abs
            );
        }
    }

    // 7. Check if norm weights are sane (not all zeros, not all ones)
    println!("\n--- Norm Weight Sanity ---");
    if let Some(out_norm) = core.tensors.get("output_norm.weight") {
        let n = out_norm.shape[0].min(hidden_size);
        let mut vals = vec![0.0f32; n];
        unsafe {
            std::ptr::copy_nonoverlapping(out_norm.data_ptr as *const f32, vals.as_mut_ptr(), n);
        }
        let all_zero = vals.iter().all(|&v| v == 0.0);
        let all_one = vals.iter().all(|&v| v == 1.0);
        let mean: f32 = vals.iter().sum::<f32>() / n as f32;
        let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!(
            "  output_norm: mean={:.4}, min={:.4}, max={:.4}, all_zero={}, all_one={}",
            mean, min, max, all_zero, all_one
        );
    }

    for l in 0..num_layers.min(3) {
        for name_suffix in &["attn_norm.weight", "norm.weight"] {
            let key = format!("blk.{}.{}", l, name_suffix);
            if let Some(t) = core.tensors.get(&key) {
                let n = t.shape[0].min(hidden_size);
                let mut vals = vec![0.0f32; n];
                unsafe {
                    std::ptr::copy_nonoverlapping(t.data_ptr as *const f32, vals.as_mut_ptr(), n);
                }
                let all_zero = vals.iter().all(|&v| v == 0.0);
                let mean: f32 = vals.iter().sum::<f32>() / n as f32;
                println!("  {}: mean={:.4}, all_zero={}", key, mean, all_zero);
            }
        }
    }

    // 8. C-MUD reasoning kernel (research §3, new work)
    println!("\n--- [8] C-MUD Reasoning Kernel (research §3) ---");
    let (cmud_ok, cmud_msg) = forge_llm::mud::cmud::cmud_kernel_selfcheck();
    if cmud_ok {
        println!("  ✅ C-MUD kernel self-check OK ({cmud_msg})");
    } else {
        println!("  ⚠️  C-MUD kernel self-check issues ({cmud_msg}) — opt-in path");
    }

    // 9. Diagnostic summary
    println!("\n--- [9] Diagnostic Summary ---");
    println!("  hidden_size     : {}", hidden_size);
    println!("  num_layers      : {}", num_layers);
    println!("  num_experts     : {}", num_experts);
    println!("  total tensors   : {}", tensor_names.len());
    println!("  null pointers   : {}", null_weight_count);
    println!("  missing tensors : {}", missing.len());
    println!("  cmud_kernel     : {}", if cmud_ok { "OK" } else { "WARN" });
    let verdict = if null_weight_count == 0 && missing.is_empty() {
        "🟢 STRUCTURALLY HEALTHY"
    } else {
        "🔴 STRUCTURAL ISSUES"
    };
    println!("  verdict         : {}", verdict);

    println!("\n=== Diagnostic Complete ===");
    Ok(())
}
