use std::collections::HashMap;

fn row_wise_absmean_ternarize(data: &mut [f32], hidden: usize) -> (Vec<u8>, Vec<f32>, HashMap<String, String>) {
    let n_rows = data.len() / hidden;
    let mut scales_f32 = Vec::with_capacity(n_rows);
    for row_i in 0..n_rows {
        let start = row_i * hidden;
        let row = &data[start..start + hidden];
        let absmean = row.iter().map(|v| v.abs()).sum::<f32>() / hidden as f32;
        scales_f32.push(absmean.max(1e-10));
    }

    let metadata: HashMap<String, String> = HashMap::from([
        ("embed_ternarized".to_string(), "row_absmean".to_string()),
    ]);

    // Ternarize data in-place
    for row_i in 0..n_rows {
        let s = scales_f32[row_i];
        let start = row_i * hidden;
        for j in 0..hidden {
            let v = data[start + j];
            data[start + j] = (v / s).round().clamp(-1.0, 1.0);
        }
    }

    let packed_scales = Vec::new();
    (packed_scales, scales_f32, metadata)
}

fn pack_ternary_rowwise(data: &[f32], _hidden: usize) -> Vec<u8> {
    let n = data.len();
    let u32_count = n.div_ceil(16);
    let mut packed = vec![0u32; u32_count];
    for i in 0..n {
        let val = data[i];
        let bit = if val > 0.5 { 1u32 } else if val < -0.5 { 2u32 } else { 0u32 };
        let u32_idx = i / 16;
        let shift = (i % 16) * 2;
        packed[u32_idx] |= bit << shift;
    }
    let bytes = unsafe {
        std::slice::from_raw_parts(packed.as_ptr() as *const u8, packed.len() * 4)
    };
    bytes.to_vec()
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: embed_ternarize <input.mud> <output.mud>");
        std::process::exit(1);
    }
    let input = &args[1];
    let output = &args[2];

    println!("🔧 Embedding Ternarizer (Row-wise AbsMean)");
    println!("  Input:  {}", input);
    println!("  Output: {}", output);

    let mf = forge_llm::mud::MudFile::load(input)?;
    let core = mf.skills.get("core").unwrap();
    let tensor = core.tensors.get("token_embd.weight")
        .ok_or_else(|| anyhow::anyhow!("token_embd.weight not found"))?;

    let vocab = tensor.shape[0];
    let hidden = tensor.shape[1];
    let total = vocab * hidden;
    println!("  Tensor: {} x {} = {:.1}M params", vocab, hidden, total as f64 / 1_000_000.0);

    let mut emb_data = vec![0.0f32; total];
    unsafe {
        std::ptr::copy_nonoverlapping(tensor.data_ptr as *const f32, emb_data.as_mut_ptr(), total);
    }

    let (_packed, scales_f32, meta) = row_wise_absmean_ternarize(&mut emb_data, hidden);

    // Quick quality check
    let mut orig_data = vec![0.0f32; total];
    unsafe {
        std::ptr::copy_nonoverlapping(tensor.data_ptr as *const f32, orig_data.as_mut_ptr(), total);
    }

    let n_rows = vocab.min(10000);
    let mut cos_sum = 0.0f32;
    let mut mse_sum = 0.0f32;
    for row_i in 0..n_rows {
        let start = row_i * hidden;
        let s = scales_f32[row_i];
        let mut dot = 0.0f32;
        let mut norm_t = 0.0f32;
        let mut norm_o = 0.0f32;
        let mut se = 0.0f32;
        for j in 0..hidden {
            let q = emb_data[start + j];
            let r = q * s;
            let o = orig_data[start + j];
            dot += o * r;
            norm_t += r * r;
            norm_o += o * o;
            se += (o - r).powi(2);
        }
        let cos = if norm_t > 0.0 && norm_o > 0.0 { dot / (norm_t.sqrt() * norm_o.sqrt()) } else { 1.0 };
        cos_sum += cos;
        mse_sum += se / hidden as f32;
    }

    let before_size = total * 4;
    let after_data = total * 2 / 8;
    let after_scales = vocab * 4; // f32 scales
    let after_size = after_data + after_scales;

    println!();
    println!("=== QUALITY (first {} rows) ===", n_rows);
    println!("  Cosine sim mean: {:.6}", cos_sum / n_rows as f32);
    println!("  MSE mean:        {:.8}", mse_sum / n_rows as f32);

    println!();
    println!("=== COMPRESSION ===");
    println!("  Before (FP32): {:.1} MB", before_size as f64 / 1_048_576.0);
    println!("  After (ternary): {:.2} MB", after_size as f64 / 1_048_576.0);
    println!("    data (2-bit): {:.2} MB", after_data as f64 / 1_048_576.0);
    println!("    scales (f32): {:.2} KB", after_scales as f64 / 1024.0);
    println!("  Ratio: {:.1}x", before_size as f64 / after_size as f64);
    println!("  Effective bits/param: {:.3}", after_size as f64 * 8.0 / total as f64);

    println!();
    println!("  Saving to {}...", output);

    let mut new_tensors = HashMap::new();
    for (name, t) in &core.tensors {
        if name == "token_embd.weight" {
            let packed = pack_ternary_rowwise(&emb_data, hidden);
            let new_t = forge_llm::mud::MudTensor {
                name: name.clone(),
                t_type: forge_llm::mud::MudTensorType::Ternary2Bit,
                shape: t.shape.clone(),
                data_ptr: std::ptr::null(),
                offset: 0,
                mmap: None,
                owned_data: Some(packed),
            };
            new_tensors.insert(name.clone(), new_t);
        } else {
            new_tensors.insert(name.clone(), t.clone());
        }
    }

    // Almacenar escalas per-row como tensor Float32
    let scales_bytes: Vec<u8> = scales_f32.iter().flat_map(|s| s.to_le_bytes()).collect();
    new_tensors.insert("embed_scales".to_string(), forge_llm::mud::MudTensor {
        name: "embed_scales".to_string(),
        t_type: forge_llm::mud::MudTensorType::Float32,
        shape: vec![vocab],
        data_ptr: std::ptr::null(),
        offset: 0,
        mmap: None,
        owned_data: Some(scales_bytes),
    });

    let mut new_skills = HashMap::new();
    let new_skill = forge_llm::mud::MudSkill {
        name: "core".to_string(),
        tensors: new_tensors,
        metadata: core.metadata.clone(),
    };
    new_skills.insert("core".to_string(), new_skill);

    let mut global_meta = mf.global_metadata.clone();
    for (k, v) in &meta {
        global_meta.insert(k.clone(), v.clone());
    }

    let new_mf = forge_llm::mud::MudFile {
        mmap: None,
        skills: new_skills,
        global_metadata: global_meta,
    };

    new_mf.save(output)?;
    println!("  ✅ Done! Saved to {}", output);

    Ok(())
}
