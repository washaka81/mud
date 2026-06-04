#![allow(clippy::needless_range_loop)]
use std::collections::HashMap;

fn row_wise_int4_quantize(
    data: &mut [f32],
    hidden: usize,
) -> (Vec<u8>, Vec<f32>, HashMap<String, String>) {
    let n_rows = data.len() / hidden;
    let mut scales_f32 = Vec::with_capacity(n_rows);
    let mut packed_data = Vec::with_capacity(n_rows * (hidden / 2));

    for row_i in 0..n_rows {
        let start = row_i * hidden;
        let row = &data[start..start + hidden];
        
        let absmax = row.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = (absmax / 7.0).max(1e-10);
        scales_f32.push(scale);

        for j in (0..hidden).step_by(2) {
            let v0 = (row[j] / scale).round().clamp(-8.0, 7.0) as i8;
            let v1 = (row[j + 1] / scale).round().clamp(-8.0, 7.0) as i8;

            let nib0 = (v0 + 8) as u8 & 0x0F;
            let nib1 = (v1 + 8) as u8 & 0x0F;
            let b = nib0 | (nib1 << 4);
            packed_data.push(b);
        }
    }

    // In-place quantize data for quality checks later
    for row_i in 0..n_rows {
        let s = scales_f32[row_i];
        let start = row_i * hidden;
        for j in 0..hidden {
            data[start + j] = (data[start + j] / s).round().clamp(-8.0, 7.0) * s;
        }
    }

    let metadata: HashMap<String, String> =
        HashMap::from([("embed_quantized".to_string(), "int4_row_wise".to_string())]);

    (packed_data, scales_f32, metadata)
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: int4_quantizer <input.mud> <output.mud>");
        std::process::exit(1);
    }
    let input = &args[1];
    let output = &args[2];

    println!("🔧 Embedding INT4 Quantizer (Row-wise)");
    println!("  Input:  {}", input);
    println!("  Output: {}", output);

    let mf = forge_llm::mud::MudFile::load(input)?;
    let core = mf.skills.get("core").unwrap();
    let tensor = core
        .tensors
        .get("token_embd.weight")
        .ok_or_else(|| anyhow::anyhow!("token_embd.weight not found"))?;

    let vocab = tensor.shape[0];
    let hidden = tensor.shape[1];
    let total = vocab * hidden;
    println!(
        "  Tensor: {} x {} = {:.1}M params",
        vocab,
        hidden,
        total as f64 / 1_000_000.0
    );

    let mut emb_data = vec![0.0f32; total];
    match tensor.t_type {
        forge_llm::mud::MudTensorType::Float32 => {
            unsafe {
                std::ptr::copy_nonoverlapping(tensor.data_ptr as *const f32, emb_data.as_mut_ptr(), total);
            }
        },
        forge_llm::mud::MudTensorType::Ternary2Bit => {
            unsafe {
                forge_llm::mud::dequantize_ternary_row(
                    tensor.data_ptr as *const u32,
                    &mut emb_data,
                    total
                );
            }
            // If it had a prq_scale, we should apply it
            if let Some(scale_tensor) = core.tensors.get("token_embd.prq_scale") {
                let scales = unsafe { std::slice::from_raw_parts(scale_tensor.data_ptr as *const f32, vocab) };
                for row_i in 0..vocab {
                    let s = scales[row_i];
                    let start = row_i * hidden;
                    for j in 0..hidden {
                        emb_data[start + j] *= s;
                    }
                }
            }
        },
        _ => {
            eprintln!("Unsupported tensor type: {:?}", tensor.t_type);
            std::process::exit(1);
        }
    }

    // Quick quality check (keep original for comparison)
    let orig_data = emb_data.clone();

    let (packed, scales_f32, meta) = row_wise_int4_quantize(&mut emb_data, hidden);

    let n_rows = vocab.min(10000);
    let mut cos_sum = 0.0f32;
    let mut mse_sum = 0.0f32;
    for row_i in 0..n_rows {
        let start = row_i * hidden;
        let mut dot = 0.0f32;
        let mut norm_t = 0.0f32;
        let mut norm_o = 0.0f32;
        let mut se = 0.0f32;
        for j in 0..hidden {
            let r = emb_data[start + j];
            let o = orig_data[start + j];
            dot += o * r;
            norm_t += r * r;
            norm_o += o * o;
            se += (o - r).powi(2);
        }
        let cos = if norm_t > 0.0 && norm_o > 0.0 {
            dot / (norm_t.sqrt() * norm_o.sqrt())
        } else {
            1.0
        };
        cos_sum += cos;
        mse_sum += se / hidden as f32;
    }

    let before_size = total * 4;
    let after_data = packed.len();
    let after_scales = vocab * 4; // f32 scales
    let after_size = after_data + after_scales;

    println!();
    println!("=== QUALITY (first {} rows) ===", n_rows);
    println!("  Cosine sim mean: {:.6}", cos_sum / n_rows as f32);
    println!("  MSE mean:        {:.8}", mse_sum / n_rows as f32);

    println!();
    println!("=== COMPRESSION ===");
    println!(
        "  Before (FP32): {:.1} MB",
        before_size as f64 / 1_048_576.0
    );
    println!(
        "  After (INT4): {:.2} MB",
        after_size as f64 / 1_048_576.0
    );
    println!(
        "    data (4-bit): {:.2} MB",
        after_data as f64 / 1_048_576.0
    );
    println!("    scales (f32): {:.2} KB", after_scales as f64 / 1024.0);
    println!("  Ratio: {:.1}x", before_size as f64 / after_size as f64);
    println!(
        "  Effective bits/param: {:.3}",
        after_size as f64 * 8.0 / total as f64
    );

    println!();
    println!("  Saving to {}...", output);

    let mut new_tensors = HashMap::new();
    for (name, t) in &core.tensors {
        if name == "token_embd.weight" {
            let new_t = forge_llm::mud::MudTensor {
                name: name.clone(),
                t_type: forge_llm::mud::MudTensorType::Int4,
                shape: t.shape.clone(),
                data_ptr: std::ptr::null(),
                offset: 0,
                mmap: None,
                owned_data: Some(packed.clone()),
            };
            new_tensors.insert(name.clone(), new_t);
        } else if name == "token_embd.prq_scale" {
            // Drop it if it existed for ternary
        } else {
            new_tensors.insert(name.clone(), t.clone());
        }
    }

    let scales_bytes: Vec<u8> = scales_f32.iter().flat_map(|s| s.to_le_bytes()).collect();
    new_tensors.insert(
        "token_embd.prq_scale".to_string(), // use same name convention for scales
        forge_llm::mud::MudTensor {
            name: "token_embd.prq_scale".to_string(),
            t_type: forge_llm::mud::MudTensorType::Float32,
            shape: vec![vocab],
            data_ptr: std::ptr::null(),
            offset: 0,
            mmap: None,
            owned_data: Some(scales_bytes),
        },
    );

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
