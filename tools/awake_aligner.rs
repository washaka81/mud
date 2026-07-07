use forge_llm::mud::{MudFile, MudTensorType};
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::env;
use std::fs::File;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        println!("Usage: awake_aligner <model_shard1.safetensors> [shard2...] <model.mud>");
        return Ok(());
    }

    println!("\x1b[1;35m╭────────────────────────────────────────────────────────────╮");
    println!("│ 🌀 MUD AWAKE ALIGNER (AWAKE-01: Global Deep Alignment)     │");
    println!("╰────────────────────────────────────────────────────────────╯\x1b[0m");

    let mud_path = &args[args.len() - 1];
    let sf_paths = &args[1..args.len() - 1];

    println!("  📥 Loading Student (MUD): {}...", mud_path);
    let mut mud_file = MudFile::load(mud_path)?;

    for sf_path in sf_paths {
        println!("  📥 Aligning with Master Shard: {}...", sf_path);
        let sf_file = File::open(sf_path)?;
        let sf_mmap = unsafe { Mmap::map(&sf_file)? };
        let tensors = SafeTensors::deserialize(&sf_mmap)?;

        if let Some(core) = mud_file.skills.get_mut("core") {
            let mud_tensor_names: Vec<String> = core.tensors.keys().cloned().collect();

            for name in mud_tensor_names {
                let sf_name = match name.as_str() {
                    "token_embd.weight" => "model.embed_tokens.weight".to_string(),
                    "output.weight" => "lm_head.weight".to_string(),
                    "output_norm.weight" => "model.norm.weight".to_string(),
                    _ if name.contains("attn_q") => name
                        .replace("blk.", "model.layers.")
                        .replace(".attn_q.weight", ".self_attn.q_proj.weight"),
                    _ if name.contains("attn_k") => name
                        .replace("blk.", "model.layers.")
                        .replace(".attn_k.weight", ".self_attn.k_proj.weight"),
                    _ if name.contains("attn_v") => name
                        .replace("blk.", "model.layers.")
                        .replace(".attn_v.weight", ".self_attn.v_proj.weight"),
                    _ if name.contains("attn_output") => name
                        .replace("blk.", "model.layers.")
                        .replace(".attn_output.weight", ".self_attn.o_proj.weight"),
                    _ if name.contains("expert.0.w1") => name
                        .replace("blk.", "model.layers.")
                        .replace(".expert.0.w1.weight", ".mlp.gate_proj.weight"),
                    _ if name.contains("expert.0.w2") => name
                        .replace("blk.", "model.layers.")
                        .replace(".expert.0.w2.weight", ".mlp.down_proj.weight"),
                    _ if name.contains("expert.0.w3") => name
                        .replace("blk.", "model.layers.")
                        .replace(".expert.0.w3.weight", ".mlp.up_proj.weight"),
                    _ if name.contains("attn_norm") => name
                        .replace("blk.", "model.layers.")
                        .replace(".attn_norm.weight", ".input_layernorm.weight"),
                    _ if name.contains("norm") && name.starts_with("blk") => name
                        .replace("blk.", "model.layers.")
                        .replace(".norm.weight", ".post_attention_layernorm.weight"),
                    _ => continue,
                };

                if let Ok(sf_tensor) = tensors.tensor(&sf_name) {
                    if name == "output.weight" || name == "token_embd.weight" {
                        println!("    💎 Keeping {} in FP32 for Signal Boost...", name);
                        let sf_data = sf_tensor.data();
                        let floats: Vec<f32> = match sf_tensor.dtype() {
                            safetensors::Dtype::BF16 => sf_data
                                .chunks_exact(2)
                                .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
                                .collect(),
                            safetensors::Dtype::F16 => sf_data
                                .chunks_exact(2)
                                .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
                                .collect(),
                            _ => vec![],
                        };
                        let bytes = unsafe {
                            std::slice::from_raw_parts(
                                floats.as_ptr() as *const u8,
                                floats.len() * 4,
                            )
                        }
                        .to_vec();
                        if let Some(t) = core.tensors.get_mut(&name) {
                            t.owned_data = Some(bytes);
                            t.t_type = MudTensorType::Float32;
                        }
                        continue;
                    }

                    println!("    ✨ Matching: {}...", name);

                    let shape = sf_tensor.shape();
                    let rows = if shape.len() > 1 { shape[0] } else { 1 };
                    let cols = if shape.len() > 1 { shape[1] } else { shape[0] };
                    let total_elements = rows * cols;

                    let sf_data = sf_tensor.data();
                    let mut new_scales = Vec::with_capacity(rows);
                    let mut ternary_data = vec![0.0f32; total_elements];

                    for r in 0..rows {
                        let mut row_absmean = 0.0f32;
                        for c in 0..cols {
                            let idx = r * cols + c;
                            let val = match sf_tensor.dtype() {
                                safetensors::Dtype::BF16 => {
                                    let b = [sf_data[idx * 2], sf_data[idx * 2 + 1]];
                                    half::bf16::from_le_bytes(b).to_f32()
                                }
                                safetensors::Dtype::F16 => {
                                    let b = [sf_data[idx * 2], sf_data[idx * 2 + 1]];
                                    half::f16::from_le_bytes(b).to_f32()
                                }
                                safetensors::Dtype::F32 => {
                                    let b = [
                                        sf_data[idx * 4],
                                        sf_data[idx * 4 + 1],
                                        sf_data[idx * 4 + 2],
                                        sf_data[idx * 4 + 3],
                                    ];
                                    f32::from_le_bytes(b)
                                }
                                _ => 0.0,
                            };
                            row_absmean += val.abs();
                            ternary_data[idx] = val;
                        }
                        let s = if row_absmean > 0.0 {
                            ((row_absmean / (cols.max(1) as f32)) * 0.707).max(1e-8)
                        } else {
                            1e-8
                        };
                        new_scales.push(s);

                        for c in 0..cols {
                            let idx = r * cols + c;
                            ternary_data[idx] = (ternary_data[idx] / s).round().clamp(-1.0, 1.0);
                        }
                    }

                    // Update weight
                    let u32_count = total_elements.div_ceil(8);
                    let mut packed = vec![0u32; u32_count];
                    for i in 0..total_elements {
                        let bit = if ternary_data[i] > 0.5 {
                            1u32
                        } else if ternary_data[i] < -0.5 {
                            2u32
                        } else {
                            0u32
                        };
                        packed[i / 8] |= bit << ((i % 8) * 4);
                    }
                    let packed_bytes = unsafe {
                        std::slice::from_raw_parts(packed.as_ptr() as *const u8, packed.len() * 4)
                    }
                    .to_vec();

                    if let Some(t) = core.tensors.get_mut(&name) {
                        t.owned_data = Some(packed_bytes);
                        t.t_type = MudTensorType::Ternary2Bit;
                    }

                    // Update scales
                    let scale_bytes = unsafe {
                        std::slice::from_raw_parts(
                            new_scales.as_ptr() as *const u8,
                            new_scales.len() * 4,
                        )
                    }
                    .to_vec();
                    let scale_name = name.replace(".weight", ".prq_scale");
                    core.tensors.insert(
                        scale_name,
                        forge_llm::mud::MudTensor {
                            name: "scale".to_string(),
                            t_type: MudTensorType::Float32,
                            shape: vec![rows],
                            data_ptr: std::ptr::null(),
                            offset: 0,
                            mmap: None,
                            owned_data: Some(scale_bytes),
                        },
                    );
                }
            }
        }
    }

    println!("  💾 Saving Globally Restored Model: {}...", mud_path);
    mud_file.save(mud_path)?;
    println!("\x1b[1;32m  🎉 THE GREAT AWAKENING: Global Model Restoration Complete! \x1b[0m");

    Ok(())
}
