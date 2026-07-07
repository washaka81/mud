#![allow(clippy::needless_range_loop)]
use forge_llm::mud::{MudFile, MudSkill, MudTensor, MudTensorType};
use rand::prelude::*;
use std::collections::HashMap;
use std::env;
use std::fs;

fn pack_ternary(data: &[f32]) -> Vec<u8> {
    let u32_count = data.len().div_ceil(8);
    let mut packed = vec![0u32; u32_count];
    for i in 0..data.len() {
        let bit = if data[i] > 0.5 {
            1u32
        } else if data[i] < -0.5 {
            2u32
        } else {
            0u32
        };
        packed[i / 8] |= bit << ((i % 8) * 4);
    }
    unsafe { std::slice::from_raw_parts(packed.as_ptr() as *const u8, packed.len() * 4) }.to_vec()
}

fn init_ternary_tensor(
    tensors: &mut HashMap<String, MudTensor>,
    rng: &mut ThreadRng,
    name: &str,
    rows: usize,
    cols: usize,
) {
    let total = rows * cols;
    let mut ternary_data = vec![0.0f32; total];
    let mut scales = vec![0.0f32; rows];

    for r in 0..rows {
        let start = r * cols;
        let mut row_abs_sum = 0.0f32;
        for j in 0..cols {
            let rv: f32 = rng.random_range(0.0..1.0);
            let val = if rv < 0.37 {
                1.0
            } else if rv < 0.74 {
                -1.0
            } else {
                0.0
            };
            ternary_data[start + j] = val;
            row_abs_sum += val.abs();
        }
        let s = row_abs_sum / cols as f32;
        scales[r] = s.max(0.01);
    }

    tensors.insert(
        name.to_string(),
        MudTensor {
            name: name.to_string(),
            t_type: MudTensorType::Ternary2Bit,
            shape: vec![rows, cols],
            data_ptr: std::ptr::null(),
            offset: 0,
            mmap: None,
            owned_data: Some(pack_ternary(&ternary_data)),
        },
    );

    let scale_name = format!("{}.prq_scale", name.replace(".weight", ""));
    tensors.insert(
        scale_name.clone(),
        MudTensor {
            name: scale_name,
            t_type: MudTensorType::Float32,
            shape: vec![rows],
            data_ptr: std::ptr::null(),
            offset: 0,
            mmap: None,
            owned_data: Some(scales.iter().flat_map(|s| s.to_le_bytes()).collect()),
        },
    );
}

fn init_f32_tensor(
    tensors: &mut HashMap<String, MudTensor>,
    name: &str,
    shape: Vec<usize>,
    val: f32,
) {
    let total: usize = shape.iter().product();
    let data = vec![val; total];
    tensors.insert(
        name.to_string(),
        MudTensor {
            name: name.to_string(),
            t_type: MudTensorType::Float32,
            shape,
            data_ptr: std::ptr::null(),
            offset: 0,
            mmap: None,
            owned_data: Some(data.iter().flat_map(|v| v.to_le_bytes()).collect()),
        },
    );
}

struct ForgeProfile {
    hidden_size: usize,
    num_layers: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    d_state: usize,
    d_conv: usize,
    ffn_hidden: usize,
    num_experts: usize,
    attn_layer_offset: usize, // e.g. 6 means 1 out of 6 layers is attention
}

fn main() -> anyhow::Result<()> {
    println!("\x1b[1;36m🔨 MUD Forge CLI: Automated Model Builder\x1b[0m");

    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        println!("Usage: mud_forge <profile> <tokenizer_dir> <output.mud> [custom parameters...]");
        println!("Profiles: pico, nano, micro, base, pro, ultra, moe_base, mamba_pure, custom");
        println!("Example: mud_forge custom models/qwen2_0.5b models/custom.mud --layers 16 --hidden 512 --experts 4");
        return Ok(());
    }

    let profile_name = &args[1];
    let tokenizer_dir = &args[2];
    let output_path = &args[3];

    let profile = match profile_name.as_str() {
        "pico" => ForgeProfile {
            hidden_size: 128,
            num_layers: 6,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 64,
            d_state: 16,
            d_conv: 4,
            ffn_hidden: 512,
            num_experts: 1,
            attn_layer_offset: 6,
        },
        "nano" => ForgeProfile {
            hidden_size: 256,
            num_layers: 12,
            num_heads: 4,
            num_kv_heads: 1,
            head_dim: 64,
            d_state: 16,
            d_conv: 4,
            ffn_hidden: 1024,
            num_experts: 1,
            attn_layer_offset: 6,
        },
        "micro" => ForgeProfile {
            hidden_size: 512,
            num_layers: 24,
            num_heads: 8,
            num_kv_heads: 2,
            head_dim: 64,
            d_state: 16,
            d_conv: 4,
            ffn_hidden: 2048,
            num_experts: 1,
            attn_layer_offset: 6,
        },
        "base" => ForgeProfile {
            hidden_size: 1024,
            num_layers: 24,
            num_heads: 16,
            num_kv_heads: 4,
            head_dim: 64,
            d_state: 16,
            d_conv: 4,
            ffn_hidden: 4096,
            num_experts: 1,
            attn_layer_offset: 6,
        },
        "pro" => ForgeProfile {
            hidden_size: 2048,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 64,
            d_state: 16,
            d_conv: 4,
            ffn_hidden: 8192,
            num_experts: 1,
            attn_layer_offset: 8,
        },
        "ultra" => ForgeProfile {
            hidden_size: 4096,
            num_layers: 40,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            d_state: 16,
            d_conv: 4,
            ffn_hidden: 14336,
            num_experts: 1,
            attn_layer_offset: 8,
        },
        "moe_base" => ForgeProfile {
            hidden_size: 1024,
            num_layers: 24,
            num_heads: 16,
            num_kv_heads: 4,
            head_dim: 64,
            d_state: 16,
            d_conv: 4,
            ffn_hidden: 2048,
            num_experts: 8,
            attn_layer_offset: 6,
        },
        "mamba_pure" => ForgeProfile {
            hidden_size: 1024,
            num_layers: 24,
            num_heads: 16,
            num_kv_heads: 4,
            head_dim: 64,
            d_state: 16,
            d_conv: 4,
            ffn_hidden: 4096,
            num_experts: 1,
            attn_layer_offset: 1000,
        },
        "custom" => {
            let mut p = ForgeProfile {
                hidden_size: 512,
                num_layers: 16,
                num_heads: 8,
                num_kv_heads: 2,
                head_dim: 64,
                d_state: 16,
                d_conv: 4,
                ffn_hidden: 2048,
                num_experts: 1,
                attn_layer_offset: 6,
            };
            let mut i = 4;
            while i < args.len() {
                match args[i].as_str() {
                    "--layers" => {
                        p.num_layers = args[i + 1].parse()?;
                        i += 2;
                    }
                    "--hidden" => {
                        p.hidden_size = args[i + 1].parse()?;
                        i += 2;
                    }
                    "--heads" => {
                        p.num_heads = args[i + 1].parse()?;
                        i += 2;
                    }
                    "--kv-heads" => {
                        p.num_kv_heads = args[i + 1].parse()?;
                        i += 2;
                    }
                    "--head-dim" => {
                        p.head_dim = args[i + 1].parse()?;
                        i += 2;
                    }
                    "--d-state" => {
                        p.d_state = args[i + 1].parse()?;
                        i += 2;
                    }
                    "--d-conv" => {
                        p.d_conv = args[i + 1].parse()?;
                        i += 2;
                    }
                    "--ffn-hidden" => {
                        p.ffn_hidden = args[i + 1].parse()?;
                        i += 2;
                    }
                    "--experts" => {
                        p.num_experts = args[i + 1].parse()?;
                        i += 2;
                    }
                    "--attn-offset" => {
                        p.attn_layer_offset = args[i + 1].parse()?;
                        i += 2;
                    }
                    _ => {
                        println!("❌ Unknown argument: {}", args[i]);
                        return Ok(());
                    }
                }
            }
            p
        }
        _ => {
            println!("❌ Unknown profile: {}. Use: pico, nano, micro, base, pro, ultra, moe_base, mamba_pure, custom", profile_name);
            return Ok(());
        }
    };

    println!("  📐 Selected Profile: {}", profile_name.to_uppercase());
    println!(
        "     Layers: {} | Hidden: {} | Attention Heads: {}",
        profile.num_layers, profile.hidden_size, profile.num_heads
    );

    let mut global_metadata = HashMap::new();
    global_metadata.insert("max_position_embeddings".to_string(), "4096".to_string());
    global_metadata.insert("rms_norm_eps".to_string(), "1e-5".to_string());
    global_metadata.insert("rope_theta".to_string(), "10000.0".to_string());
    global_metadata.insert("model.type".to_string(), "jamba_hybrid".to_string());
    global_metadata.insert("arch".to_string(), "jamba-hybrid-v1".to_string());
    global_metadata.insert("hidden_size".to_string(), profile.hidden_size.to_string());
    global_metadata.insert("num_layers".to_string(), profile.num_layers.to_string());
    global_metadata.insert("num_heads".to_string(), profile.num_heads.to_string());
    global_metadata.insert("num_kv_heads".to_string(), profile.num_kv_heads.to_string());
    global_metadata.insert("head_dim".to_string(), profile.head_dim.to_string());
    global_metadata.insert("d_state".to_string(), profile.d_state.to_string());
    global_metadata.insert("d_conv".to_string(), profile.d_conv.to_string());
    global_metadata.insert("ffn_hidden".to_string(), profile.ffn_hidden.to_string());
    global_metadata.insert("num_experts".to_string(), profile.num_experts.to_string());
    global_metadata.insert("top_k".to_string(), "1".to_string());

    let tokenizer_path = format!("{}/tokenizer.json", tokenizer_dir);
    if let Ok(content) = fs::read_to_string(&tokenizer_path) {
        println!("  ✅ Injecting tokenizer from {}", tokenizer_path);
        global_metadata.insert("tokenizer.tokens".to_string(), content);
    } else {
        println!(
            "  ⚠️ Warning: tokenizer.json not found in {}",
            tokenizer_dir
        );
    }

    let merges_path = format!("{}/merges.txt", tokenizer_dir);
    if let Ok(content) = fs::read_to_string(&merges_path) {
        println!("  ✅ Injecting merges from {}", merges_path);
        global_metadata.insert("tokenizer.merges".to_string(), content);
    }

    let mut tensors = HashMap::new();
    let mut rng = rand::rng();

    // Default to Qwen/Llama size if we don't parse it
    let vocab_size = 151643;
    println!("  🧱 Initializing Embeddings (Vocab: {})...", vocab_size);

    init_ternary_tensor(
        &mut tensors,
        &mut rng,
        "token_embd.weight",
        vocab_size,
        profile.hidden_size,
    );
    init_f32_tensor(
        &mut tensors,
        "output_norm.weight",
        vec![profile.hidden_size],
        1.0,
    );
    init_ternary_tensor(
        &mut tensors,
        &mut rng,
        "output.weight",
        vocab_size,
        profile.hidden_size,
    );

    println!(
        "  🧱 Initializing Jamba Hybrid Layers (1:{} ratio)...",
        profile.attn_layer_offset - 1
    );
    for l in 0..profile.num_layers {
        init_f32_tensor(
            &mut tensors,
            &format!("blk.{}.norm.weight", l),
            vec![profile.hidden_size],
            1.0,
        );

        if l % profile.attn_layer_offset == 0 {
            // Attention + MoE
            init_f32_tensor(
                &mut tensors,
                &format!("blk.{}.attn_norm.weight", l),
                vec![profile.hidden_size],
                1.0,
            );
            init_ternary_tensor(
                &mut tensors,
                &mut rng,
                &format!("blk.{}.attn_q.weight", l),
                profile.hidden_size,
                profile.hidden_size,
            );
            init_ternary_tensor(
                &mut tensors,
                &mut rng,
                &format!("blk.{}.attn_k.weight", l),
                profile.num_kv_heads * profile.head_dim,
                profile.hidden_size,
            );
            init_ternary_tensor(
                &mut tensors,
                &mut rng,
                &format!("blk.{}.attn_v.weight", l),
                profile.num_kv_heads * profile.head_dim,
                profile.hidden_size,
            );
            init_ternary_tensor(
                &mut tensors,
                &mut rng,
                &format!("blk.{}.attn_output.weight", l),
                profile.hidden_size,
                profile.hidden_size,
            );

            for e in 0..profile.num_experts {
                init_ternary_tensor(
                    &mut tensors,
                    &mut rng,
                    &format!("blk.{}.expert.{}.w1.weight", l, e),
                    profile.ffn_hidden,
                    profile.hidden_size,
                );
                init_ternary_tensor(
                    &mut tensors,
                    &mut rng,
                    &format!("blk.{}.expert.{}.w2.weight", l, e),
                    profile.hidden_size,
                    profile.ffn_hidden,
                );
                init_ternary_tensor(
                    &mut tensors,
                    &mut rng,
                    &format!("blk.{}.expert.{}.w3.weight", l, e),
                    profile.ffn_hidden,
                    profile.hidden_size,
                );
            }
            init_f32_tensor(
                &mut tensors,
                &format!("blk.{}.gate.weight", l),
                vec![profile.num_experts, profile.hidden_size],
                1.0,
            );
        } else {
            // Mamba / SSM
            init_ternary_tensor(
                &mut tensors,
                &mut rng,
                &format!("blk.{}.ssm_in.weight", l),
                profile.hidden_size * 2,
                profile.hidden_size,
            );
            init_f32_tensor(
                &mut tensors,
                &format!("blk.{}.ssm_in.bias", l),
                vec![profile.hidden_size * 2],
                0.0,
            );
            init_ternary_tensor(
                &mut tensors,
                &mut rng,
                &format!("blk.{}.ssm_out.weight", l),
                profile.hidden_size,
                profile.hidden_size,
            );
            init_f32_tensor(
                &mut tensors,
                &format!("blk.{}.ssm_out.bias", l),
                vec![profile.hidden_size],
                0.0,
            );
            init_ternary_tensor(
                &mut tensors,
                &mut rng,
                &format!("blk.{}.ssm_x.weight", l),
                32 + 2 * profile.d_state,
                profile.hidden_size,
            );
            init_ternary_tensor(
                &mut tensors,
                &mut rng,
                &format!("blk.{}.ssm_dt.weight", l),
                profile.hidden_size,
                32,
            );
            init_f32_tensor(
                &mut tensors,
                &format!("blk.{}.ssm_dt.bias", l),
                vec![profile.hidden_size],
                0.0,
            );
            init_f32_tensor(
                &mut tensors,
                &format!("blk.{}.ssm_a", l),
                vec![profile.hidden_size, profile.d_state],
                -1.0,
            );
            init_f32_tensor(
                &mut tensors,
                &format!("blk.{}.ssm_d", l),
                vec![profile.hidden_size],
                1.0,
            );
            init_f32_tensor(
                &mut tensors,
                &format!("blk.{}.ssm_conv1d.weight", l),
                vec![profile.hidden_size, profile.d_conv],
                0.25,
            );
            init_f32_tensor(
                &mut tensors,
                &format!("blk.{}.ssm_conv1d.bias", l),
                vec![profile.hidden_size],
                0.0,
            );
        }
    }

    let mud_skill = MudSkill {
        name: "core".to_string(),
        tensors,
        metadata: HashMap::new(),
    };

    let mud_file = MudFile {
        mmap: None,
        skills: HashMap::from([("core".to_string(), mud_skill)]),
        global_metadata,
    };

    println!(
        "  💾 Compiling ternary manifolds and saving to {}...",
        output_path
    );
    mud_file.save(output_path)?;
    println!("\x1b[1;32m🏁 Success! Model forged securely.\x1b[0m");

    Ok(())
}
