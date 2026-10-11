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
            15u32
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
            data_base: 0,
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
            data_base: 0,
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
            data_base: 0,
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
    ffn_hidden: usize,
}

fn main() -> anyhow::Result<()> {
    println!("\x1b[1;36m🔨 MUD Forge — Dense Ternary Transformer from Scratch\x1b[0m");

    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        println!("Usage: mud_forge <profile> <tokenizer_dir> <output.mud> [options]");
        println!("Profiles: pico, nano, micro, base, pro, ultra, custom");
        println!("Options:");
        println!("  --vocab-size N     Set vocabulary size (default: 151643)");
        println!("  --layers N         Override number of layers");
        println!("  --hidden N         Override hidden size");
        println!("  --heads N          Override attention heads");
        println!("  --kv-heads N       Override key-value heads");
        println!("  --head-dim N       Override head dimension");
        println!("  --ffn-hidden N     Override FFN intermediate size");
        println!("Example: mud_forge nano models/qwen2_0.5b models/scratch.mud --vocab-size 32000");
        return Ok(());
    }

    let profile_name = &args[1];
    let mut tokenizer_dir_idx = 2;
    let output_path = &args[3];

    let mut profile = match profile_name.as_str() {
        "pico" => ForgeProfile {
            hidden_size: 128,
            num_layers: 6,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 64,
            ffn_hidden: 512,
        },
        "nano" => ForgeProfile {
            hidden_size: 256,
            num_layers: 12,
            num_heads: 4,
            num_kv_heads: 1,
            head_dim: 64,
            ffn_hidden: 1024,
        },
        "micro" => ForgeProfile {
            hidden_size: 512,
            num_layers: 24,
            num_heads: 8,
            num_kv_heads: 2,
            head_dim: 64,
            ffn_hidden: 2048,
        },
        "base" => ForgeProfile {
            hidden_size: 1024,
            num_layers: 24,
            num_heads: 16,
            num_kv_heads: 4,
            head_dim: 64,
            ffn_hidden: 4096,
        },
        "pro" => ForgeProfile {
            hidden_size: 2048,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 64,
            ffn_hidden: 8192,
        },
        "ultra" => ForgeProfile {
            hidden_size: 4096,
            num_layers: 40,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            ffn_hidden: 14336,
        },
        "custom" => ForgeProfile {
            hidden_size: 512,
            num_layers: 16,
            num_heads: 8,
            num_kv_heads: 2,
            head_dim: 64,
            ffn_hidden: 2048,
        },
        _ => {
            println!(
                "❌ Unknown profile: {}. Use: pico, nano, micro, base, pro, ultra, custom",
                profile_name
            );
            return Ok(());
        }
    };

    let mut vocab_size: usize = 151643;

    let mut i = 4;
    while i < args.len() {
        match args[i].as_str() {
            "--tokenizer-dir" => {
                tokenizer_dir_idx = i + 1;
                i += 2;
            }
            "--vocab-size" => {
                vocab_size = args[i + 1].parse()?;
                i += 2;
            }
            "--layers" => {
                profile.num_layers = args[i + 1].parse()?;
                i += 2;
            }
            "--hidden" => {
                profile.hidden_size = args[i + 1].parse()?;
                i += 2;
            }
            "--heads" => {
                profile.num_heads = args[i + 1].parse()?;
                i += 2;
            }
            "--kv-heads" => {
                profile.num_kv_heads = args[i + 1].parse()?;
                i += 2;
            }
            "--head-dim" => {
                profile.head_dim = args[i + 1].parse()?;
                i += 2;
            }
            "--ffn-hidden" => {
                profile.ffn_hidden = args[i + 1].parse()?;
                i += 2;
            }
            _ => {
                println!("❌ Unknown option: {}", args[i]);
                return Ok(());
            }
        }
    }

    println!("  📐 Profile: {} (arch=dense-transformer)", profile_name.to_uppercase());
    println!("     Layers: {} | Hidden: {} | Heads: {} | KV: {} | FFN: {} | Vocab: {}",
        profile.num_layers, profile.hidden_size, profile.num_heads,
        profile.num_kv_heads, profile.ffn_hidden, vocab_size);

    let mut global_metadata = HashMap::new();
    global_metadata.insert("hidden_size".to_string(), profile.hidden_size.to_string());
    global_metadata.insert("num_hidden_layers".to_string(), profile.num_layers.to_string());
    global_metadata.insert("num_layers".to_string(), profile.num_layers.to_string());
    global_metadata.insert("num_experts".to_string(), "1".to_string());
    global_metadata.insert("num_attention_heads".to_string(), profile.num_heads.to_string());
    global_metadata.insert("num_key_value_heads".to_string(), profile.num_kv_heads.to_string());
    global_metadata.insert("intermediate_size".to_string(), profile.ffn_hidden.to_string());
    global_metadata.insert("vocab_size".to_string(), vocab_size.to_string());
    global_metadata.insert("max_position_embeddings".to_string(), "4096".to_string());
    global_metadata.insert("rms_norm_eps".to_string(), "1e-5".to_string());
    global_metadata.insert("rope_theta".to_string(), "10000.0".to_string());
    global_metadata.insert("arch".to_string(), "transformer".to_string());

    // 2026-08-03 self-describing layout metadata — source of truth for the
    // integrity gate + CLI health + trainer. Lets them resolve layout WITHOUT
    // inferring it from tensor-name heuristics (universal agnosticism). Readers
    // fall back to `moe_load::dense_ffn_names_for_train` when a key is absent
    // (backwards-compat with older .mud files that predate this metadata).
    global_metadata.insert("mud:layout".to_string(), "dense".to_string());
    global_metadata.insert("mud:ffn_naming".to_string(), "ffn_up/gate/down".to_string());
    global_metadata.insert("mud:weight_type".to_string(), "ternary2bit".to_string());
    global_metadata.insert("mud:quant".to_string(), "elut4bit_prq".to_string());
    global_metadata.insert("mud:expert_count".to_string(), "1".to_string());

    match load_tokenizer_from_dir(&args[tokenizer_dir_idx]) {
        Ok((tokens, merges_opt)) => {
            println!("  ✅ Loaded tokenizer from {}", &args[tokenizer_dir_idx]);
            global_metadata.insert("tokenizer.tokens".to_string(), tokens);
            if let Some(m) = merges_opt {
                global_metadata.insert("tokenizer.merges".to_string(), m);
            }
        }
        Err(e) => {
            println!("  ⚠️  Failed to load tokenizer: {} (model will lack tokenizer)", e);
        }
    }

    let mut tensors = HashMap::new();
    let mut rng = rand::rng();

    println!("  🧱 Embeddings (vocab={}) + output_norm...", vocab_size);
    init_ternary_tensor(&mut tensors, &mut rng, "token_embd.weight", vocab_size, profile.hidden_size);
    init_ternary_tensor(&mut tensors, &mut rng, "output.weight", vocab_size, profile.hidden_size);
    init_f32_tensor(&mut tensors, "output_norm.weight", vec![profile.hidden_size], 1.0);

    let hidden = profile.hidden_size;
    let kv_dim = profile.num_kv_heads * profile.head_dim;

    println!("  🧱 Initializing {} dense transformer layers...", profile.num_layers);
    for l in 0..profile.num_layers {
        // Norms
        init_f32_tensor(&mut tensors, &format!("blk.{}.attn_norm.weight", l), vec![hidden], 1.0);
        init_f32_tensor(&mut tensors, &format!("blk.{}.ffn_norm.weight", l), vec![hidden], 1.0);

        // Attention QKV + O (every layer, dense transformer)
        init_ternary_tensor(&mut tensors, &mut rng, &format!("blk.{}.attn_q.weight", l), hidden, hidden);
        init_ternary_tensor(&mut tensors, &mut rng, &format!("blk.{}.attn_k.weight", l), kv_dim, hidden);
        init_ternary_tensor(&mut tensors, &mut rng, &format!("blk.{}.attn_v.weight", l), kv_dim, hidden);
        init_ternary_tensor(&mut tensors, &mut rng, &format!("blk.{}.attn_output.weight", l), hidden, hidden);

        // Dense FFN (SwiGLU: up/gate/down)
        init_ternary_tensor(&mut tensors, &mut rng, &format!("blk.{}.ffn_up.weight", l), profile.ffn_hidden, hidden);
        init_ternary_tensor(&mut tensors, &mut rng, &format!("blk.{}.ffn_gate.weight", l), profile.ffn_hidden, hidden);
        init_ternary_tensor(&mut tensors, &mut rng, &format!("blk.{}.ffn_down.weight", l), hidden, profile.ffn_hidden);

        // mHC scales (JEPA homeostatic control): alpha=0.85, beta=0.15, radius=√hidden
        init_f32_tensor(&mut tensors, &format!("blk.{}.mhc_alpha", l), vec![hidden], 0.85);
        init_f32_tensor(&mut tensors, &format!("blk.{}.mhc_beta", l), vec![hidden], 0.15);
        // FIX (2026-08-04): mHC radius was `sqrt(hidden) * 8.0` (≈90.5 for
        // hidden=128) — 10.8× too large. The `8.0` multiplier originated from
        // a BF16-embedding model (max_emb≈8.0, per 2026-07-01 report) but
        // forge-initialized ternary tensors have max_emb≈0.74 (PRQ scale≈0.74).
        // The inflated radius allowed activations to explode (cog≈564, varh≈7.9)
        // producing loss above random (7.09 vs 6.24) with zero weight movement
        // (ΔW=0) because gradients were too small to overcome the STE deadzone.
        // Correct: 1.5·√hidden ≈ 16.97 (matches slime.rs workspace fallback).
        let radius = ((hidden as f32).sqrt() * 1.5).max(5.0);
        init_f32_tensor(&mut tensors, &format!("blk.{}.mhc_radius", l), vec![1], radius);
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
        path: Some(output_path.clone()),
    };

    println!("  💾 Saving to {}...", output_path);
    mud_file.save(output_path)?;
    println!("\x1b[1;32m🏁  Done! {} layers, {} params (ternary, packed ELUT)\x1b[0m",
        profile.num_layers,
        estimate_params(&profile, vocab_size));
    println!("  Next: cargo run --release --bin run_trainer -- {} --epochs 5", output_path);

    Ok(())
}

fn load_tokenizer_from_dir(dir: &str) -> Result<(String, Option<String>), Box<dyn std::error::Error>> {
    use std::collections::BTreeMap;

    let json_path = format!("{}/tokenizer.json", dir);
    let content = fs::read_to_string(&json_path)?;

    if content.trim_start().starts_with('{') {
        let v: serde_json::Value = serde_json::from_str(&content)?;
        let model = v.get("model").ok_or("no 'model' in tokenizer.json")?;
        let vocab_map = model.get("vocab").and_then(|m| m.as_object())
            .ok_or("no 'model.vocab' in tokenizer.json")?;
        let mut by_id: BTreeMap<u32, &str> = BTreeMap::new();
        for (token, id_val) in vocab_map {
            let id = id_val.as_u64().ok_or("non-integer vocab id")? as u32;
            by_id.insert(id, token.as_str());
        }
        let tokens_str: String = by_id.values().map(|t| format!("{}\n", t)).collect();
        let merges_str = model.get("merges").and_then(|m| m.as_array()).map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .map(|s| format!("{}\n", s))
                .collect::<String>()
        });
        Ok((tokens_str, merges_str))
    } else {
        let merges_path = format!("{}/merges.txt", dir);
        let merges = fs::read_to_string(&merges_path).ok();
        Ok((content, merges))
    }
}

fn estimate_params(p: &ForgeProfile, vocab: usize) -> String {
    let emb = vocab * p.hidden_size;
    let out = vocab * p.hidden_size;
    let per_layer_q = p.hidden_size * p.hidden_size;
    let per_layer_k = (p.num_kv_heads * p.head_dim) * p.hidden_size;
    let per_layer_v = per_layer_k;
    let per_layer_o = p.hidden_size * p.hidden_size;
    let per_layer_up = p.ffn_hidden * p.hidden_size;
    let per_layer_gate = p.ffn_hidden * p.hidden_size;
    let per_layer_down = p.hidden_size * p.ffn_hidden;
    let norms = p.hidden_size * 2;
    let per_layer = per_layer_q + per_layer_k + per_layer_v + per_layer_o
        + per_layer_up + per_layer_gate + per_layer_down + norms;
    let total = emb + out + per_layer * p.num_layers;
    let mb = (total as f64 * 2.0) / (8.0 * 1024.0 * 1024.0); // 2 bits per param (ternary)
    format!("{}M (~{:.1}MiB ELUT)", total / 1_000_000, mb)
}
