use forge_llm::mud::{MudFile, MudSkill, MudTensor, MudTensorType};
use rand::prelude::*;
use std::collections::HashMap;
use std::fs;

fn pack_ternary(data: &[f32]) -> Vec<u8> {
    let u32_count = data.len().div_ceil(16);
    let mut packed = vec![0u32; u32_count];
    for i in 0..data.len() {
        let bit = if data[i] > 0.5 {
            1u32
        } else if data[i] < -0.5 {
            2u32
        } else {
            0u32
        };
        packed[i / 16] |= bit << ((i % 16) * 2);
    }
    unsafe { std::slice::from_raw_parts(packed.as_ptr() as *const u8, packed.len() * 4) }.to_vec()
}

#[allow(clippy::needless_range_loop)]
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

fn main() -> anyhow::Result<()> {
    println!("\x1b[1;36m🚀 Forge LLM: Blank Model Generator (Jamba Hybrid)\x1b[0m");

    let hidden_size = 256;
    let num_layers = 6;
    let num_heads = 4;
    let num_kv_heads = 1;
    let head_dim = 64;
    let d_state = 16;
    let d_conv = 4;
    let ffn_hidden = 1024;
    let num_experts = 1;
    let output_path = "models/blank_micro.mud";

    let mut global_metadata = HashMap::new();
    global_metadata.insert("arch".to_string(), "jamba-hybrid-v1".to_string());
    global_metadata.insert("hidden_size".to_string(), hidden_size.to_string());
    global_metadata.insert("num_layers".to_string(), num_layers.to_string());
    global_metadata.insert("num_heads".to_string(), num_heads.to_string());
    global_metadata.insert("num_kv_heads".to_string(), num_kv_heads.to_string());
    global_metadata.insert("head_dim".to_string(), head_dim.to_string());
    global_metadata.insert("d_state".to_string(), d_state.to_string());
    global_metadata.insert("d_conv".to_string(), d_conv.to_string());
    global_metadata.insert("ffn_hidden".to_string(), ffn_hidden.to_string());
    global_metadata.insert("num_experts".to_string(), num_experts.to_string());
    global_metadata.insert("top_k".to_string(), "1".to_string());

    let tokenizer_path = "models/qwen2_0.5b/tokenizer.json";
    if let Ok(content) = fs::read_to_string(tokenizer_path) {
        println!("  ✅ Injecting tokenizer from {}", tokenizer_path);
        global_metadata.insert("tokenizer.tokens".to_string(), content);
    }

    let merges_path = "models/qwen2_0.5b/merges.txt";
    if let Ok(content) = fs::read_to_string(merges_path) {
        println!("  ✅ Injecting merges from {}", merges_path);
        global_metadata.insert("tokenizer.merges".to_string(), content);
    }

    let mut tensors = HashMap::new();
    let mut rng = rand::rng();

    println!("  🧱 Initializing Embeddings...");
    let vocab_size = 151643;
    init_ternary_tensor(
        &mut tensors,
        &mut rng,
        "token_embd.weight",
        vocab_size,
        hidden_size,
    );
    init_f32_tensor(&mut tensors, "output_norm.weight", vec![hidden_size], 1.0);
    init_ternary_tensor(
        &mut tensors,
        &mut rng,
        "output.weight",
        vocab_size,
        hidden_size,
    );

    println!("  🧱 Initializing Layers (Jamba 1:5 ratio)...");
    for l in 0..num_layers {
        init_f32_tensor(
            &mut tensors,
            &format!("blk.{}.norm.weight", l),
            vec![hidden_size],
            1.0,
        );

        if l % 6 == 0 {
            init_f32_tensor(
                &mut tensors,
                &format!("blk.{}.attn_norm.weight", l),
                vec![hidden_size],
                1.0,
            );
            init_ternary_tensor(
                &mut tensors,
                &mut rng,
                &format!("blk.{}.attn_q.weight", l),
                hidden_size,
                hidden_size,
            );
            init_ternary_tensor(
                &mut tensors,
                &mut rng,
                &format!("blk.{}.attn_k.weight", l),
                num_kv_heads * head_dim,
                hidden_size,
            );
            init_ternary_tensor(
                &mut tensors,
                &mut rng,
                &format!("blk.{}.attn_v.weight", l),
                num_kv_heads * head_dim,
                hidden_size,
            );
            init_ternary_tensor(
                &mut tensors,
                &mut rng,
                &format!("blk.{}.attn_output.weight", l),
                hidden_size,
                hidden_size,
            );

            init_ternary_tensor(
                &mut tensors,
                &mut rng,
                &format!("blk.{}.expert.0.w1.weight", l),
                ffn_hidden,
                hidden_size,
            );
            init_ternary_tensor(
                &mut tensors,
                &mut rng,
                &format!("blk.{}.expert.0.w2.weight", l),
                hidden_size,
                ffn_hidden,
            );
            init_ternary_tensor(
                &mut tensors,
                &mut rng,
                &format!("blk.{}.expert.0.w3.weight", l),
                ffn_hidden,
                hidden_size,
            );
            init_ternary_tensor(
                &mut tensors,
                &mut rng,
                &format!("blk.{}.gate.weight", l),
                num_experts,
                hidden_size,
            );
        } else {
            init_ternary_tensor(
                &mut tensors,
                &mut rng,
                &format!("blk.{}.ssm_in.weight", l),
                hidden_size * 2,
                hidden_size,
            );
            init_ternary_tensor(
                &mut tensors,
                &mut rng,
                &format!("blk.{}.ssm_out.weight", l),
                hidden_size,
                hidden_size,
            );
            init_ternary_tensor(
                &mut tensors,
                &mut rng,
                &format!("blk.{}.ssm_x.weight", l),
                32 + 2 * d_state,
                hidden_size,
            );
            init_ternary_tensor(
                &mut tensors,
                &mut rng,
                &format!("blk.{}.ssm_dt.weight", l),
                hidden_size,
                32,
            );
            init_f32_tensor(
                &mut tensors,
                &format!("blk.{}.ssm_a", l),
                vec![hidden_size, d_state],
                -1.0,
            );
            init_f32_tensor(
                &mut tensors,
                &format!("blk.{}.ssm_d", l),
                vec![hidden_size],
                1.0,
            );
            init_f32_tensor(
                &mut tensors,
                &format!("blk.{}.ssm_conv1d.weight", l),
                vec![hidden_size, d_conv],
                0.25,
            );
            init_f32_tensor(
                &mut tensors,
                &format!("blk.{}.ssm_conv1d.bias", l),
                vec![hidden_size],
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

    println!("  💾 Saving to {}...", output_path);
    mud_file.save(output_path)?;
    println!("\x1b[1;32m🏁 Success! Blank Jamba Hybrid model created.\x1b[0m");

    Ok(())
}
