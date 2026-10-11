// Temporary fixture generator: writes a small, HEALTHY SmolLM2-format model
// (safetensors + config.json + tokenizer.json) so we can validate that the
// universal_converter produces a healthy .mud from a healthy source.
//
// This isolates the converter from the "collapsed .mud" bug: if the converter
// yields non-zero norms/weights from this fixture, the collapsed smollm2.mud
// came from a collapsed source — and the fix is input-health validation.
use std::collections::BTreeMap;
use std::fs;

fn main() -> anyhow::Result<()> {
    let out_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "fixtures/smollm2_mini".to_string());
    std::fs::create_dir_all(&out_dir)?;

    let hidden = 64usize;
    let ffn = 128usize;
    let layers = 2usize;
    let vocab = 128usize;
    let heads = 4usize;
    let kv = 4usize;
    let _ = (heads, kv);

    let mut rng = SplitMix64::new(0x1234_5678_9abc_def1);

    // Gaussian-ish small weights so the model is clearly non-collapsed.
    let rand_weight = |rng: &mut SplitMix64, rows: usize, cols: usize| -> Vec<f32> {
        let mut v = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            // Box-Muller-lite: sum of uniforms ~ small normal
            let a = rng.next() as f32 / u64::MAX as f32 - 0.5;
            let b = rng.next() as f32 / u64::MAX as f32 - 0.5;
            let c = rng.next() as f32 / u64::MAX as f32 - 0.5;
            let s = (a + b + c) * 0.2;
            v.push(s);
        }
        v
    };
    // RMSNorm weights ~1.0 with small variation (never zero).
    let rand_norm = |rng: &mut SplitMix64, n: usize| -> Vec<f32> {
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            let d = rng.next() as f32 / u64::MAX as f32 - 0.5;
            v.push(1.0 + d * 0.1);
        }
        v
    };

    let mut tensors: BTreeMap<String, Vec<f32>> = BTreeMap::new();

    tensors.insert(
        "model.embed_tokens.weight".into(),
        rand_weight(&mut rng, vocab, hidden),
    );
    tensors.insert(
        "lm_head.weight".into(),
        rand_weight(&mut rng, vocab, hidden),
    );
    tensors.insert("model.norm.weight".into(), rand_norm(&mut rng, hidden));

    for l in 0..layers {
        let p = format!("model.layers.{}", l);
        tensors.insert(
            format!("{}.input_layernorm.weight", p),
            rand_norm(&mut rng, hidden),
        );
        tensors.insert(
            format!("{}.post_attention_layernorm.weight", p),
            rand_norm(&mut rng, hidden),
        );
        tensors.insert(
            format!("{}.self_attn.q_proj.weight", p),
            rand_weight(&mut rng, hidden, hidden),
        );
        tensors.insert(
            format!("{}.self_attn.k_proj.weight", p),
            rand_weight(&mut rng, kv * (hidden / heads), hidden),
        );
        tensors.insert(
            format!("{}.self_attn.v_proj.weight", p),
            rand_weight(&mut rng, kv * (hidden / heads), hidden),
        );
        tensors.insert(
            format!("{}.self_attn.o_proj.weight", p),
            rand_weight(&mut rng, hidden, hidden),
        );
        tensors.insert(
            format!("{}.mlp.gate_proj.weight", p),
            rand_weight(&mut rng, ffn, hidden),
        );
        tensors.insert(
            format!("{}.mlp.up_proj.weight", p),
            rand_weight(&mut rng, ffn, hidden),
        );
        tensors.insert(
            format!("{}.mlp.down_proj.weight", p),
            rand_weight(&mut rng, hidden, ffn),
        );
    }

    // Serialize safetensors
    let mut st_tensors: BTreeMap<String, safetensors::tensor::TensorView<'_>> = BTreeMap::new();
    let mut owned: Vec<Vec<f32>> = Vec::new();
    for (name, data) in tensors.iter() {
        let shape: Vec<usize> = if name.contains("embed_tokens") || name.contains("lm_head") {
            vec![vocab, hidden]
        } else if name.contains("self_attn.q_proj") || name.contains("self_attn.o_proj") {
            vec![hidden, hidden]
        } else if name.contains("self_attn.k_proj") || name.contains("self_attn.v_proj") {
            vec![kv * (hidden / heads), hidden]
        } else if name.contains("gate_proj") || name.contains("up_proj") {
            vec![ffn, hidden]
        } else if name.contains("down_proj") {
            vec![hidden, ffn]
        } else {
            vec![hidden]
        };
        owned.push(data.clone());
        let buf = unsafe {
            std::slice::from_raw_parts(
                owned.last().unwrap().as_ptr() as *const u8,
                owned.last().unwrap().len() * 4,
            )
        };
        st_tensors.insert(
            name.clone(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape, buf)?,
        );
    }
    let st_bytes = safetensors::serialize(&st_tensors, &None)?;
    fs::write(format!("{}/model.safetensors", out_dir), st_bytes)?;

    let config = serde_json::json!({
        "architectures": ["SmolLM2ForCausalLM"],
        "model_type": "smollm2",
        "num_hidden_layers": layers,
        "hidden_size": hidden,
        "intermediate_size": ffn,
        "num_attention_heads": heads,
        "num_key_value_heads": kv,
        "vocab_size": vocab,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 2048,
        "tie_word_embeddings": true,
        "head_dim": hidden / heads
    });
    fs::write(
        format!("{}/config.json", out_dir),
        serde_json::to_string_pretty(&config)?,
    )?;

    // Minimal tokenizer: vocab tokens + merges empty
    let mut tokens = Vec::new();
    for i in 0..vocab {
        tokens.push(format!("<t{}>", i));
    }
    fs::write(
        format!("{}/tokenizer.json", out_dir),
        serde_json::to_string_pretty(&serde_json::json!({
            "model": { "vocab": tokens.iter().enumerate().map(|(i,t)| (t.clone(), i)).collect::<std::collections::HashMap<_,_>>() },
            "merges": []
        }))?,
    )?;

    println!("Wrote healthy fixture to {}", out_dir);
    Ok(())
}

struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}
