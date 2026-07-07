use forge_llm::mud::{
    slime::{float_to_half_bits, half_to_float_bits, SlimeWorkspace},
    MudFile,
};
use std::env;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    let model_path = if args.len() > 1 {
        &args[1]
    } else {
        "model_trained.mud"
    };

    println!("📡 MUD DOPPLER RADAR (SlimeRegister Ultrasound)");
    println!("Loading Model: {}", model_path);
    let mud = MudFile::load(model_path)?;
    println!("Initializing SlimeWorkspace...");

    let m = &mud.global_metadata;
    let hidden = m
        .get("hidden_size")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(2048) as usize;
    let max_pos = m
        .get("max_position_embeddings")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(2048) as usize;
    let n_heads = m
        .get("num_attention_heads")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(32) as usize;
    let n_kv_heads = m
        .get("num_key_value_heads")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(32) as usize;
    let num_layers = m
        .get("num_hidden_layers")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(30) as usize;
    let head_dim = hidden / n_heads;
    let ffn_mid = m
        .get("intermediate_size")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(5632) as usize;
    let max_emb = 128.0; // Simulated

    let mut ws = SlimeWorkspace::new(
        hidden, max_pos, n_heads, n_kv_heads, head_dim, ffn_mid, num_layers, max_emb,
    );

    // Simulate a random "noisy" input semantic state
    for i in 0..hidden {
        ws.registers[i].matmul_accum = (i % 100) as f32 / 10.0;
        ws.registers[i].jepa_packed = float_to_half_bits(((i * 7) % 50) as f32 / 10.0 - 2.5);
    }

    println!(
        "\n[ INITIATING DOPPLER SWEEP ACROSS {} LAYERS ]\n",
        num_layers
    );

    let width = 64; // Radar width in chars
    let chunk_size = hidden / width;

    for layer in 0..num_layers {
        print!("L{:02} | ", layer);

        // Simulate layer forward pass: Noise reduction & JEPA gating
        let mut row_output = String::new();

        for w in 0..width {
            let start = w * chunk_size;
            let end = start + chunk_size;

            let mut avg_accum = 0.0;
            let mut avg_jepa = 0.0;
            for i in start..end {
                avg_accum += ws.registers[i].matmul_accum.abs();
                avg_jepa += half_to_float_bits(ws.registers[i].jepa_packed);

                // Simulate mathematically: Accumulator decays noise, JEPA focuses it
                let z = half_to_float_bits(ws.registers[i].jepa_packed);
                let gate = 1.0 / (1.0 + (-z).exp());
                ws.registers[i].matmul_accum = ws.registers[i].matmul_accum * 0.9 * gate;
                ws.registers[i].jepa_packed = float_to_half_bits(z * 1.05); // JEPA signal powers up
            }
            avg_accum /= chunk_size as f32;
            avg_jepa /= chunk_size as f32;

            // Character represents intensity (Accumulator)
            let char_repr = if avg_accum < 0.1 {
                ' '
            } else if avg_accum < 0.5 {
                '░'
            } else if avg_accum < 1.0 {
                '▒'
            } else if avg_accum < 2.0 {
                '▓'
            } else {
                '█'
            };

            // Color represents Doppler Phase (JEPA Z-score)
            // Red = negative shift, Blue = positive shift, Green = neutral/focused
            let color = if avg_jepa < -1.0 {
                "\x1b[31m"
            }
            // Red
            else if avg_jepa > 1.0 {
                "\x1b[34m"
            }
            // Blue
            else {
                "\x1b[32m"
            }; // Green

            row_output.push_str(&format!("{}{}\x1b[0m", color, char_repr));
        }
        println!("{} |", row_output);
        std::thread::sleep(std::time::Duration::from_millis(50)); // Radar sweep delay
    }

    println!("\n🎯 DOPPLER FILTER COMPLETE. SIGNAL ISOLATED.");
    Ok(())
}
