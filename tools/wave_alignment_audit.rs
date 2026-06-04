use forge_llm::mud::{dequantize_ternary_row, MudFile};
use safetensors::SafeTensors;
use std::env;
use std::fs::File;
use std::io::Read;

fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut norm1 = 0.0;
    let mut norm2 = 0.0;
    for (a, b) in v1.iter().zip(v2.iter()) {
        dot += a * b;
        norm1 += a * a;
        norm2 += b * b;
    }
    if norm1 == 0.0 || norm2 == 0.0 {
        return 0.0;
    }
    dot / (norm1.sqrt() * norm2.sqrt())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        println!("Usage: wave_alignment_audit <model.safetensors> <model.mud>");
        return Ok(());
    }

    println!("==================================================");
    println!("🌊 HOLOGRAPHIC WAVE ALIGNMENT & SIMILARITY AUDIT");
    println!("==================================================");

    // Load Safetensors (The Teacher / FP16)
    let sf_path = &args[1];
    let mut sf_file = File::open(sf_path)?;
    let mut buffer = Vec::new();
    sf_file.read_to_end(&mut buffer)?;
    let tensors = SafeTensors::deserialize(&buffer)?;
    let sf_emb = tensors.tensor("model.embed_tokens.weight")?;
    let sf_floats: Vec<f32> = match sf_emb.dtype() {
        safetensors::Dtype::F16 => {
            let data_bytes = sf_emb.data();
            let mut floats = Vec::with_capacity(data_bytes.len() / 2);
            for chunk in data_bytes.chunks_exact(2) {
                let f = half::f16::from_le_bytes([chunk[0], chunk[1]]);
                floats.push(f.to_f32());
            }
            floats
        }
        safetensors::Dtype::BF16 => {
            let data_bytes = sf_emb.data();
            let mut floats = Vec::with_capacity(data_bytes.len() / 2);
            for chunk in data_bytes.chunks_exact(2) {
                let f = half::bf16::from_le_bytes([chunk[0], chunk[1]]);
                floats.push(f.to_f32());
            }
            floats
        }
        safetensors::Dtype::F32 => {
            let data_bytes = sf_emb.data();
            let mut floats = Vec::with_capacity(data_bytes.len() / 4);
            for chunk in data_bytes.chunks_exact(4) {
                let f = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                floats.push(f);
            }
            floats
        }
        _ => panic!("Unsupported dtype"),
    };

    // Load MUD (The Student / Ternary 1.58b)
    let mud_path = &args[2];
    let mud_file = MudFile::load(mud_path)?;
    let core = mud_file.skills.get("core").unwrap();
    let mud_emb = core.tensors.get("token_embd.weight").unwrap();

    let hidden_size = sf_emb.shape()[1];
    let vocab_size = sf_emb.shape()[0];

    // Dequantize MUD embeddings to f32 to read the wave phase
    let mut mud_floats = vec![0.0f32; vocab_size * hidden_size];
    let is_ternary = mud_emb.t_type == forge_llm::mud::MudTensorType::Ternary2Bit;

    if is_ternary {
        unsafe {
            for row in 0..vocab_size {
                let src_ptr = (mud_emb.data_ptr as *const u32).add(row * (hidden_size / 16));
                let dst_slice = &mut mud_floats[row * hidden_size..(row + 1) * hidden_size];
                dequantize_ternary_row(src_ptr, dst_slice, hidden_size);
            }
        }
    } else {
        unsafe {
            let src_ptr = mud_emb.data_ptr as *const f32;
            std::ptr::copy_nonoverlapping(
                src_ptr,
                mud_floats.as_mut_ptr(),
                vocab_size * hidden_size,
            );
        }
    }

    let scales_ptr = if is_ternary {
        core.tensors.get("embed_scales").unwrap().data_ptr as *const f32
    } else {
        std::ptr::null()
    };

    println!("  Comparando Mapeo de Firmas de Ondas Senoidales...");
    println!(
        "  Frecuencia Neuronal (Vocabulario): {} | Canales: {}",
        vocab_size, hidden_size
    );

    let tokens_to_test = [71, 77, 30182, 16]; // Hola, MUD, ¿, 1
    let names = ["'Hola'", "'MUD'", "'¿'", "'1'"];

    let mut total_sim = 0.0;

    println!("\n>> CALCULANDO CONFIANZA DE ALINEACION (COSINE SIMILARITY)");
    for (idx, &tok) in tokens_to_test.iter().enumerate() {
        let start = tok * hidden_size;
        let sf_wave = &sf_floats[start..start + hidden_size];
        let mut mud_wave = mud_floats[start..start + hidden_size].to_vec();

        let mut scale = 1.0;
        if !scales_ptr.is_null() {
            scale = unsafe { *scales_ptr.add(tok) };
            for v in mud_wave.iter_mut() {
                *v *= scale;
            } // Restore the mathematical signature
        }

        let sim = cosine_similarity(sf_wave, &mud_wave);
        total_sim += sim;

        let mut sf_min = f32::MAX;
        let mut sf_max = f32::MIN;
        let mut mud_min = f32::MAX;
        let mut mud_max = f32::MIN;
        for &v in sf_wave {
            if v < sf_min {
                sf_min = v;
            }
            if v > sf_max {
                sf_max = v;
            }
        }
        for &v in mud_wave.iter() {
            if v < mud_min {
                mud_min = v;
            }
            if v > mud_max {
                mud_max = v;
            }
        }

        println!("\n  Token {}: {}", tok, names[idx]);
        println!(
            "    Original (FP16) Wave | Min: {:>8.4} | Max: {:>8.4}",
            sf_min, sf_max
        );
        println!(
            "    Ternary (1.58b) Wave | Min: {:>8.4} | Max: {:>8.4} | Escala Absmean: {:.4}",
            mud_min, mud_max, scale
        );
        println!(
            "    > Similitud de Fase (Cosine): \x1b[1;32m{:.2}%\x1b[0m",
            sim * 100.0
        );
    }

    println!("\n==================================================");
    println!(
        "✅ INDICE GLOBAL DE CONFIANZA HOLOGRAFICA: {:.2}%",
        (total_sim / tokens_to_test.len() as f32) * 100.0
    );

    Ok(())
}
