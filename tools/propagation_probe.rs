use forge_llm::model::tokenizer::Tokenizer;
use forge_llm::mud::{inference::MudInference, MudFile};
use forge_llm::vulkan::VulkanContext;
use std::env;
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        println!("Usage: propagation_probe <model.mud> <palabra>");
        return Ok(());
    }

    println!("==================================================");
    println!("🔬 MUD SEMANTIC PROPAGATION PROBE - IN-SITU TRACKER");
    println!("==================================================");

    println!("  Cargando Modelo: {}", args[1]);
    let vk = Arc::new(VulkanContext::new().unwrap());
    let mud_file = MudFile::load(&args[1])?;
    let mut engine = MudInference::new(&mud_file, Some(vk))?;

    // Cargar Tokenizer para mapear el lenguaje humano
    let tokens_str = mud_file
        .global_metadata
        .get("tokenizer_tokens")
        .map(|s| s.as_str())
        .unwrap_or("");
    let merges_str = mud_file
        .global_metadata
        .get("tokenizer_merges")
        .map(|s| s.as_str())
        .unwrap_or("");
    let tokenizer = Tokenizer::from_mud_metadata(tokens_str, merges_str);
    let word = &args[2];
    let tokens = tokenizer.encode(word);

    if tokens.is_empty() {
        println!("Error: La palabra no produjo tokens.");
        return Ok(());
    }
    let token_id = tokens[0];

    let hidden = engine.model.hidden_size;
    let mut x = vec![0.0f32; hidden];

    // Extraer la firma de pensamiento real (Embedding)
    engine.embed_token(token_id, &mut x);

    // Forzamos la variable de entorno para que inference.rs despierte la sonda
    env::set_var("MUD_TRACE_PROPAGATION", "1");

    println!(
        "\n[Inyectando Semántica: '{}' (Token ID: {})]",
        word, token_id
    );

    println!("\n\x1b[1;36m>> TRAZANDO HUELLA DE LA PALABRA\x1b[0m");
    // Hacemos el forward de la capa
    engine.step(&mut x, "semantic_probe", &[], 0);

    let mut min = f32::MAX;
    let mut max = f32::MIN;
    let mut sum = 0.0;
    for &v in x.iter() {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
        sum += v;
    }
    let mean = sum / hidden as f32;
    let var = x.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / hidden as f32;

    println!("\n  \x1b[1;32m[FIRMA FINAL DE '{}'] Pico Mín: {:.4} | Pico Máx: {:.4} | Sigma: {:.4}\x1b[0m", word, min, max, var.sqrt());

    Ok(())
}
