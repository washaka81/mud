use forge_llm::mud::MudFile;
use forge_llm::model::tokenizer::Tokenizer;
use std::env;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    let text = if args.len() > 1 {
        args[1..].join(" ")
    } else {
        "hola MUD engine".to_string()
    };

    println!("=== MUD Tokenizer Audit ===");
    
    let model_path = "models/core_skills.mud";
    println!("Cargando vocabulario del modelo MUD: {}...", model_path);
    
    let mud_file = MudFile::load(model_path)?;
    
    let tokens_str = mud_file.global_metadata.get("tokenizer.tokens")
        .ok_or_else(|| anyhow::anyhow!("No tokenizer tokens in MUD metadata"))?;
    let merges_str = mud_file.global_metadata.get("tokenizer.merges").map(|s: &String| s.as_str()).unwrap_or("");
    
    let tokenizer = Tokenizer::from_mud_metadata(tokens_str, merges_str);
    
    println!("\nTexto de entrada: \"{}\"", text);
    println!("--------------------------------------------------");
    
    let tokens = tokenizer.encode(&text);
    
    println!("{:<10} | {:<20} | {:<20}", "ID", "Token (Literal)", "Reconstruido");
    println!("--------------------------------------------------");
    
    for id in &tokens {
        let literal = tokenizer.id_to_token.get(*id as usize).cloned().unwrap_or_else(|| "UNKNOWN".to_string());
        let decoded_part = tokenizer.decode(&[*id]);
        println!("{:<10} | {:<20?} | {:<20?}", id, literal, decoded_part);
    }
    
    println!("--------------------------------------------------");
    let final_decoded = tokenizer.decode(&tokens);
    println!("Texto final decodificado: \"{}\"", final_decoded);
    
    if final_decoded.to_lowercase().trim() == text.to_lowercase().trim() {
        println!("\nVERIFICACIÓN: COINCIDENCIA");
    } else {
        println!("\nVERIFICACIÓN: Diferencia detectada (esto es normal si se usó fallback por caracteres)");
    }

    Ok(())
}
