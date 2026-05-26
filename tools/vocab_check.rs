use forge_llm::mud::MudFile;
use forge_llm::model::tokenizer::Tokenizer;

fn main() -> anyhow::Result<()> {
    let mud_path = "models/core_skills.mud";
    let mud_file = MudFile::load(mud_path)?;
    
    let tokens_str = mud_file.global_metadata.get("tokenizer.tokens").expect("No tokens");
    let empty = "".to_string();
    let merges_str = mud_file.global_metadata.get("tokenizer.merges").unwrap_or(&empty);
    let tokenizer = Tokenizer::from_mud_metadata(tokens_str, merges_str);
    
    println!("=== Tokenizer Vocab Check ===");
    println!("Total tokens: {}", tokenizer.id_to_token.len());
    
    let test_words = vec!["the", " the", " to", " and", "a", " is", "in", "che", "WISE", "Aution", "iveness", "ardless"];
    for word in test_words {
        let ids = tokenizer.encode(word);
        print!("Word: '{:?}' -> IDs: {:?} -> Tokens: ", word, ids);
        for &id in &ids {
            print!("'{}' ", tokenizer.id_to_token.get(id as usize).unwrap_or(&"??".to_string()));
        }
        println!();
    }
    
    Ok(())
}
