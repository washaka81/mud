use crate::gguf::GGUFModel;
use std::path::Path;

#[test]
fn test_tokenizer_details() {
    let model_path = "models/qwen2.5-coder-1.5b-instruct-q4_0.gguf";
    if !Path::new(model_path).exists() { return; }

    let model = GGUFModel::load(model_path).unwrap();
    
    if let Some(crate::gguf::MetadataValue::String(m)) = model.metadata.get("tokenizer.ggml.model") {
        println!("Tokenizer Model: {}", m);
    }
    
    if let Some(tokens) = model.get_metadata_array("tokenizer.ggml.tokens") {
        println!("Vocab Size: {}", tokens.len());
        // Print first 5 tokens
        for (i, t) in tokens.iter().take(5).enumerate() {
            println!("  Token {}: {:?}", i, t);
        }
    }

    if let Some(merges) = model.get_metadata_array("tokenizer.ggml.merges") {
        println!("Merges Count: {}", merges.len());
        for (i, m) in merges.iter().take(5).enumerate() {
            println!("  Merge {}: {:?}", i, m);
        }
    }
}
