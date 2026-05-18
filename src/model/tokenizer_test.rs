use crate::gguf::GGUFModel;
use crate::model::tokenizer::Tokenizer;
use std::path::Path;

#[test]
fn test_tokenizer_prompt() {
    let model_path = "models/qwen2.5-coder-1.5b-instruct-q4_0.gguf";
    if !Path::new(model_path).exists() { return; }

    let model = GGUFModel::load(model_path).unwrap();
    let tokenizer = Tokenizer::from_gguf(&model).unwrap();
    
    let text = "def fast_fibonacci(n):";
    let ids = tokenizer.encode(text);
    println!("Prompt: {}", text);
    for id in ids {
        println!("  ID {}: {:?}", id, tokenizer.id_to_token[id as usize]);
    }
}
