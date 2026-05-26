use crate::gguf::GGUFModel;
use crate::model::tokenizer::Tokenizer;
use std::path::Path;

#[test]
fn test_tokenizer_prompt() {
    let model_path = "models/MUD2.5-coder-1.5b-instruct-q4_0.gguf";
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

#[test]
fn test_auto_concordance_gpt_spaces() {
    // GPT-style BPE uses 'Ġ' for spaces.
    let tokens = "hello\nĠworld\n!\n<|im_start|>\n<|im_end|>";
    let merges = "";
    
    let tokenizer = Tokenizer::from_mud_metadata(tokens, merges);
    
    // Check space_char detection.
    assert_eq!(tokenizer.space_char, Some('Ġ'));
    
    // Check special control token detection.
    assert!(tokenizer.special_tokens.contains_key("<|im_start|>"));
    assert!(tokenizer.special_tokens.contains_key("<|im_end|>"));
    
    // Let's test decoding tokens with space prefix
    // 0: "hello"
    // 1: "Ġworld"
    // 2: "!"
    let decoded = tokenizer.decode(&[0, 1, 2]);
    assert_eq!(decoded, "hello world!");
}

#[test]
fn test_auto_concordance_sp_spaces() {
    // SentencePiece-style BPE uses '\u{2581}' for spaces.
    let tokens = "hello\n\u{2581}world\n!\n[PAD]\n[CLS]";
    let merges = "";
    
    let tokenizer = Tokenizer::from_mud_metadata(tokens, merges);
    
    // Check space_char detection (space prefix character is U+2581)
    assert_eq!(tokenizer.space_char, Some('\u{2581}'));
    
    // Check special control token detection
    assert!(tokenizer.special_tokens.contains_key("[PAD]"));
    assert!(tokenizer.special_tokens.contains_key("[CLS]"));
    
    // Let's test decoding tokens with SP space prefix
    // 0: "hello"
    // 1: "\u{2581}world"
    // 2: "!"
    let decoded = tokenizer.decode(&[0, 1, 2]);
    assert_eq!(decoded, "hello world!");
}
