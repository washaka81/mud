use forge_llm::model::tokenizer::Tokenizer;
use forge_llm::mud::MudFile;

fn main() -> anyhow::Result<()> {
    let mud = MudFile::load("models/smollm2.mud")?;
    let tokens_str = mud.global_metadata.get("tokenizer.tokens").unwrap();
    let merges_str = mud.global_metadata.get("tokenizer.merges").map(|s| s.as_str()).unwrap_or("");

    let tokenizer = Tokenizer::from_mud_metadata(tokens_str, merges_str);

    println!("space_char: {:?}", tokenizer.space_char);
    println!("vocab_size: {}", tokenizer.vocab.len());

    let test_texts = vec![
        "hola, como estas?",
        "<|im_start|>user\nhola, como estas?<|im_end|>\n<|im_start|>assistant\n",
        "El modelo Forge LLM es modular.",
    ];

    for text in test_texts {
        println!("\n--- Testing text: {:?} ---", text);
        let ids = tokenizer.encode(text);
        println!("Token IDs: {:?}", ids);
        for &id in &ids {
            let tok_raw = tokenizer.id_to_token.get(id as usize).cloned().unwrap_or_default();
            let decoded = tokenizer.decode(&[id]);
            println!("  ID {:<6} | Raw token: {:<20} | Decoded: {:?}", id, format!("{:?}", tok_raw), decoded);
        }
        let full_decoded = tokenizer.decode(&ids);
        println!("Full Decoded: {:?}", full_decoded);
    }

    Ok(())
}
