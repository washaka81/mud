use forge_llm::model::tokenizer::Tokenizer;
use forge_llm::mud::MudFile;

fn main() -> anyhow::Result<()> {
    let mud = MudFile::load("models/smollm2.mud")?;
    let tokens_str = mud.global_metadata.get("tokenizer.tokens").unwrap();
    let merges_str = mud.global_metadata.get("tokenizer.merges").map(|s| s.as_str()).unwrap_or("");
    let tokenizer = Tokenizer::from_mud_metadata(tokens_str, merges_str);

    let test_texts = vec![
        "hola como estas",
        "hola\ncomo\nestas",
        "hola, como estas?",
    ];

    for text in test_texts {
        println!("\nText: {:?}", text);
        let ids = tokenizer.encode(text);
        for &id in &ids {
            let tok = tokenizer.id_to_token.get(id as usize).unwrap();
            let dec = tokenizer.decode(&[id]);
            println!("  ID {:<6} raw: {:<15} decoded: {:?}", id, format!("{:?}", tok), dec);
        }
    }

    Ok(())
}
