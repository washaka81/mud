use forge_llm::mud::MudFile;
use forge_llm::model::tokenizer::Tokenizer;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: debug_tokens <model.mud>");
        return Ok(());
    }

    let mud = MudFile::load(&args[1])?;
    let tokens_str = mud.global_metadata.get("tokenizer.tokens").unwrap().as_str();
    let merges_str = mud.global_metadata.get("tokenizer.merges").map(|s| s.as_str()).unwrap_or("");
    let tokenizer = Tokenizer::from_mud_metadata(tokens_str, merges_str);

    let prompt = "Hola, como estas?";
    let tokens = tokenizer.encode(prompt);
    println!("Prompt: '{}'", prompt);
    println!("Tokens: {:?}", tokens);
    for t in &tokens {
        println!("  {} -> '{}'", t, tokenizer.decode(&[*t]));
    }

    Ok(())
}
