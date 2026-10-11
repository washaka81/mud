use forge_llm::model::tokenizer::Tokenizer;
use forge_llm::mud::MudFile;

fn main() {
    let mud_file = MudFile::load("models/bitnet-b1.58-2B-4T.mud").unwrap();
    let tok_str = mud_file.global_metadata.get("tokenizer.tokens").unwrap();
    let mer_str = mud_file.global_metadata.get("tokenizer.merges").unwrap();
    let tokenizer = Tokenizer::from_mud_metadata(tok_str, mer_str);
    let text = "Hola, cómo estás?";
    let tokens = tokenizer.encode_simple(text);
    println!("Tokens: {:?}", tokens);
    for t in &tokens {
        println!("Token {}: {:?}", t, tokenizer.decode_simple(&[*t]));
    }
}
