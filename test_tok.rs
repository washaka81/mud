use forge_llm::model::tokenizer::Tokenizer;
fn main() {
    let tk = Tokenizer::from_mud_metadata(&forge_llm::mud::MudFile::load("weights/checkpoints/model_latest_checkpoint.mud").unwrap()).unwrap();
    let text = " 20Stress  Can Multi b aspot noCommon socialRom up";
    let ids = tk.encode(text);
    println!("IDs: {:?}", ids);
    for id in ids {
        println!("{}: {}", id, tk.decode(&[id]));
    }
}
