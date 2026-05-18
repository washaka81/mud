use crate::gguf::GGUFModel;
use std::path::Path;

#[test]
fn test_dump_metadata() {
    let model_path = "models/MUD2.5-coder-1.5b-instruct-q4_0.gguf";
    if !Path::new(model_path).exists() { return; }

    let model = GGUFModel::load(model_path).unwrap();
    println!("Metadata Keys:");
    let mut keys: Vec<_> = model.metadata.keys().collect();
    keys.sort();
    for key in keys {
        println!("  {}: {:?}", key, model.metadata[key]);
    }
}
