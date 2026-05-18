use crate::gguf::GGUFModel;
use std::path::Path;

#[test]
fn test_load_real_model() {
    let model_path = "models/qwen2.5-coder-1.5b-instruct-q4_0.gguf";
    
    if !Path::new(model_path).exists() { return; }

    let model = GGUFModel::load(model_path).unwrap();
    println!("Modelo cargado. Total de tensores: {}", model.tensors.len());
    
    let mut keys: Vec<_> = model.tensors.keys().cloned().collect();
    keys.sort();
    for key in keys {
        if key.contains("embd") || key.contains("output.weight") || key.contains("output_norm") {
            let t = &model.tensors[&key];
            println!("Tensor: {:<40} | Tipo: {:?} | Shape: {:?}", key, t.t_type, t.shape);
        }
    }
}
