use crate::gguf::GGUFModel;
use std::path::Path;
use std::io::Write;

#[test]
fn test_dump_tensors() {
    let model_path = "models/MUD2.5-coder-1.5b-instruct-q4_0.gguf";
    if !Path::new(model_path).exists() { return; }

    let model = GGUFModel::load(model_path).unwrap();
    let mut file = std::fs::File::create("all_tensors.txt").unwrap();
    let mut keys: Vec<_> = model.tensors.keys().cloned().collect();
    keys.sort();
    for key in keys {
        let t = &model.tensors[&key];
        writeln!(file, "{} | {:?} | {:?}", key, t.t_type, t.shape).unwrap();
    }
}
