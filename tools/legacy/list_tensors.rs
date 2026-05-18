use forge_llm::gguf::GGUFModel;
use std::fs::File;
use std::io::Write;

fn main() -> anyhow::Result<()> {
    let model = GGUFModel::load("models/qwen2.5-coder-1.5b-instruct-q4_0.gguf")?;
    let mut f = File::create("all_tensors.txt")?;
    let mut names: Vec<_> = model.tensors.keys().collect();
    names.sort();
    for name in names {
        writeln!(f, "{}", name)?;
    }
    println!("Listado {} tensores en all_tensors.txt", model.tensors.len());
    Ok(())
}
