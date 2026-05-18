use forge_llm::gguf::GGUFModel;
use std::collections::BTreeMap;

fn main() -> anyhow::Result<()> {
    let model = GGUFModel::load("models/MUD2.5-coder-1.5b-instruct-q4_0.gguf")?;
    
    let mut map = BTreeMap::new();
    for (name, t) in &model.tensors {
        let offset = unsafe { t.data_ptr.offset_from(model.mmap.as_ptr()) };
        let size = calculate_size(t);
        map.insert(offset, (name.clone(), size));
    }

    println!("Tensor Memory Map:");
    println!("{:<12} | {:<12} | {:<40}", "Offset", "Size", "Name");
    println!("------------------------------------------------------------------");
    
    let mut total_mapped_size = 0;
    for (offset, (name, size)) in &map {
        println!("{:<12} | {:<12} | {}", offset, size, name);
        total_mapped_size += size;
    }
    
    println!("\nTotal Mapped Size: {} bytes ({:.2} MB)", total_mapped_size, total_mapped_size as f64 / 1024.0 / 1024.0);
    println!("Mmap total size:   {} bytes ({:.2} MB)", model.mmap.len(), model.mmap.len() as f64 / 1024.0 / 1024.0);
    
    Ok(())
}

fn calculate_size(t: &forge_llm::gguf::Tensor) -> usize {
    let elements = t.shape.iter().product::<usize>();
    match t.t_type {
        forge_llm::gguf::TensorType::F32 => elements * 4,
        forge_llm::gguf::TensorType::F16 => elements * 2,
        forge_llm::gguf::TensorType::Q4_0 => (elements / 32) * 18,
        _ => elements, // Simplificado
    }
}
