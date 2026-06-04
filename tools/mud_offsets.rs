use forge_llm::mud::{MudFile, MudTensorType};

fn main() {
    println!("🔍 ANALIZANDO OFFSETS DEL ARCHIVO MUD...");
    let model = MudFile::load("models/jamba_v1_master.mud").expect("Failed to load mud");
    let core = model.skills.get("core").unwrap();

    let mut tensors: Vec<_> = core.tensors.values().collect();
    tensors.sort_by_key(|t| t.offset);

    for t in tensors.iter().take(20) {
        let elements: usize = t.shape.iter().product();
        let expected_size = match t.t_type {
            MudTensorType::Ternary2Bit => elements.div_ceil(16) * 4,
            MudTensorType::Float32 => elements * 4,
            MudTensorType::Float16 => elements * 2,
            MudTensorType::Int4 => elements.div_ceil(2),
        };
        println!(
            "Tensor: {} | Type: {:?} | Shape: {:?} | Elements: {} | Expected Size: {} | Offset: {}",
            t.name, t.t_type, t.shape, elements, expected_size, t.offset
        );
    }
}
