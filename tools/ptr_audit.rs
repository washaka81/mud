use forge_llm::ai::MudFile;

fn main() -> anyhow::Result<()> {
    println!("=== MUD POINTER & MEMORY AUDIT ===");
    let model = MudFile::load("models/core_skills.ai")?;
    
    let core = model.skills.get("core").expect("Core skill not found");
    
    println!("Mmap total size: {} bytes", model.mmap.as_ref().unwrap().len());
    
    let mut t_names: Vec<_> = core.tensors.keys().collect();
    t_names.sort();

    println!("\n{:<40} | {:<12} | {:<10}", "Tensor Name", "Type", "Offset");
    println!("{}", "-".repeat(68));

    for name in t_names {
        let t = core.tensors.get(name).unwrap();
        let offset = unsafe { (t.data_ptr as *const u8).offset_from(model.mmap.as_ref().unwrap().as_ptr()) };
        println!("{:<40} | {:<12?} | {:<10}", name, t.t_type, offset);
    }
    
    Ok(())
}
