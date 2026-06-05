use forge_llm::mud::MudFile;

fn main() -> anyhow::Result<()> {
    let mut mud = MudFile::load("models/Phi-4-mini.mud")?;
    println!("Loaded MUD. Head dim is currently: {:?}", mud.global_metadata.get("head_dim"));
    
    // Set head_dim to 128
    mud.global_metadata.insert("head_dim".to_string(), "128".to_string());
    
    println!("Saving updated MUD...");
    mud.save("models/Phi-4-mini-fixed.mud")?;
    
    println!("Done. Replacing file...");
    std::fs::rename("models/Phi-4-mini-fixed.mud", "models/Phi-4-mini.mud")?;
    
    Ok(())
}
