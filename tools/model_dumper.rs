use forge_llm::ai::MudFile;
use std::fs::File;
use std::io::Write;

fn main() -> anyhow::Result<()> {
    let model_path = "models/core_skills.ai";
    println!("Dumping MUD model metadata and skill info from: {}", model_path);
    
    let model = MudFile::load(model_path)?;
    let mut f = File::create("mud_disassembly.txt")?;

    writeln!(f, "=== MUD MODEL DISASSEMBLY ===")?;
    writeln!(f, "\n--- GLOBAL METADATA ---")?;
    
    let mut keys: Vec<_> = model.global_metadata.keys().collect();
    keys.sort();
    for key in keys {
        writeln!(f, "{:<40}: {:?}", key, model.global_metadata.get(key).unwrap())?;
    }

    for (skill_name, skill) in &model.skills {
        writeln!(f, "\n--- SKILL MODULE: {} ---", skill_name)?;
        let mut t_names: Vec<_> = skill.tensors.keys().collect();
        t_names.sort();
        
        for name in t_names {
            let t = skill.tensors.get(name).unwrap();
            writeln!(f, "{:<40} | Type: {:<12?} | Shape: {:?}", name, t.t_type, t.shape)?;
        }
    }

    println!("Disassembly completed. See mud_disassembly.txt");
    Ok(())
}
