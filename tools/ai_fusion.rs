use forge_llm::ai::{MudFile, MudSkill};
use std::collections::HashMap;
use std::env;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        println!("Usage: mud_fusion <output.ai> <input1.ai> <input2.ai> ...");
        return Ok(());
    }

    let output_path = &args[1];
    let input_paths = &args[2..];

    println!("=== MUD Model Fusion Tool ===");
    println!("Merging {} models into {}...", input_paths.len(), output_path);

    let mut fused_metadata: HashMap<String, String> = HashMap::new();
    let mut fused_skills: HashMap<String, MudSkill> = HashMap::new();

    let mut total_experts = 0;

    for path in input_paths {
        println!("  Reading {}...", path);
        let mud = MudFile::load(path)?;
        
        // 1. Merge Metadata (Keep latest)
        for (k, v) in &mud.global_metadata {
            fused_metadata.insert(k.clone(), v.clone());
        }

        // 2. Merge Skills
        for (skill_name, skill) in mud.skills {
            if !fused_skills.contains_key(&skill_name) {
                fused_skills.insert(skill_name.clone(), skill.clone());
                // Initial expert count for core
                if skill_name == "core" {
                    total_experts = mud.global_metadata.get("num_experts")
                        .and_then(|s: &String| s.parse::<usize>().ok()).unwrap_or(0);
                }
            } else {
                // COLLISION: Special logic for "core" MoE experts
                if skill_name == "core" {
                    println!("    Detected colliding 'core' skill. Concatenating experts...");
                    let incoming_experts = mud.global_metadata.get("num_experts")
                        .and_then(|s: &String| s.parse::<usize>().ok()).unwrap_or(0);
                    
                    let target_skill = fused_skills.get_mut("core").unwrap();
                    
                    for i in 0..incoming_experts {
                        let new_id = total_experts + i;
                        // Copy expert weights (w1, w2, w3) with remapped names
                        for w in &["w1", "w2", "w3"] {
                            let old_name = format!("blk.0.expert.{}.{}.weight", i, w);
                            let new_name = format!("blk.0.expert.{}.{}.weight", new_id, w);
                            if let Some(t) = skill.tensors.get(&old_name) {
                                let mut new_t = t.clone();
                                new_t.name = new_name.clone();
                                target_skill.tensors.insert(new_name, new_t);
                            }
                        }
                    }
                    total_experts += incoming_experts;
                } else {
                    println!("    Colliding skill '{}' ignored (keeping first).", skill_name);
                }
            }
        }
    }

    // Update global expert count
    fused_metadata.insert("num_experts".to_string(), total_experts.to_string());
    
    let fused_model = MudFile {
        mmap: None,
        skills: fused_skills,
        global_metadata: fused_metadata,
    };

    println!("  Finalizing fusion. Total Experts: {}", total_experts);
    fused_model.save(output_path)?;
    println!("✅ Fusion complete. New model saved to {}", output_path);

    Ok(())
}
