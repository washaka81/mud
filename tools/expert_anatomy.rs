use forge_llm::mud::MudFile;

fn main() -> anyhow::Result<()> {
    let model_path = "models/core_skills.mud";
    println!("=== MUD MOE EXPERT ANATOMY: {} ===", model_path);
    
    let model = MudFile::load(model_path)?;
    let core = model.skills.get("core").expect("No core skill");
    
    let n_layers = 2; // Fixed for v37
    let n_experts = 8;

    for l in 0..n_layers {
        println!("\n[CAPA {}]", l);
        println!("{:<10} | {:<12} | {:<12} | {:<12} | {:<10}", 
                 "Experto", "W1 Magnitude", "W2 Magnitude", "W3 Magnitude", "Status");
        println!("{}", "-".repeat(70));

        for e in 0..n_experts {
            let m1 = get_mean_magnitude(core, &format!("blk.{}.expert.{}.w1.weight", l, e));
            let m2 = get_mean_magnitude(core, &format!("blk.{}.expert.{}.w2.weight", l, e));
            let m3 = get_mean_magnitude(core, &format!("blk.{}.expert.{}.w3.weight", l, e));
            
            let avg_mag = (m1 + m2 + m3) / 3.0;
            let status = if avg_mag < 0.05 { "💀 DEAD" } 
                        else if avg_mag < 0.15 { "💤 COMA" } 
                        else { "🔥 ALIVE" };

            println!("{:<10} | {:<12.6} | {:<12.6} | {:<12.6} | {}", 
                     format!("Exp #{}", e), m1, m2, m3, status);
        }
    }
    
    Ok(())
}

fn get_mean_magnitude(skill: &forge_llm::mud::MudSkill, name: &str) -> f32 {
    if let Some(t) = skill.tensors.get(name) {
        let n_elements = t.shape.iter().copied().product::<usize>();
        let n_u32 = (n_elements + 15) / 16;
        let data_ptr = t.data_ptr as *const u32;
        let packed_data = unsafe { std::slice::from_raw_parts(data_ptr, n_u32) };
        
        let mut sum_mag = 0.0f32;
        let mut count = 0;
        
        for &val in packed_data {
            for i in 0..16 {
                if count >= n_elements { break; }
                let bits = (val >> (i * 2)) & 3;
                let v: f32 = if bits == 1 { 1.0 } else if bits == 2 { -1.0 } else { 0.0 };
                sum_mag += v.abs();
                count += 1;
            }
        }
        sum_mag / n_elements as f32
    } else {
        0.0
    }
}
