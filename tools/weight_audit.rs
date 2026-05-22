use forge_llm::mud::MudFile;

fn main() -> anyhow::Result<()> {
    let model_path = "models/core_skills.mud";
    println!("=== MUD Weight & Sigma Audit: {} ===", model_path);
    
    let model = MudFile::load(model_path)?;
    let core = model.skills.get("core").expect("No core skill");
    
    for (name, tensor) in &core.tensors {
        if tensor.t_type == forge_llm::mud::MudTensorType::Ternary2Bit {
            // Read packed u32 values
            let n_elements = tensor.shape.iter().copied().product::<usize>();
            let n_u32 = (n_elements + 15) / 16;
            let data_ptr = tensor.data_ptr as *const u32;
            let packed_data = unsafe { std::slice::from_raw_parts(data_ptr, n_u32) };
            
            let mut counts = [0usize; 3]; // 0: 0, 1: +1, 2: -1
            for &val in packed_data {
                for i in 0..16 {
                    let bits = (val >> (i * 2)) & 3;
                    if bits == 1 { counts[1] += 1; }
                    else if bits == 2 { counts[2] += 1; }
                    else { counts[0] += 1; }
                }
            }
            
            let total = counts[0] + counts[1] + counts[2];
            let fill_rate = (counts[1] + counts[2]) as f32 / total as f32;
            
            // Calculate pseudo-sigma (std dev of -1, 0, 1 distribution)
            // Mean is roughly 0 if balanced
            let variance = (counts[1] as f32 * 1.0 + counts[2] as f32 * 1.0) / total as f32;
            let sigma = variance.sqrt();
            
            println!("{:<40} | Sigma: {:.4} | Fill: {:.1}% | Pos: {:>5} | Neg: {:>5} | Zero: {:>5}", 
                     name, sigma, fill_rate * 100.0, counts[1], counts[2], counts[0]);
        }
    }
    
    Ok(())
}
