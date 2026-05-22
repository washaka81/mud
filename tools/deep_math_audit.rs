use forge_llm::mud::MudFile;

fn main() -> anyhow::Result<()> {
    let model_path = "models/core_skills.mud";
    println!("=== MUD ADVANCED STATISTICAL AUDIT: {} ===", model_path);
    
    let model = MudFile::load(model_path)?;
    let core = model.skills.get("core").expect("No core skill found");
    
    for (name, tensor) in &core.tensors {
        if tensor.t_type == forge_llm::mud::MudTensorType::Ternary2Bit {
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
            
            let n = (counts[0] + counts[1] + counts[2]) as f32;
            
            // 1. Mean (Expectation) - Should be close to 0 for balanced weights
            let mean = (counts[1] as f32 - counts[2] as f32) / n;
            
            // 2. Variance and Sigma
            let variance = (counts[1] as f32 * (1.0 - mean).powi(2) + 
                            counts[2] as f32 * (-1.0 - mean).powi(2) + 
                            counts[0] as f32 * (0.0 - mean).powi(2)) / n;
            let sigma = variance.sqrt();
            
            // 3. Skewness (Asimetría) - Measures lack of symmetry
            // Standardized 3rd moment
            let skewness = (counts[1] as f32 * (1.0 - mean).powi(3) + 
                            counts[2] as f32 * (-1.0 - mean).powi(3) + 
                            counts[0] as f32 * (0.0 - mean).powi(3)) / (n * sigma.powi(3));
            
            // 4. Kurtosis (Curtosis) - Measures thickness of tails (outliers)
            // Standardized 4th moment - 3 (Excess Kurtosis)
            let kurtosis = (counts[1] as f32 * (1.0 - mean).powi(4) + 
                            counts[2] as f32 * (-1.0 - mean).powi(4) + 
                            counts[0] as f32 * (0.0 - mean).powi(4)) / (n * sigma.powi(4)) - 3.0;

            println!("{:<35} | Sigma: {:.4} | Skew: {:>6.2} | Kurt: {:>6.2} | Mean: {:>6.3}", 
                     name, sigma, skewness, kurtosis, mean);
            
            // Interpretation based on 365 Data Science principles
            if kurtosis > 1.0 { print!("  [Leptokurtic: Heavy Tails] "); }
            if skewness.abs() > 0.5 { print!("  [High Skewness: Asymmetric] "); }
            if mean.abs() > 0.1 { print!("  [Bias Detected] "); }
            if kurtosis.abs() > 0.1 || skewness.abs() > 0.1 || mean.abs() > 0.1 { println!(); }
        }
    }
    
    Ok(())
}
