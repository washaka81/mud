use forge_llm::mud::{MudFile, MudTensorType, dequantize_ternary_row};
use std::env;

fn calculate_stats(data: &[f32]) -> (f32, f32, f32, f32, usize) {
    if data.is_empty() { return (0.0, 0.0, 0.0, 0.0, 0); }
    
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    let mut sum = 0.0;
    let mut zeros = 0;
    
    for &v in data {
        if v < min { min = v; }
        if v > max { max = v; }
        sum += v;
        if v.abs() < 1e-6 { zeros += 1; }
    }
    
    let mean = sum / data.len() as f32;
    let mut var_sum = 0.0;
    for &v in data {
        var_sum += (v - mean).powi(2);
    }
    let variance = var_sum / data.len() as f32;
    
    (min, max, mean, variance, zeros)
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        println!("Usage: tensor_microscope <model.mud> <tensor_name_substring>");
        return Ok(());
    }

    let model_path = &args[1];
    let query = &args[2];

    println!("🔬 MUD Tensor Microscope - Nanometric Analysis");
    println!("Loading: {}", model_path);
    let mud_file = MudFile::load(model_path)?;
    let core = mud_file.skills.get("core").expect("Core skill not found");

    for (name, tensor) in &core.tensors {
        if name.contains(query) {
            println!("\n========================================");
            println!("🎯 Tensor: {}", name);
            println!("   Type:   {:?}", tensor.t_type);
            println!("   Shape:  {:?}", tensor.shape);
            
            let elements: usize = tensor.shape.iter().product();
            if elements == 0 { continue; }

            match tensor.t_type {
                MudTensorType::Float32 => {
                    let slice = unsafe { std::slice::from_raw_parts(tensor.data_ptr as *const f32, elements) };
                    let (min, max, mean, var, zeros) = calculate_stats(slice);
                    println!("   Min:    {:.6}", min);
                    println!("   Max:    {:.6}", max);
                    println!("   Mean:   {:.6}", mean);
                    println!("   Var:    {:.6}", var);
                    println!("   Sigma:  {:.6}", var.sqrt());
                    println!("   Sparsity: {:.2}%", (zeros as f32 / elements as f32) * 100.0);
                    
                    println!("\n   [Nanometric Sample (First 10)]");
                    for i in 0..10.min(elements) {
                        print!("{:>8.4} ", slice[i]);
                    }
                    println!();
                },
                MudTensorType::Ternary2Bit => {
                    let mut shadow = vec![0.0f32; elements];
                    unsafe {
                        dequantize_ternary_row(tensor.data_ptr as *const u32, &mut shadow, elements);
                    }
                    let (min, max, mean, var, zeros) = calculate_stats(&shadow);
                    println!("   Min:    {:.6}", min);
                    println!("   Max:    {:.6}", max);
                    println!("   Mean:   {:.6}", mean);
                    println!("   Var:    {:.6}", var);
                    println!("   Sigma:  {:.6}", var.sqrt());
                    println!("   Sparsity: {:.2}%", (zeros as f32 / elements as f32) * 100.0);
                    
                    // Count exact ternary states
                    let mut pos = 0; let mut neg = 0; let mut zer = 0;
                    for &v in &shadow {
                        if v > 0.5 { pos += 1; }
                        else if v < -0.5 { neg += 1; }
                        else { zer += 1; }
                    }
                    println!("   States: [+1: {:.1}%]  [-1: {:.1}%]  [0: {:.1}%]", 
                        (pos as f32 / elements as f32) * 100.0,
                        (neg as f32 / elements as f32) * 100.0,
                        (zer as f32 / elements as f32) * 100.0
                    );

                    println!("\n   [Nanometric Sample (First 10)]");
                    for i in 0..10.min(elements) {
                        print!("{:>4.0} ", shadow[i]);
                    }
                    println!();
                },
                MudTensorType::Float16 => {
                    println!("   Error: Microscope currently does not support f16 decoding natively.");
                }
            }
        }
    }

    Ok(())
}
