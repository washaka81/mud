use forge_llm::mud::{MudFile, MudTensorType};
use std::env;
use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        println!("Usage: mud_calibrator <model.mud> <tensor_scale_name> [new_scale_value_float]");
        println!("Example: mud_calibrator models/smollm2_mud.mud blk.0.expert.0.w1.scale 0.05");
        return Ok(());
    }

    let model_path = &args[1];
    let tensor_name = &args[2];
    
    let mud_file = MudFile::load(model_path)?;
    let core = mud_file.skills.get("core").expect("Core skill not found");
    
    let tensor = core.tensors.get(tensor_name).ok_or_else(|| anyhow::anyhow!("Tensor not found"))?;

    if tensor.t_type != MudTensorType::Float32 {
        println!("Error: Calibration tool currently targets FP32 scaling/norm tensors. This is {:?}", tensor.t_type);
        return Ok(());
    }

    let elements: usize = tensor.shape.iter().product();
    let slice = unsafe { std::slice::from_raw_parts(tensor.data_ptr as *const f32, elements) };
    
    println!("🔧 MUD Nanometric Calibrator");
    println!("Target: {}", tensor_name);
    println!("Current Values (First 5): {:?}", &slice[0..5.min(elements)]);
    
    if args.len() == 4 {
        let new_val: f32 = args[3].parse()?;
        println!("Applying nanometric adjustment: {} -> {}", slice[0], new_val);
        
        // Let's modify the file directly
        let mut file_mut = OpenOptions::new().read(true).write(true).open(model_path)?;
        
        let base_ptr = mud_file.mmap.as_ref().unwrap().as_ptr() as usize;
        let tensor_ptr = tensor.data_ptr as usize;
        let abs_file_offset = tensor_ptr - base_ptr;

        if file_mut.seek(SeekFrom::Start(abs_file_offset as u64)).is_ok() {
            let new_data = vec![new_val; elements];
            let bytes = unsafe { std::slice::from_raw_parts(new_data.as_ptr() as *const u8, new_data.len() * 4) };
            file_mut.write_all(bytes)?;
            println!("✅ Calibration applied successfully to disk.");
        } else {
            println!("❌ Failed to seek into file.");
        }
    } else {
        println!("\nTo adjust this tensor, provide a new float value as the 3rd argument.");
        println!("Tip: If logit variance is too high, lower the w2.scale or attn_output.scale.");
    }

    Ok(())
}
