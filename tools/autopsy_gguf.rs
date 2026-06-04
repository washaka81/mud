use std::fs::File;
use std::io::Read;

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("Usage: autopsy_gguf <gguf_file>");
    let mut file = File::open(&path)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;
    
    let gguf = gguf::GGUFFile::read(&data).map_err(|e| anyhow::anyhow!(e))?.unwrap();
    println!("GGUF Version: {}", gguf.header.version);
    
    for tensor in gguf.tensors.iter() {
        if tensor.name == "blk.0.ffn_gate.weight" || tensor.name == "blk.0.ffn_down.weight" {
            println!("Tensor: {} | Type: {:?}", tensor.name, tensor.tensor_type);
            // gguf crate provides raw bytes for data
            // We need to parse f16
            let offset = tensor.offset;
            let bytes = &data[gguf.tensor_data_offset as usize + offset as usize..];
            
            // read first 20 f16 values
            println!("First 20 values:");
            for i in 0..20 {
                let f16_bytes = [bytes[i*2], bytes[i*2+1]];
                let f16_val = half::f16::from_le_bytes(f16_bytes);
                println!("  {:.4}", f16_val.to_f32());
            }
            
            // Check ternary ratio
            let mut ternary_count = 0;
            let mut total_count = 0;
            let mut abs_max = 0.0f32;
            let num_elements = tensor.dimensions.iter().product::<usize>();
            let max_check = std::cmp::min(num_elements, 1000000);
            
            for i in 0..max_check {
                let f16_bytes = [bytes[i*2], bytes[i*2+1]];
                let val = half::f16::from_le_bytes(f16_bytes).to_f32();
                if val.abs() > abs_max {
                    abs_max = val.abs();
                }
                
                // Usually ternary is represented exactly as -1.0, 0.0, 1.0, 
                // OR it has a scale, so we check if it clusters to 3 values.
                // Wait, if it's perfectly quantized, let's just count unique?
                // For simplicity, just find abs_max.
            }
            println!("Abs Max (first 1M elements): {:.4}", abs_max);
        }
    }
    
    Ok(())
}
