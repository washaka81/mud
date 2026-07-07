use forge_llm::mud::{MudFile, MudTensorType};


fn main() -> anyhow::Result<()> {
    let mut mud = MudFile::load("smollm2.mud")?;
    
    let core_skill = mud.skills.get_mut("core").unwrap();
    for (_name, tensor) in core_skill.tensors.iter_mut() {
        if tensor.t_type == MudTensorType::Ternary2Bit {
            let total = tensor.shape.iter().product::<usize>();
            let mut w_fp32 = vec![0.0f32; total];
            
            for item in w_fp32.iter_mut().take(total) {
                let rv: f32 = rand::random::<f32>();
                *item = if rv < 0.37 { 1.0 } else if rv < 0.74 { -1.0 } else { 0.0 };
            }
            
            let u32_count = total.div_ceil(8);
            let mut packed = vec![0u32; u32_count];
            for (i, val) in w_fp32.iter().enumerate() {
                let bit = if *val > 0.5 { 1u32 } else if *val < -0.5 { 15u32 } else { 0u32 };
                packed[i / 8] |= bit << ((i % 8) * 4);
            }
            
            tensor.owned_data = Some(unsafe {
                std::slice::from_raw_parts(packed.as_ptr() as *const u8, packed.len() * 4)
            }.to_vec());
            tensor.data_ptr = tensor.owned_data.as_ref().unwrap().as_ptr();
        }
    }
    
    mud.save("smollm2.mud")?;
    println!("Reset all ternary weights in smollm2.mud");
    Ok(())
}
