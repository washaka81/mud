use forge_llm::mud::MudFile;

fn main() -> anyhow::Result<()> {
    let mud_file = MudFile::load("weights/checkpoints/model_latest_checkpoint.mud")?;
    let core = mud_file.skills.get("core").unwrap();
    let emb = core.tensors.get("output.weight").unwrap();
    let slice = unsafe { std::slice::from_raw_parts(emb.data_ptr as *const f32, 576) };
    
    let mut reg_f32 = vec![0.0f32; 576];
    for (i, reg) in reg_f32.iter_mut().enumerate().take(576) {
        *reg = (i as f32 * 1.5).sin() * 1.87 * 1.414;
    }
    
    let mut dot = 0.0;
    for (r, s) in reg_f32.iter().zip(slice.iter()) {
        dot += r * s;
    }
    
    println!("Logit 0: {}", dot);
    
    // Print first 5 values of emb to verify
    for (i, &val) in slice.iter().enumerate().take(5) {
        println!("  Row 0, Col {}: {}", i, val);
    }
    
    Ok(())
}
