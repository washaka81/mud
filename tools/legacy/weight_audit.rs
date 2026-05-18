use forge_llm::gguf::GGUFModel;

fn main() -> anyhow::Result<()> {
    let model = GGUFModel::load("models/MUD2.5-coder-1.5b-instruct-q4_0.gguf")?;
    let t = model.tensors.get("output.weight").unwrap();
    let ptr = t.data_ptr as *const u8;
    
    println!("Auditing Token ID 10 Row (48 blocks):");
    let token_id = 10;
    let row_ptr = unsafe { ptr.add(token_id * 48 * 18) };
    let mut nans = Vec::new();
    for i in 0..48 {
        let block_ptr = unsafe { row_ptr.add(i * 18) } as *const forge_llm::asm::BlockQ4_0;
        let block = unsafe { &*block_ptr };
        let d = block.d.to_f32();
        if d.is_nan() || d.is_infinite() { nans.push(i); }
    }
    if !nans.is_empty() {
        println!("  Token {}: NaNs at blocks {:?}", token_id, nans);
    } else {
        println!("  Token {}: OK", token_id);
    }
    Ok(())
}
