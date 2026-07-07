use forge_llm::mud::MudFile;

fn main() {
    let mud = MudFile::load("smollm2.mud").unwrap();
    let core = mud.skills.get("core").unwrap();
    let emb = core.tensors.get("token_embd.weight").unwrap();
    let vocab_size: usize = mud.global_metadata.get("vocab_size").unwrap().parse().unwrap();
    let hidden_size: usize = mud.global_metadata.get("hidden_size").unwrap().parse().unwrap();
    
    let emb_ptr = emb.owned_data.as_ref().map(|d| d.as_ptr()).unwrap_or(emb.data_ptr);
    let emb_slice = unsafe { std::slice::from_raw_parts(emb_ptr as *const f32, vocab_size * hidden_size) };
    let max_emb = emb_slice.iter().map(|v| v.abs()).fold(0.0f32, |a, b| a.max(b));
    println!("Actual max_emb of smollm2.mud: {}", max_emb);
}
