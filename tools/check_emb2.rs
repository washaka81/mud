use forge_llm::mud::MudFile;

fn main() {
    let mud = MudFile::load("weights/checkpoints/model_latest_checkpoint.mud").unwrap();
    let core = mud.skills.get("core").unwrap();
    let emb = core.tensors.get("token_embd.weight").unwrap();
    let data = unsafe { std::slice::from_raw_parts(emb.data_ptr as *const f32, emb.shape[0] * emb.shape[1]) };
    let max_emb = data.iter().map(|v| v.abs()).fold(0.0f32, |a, b| a.max(b));
    println!("Actual max_emb: {}", max_emb);
}
