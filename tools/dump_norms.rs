use forge_llm::mud::MudFile;

fn main() {
    let mud = MudFile::load("weights/checkpoints/model_latest_checkpoint.mud").unwrap();
    let core = mud.skills.get("core").unwrap();
    let t = core.tensors.get("blk.0.attn_norm.weight").unwrap();
    let ptr = t.data_ptr as *const f32;
    let vals = unsafe { std::slice::from_raw_parts(ptr, 10) };
    println!("blk.0.attn_norm.weight[..10] = {:?}", vals);
}
