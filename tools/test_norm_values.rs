fn main() {
    let mud = forge_llm::mud::MudFile::load("./smollm2.mud").unwrap();
    let core = mud.skills.get("core").unwrap();
    let t = core.tensors.get("blk.0.attn_norm.weight").unwrap();
    let slice = unsafe { std::slice::from_raw_parts(t.data_ptr as *const f32, 10) };
    println!("blk.0.attn_norm.weight first 10: {:?}", slice);
}
