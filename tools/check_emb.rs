use forge_llm::mud::MudFile;

fn main() {
    let mud = MudFile::load("./smollm2.mud").unwrap();
    let core = mud.skills.get("core").unwrap();
    let emb = core.tensors.get("token_embd.weight").unwrap();
    println!("Type of token_embd.weight: {:?}", emb.t_type);
}
