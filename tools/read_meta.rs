use forge_llm::mud::MudFile;
fn main() {
    let mud = MudFile::load("models/Phi-4-mini.mud").unwrap();
    for (k, v) in mud.global_metadata {
        println!("{}: {}", k, v);
    }
}
