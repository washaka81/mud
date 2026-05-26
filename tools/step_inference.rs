use forge_llm::mud::MudFile;

fn main() {
    let mud_file = MudFile::load("models/core_skills.mud").unwrap();
    println!("Tensors in core_skills:");
    if let Some(core) = mud_file.skills.get("core") {
        for (name, t) in &core.tensors {
            println!("{}: {:?}", name, t.shape);
        }
    }
}
