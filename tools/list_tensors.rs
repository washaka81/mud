fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "model.mud".to_string());
    let mut mud = forge_llm::mud::Mud::load(&path).unwrap();
    let skill = mud.skills.get("core").unwrap();
    for (name, tensor) in &skill.tensors {
        println!("{}: {:?}", name, tensor.t_type);
    }
}
