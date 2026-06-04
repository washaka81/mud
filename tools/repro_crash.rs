use forge_llm::mud::{inference::MudInference, MudFile};

fn main() -> anyhow::Result<()> {
    let mud_path = "models/pico_test.mud";
    let mud = MudFile::load(mud_path)?;
    println!("Loading model...");
    let mut infer = MudInference::new(&mud, None)?;

    let mut current_x = vec![0.0f32; infer.model.hidden_size];
    let mut pos = 0usize;
    println!("Prompting...");
    infer.prompt("hola", &mut current_x, &mut pos);

    println!("Generating...");
    infer.generate(&current_x, 10, "hola", &mut pos, 0, |token_id, text| {
        println!("Token: {} -> {}", token_id, text);
    });

    Ok(())
}
