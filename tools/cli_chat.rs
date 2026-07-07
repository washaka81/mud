use forge_llm::mud::{MudFile, SlimeWorkspace, slime_forward::evaluate_slime_block};
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).expect("Usage: cli_chat <model.mud> [prompt]");
    let prompt = args.get(2).map(|s| s.as_str()).unwrap_or("Hola");

    let mud = MudFile::load(model_path).unwrap();
    let hidden = mud.global_metadata.get("hidden_size").and_then(|s| s.parse().ok()).unwrap();
    let n_heads = mud.global_metadata.get("num_heads").and_then(|s| s.parse().ok()).unwrap();
    let n_kv_heads = mud.global_metadata.get("num_kv_heads").and_then(|s| s.parse().ok()).unwrap();
    let max_pos = mud.global_metadata.get("max_position_embeddings").and_then(|s| s.parse().ok()).unwrap();
    let vocab_size = mud.global_metadata.get("vocab_size").and_then(|s| s.parse().ok()).unwrap();
    let n_layers = mud.global_metadata.get("num_layers").and_then(|s| s.parse().ok()).unwrap();
    let head_dim = hidden / n_heads;

    let mut ws = SlimeWorkspace::new(hidden, max_pos, n_kv_heads, head_dim);

    // Mock forward pass for testing
    println!("Loaded {}, Prompt: {}", model_path, prompt);
    println!("Model config: hidden={}, heads={}, kv_heads={}, max_pos={}, layers={}, vocab={}", 
        hidden, n_heads, n_kv_heads, max_pos, n_layers, vocab_size);
}
