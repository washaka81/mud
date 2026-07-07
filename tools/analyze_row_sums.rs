use forge_llm::mud::MudFile;
use forge_llm::model::tokenizer::Tokenizer;

fn main() {
    let path = "weights/checkpoints/model_latest_checkpoint.mud";
    let mud = match MudFile::load(path) {
        Ok(m) => m,
        Err(e) => {
            println!("Could not load model: {}", e);
            return;
        }
    };
    
    let core_skill = mud.skills.get("core").expect("No core skill found");
    
    let tensor = if let Some(t) = core_skill.tensors.get("output.weight") {
        t
    } else if let Some(t) = core_skill.tensors.get("token_embd.weight") {
        t
    } else {
        panic!("No output weight found");
    };

    let data = tensor.owned_data.as_deref().unwrap_or_else(|| {
        unsafe { std::slice::from_raw_parts(tensor.data_ptr, tensor.data_size()) }
    });

    let hidden = mud.global_metadata.get("hidden_size").and_then(|s| s.parse::<usize>().ok()).unwrap_or(576);
    let vocab_size = tensor.data_size() / (4 * hidden); // f32 is 4 bytes
    
    let f32_data: &[f32] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const f32, tensor.data_size() / 4)
    };

    let mut row_sums = Vec::new();
    for v in 0..vocab_size {
        let mut sum = 0.0;
        let start = v * hidden;
        for h in 0..hidden {
            sum += f32_data[start + h];
        }
        row_sums.push((v, sum));
    }

    row_sums.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let bpe_tokens = mud.global_metadata.get("tokenizer.tokens").unwrap();
    let merges = mud.global_metadata.get("tokenizer.merges").map(|s| s.as_str()).unwrap_or("");
    let vocab = Tokenizer::from_mud_metadata(bpe_tokens, merges);
    
    println!("Top 50 tokens by row sum:");
    for (v, sum) in row_sums.into_iter().take(50) {
        let token = vocab.decode(&[v as u32]);
        println!("{:>6} | {:>10.4} | {}", v, sum, token.replace("\n", "\\n"));
    }
}
