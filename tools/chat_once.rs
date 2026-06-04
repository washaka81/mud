use forge_llm::mud::inference::MudInference;
use forge_llm::mud::MudFile;
use std::io::Write;

fn apply_chat_template(inference: &MudInference, user_msg: &str) -> String {
    // Check if the model has a BitNet-style chat template (simple role: prefix format)
    let tmpl = &inference.chat_template;

    if !tmpl.is_empty() {
        // BitNet / LLaMA-3 template:
        // "User: {content}<|eot_id|>Assistant: "
        // We implement the simple version of the Jinja template:
        // {% for message in messages %}{{ role|capitalize }}: {{ content }}<|eot_id|>{% endfor %}
        // {% if add_generation_prompt %}{{ 'Assistant: ' }}{% endif %}
        let bos = &inference.bos_token;
        let eot = &inference.eos_token;
        format!("{}User: {}{}\nAssistant:", bos, user_msg.trim(), eot)
    } else {
        // Generic fallback — works for Qwen, Mistral, etc.
        format!("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", user_msg)
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).cloned().unwrap_or_else(|| "models/qwen2_0.5b.mud".to_string());
    let user_prompt = args.get(2).cloned().unwrap_or_else(|| "Hello! Who are you?".to_string());

    println!("Loading MUD Model: {}", model_path);

    let mud_file = MudFile::load(&model_path)?;
    let vk = None;
    let mut inference = MudInference::new(&mud_file, vk)?;

    // Apply the model's native chat template
    let full_prompt = apply_chat_template(&inference, &user_prompt);

    // Log template info
    if !inference.chat_template.is_empty() {
        println!("📋 Chat template: active (bos='{}', eos='{}')", inference.bos_token, inference.eos_token);
    } else {
        println!("📋 Chat template: fallback ChatML");
    }
    println!("\nUser: {}", user_prompt);

    let mut x = vec![0.0f32; inference.model.hidden_size];
    let mut conversation_pos = 0;

    inference.prompt(&full_prompt, &mut x, &mut conversation_pos);

    print!("Assistant:");
    std::io::stdout().flush()?;

    let eos = inference.eos_token.clone();
    let (_tokens, _) = inference.generate(&x, 100, &full_prompt, &mut conversation_pos, 0, |_id, text| {
        // Stop streaming on EOS token
        if !eos.is_empty() && text.contains(&eos) {
            return;
        }
        print!("{}", text);
        let _ = std::io::stdout().flush();
    });
    println!();

    Ok(())
}
