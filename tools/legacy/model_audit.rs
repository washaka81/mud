use forge_llm::gguf::GGUFModel;
use forge_llm::asm::BlockQ4_0;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let model_path = "models/qwen2.5-coder-1.5b-instruct-q4_0.gguf";
    if !Path::new(model_path).exists() { return Ok(()); }

    println!("=== Forge LLM Model Integrity Audit ===");
    let model = GGUFModel::load(model_path)?;
    println!("  Model Alignment: {}", model.alignment);

    println!("\n[1. Layer 0 Tensor Check]");
    let layer0_tensors = [
        "blk.0.attn_q.weight",
        "blk.0.attn_q.bias",
        "blk.0.attn_k.weight",
        "blk.0.attn_k.bias",
        "blk.0.attn_v.weight",
        "blk.0.attn_v.bias",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "blk.0.attn_norm.weight",
        "blk.0.ffn_norm.weight",
    ];

    for name in layer0_tensors {
        if let Some(t) = model.tensors.get(name) {
            print!("  {:<25}: RawType={}, Type={:?}, Shape={:?}, First 4 vals: ", name, t.raw_type, t.t_type, t.shape);
            if t.t_type == forge_llm::gguf::TensorType::F32 {
                let ptr = t.data_ptr as *const f32;
                let slice = unsafe { std::slice::from_raw_parts(ptr, 4) };
                println!("{:?}", slice);
            }
        }
    }

    if let Some(t) = model.tensors.get("blk.0.attn_norm.weight") {
        let ptr = t.data_ptr as *const f32;
        let slice = unsafe { std::slice::from_raw_parts(ptr, 1536) };
        let zero_count = slice.iter().filter(|&&v| v == 0.0).count();
        println!("  blk.0.attn_norm.weight zero count: {}", zero_count);
    }
    let globals = ["token_embd.weight", "output.weight", "output_norm.weight"];
    for name in globals {
        if let Some(t) = model.tensors.get(name) {
             print!("  {:<25}: Type={:?}, Shape={:?}, First 4 vals: ", name, t.t_type, t.shape);
             if t.t_type == forge_llm::gguf::TensorType::F32 {
                let ptr = t.data_ptr as *const f32;
                println!("{:?}", unsafe { std::slice::from_raw_parts(ptr, 4) });
            } else {
                let ptr = t.data_ptr as *const BlockQ4_0;
                let block = unsafe { &*ptr };
                println!("d={:.6}", block.d.to_f32());
            }
        }
    }

    Ok(())
}
