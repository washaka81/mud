use candle_core::quantized::gguf_file::Content;
use candle_core::quantized::GgmlDType;
use memmap2::MmapOptions;
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use std::fs::File;

fn ggml_to_safetensors_dtype(ggml_dtype: GgmlDType) -> Dtype {
    match ggml_dtype {
        GgmlDType::F32 => Dtype::F32,
        GgmlDType::F16 => Dtype::F16,
        _ => panic!(
            "Unsupported ggml dtype for safetensors conversion: {:?}",
            ggml_dtype
        ),
    }
}

fn dtype_byte_size(dtype: Dtype) -> usize {
    match dtype {
        Dtype::F32 => 4,
        Dtype::F16 => 2,
        _ => panic!("Unsupported dtype size"),
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: gguf_to_safetensors <input.gguf> <output.safetensors>");
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];

    println!("🔍 Loading GGUF: {}", input_path);
    let mut file = File::open(input_path)?;
    let content = Content::read(&mut file).map_err(|e| anyhow::anyhow!(e))?;

    println!(
        "✅ Parsed GGUF metadata. Tensor Count: {}",
        content.tensor_infos.len()
    );

    let mmap = unsafe { MmapOptions::new().map(&file)? };

    let mut tensors = Vec::new();
    let data_offset = content.tensor_data_offset as usize;

    for (name, info) in &content.tensor_infos {
        let dtype = ggml_to_safetensors_dtype(info.ggml_dtype);
        let shape: Vec<usize> = info.shape.dims().to_vec();

        let elem_count: usize = shape.iter().product();
        let bytes_len = elem_count * dtype_byte_size(dtype);

        let start = data_offset + info.offset as usize;
        let end = start + bytes_len;

        let tensor_data = &mmap[start..end];
        let view =
            TensorView::new(dtype, shape, tensor_data).map_err(|e| anyhow::anyhow!("{:?}", e))?;
        tensors.push((name.clone(), view));
    }

    // Sort tensors to ensure deterministic safetensors
    tensors.sort_by(|a, b| a.0.cmp(&b.0));

    let iter = tensors.iter().map(|(k, v)| (k.clone(), v));
    println!("💾 Serializing to safetensors: {}", output_path);
    serialize_to_file(iter, &None, std::path::Path::new(output_path))
        .map_err(|e| anyhow::anyhow!("{:?}", e))?;

    println!("🎉 Done!");
    Ok(())
}
