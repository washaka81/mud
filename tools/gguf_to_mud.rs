use std::fs::File;
use std::io::Read;

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("Usage: gguf_to_mud <gguf_file>");
    let mut file = File::open(&path)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;

    let gguf = gguf::GGUFFile::read(&data)
        .map_err(|e| anyhow::anyhow!(e))?
        .unwrap();
    println!("GGUF Version: {}", gguf.header.version);
    println!("Metadata KVs: {}", gguf.header.metadata.len());
    println!("Tensors: {}", gguf.tensors.len());
    for tensor in gguf.tensors.iter().take(10) {
        println!(
            "Tensor: {} | Type: {:?} | Elements: {:?}",
            tensor.name, tensor.tensor_type, tensor.dimensions
        );
    }

    Ok(())
}
