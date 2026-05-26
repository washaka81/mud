use safetensors::SafeTensors;
use memmap2::Mmap;

fn main() -> anyhow::Result<()> {
    let file = std::fs::File::open("models/mud_fast_ckpt.safetensors")?;
    let mmap = unsafe { Mmap::map(&file)? };
    let safe_tensors = SafeTensors::deserialize(&mmap)?;
    
    println!("Number of tensors: {}", safe_tensors.tensors().len());
    
    // Check type of tensors()
    let t_list = safe_tensors.tensors();
    if let Some((name, view)) = t_list.first() {
        println!("Type of name: {}, Type of view shape: {:?}", std::any::type_name_of_val(name), view.shape());
    }
    
    Ok(())
}
