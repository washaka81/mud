fn main() {
    let file = std::fs::OpenOptions::new().read(true).write(true).open("./smollm2.mud").unwrap();
    let mut mmap = unsafe { memmap2::MmapOptions::new().map_mut(&file).unwrap() };
    
    // We will find the offset manually by parsing the header!
    let mud = forge_llm::mud::MudFile::load("./smollm2.mud").unwrap();
    let core = mud.skills.get("core").unwrap();
    
    let mut fixed = 0;
    for (name, tensor) in &core.tensors {
        if tensor.t_type == forge_llm::mud::MudTensorType::Float32 {
            let offset = tensor.data_ptr as usize - mud.mmap.as_ref().unwrap().as_ptr() as usize;
            let num_elements = tensor.shape.iter().product::<usize>();
            let slice: &mut [f32] = unsafe { std::slice::from_raw_parts_mut(mmap.as_mut_ptr().add(offset) as *mut f32, num_elements) };
            
            let mut layer_fixed = 0;
            for val in slice.iter_mut() {
                if !val.is_finite() {
                    *val = 1.0;
                    layer_fixed += 1;
                }
            }
            if layer_fixed > 0 {
                println!("Fixed {} NaNs in tensor {}", layer_fixed, name);
                fixed += layer_fixed;
            }
        }
    }
    
    println!("Total NaNs fixed: {}", fixed);
    mmap.flush().unwrap();
}
