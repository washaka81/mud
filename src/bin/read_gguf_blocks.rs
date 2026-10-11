use std::fs::File;
use std::io::Read;

fn main() {
    let mut f = match File::open("models/ternary_bonsai_1.7b/Ternary-Bonsai-1.7B-Q2_0.gguf") {
        Ok(file) => file,
        Err(_) => return,
    };
    let mut header = vec![0u8; 1024 * 1024]; // 1MB should cover metadata
    if f.read_exact(&mut header).is_err() {
        return;
    }
    
    // Quick and dirty manual find of token_embd.weight
    let search_str = b"token_embd.weight";
    for i in 0..header.len() - search_str.len() {
        if &header[i..i+search_str.len()] == search_str {
            println!("Found at {}", i);
            break;
        }
    }
}
