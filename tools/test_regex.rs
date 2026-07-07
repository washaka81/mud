fn main() {
    let re = fancy_regex::Regex::new(r"'(?i)[sdmt]|(?i)ll|(?i)ve|(?i)re|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+").unwrap();
    let text = std::fs::read_to_string("training/corpus/github_repos/rust-main/tests/codegen-llvm/abi-main-signature-16bit-c-int.rs").unwrap();
    println!("Testing regex on text of length {}", text.len());
    let mut count = 0;
    for mat in re.find_iter(&text) {
        count += 1;
    }
    println!("Found {} matches", count);
}
