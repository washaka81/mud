fn main() {
    let mut files = Vec::new();
    collect_files(std::path::Path::new("training/corpus"), &mut files);
    collect_files(std::path::Path::new("src"), &mut files);
    collect_files(std::path::Path::new("docs"), &mut files);
    println!("Files: {:?}", files);
}

fn collect_files(dir: &std::path::Path, files: &mut Vec<std::path::PathBuf>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if let Some(path_str) = path.to_str() {
                    if !path_str.contains("target") && !path_str.contains(".git") && !path_str.contains("downloads") {
                        collect_files(&path, files);
                    }
                }
            } else if let Some(ext) = path.extension() {
                let ext_str = ext.to_string_lossy();
                if ext_str == "txt" || ext_str == "rs" || ext_str == "md" {
                    files.push(path);
                }
            }
        }
    }
}
