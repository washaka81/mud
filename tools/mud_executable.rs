use std::fs::File;
use std::io::Write;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: mud_executable <forge_llm_binary> <model.mud> <output.run>");
        std::process::exit(1);
    }

    let bin_path = &args[1];
    let mud_path = &args[2];
    let out_path = &args[3];

    println!("Building MUD-Executable: {}", out_path);
    println!("  Engine : {}", bin_path);
    println!("  Payload: {}", mud_path);

    let bin_meta = std::fs::metadata(bin_path)?;
    let mud_offset = bin_meta.len();

    let mut out_file = File::create(out_path)?;

    // Copy the engine binary
    let mut bin_file = File::open(bin_path)?;
    std::io::copy(&mut bin_file, &mut out_file)?;

    // Copy the MUD payload
    let mut mud_file = File::open(mud_path)?;
    std::io::copy(&mut mud_file, &mut out_file)?;

    // Append Trailer
    out_file.write_all(&mud_offset.to_le_bytes())?;
    out_file.write_all(b"MUDEXEC\0")?;

    out_file.sync_all()?;

    // Make executable
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(out_path)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(out_path, perms)?;
    }

    println!("✅ MUD-Executable created successfully at: {}", out_path);
    println!("  Run it directly: ./{}", out_path);

    Ok(())
}
