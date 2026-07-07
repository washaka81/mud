use forge_llm::mud::MudFile;

fn main() -> anyhow::Result<()> {
    let mud = MudFile::load("models/bitnet-b1.58-2B-4T.mud")?;
    println!(
        "rms_norm_eps: {:?}",
        mud.global_metadata.get("rms_norm_eps")
    );
    Ok(())
}
