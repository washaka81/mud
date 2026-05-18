use forge_llm::ai::store::MudStore;
use std::io::Write;

fn main() -> anyhow::Result<()> {
    println!("=== MUD KNOWLEDGE PACKAGER ===");
    
    let store = MudStore::open("models/knowledge.db")?;
    let unassimilated = store.get_unassimilated()?;
    
    if unassimilated.is_empty() {
        println!("No new facts to package. Everything is up to date.");
        return Ok(());
    }

    println!("Packaging {} new facts for assimilation...", unassimilated.len());
    
    // In a real MUD scenario, this would create a new .ai skill block
    // For the prototype, we append to a 'knowledge_package.txt'
    let mut pkg_file = std::fs::File::create("models/knowledge_package.txt")?;
    let mut ids = Vec::new();

    for (id, content) in unassimilated {
        writeln!(pkg_file, "FACT_ID: {}\nCONTENT: {}\n---", id, content)?;
        ids.push(id);
    }

    store.mark_as_packed(&ids)?;
    println!("Package created: models/knowledge_package.txt");
    println!("Assimilated facts marked as 'Packed' in DB.");

    Ok(())
}
