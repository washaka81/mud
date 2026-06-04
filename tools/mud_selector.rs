use comfy_table::modifiers::UTF8_ROUND_CORNERS;
use comfy_table::presets::UTF8_FULL;
use comfy_table::{Attribute, Cell, Color, Table};
use forge_llm::mud::MudFile;
use std::fs;
use std::io::Write;

fn main() -> anyhow::Result<()> {
    let mut paths = Vec::new();
    if let Ok(entries) = fs::read_dir("models") {
        for entry in entries.flatten() {
            if entry.path().extension().is_some_and(|ext| ext == "mud") {
                paths.push(entry.path());
            }
        }
    }

    if paths.is_empty() {
        println!("\x1b[1;31mNo se encontraron modelos .mud en la carpeta models/\x1b[0m");
        return Ok(());
    }

    paths.sort();

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_header(vec![
            Cell::new("ID")
                .add_attribute(Attribute::Bold)
                .fg(Color::Yellow),
            Cell::new("Model Name")
                .add_attribute(Attribute::Bold)
                .fg(Color::Cyan),
            Cell::new("Size (MB)")
                .add_attribute(Attribute::Bold)
                .fg(Color::Magenta),
            Cell::new("Hidden Size").add_attribute(Attribute::Bold),
            Cell::new("Epochs Trained")
                .add_attribute(Attribute::Bold)
                .fg(Color::Green),
            Cell::new("Loss (Variance)")
                .add_attribute(Attribute::Bold)
                .fg(Color::Red),
            Cell::new("Cohesion Score")
                .add_attribute(Attribute::Bold)
                .fg(Color::Yellow),
            Cell::new("Last Sync")
                .add_attribute(Attribute::Bold)
                .fg(Color::DarkGrey),
        ]);

    for (i, path) in paths.iter().enumerate() {
        let file_name = path.file_name().unwrap().to_string_lossy();
        let size_mb = fs::metadata(path).map(|m| m.len() / 1_048_576).unwrap_or(0);

        let (hidden, epoch, loss, cohesion, last_sync) = match MudFile::load(path.to_str().unwrap())
        {
            Ok(model) => {
                let h = model
                    .global_metadata
                    .get("hidden_size")
                    .cloned()
                    .unwrap_or_else(|| "-".to_string());
                let ep = model
                    .global_metadata
                    .get("trainer.current_epoch")
                    .cloned()
                    .unwrap_or_else(|| "0".to_string());
                let lss = model
                    .global_metadata
                    .get("trainer.last_loss")
                    .cloned()
                    .unwrap_or_else(|| "N/A".to_string());
                let coh = model
                    .global_metadata
                    .get("trainer.cohesion_score")
                    .cloned()
                    .unwrap_or_else(|| "Untested".to_string());
                let sync = model
                    .global_metadata
                    .get("trainer.last_sync")
                    .cloned()
                    .unwrap_or_else(|| "Never".to_string());
                (h, ep, lss, coh, sync)
            }
            Err(_) => (
                "-".to_string(),
                "-".to_string(),
                "-".to_string(),
                "CORRUPTED".to_string(),
                "-".to_string(),
            ),
        };

        table.add_row(vec![
            Cell::new(i.to_string()).fg(Color::Yellow),
            Cell::new(file_name).fg(Color::Cyan),
            Cell::new(size_mb.to_string()).fg(Color::Magenta),
            Cell::new(hidden),
            Cell::new(epoch).fg(Color::Green),
            Cell::new(loss).fg(Color::Red),
            Cell::new(cohesion).fg(Color::Yellow),
            Cell::new(last_sync).fg(Color::DarkGrey),
        ]);
    }

    println!("\n\x1b[1;36m 🧠 MUD COGNITIVE MODEL SELECTOR \x1b[0m");
    println!("\x1b[1;30m──────────────────────────────────────────────────────────────────────────────────────────\x1b[0m\n");
    println!("{}", table);

    print!(
        "\n\x1b[1;32mSelecciona el ID del modelo para usar (0-{}): \x1b[0m",
        paths.len() - 1
    );
    std::io::stdout().flush()?;

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let selection: Result<usize, _> = input.trim().parse();

    match selection {
        Ok(idx) if idx < paths.len() => {
            let selected = paths[idx].to_string_lossy();
            println!("\x1b[1;32m✅ Modelo anclado: {}\x1b[0m", selected);
            let mut env_file = fs::File::create(".mud_env")?;
            writeln!(env_file, "export MODEL_PATH=\"{}\"", selected)?;
            println!("\x1b[1;33mEjecuta \x1b[1;37msource .mud_env\x1b[0m\x1b[1;33m para aplicar en esta terminal.\x1b[0m");
        }
        _ => {
            println!("\x1b[1;31m❌ Selección inválida.\x1b[0m");
            std::process::exit(1);
        }
    }

    Ok(())
}
