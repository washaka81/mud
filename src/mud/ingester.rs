use std::fs;
use std::path::Path;
use crate::mud::inference::MudInference;

/// Module for ingesting local documents into the MUD Knowledge Index.
pub struct MudIngester;

impl MudIngester {
    /// Reads a file or directory and adds its content to the MUD engine's index.
    pub fn ingest(path: &str, engine: &MudInference) -> anyhow::Result<usize> {
        let p = Path::new(path);
        let mut total_count = 0;

        if p.is_dir() {
            for entry in fs::read_dir(p)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_file() {
                    match ingest_file(&path, engine) {
                        Ok(n) => total_count += n,
                        Err(e) => println!("  ⚠️ [Ingester] Skipping {}: {}", path.display(), e),
                    }
                }
            }
        } else {
            total_count = ingest_file(p, engine)?;
        }

        Ok(total_count)
    }
}

use std::process::Command;

fn ingest_file(path: &Path, engine: &MudInference) -> anyhow::Result<usize> {
    let filename = path.file_name().unwrap().to_string_lossy();
    let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    
    let content = if extension.to_lowercase() == "pdf" {
        println!("  [Ingester] Extracting text from PDF: {}...", filename);
        let output = Command::new("pdftotext")
            .arg("-layout")
            .arg(path)
            .arg("-") // output to stdout
            .output()?;
        
        if !output.status.success() {
            anyhow::bail!("Failed to extract text from PDF: {}", String::from_utf8_lossy(&output.stderr));
        }
        String::from_utf8_lossy(&output.stdout).to_string()
    } else {
        fs::read_to_string(path)?
    };

    println!("  [Ingester] Ingesting {}...", filename);

    // Better chunking: 800 characters with paragraph logic
    let chunks: Vec<String> = content
        .split("\n\n")
        .flat_map(|s| {
            if s.len() > 800 {
                s.chars().collect::<Vec<char>>()
                    .chunks(800)
                    .map(|c| c.iter().collect::<String>())
                    .collect::<Vec<String>>()
            } else {
                vec![s.to_string()]
            }
        })
        .filter(|s| !s.trim().is_empty())
        .collect();

    let mut added = 0;
    let total_chunks = chunks.len();

    for (_i, chunk) in chunks.into_iter().enumerate() {
        let embedding = generate_real_embedding(&chunk, engine);
        let fact = format!("Source: {} | Content: {}", filename, chunk.replace("\n", " "));
        
        // 1. Update Knowledge Graph (Neural Bridges)
        let mut graph = engine.model.knowledge_graph.write().unwrap();
        graph.add_node(fact.clone(), embedding.clone());
        
        // 2. Persist to MudStore with Embedding (for future dynamic loading)
        engine.store.add_fact_with_embedding(&fact, &filename, &embedding)?;

        added += 1;
        if added % 50 == 0 {
            println!("    [Ingester] Processed {}/{} chunks...", added, total_chunks);
        }
    }

    // 3. Recalculate PageRank ONCE per file (The Google Algorithm)
    println!("    [Ingester] Finalizing knowledge bridges (Google Algorithm)...");
    engine.model.knowledge_graph.write().unwrap().recalculate_ranks();

    Ok(added)
}

/// Generates a semantic embedding by averaging the model's token embeddings for the chunk.
fn generate_real_embedding(text: &str, engine: &MudInference) -> Vec<f32> {
    let tokens = engine.tokenizer.encode(text);
    if tokens.is_empty() {
        return vec![0.0f32; engine.model.hidden_size];
    }

    let mut mean_embedding = vec![0.0f32; engine.model.hidden_size];
    let mut temp_x = vec![0.0f32; engine.model.hidden_size];

    for &token in &tokens {
        engine.embed_token(token, &mut temp_x);
        for i in 0..engine.model.hidden_size {
            mean_embedding[i] += temp_x[i];
        }
    }

    // Average and Normalize
    let n = tokens.len() as f32;
    let mut mag = 0.0f32;
    for i in 0..engine.model.hidden_size {
        mean_embedding[i] /= n;
        mag += mean_embedding[i] * mean_embedding[i];
    }
    
    mag = mag.sqrt();
    if mag > 1e-9 {
        for v in mean_embedding.iter_mut() { *v /= mag; }
    }

    mean_embedding
}
