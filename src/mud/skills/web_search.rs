use crate::mud::skills::MudSkill;
use crate::mud::inference::MudInference;
use std::process::Command;
use serde_json::Value;

/// Implementation of the Web Search & Ingestion skill.
/// Autonomously crawls URLs to fetch high-authority knowledge.
pub struct WebSearchSkill;

impl WebSearchSkill {
    pub fn new() -> Self {
        Self
    }
}

impl MudSkill for WebSearchSkill {
    fn name(&self) -> &str {
        "web_search"
    }

    fn should_activate(&self, _x: &[f32], context: &str) -> bool {
        let keywords = ["search", "google", "investiga", "who is", "what is the latest", "http", "www"];
        keywords.iter().any(|&k| context.to_lowercase().contains(k))
    }

    fn execute_autonomous_action(&self, context: &str, engine: &MudInference) {
        // Detect potential URLs or specific research topics
        let words: Vec<&str> = context.split_whitespace().collect();
        for word in words {
            if word.starts_with("http") {
                println!("  [MUD Auto-Action] Researching URL: {}...", word);
                self.ingest_from_url(word, engine);
            }
        }
        
        // Factual Research for IBM Watson or similar
        if context.to_lowercase().contains("watson") && context.to_lowercase().contains("ibm") {
            println!("  [MUD Auto-Action] Researching IBM Watson via Authority Bridge...");
            self.ingest_from_url("https://www.ibm.com/watson/about", engine);
        }
    }
}

impl WebSearchSkill {
    fn ingest_from_url(&self, url: &str, engine: &MudInference) {
        let output = Command::new(".venv/bin/python")
            .arg("tools/web_bridge.py")
            .arg(url)
            .output();

        if let Ok(out) = output {
            let json_str = String::from_utf8_lossy(&out.stdout);
            if let Ok(v) = serde_json::from_str::<Value>(&json_str) {
                if let Some(content) = v["content"].as_str() {
                    println!("  [MUD Auto-Action] Web Knowledge extracted ({} chars). Assembling...", content.len());
                    
                    // Inject into Store
                    let _ = engine.store.add_fact(content, url);
                    
                    // Inject summary into Graph
                    let mut graph = engine.model.knowledge_graph.write().unwrap();
                    let embedding = vec![0.8; engine.model.hidden_size]; 
                    let summary = format!("Web Resource ({}): {}", url, &content[..content.len().min(200)]);
                    graph.add_node(summary, embedding);
                    
                    println!("  [MUD Auto-Action] Web Knowledge assimilated into MKG and MKS.");
                }
            }
        }
    }
}
