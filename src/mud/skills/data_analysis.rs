use crate::mud::skills::MudSkill;
use crate::mud::inference::MudInference;
use std::fs;
use std::path::Path;

/// Implementation of the Data Analysis skill.
/// Autonomously handles CSV parsing and tabular data summarization.
pub struct DataAnalysisSkill;

impl DataAnalysisSkill {
    pub fn new() -> Self {
        Self
    }
}

impl MudSkill for DataAnalysisSkill {
    fn name(&self) -> &str {
        "data_analysis"
    }

    fn should_activate(&self, _x: &[f32], context: &str) -> bool {
        let keywords = ["csv", "table", "analyze", "data", "rows", "columns", "summary"];
        keywords.iter().any(|&k| context.to_lowercase().contains(k)) || context.contains(".csv")
    }

    fn execute_autonomous_action(&self, context: &str, engine: &MudInference) {
        for word in context.split_whitespace() {
            if word.ends_with(".csv") {
                let clean_path = word.trim_matches(|c| c == '(' || c == ')' || c == ',' || c == '\"' || c == '\'');
                if Path::new(clean_path).exists() {
                    println!("  [MUD Auto-Action] Analyzing data file: {}...", clean_path);
                    if let Ok(summary) = self.analyze_csv(clean_path) {
                        // Inject summary into Knowledge Graph
                        let mut graph = engine.model.knowledge_graph.write().unwrap();
                        let embedding = vec![0.5; engine.model.hidden_size]; // Placeholder embedding
                        graph.add_node(summary, embedding);
                        println!("  [MUD Auto-Action] Data summary added to Knowledge Graph.");
                    }
                }
            }
        }
    }
}

impl DataAnalysisSkill {
    fn analyze_csv(&self, path: &str) -> anyhow::Result<String> {
        let content = fs::read_to_string(path)?;
        let lines: Vec<&str> = content.lines().collect();
        if lines.is_empty() { return Ok("Empty CSV file.".to_string()); }

        let header = lines[0];
        let row_count = lines.len().saturating_sub(1);
        let columns: Vec<&str> = header.split(',').map(|s| s.trim()).collect();

        Ok(format!(
            "Data Summary for {}: Columns: {:?}. Total rows: {}. First 3 rows of data present.",
            path, columns, row_count
        ))
    }
}
