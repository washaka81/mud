use crate::mud::inference::MudInference;
use crate::mud::skills::MudSkill;
use crate::mud::workspace_agent::AgentWorkspace;

/// Implementation of the Data Analysis skill.
/// Autonomously handles CSV parsing and tabular data summarization.
///
/// R-37 (SEC-SKILL-01): this skill never touches the filesystem directly.
/// Every read goes through [`AgentWorkspace::read_file`], which confines the
/// path to the mounted workspace root (rejects absolute paths, `..` and
/// symlink escapes — see `workspace_agent::resolve_inside_root`). Without a
/// mounted workspace the skill is inert (fail-closed).
pub struct DataAnalysisSkill {
    /// Workspace the skill is allowed to read from. `None` => no filesystem
    /// access at all.
    workspace: Option<AgentWorkspace>,
}

impl Default for DataAnalysisSkill {
    fn default() -> Self {
        Self::new()
    }
}

impl DataAnalysisSkill {
    pub fn new() -> Self {
        Self { workspace: None }
    }

    /// Confined constructor: reads are limited to `root`.
    pub fn with_workspace<P: AsRef<std::path::Path>>(root: P) -> std::io::Result<Self> {
        Ok(Self {
            workspace: Some(AgentWorkspace::mount(root)?),
        })
    }
}

impl MudSkill for DataAnalysisSkill {
    fn name(&self) -> &str {
        "data_analysis"
    }

    fn should_activate(&self, _x: &[f32], context: &str) -> bool {
        let keywords = [
            "csv", "table", "analyze", "data", "rows", "columns", "summary",
        ];
        keywords.iter().any(|&k| context.to_lowercase().contains(k)) || context.contains(".csv")
    }

    fn execute_autonomous_action(&self, context: &str, _engine: &MudInference) {
        // R-37: no workspace mounted => never read anything.
        let Some(ws) = &self.workspace else {
            return;
        };
        for word in context.split_whitespace() {
            if word.ends_with(".csv") {
                let clean_path = word
                    .trim_matches(|c| c == '(' || c == ')' || c == ',' || c == '\"' || c == '\'');
                match ws.read_file(clean_path) {
                    Ok(content) => {
                        println!("  [MUD Auto-Action] Analyzing data file: {}...", clean_path);
                        if let Ok(_summary) = self.analyze_csv(clean_path, &content) {
                            println!("  [MUD Auto-Action] Data analyzed. (Knowledge Graph disabled)");
                        }
                    }
                    // Outside the root / unreadable => log and skip (fail-fast P-17
                    // without stderr spam; code intentionally stays inert).
                    Err(_) => {
                        log::debug!(
                            "data_analysis: refused to read '{clean_path}' (outside workspace root or unreadable)"
                        );
                    }
                }
            }
        }
    }
}

impl DataAnalysisSkill {
    fn analyze_csv(&self, path: &str, content: &str) -> anyhow::Result<String> {
        let lines: Vec<&str> = content.lines().collect();
        if lines.is_empty() {
            return Ok("Empty CSV file.".to_string());
        }

        let header = lines[0];
        let row_count = lines.len().saturating_sub(1);
        let columns: Vec<&str> = header.split(',').map(|s| s.trim()).collect();

        Ok(format!(
            "Data Summary for {}: Columns: {:?}. Total rows: {}. First 3 rows of data present.",
            path, columns, row_count
        ))
    }
}
