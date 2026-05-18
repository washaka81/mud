use crate::ai::skills::MudSkill;
use crate::ai::inference::MudInference;

/// Implementation of the ASCII Plotting skill.
/// Renders simple bar charts from numeric data.
pub struct PlottingSkill;

impl PlottingSkill {
    pub fn new() -> Self {
        Self
    }
}

impl MudSkill for PlottingSkill {
    fn name(&self) -> &str {
        "plotting"
    }

    fn should_activate(&self, _x: &[f32], context: &str) -> bool {
        let keywords = ["plot", "chart", "graph", "histogram", "visualize data"];
        keywords.iter().any(|&k| context.to_lowercase().contains(k))
    }

    fn execute_autonomous_action(&self, context: &str, _engine: &MudInference) {
        // Detect numeric lists for plotting: e.g. [1, 5, 3, 8]
        if let Some(start) = context.find('[') {
            if let Some(end) = context.find(']') {
                let nums_str = &context[start + 1..end];
                let values: Vec<f32> = nums_str.split(',')
                    .filter_map(|s| s.trim().parse::<f32>().ok())
                    .collect();

                if !values.is_empty() {
                    println!("\n  [MUD Chart] ASCII Visualization:");
                    self.render_bar_chart(&values);
                }
            }
        }
    }
}

impl PlottingSkill {
    fn render_bar_chart(&self, values: &[f32]) {
        let max_val = values.iter().cloned().fold(0.0f32, f32::max);
        let max_width = 40;

        for (i, &val) in values.iter().enumerate() {
            let bar_len = ((val / max_val) * max_width as f32) as usize;
            let bar = "█".repeat(bar_len);
            println!("  Val {:02} | {:<40} | {:.2}", i + 1, bar, val);
        }
        println!();
    }
}
