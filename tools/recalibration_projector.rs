use forge_llm::mud::{MudFile, MudTensorType};
use forge_llm::hardware::HardwareProfile;
use comfy_table::modifiers::UTF8_ROUND_CORNERS;
use comfy_table::presets::UTF8_FULL;
use comfy_table::{Table, Cell, Color, Attribute};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).map(|s| s.as_str()).unwrap_or("models/core_skills.mud");
    
    let mud = MudFile::load(model_path)?;
    let _hw = HardwareProfile::detect();

    let mut ternary_weights = 0usize;
    let mut zero_weights = 0usize;
    let mut scales_sum = 0.0;
    let mut scale_count = 0;

    if let Some(core) = mud.skills.get("core") {
        for (name, tensor) in &core.tensors {
            if tensor.t_type == MudTensorType::Ternary2Bit {
                let elements: usize = tensor.shape.iter().product();
                ternary_weights += elements;
                let u32_count = elements.div_ceil(16);
                let ptr = tensor.data_ptr as *const u32;
                let sample_size = u32_count.min(1000);
                let mut local_zero = 0;
                for i in 0..sample_size {
                    let val = unsafe { *ptr.add(i) };
                    for j in 0..16 {
                        if (val >> (j * 2)) & 3 == 0 { local_zero += 1; }
                    }
                }
                zero_weights += (local_zero as f32 * (u32_count as f32 / sample_size as f32)) as usize;
            } else if name.ends_with(".scale") {
                let val = unsafe { *(tensor.data_ptr as *const f32) };
                scales_sum += val;
                scale_count += 1;
            }
        }
    }

    if ternary_weights == 0 { ternary_weights = 1; }
    let avg_sparsity = (zero_weights as f32 / ternary_weights as f32) * 100.0;
    let model_certainty = (100.0 - ((avg_sparsity - 33.0).abs() / 100.0 * 100.0)).clamp(0.0, 100.0);

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_header(vec![
            Cell::new("MUD DETERMINISTIC RECALIBRATION PROJECTOR").add_attribute(Attribute::Bold).fg(Color::Yellow),
            Cell::new("ANALYSIS").add_attribute(Attribute::Bold).fg(Color::Cyan),
        ]);

    table.add_row(vec!["Target Model", model_path]);
    table.add_row(vec!["Ternary Weights", &format!("{}M", ternary_weights / 1_000_000)]);
    table.add_row(vec!["Average Sparsity", &format!("{:.2}%", avg_sparsity)]);
    table.add_row(vec![
        Cell::new("Model Certainty Score").fg(Color::Magenta),
        Cell::new(format!("{:.2}%", model_certainty)).fg(Color::Magenta),
    ]);

    let initial_divergence = 100.0 - model_certainty; 
    let reduction_per_epoch = (model_certainty / 100.0) * 0.85;
    let mut current_div = initial_divergence;
    let mut epochs_required = 0;
    let mut trajectory = Vec::new();

    while current_div > 5.0 && epochs_required < 10 {
        epochs_required += 1;
        current_div *= 1.0 - reduction_per_epoch;
        trajectory.push((100.0 - current_div).clamp(0.0, 100.0));
    }

    table.add_row(vec![
        Cell::new("OPTIMAL EPOCHS REQUIRED").add_attribute(Attribute::Bold).fg(Color::Green),
        Cell::new(epochs_required.to_string()).add_attribute(Attribute::Bold).fg(Color::Green),
    ]);
    let final_prob = trajectory.last().unwrap_or(&0.0);
    table.add_row(vec![
        Cell::new("FINAL COHERENCE PROBABILITY").fg(Color::Cyan),
        Cell::new(format!("{:.1}%", final_prob)).fg(Color::Cyan),
    ]);

    println!("\n  🔮 RECALIBRATION PROJECTION REPORT");
    println!("{}", table);

    println!("\n📈 Convergence Confidence Map:");
    for (i, prob) in trajectory.iter().enumerate() {
        let bar_len = (*prob / 5.0) as usize;
        let bar = "█".repeat(bar_len) + &"░".repeat(20usize.saturating_sub(bar_len));
        let color = if *prob > 95.0 { "\x1b[1;32m" } else if *prob > 75.0 { "\x1b[1;33m" } else { "\x1b[1;31m" };
        println!("   Epoch {:>2} | {}{:>5.1}% [{}]\x1b[0m", i + 1, color, prob, bar);
    }
    
    Ok(())
}
