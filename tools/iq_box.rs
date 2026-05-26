use comfy_table::modifiers::UTF8_ROUND_CORNERS;
use comfy_table::presets::UTF8_FULL;
use comfy_table::{Table, Cell, Color, Attribute};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let iq = args.get(1).cloned().unwrap_or_else(|| "8.87".to_string());
    let label = args.get(2).cloned().unwrap_or_else(|| "COGNICIÓN FRAGMENTADA".to_string());
    
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS);

    let iq_val = iq.parse::<f32>().unwrap_or(0.0);
    let iq_color = if iq_val < 15.0 { Color::Yellow } else if iq_val < 100.0 { Color::Cyan } else { Color::Green };

    table.add_row(vec![
        Cell::new("IQ SCORE").add_attribute(Attribute::Bold),
        Cell::new(iq).fg(iq_color).add_attribute(Attribute::Bold),
    ]);
    table.add_row(vec![
        Cell::new("STATE"),
        Cell::new(label).fg(iq_color),
    ]);

    println!("{}", table);
}
