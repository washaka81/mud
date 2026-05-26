use comfy_table::modifiers::UTF8_ROUND_CORNERS;
use comfy_table::presets::UTF8_FULL;
use comfy_table::{Table, Cell, Color, Attribute};

fn main() {
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_header(vec![
            Cell::new("MUD ENGINE").add_attribute(Attribute::Bold).fg(Color::Magenta),
            Cell::new("VERSION 1.58b").add_attribute(Attribute::Bold).fg(Color::Cyan),
        ]);

    table.add_row(vec!["Architecture", "Ternary 1.58-bit MoE"]);
    table.add_row(vec!["Status", "Ready for Inference"]);
    table.add_row(vec!["Optimization", "Zero-Allocation / Hybrid Zero-Copy"]);

    println!("{}", table);
}
