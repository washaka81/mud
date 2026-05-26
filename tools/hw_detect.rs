use forge_llm::hardware::HardwareProfile;
use comfy_table::modifiers::UTF8_ROUND_CORNERS;
use comfy_table::presets::UTF8_FULL;
use comfy_table::{Table, Cell, Color, Attribute};

fn main() {
    let hw = HardwareProfile::detect();
    
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .apply_modifier(UTF8_ROUND_CORNERS)
        .set_header(vec![
            Cell::new("PROPERTY").add_attribute(Attribute::Bold).fg(Color::Magenta),
            Cell::new("DETECTED VALUE").add_attribute(Attribute::Bold).fg(Color::Cyan),
        ]);

    table.add_row(vec!["CPU Brand", &hw.cpu_brand.trim()]);
    table.add_row(vec!["Total Cores", &hw.total_cores.to_string()]);
    
    if hw.is_intel_hybrid {
        table.add_row(vec![
            Cell::new("Architecture").fg(Color::Yellow),
            Cell::new("Intel Hybrid (P+E Cores)").fg(Color::Yellow),
        ]);
        table.add_row(vec!["Performance (P) Threads", &(hw.p_cores).to_string()]);
        table.add_row(vec!["Efficiency (E) Threads", &(hw.e_cores).to_string()]);
        table.add_row(vec![
            Cell::new("Optimized Work Pool").fg(Color::Green),
            Cell::new(format!("{} threads", hw.preferred_threads)).fg(Color::Green),
        ]);
    } else {
        table.add_row(vec!["Architecture", "Standard Uniform"]);
        table.add_row(vec!["Optimized Threads", &hw.preferred_threads.to_string()]);
    }

    let avx2_cell = if hw.has_avx2 { Cell::new("TRUE").fg(Color::Green) } else { Cell::new("FALSE").fg(Color::Red) };
    let avx512_cell = if hw.has_avx512 { Cell::new("TRUE").fg(Color::Green) } else { Cell::new("FALSE").fg(Color::Red) };

    table.add_row(vec![Cell::new("SIMD: AVX2"), avx2_cell]);
    table.add_row(vec![Cell::new("SIMD: AVX-512"), avx512_cell]);
    table.add_row(vec!["L3 Cache (Est)", &format!("{} KB", hw.l3_cache_kb)]);

    println!("\n  🔍 MUD HARDWARE TOPOLOGY REPORT");
    println!("{}", table);
}
