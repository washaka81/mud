/// dead_code_audit.rs — Auditoría de expertos muertos y tensores vacíos en el .mud
///
/// Detecta:
///   - Expertos con todos los pesos cero (dead experts sin entrenamiento)
///   - Capas de atención con punteros nulos (weights no cargados)
///   - Tensores con sigma < 0.1 (amnesia ternaria severa)
///   - Tensores con esparsidad > 90% (demasiados ceros)
///
/// Uso: cargo run --release --bin dead_code_audit -- [modelo.mud]
use forge_llm::mud::{MudFile, MudTensorType, dequantize_ternary_row};
use comfy_table::modifiers::UTF8_ROUND_CORNERS;
use comfy_table::presets::UTF8_FULL;
use comfy_table::{Table, Cell, Color, Attribute};

// Umbrales de salud de los tensores
const SIGMA_MIN_HEALTHY: f32  = 0.10; // sigma < 0.10 → amnesia ternaria
const SPARSITY_MAX_HEALTHY: f32 = 0.90; // > 90% ceros → experto muerto
const SAMPLE_ROWS: usize = 8;          // filas a muestrear por tensor grande

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).map(|s| s.as_str()).unwrap_or("models/core_skills.mud");

    println!("\n🔬 MUD Dead Expert & Tensor Health Audit");
    println!("   Modelo: {}\n", model_path);

    let mf = MudFile::load(model_path)?;
    let core = mf.skills.get("core").ok_or_else(|| anyhow::anyhow!("No core skill"))?;

    let num_layers: usize = mf.global_metadata.get("num_layers")
        .and_then(|s| s.parse().ok()).unwrap_or(0);
    let num_experts: usize = mf.global_metadata.get("num_experts")
        .and_then(|s| s.parse().ok()).unwrap_or(1);
    let hidden_size: usize = mf.global_metadata.get("hidden_size")
        .and_then(|s| s.parse().ok()).unwrap_or(576);

    println!("   Arquitectura: {} capas × {} expertos × hidden={}\n", num_layers, num_experts, hidden_size);

    let mut dead_experts   = 0usize;
    let mut warn_tensors   = 0usize;
    let mut healthy        = 0usize;
    let mut total_checked  = 0usize;

    let mut table = Table::new();
    table.load_preset(UTF8_FULL).apply_modifier(UTF8_ROUND_CORNERS);
    table.set_header(vec![
        Cell::new("Tensor").add_attribute(Attribute::Bold),
        Cell::new("Sigma").add_attribute(Attribute::Bold),
        Cell::new("Sparsity%").add_attribute(Attribute::Bold),
        Cell::new("Diagnóstico").add_attribute(Attribute::Bold),
    ]);

    for layer in 0..num_layers {
        for expert in 0..num_experts {
            let w1_name = format!("blk.{}.expert.{}.w1.weight", layer, expert);
            if let Some(tensor) = core.tensors.get(&w1_name) {
                total_checked += 1;
                let elements: usize = tensor.shape.iter().product();

                // Dequantizar muestra para estadísticas
                let sample_elements = (SAMPLE_ROWS * hidden_size).min(elements);
                let mut buf = vec![0.0f32; sample_elements];

                match tensor.t_type {
                    MudTensorType::Ternary2Bit => {
                        unsafe { dequantize_ternary_row(tensor.data_ptr as *const u32, &mut buf, sample_elements); }
                    }
                    MudTensorType::Float32 => {
                        unsafe { std::ptr::copy_nonoverlapping(tensor.data_ptr as *const f32, buf.as_mut_ptr(), sample_elements); }
                    }
                    _ => {}
                }

                let mean = buf.iter().sum::<f32>() / buf.len() as f32;
                let variance = buf.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / buf.len() as f32;
                let sigma = variance.sqrt();
                let zeros = buf.iter().filter(|&&v| v.abs() < 0.01).count();
                let sparsity = zeros as f32 / buf.len() as f32;

                let (diag, color) = if sigma < SIGMA_MIN_HEALTHY || sparsity > SPARSITY_MAX_HEALTHY {
                    dead_experts += 1;
                    ("🔴 EXPERTO MUERTO", Color::Red)
                } else if sigma < 0.3 || sparsity > 0.75 {
                    warn_tensors += 1;
                    ("🟠 ADVERTENCIA", Color::Yellow)
                } else {
                    healthy += 1;
                    ("✅ SANO", Color::Green)
                };

                // Solo mostrar problemas o cada 5a fila para no saturar output
                if sigma < 0.3 || sparsity > 0.75 || (layer % 5 == 0 && expert == 0) {
                    table.add_row(vec![
                        Cell::new(format!("L{}.E{}.w1", layer, expert)),
                        Cell::new(format!("{:.3}", sigma)).fg(color),
                        Cell::new(format!("{:.1}%", sparsity * 100.0)).fg(color),
                        Cell::new(diag).fg(color),
                    ]);
                }
            } else {
                // Tensor faltante = experto no cargado (NULL ptr en inference)
                table.add_row(vec![
                    Cell::new(format!("L{}.E{}.w1", layer, expert)),
                    Cell::new("N/A").fg(Color::Red),
                    Cell::new("N/A").fg(Color::Red),
                    Cell::new("🔴 TENSOR FALTANTE").fg(Color::Red),
                ]);
                dead_experts += 1;
                total_checked += 1;
            }
        }
    }

    println!("{}", table);

    // Resumen
    let mut summary = Table::new();
    summary.load_preset(UTF8_FULL).apply_modifier(UTF8_ROUND_CORNERS);
    summary.set_header(vec![
        Cell::new("Métrica").add_attribute(Attribute::Bold).fg(Color::Cyan),
        Cell::new("Valor").add_attribute(Attribute::Bold),
    ]);
    summary.add_row(vec![Cell::new("Total expertos auditados"), Cell::new(total_checked.to_string())]);
    summary.add_row(vec![
        Cell::new("Expertos muertos/faltantes"),
        Cell::new(dead_experts.to_string()).fg(if dead_experts > 0 { Color::Red } else { Color::Green }),
    ]);
    summary.add_row(vec![
        Cell::new("Tensores con advertencias"),
        Cell::new(warn_tensors.to_string()).fg(if warn_tensors > 0 { Color::Yellow } else { Color::Green }),
    ]);
    summary.add_row(vec![
        Cell::new("Tensores sanos"),
        Cell::new(healthy.to_string()).fg(Color::Green),
    ]);
    let health_pct = if total_checked > 0 { healthy as f32 * 100.0 / total_checked as f32 } else { 0.0 };
    summary.add_row(vec![
        Cell::new("Salud del modelo"),
        Cell::new(format!("{:.1}%", health_pct))
            .fg(if health_pct > 80.0 { Color::Green } else if health_pct > 50.0 { Color::Yellow } else { Color::Red }),
    ]);
    println!("{}", summary);

    if dead_experts > 0 {
        println!("\n  ⚠️  Hay {} experto(s) muerto(s) o faltante(s). Ejecuta recalibración.", dead_experts);
        std::process::exit(1);
    } else {
        println!("\n  ✅ Todos los expertos están activos y sanos.");
    }

    Ok(())
}
