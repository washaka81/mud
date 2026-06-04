/// training_health.rs — Auditoría de salud del modelo post-entrenamiento
///
/// Verifica que la distribución ternaria de TODOS los expertos se mantenga
/// en el rango saludable {37% +1, 26% 0, 37% -1} y que el sigma esté
/// entre 0.75 y 0.95 (con objetivo de 0.86 para formato MUD ternario).
///
/// También verifica:
///   - Que el weight decay no haya colapsado pesos a cero (BUG-6 check)
///   - Coherencia entre capas: sigma no debe variar > 0.2 entre capas consecutivas
///   - Escala promedio por capa (indica si las dequantizaciones son válidas)
///
/// Uso: cargo run --release --bin training_health -- [modelo.mud]
use forge_llm::mud::{MudFile, MudTensorType, dequantize_ternary_row};
use comfy_table::modifiers::UTF8_ROUND_CORNERS;
use comfy_table::presets::UTF8_FULL;
use comfy_table::{Table, Cell, Color, Attribute};

// Rangos saludables según MUD_OVERVIEW.md §6
const SIGMA_MIN: f32 = 0.75;
const SIGMA_MAX: f32 = 0.95;
const POS_TARGET: f32 = 0.37;  // fracción ideal de +1
const ZERO_TARGET: f32 = 0.26; // fracción ideal de 0
const NEG_TARGET: f32 = 0.37;  // fracción ideal de -1
const TOLERANCE: f32 = 0.12;   // ±12% de tolerancia sobre cada fracción

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).map(|s| s.as_str()).unwrap_or("models/core_skills.mud");

    println!("\n🩺 MUD Training Health — Auditoría de Distribución Ternaria Post-Entrenamiento");
    println!("   Modelo: {}\n", model_path);

    let mf = MudFile::load(model_path)?;
    let core = mf.skills.get("core").ok_or_else(|| anyhow::anyhow!("No core skill"))?;

    let num_layers: usize = mf.global_metadata.get("num_layers")
        .and_then(|s| s.parse().ok()).unwrap_or(0);
    let num_experts: usize = mf.global_metadata.get("num_experts")
        .and_then(|s| s.parse().ok()).unwrap_or(1);

    let mut global_pos   = 0u64;
    let mut global_zero  = 0u64;
    let mut global_neg   = 0u64;
    let mut global_total = 0u64;
    let mut all_sigmas: Vec<f32> = Vec::new();
    let mut layer_sigmas: Vec<f32> = Vec::new();
    let mut issues = 0usize;

    let mut table = Table::new();
    table.load_preset(UTF8_FULL).apply_modifier(UTF8_ROUND_CORNERS);
    table.set_header(vec![
        Cell::new("Capa").add_attribute(Attribute::Bold),
        Cell::new("Experto").add_attribute(Attribute::Bold),
        Cell::new("Peso").add_attribute(Attribute::Bold),
        Cell::new("Sigma").add_attribute(Attribute::Bold),
        Cell::new("+1%").add_attribute(Attribute::Bold),
        Cell::new("0%").add_attribute(Attribute::Bold),
        Cell::new("-1%").add_attribute(Attribute::Bold),
        Cell::new("Estado").add_attribute(Attribute::Bold),
    ]);

    for layer in 0..num_layers {
        let mut layer_sigma_acc = 0.0f32;
        let mut layer_count = 0usize;

        for expert in 0..num_experts {
            for weight_key in &["w1", "w2", "w3"] {
                let name = format!("blk.{}.expert.{}.{}.weight", layer, expert, weight_key);
                let tensor = match core.tensors.get(&name) {
                    Some(t) => t,
                    None => continue,
                };

                let elements: usize = tensor.shape.iter().product();
                let mut buf = vec![0.0f32; elements];

                match tensor.t_type {
                    MudTensorType::Ternary2Bit => {
                        unsafe { dequantize_ternary_row(tensor.data_ptr as *const u32, &mut buf, elements); }
                    }
                    MudTensorType::Float32 => {
                        unsafe { std::ptr::copy_nonoverlapping(tensor.data_ptr as *const f32, buf.as_mut_ptr(), elements); }
                    }
                    _ => continue,
                }

                let pos   = buf.iter().filter(|&&v| v >  0.5).count() as u64;
                let zero  = buf.iter().filter(|&&v| v.abs() <= 0.5).count() as u64;
                let neg   = buf.iter().filter(|&&v| v < -0.5).count() as u64;
                let total = buf.len() as u64;

                global_pos   += pos;
                global_zero  += zero;
                global_neg   += neg;
                global_total += total;

                let mean = buf.iter().sum::<f32>() / total as f32;
                let var  = buf.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / total as f32;
                let sigma = var.sqrt();
                all_sigmas.push(sigma);
                layer_sigma_acc += sigma;
                layer_count += 1;

                let pos_frac  = pos  as f32 / total as f32;
                let zero_frac = zero as f32 / total as f32;
                let neg_frac  = neg  as f32 / total as f32;

                let sigma_ok = sigma >= SIGMA_MIN && sigma <= SIGMA_MAX;
                let pos_ok   = (pos_frac  - POS_TARGET).abs()  < TOLERANCE;
                let zero_ok  = (zero_frac - ZERO_TARGET).abs() < TOLERANCE;
                let neg_ok   = (neg_frac  - NEG_TARGET).abs()  < TOLERANCE;
                let all_ok   = sigma_ok && pos_ok && zero_ok && neg_ok;

                // BUG-6 check: si el sigma es muy bajo y muchos ceros, weight decay colapsó los pesos
                let bug6_collapse = sigma < 0.3 && zero_frac > 0.7;

                let (status_str, status_color) = if bug6_collapse {
                    issues += 1;
                    ("🔴 BUG-6: DECAY COLLAPSE", Color::Red)
                } else if !all_ok {
                    issues += 1;
                    ("🟠 DISTRIBUCIÓN SESGADA", Color::Yellow)
                } else {
                    ("✅", Color::Green)
                };

                // Solo mostrar filas con problemas o primera/última capa
                if !all_ok || bug6_collapse || layer == 0 || layer == num_layers - 1 {
                    let sigma_color = if sigma_ok { Color::Green } else { Color::Red };
                    let pos_color  = if pos_ok  { Color::Green } else { Color::Yellow };
                    let zero_color = if zero_ok { Color::Green } else { Color::Yellow };
                    let neg_color  = if neg_ok  { Color::Green } else { Color::Yellow };

                    table.add_row(vec![
                        Cell::new(layer.to_string()),
                        Cell::new(expert.to_string()),
                        Cell::new(*weight_key),
                        Cell::new(format!("{:.3}", sigma)).fg(sigma_color),
                        Cell::new(format!("{:.1}%", pos_frac * 100.0)).fg(pos_color),
                        Cell::new(format!("{:.1}%", zero_frac * 100.0)).fg(zero_color),
                        Cell::new(format!("{:.1}%", neg_frac * 100.0)).fg(neg_color),
                        Cell::new(status_str).fg(status_color),
                    ]);
                }
            }
        }

        if layer_count > 0 {
            layer_sigmas.push(layer_sigma_acc / layer_count as f32);
        }
    }

    println!("{}", table);

    // Coherencia entre capas (sigma no debe variar > 0.2 entre capas adyacentes)
    println!("\n📊 Coherencia sigma entre capas:");
    let mut coherence_issues = 0usize;
    for i in 1..layer_sigmas.len() {
        let delta = (layer_sigmas[i] - layer_sigmas[i - 1]).abs();
        if delta > 0.2 {
            println!("  🟠 Salto sigma entre capa {} y {}: Δ={:.3}", i-1, i, delta);
            coherence_issues += 1;
        }
    }
    if coherence_issues == 0 {
        println!("  ✅ Sigma estable entre todas las capas.");
    }

    // Resumen global
    let global_pos_frac  = global_pos  as f32 / global_total as f32;
    let global_zero_frac = global_zero as f32 / global_total as f32;
    let global_neg_frac  = global_neg  as f32 / global_total as f32;
    let mean_sigma = all_sigmas.iter().sum::<f32>() / all_sigmas.len().max(1) as f32;
    let min_sigma  = all_sigmas.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_sigma  = all_sigmas.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let mut summary = Table::new();
    summary.load_preset(UTF8_FULL).apply_modifier(UTF8_ROUND_CORNERS);
    summary.set_header(vec![
        Cell::new("Métrica Global").add_attribute(Attribute::Bold).fg(Color::Cyan),
        Cell::new("Valor").add_attribute(Attribute::Bold),
        Cell::new("Objetivo").add_attribute(Attribute::Bold),
    ]);
    summary.add_row(vec![
        Cell::new("Total pesos analizados"),
        Cell::new(format!("{}", global_total)),
        Cell::new("—"),
    ]);
    let pos_col = if (global_pos_frac - POS_TARGET).abs() < TOLERANCE { Color::Green } else { Color::Red };
    summary.add_row(vec![
        Cell::new("Fracción +1 (global)"),
        Cell::new(format!("{:.2}%", global_pos_frac * 100.0)).fg(pos_col),
        Cell::new(format!("{:.0}% ± {:.0}%", POS_TARGET*100.0, TOLERANCE*100.0)),
    ]);
    let zero_col = if (global_zero_frac - ZERO_TARGET).abs() < TOLERANCE { Color::Green } else { Color::Red };
    summary.add_row(vec![
        Cell::new("Fracción 0  (global)"),
        Cell::new(format!("{:.2}%", global_zero_frac * 100.0)).fg(zero_col),
        Cell::new(format!("{:.0}% ± {:.0}%", ZERO_TARGET*100.0, TOLERANCE*100.0)),
    ]);
    let neg_col = if (global_neg_frac - NEG_TARGET).abs() < TOLERANCE { Color::Green } else { Color::Red };
    summary.add_row(vec![
        Cell::new("Fracción -1 (global)"),
        Cell::new(format!("{:.2}%", global_neg_frac * 100.0)).fg(neg_col),
        Cell::new(format!("{:.0}% ± {:.0}%", NEG_TARGET*100.0, TOLERANCE*100.0)),
    ]);
    let sigma_col = if mean_sigma >= SIGMA_MIN && mean_sigma <= SIGMA_MAX { Color::Green } else { Color::Red };
    summary.add_row(vec![
        Cell::new("Sigma media global"),
        Cell::new(format!("{:.3} (min {:.3} / max {:.3})", mean_sigma, min_sigma, max_sigma)).fg(sigma_col),
        Cell::new(format!("{:.2} – {:.2}", SIGMA_MIN, SIGMA_MAX)),
    ]);
    summary.add_row(vec![
        Cell::new("Problemas detectados"),
        Cell::new(format!("{}", issues + coherence_issues))
            .fg(if issues + coherence_issues == 0 { Color::Green } else { Color::Red }),
        Cell::new("0"),
    ]);
    println!("{}", summary);

    if issues + coherence_issues > 0 {
        println!("\n  ⚠️  El modelo necesita recalibración. Ejecuta: ./mud.sh align");
        std::process::exit(1);
    } else {
        println!("\n  ✅ Distribución ternaria sana. El entrenamiento no colapsó los pesos.");
    }

    Ok(())
}
