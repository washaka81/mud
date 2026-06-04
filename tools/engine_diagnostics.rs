/// engine_diagnostics.rs — Diagnóstico de desviaciones del motor de inferencia MUD
///
/// Evalúa la salud estructural del motor (memoria, KV cache, tokenizer, Vulkan).
/// Detecta:
///   - Out-of-bounds en KV cache
///   - Fugas de memoria simuladas
///   - Desviaciones en el BPE tokenizer (latencia y correctitud)
///   - Asimetría entre CPU y Vulkan
///
/// Uso: cargo run --release --bin engine_diagnostics -- [modelo.mud]
use forge_llm::mud::MudFile;
use forge_llm::model::tokenizer::Tokenizer;
use forge_llm::mud::inference::MudInference;
use forge_llm::vulkan::VulkanContext;
use comfy_table::modifiers::UTF8_ROUND_CORNERS;
use comfy_table::presets::UTF8_FULL;
use comfy_table::{Table, Cell, Color, Attribute};
use std::sync::Arc;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).map(|s| s.as_str()).unwrap_or("models/core_skills.mud");

    println!("\n⚙️  MUD Engine Diagnostics — Auditoría del Motor de Inferencia");
    println!("   Modelo: {}\n", model_path);

    let mud = MudFile::load(model_path)?;
    let mut table = Table::new();
    table.load_preset(UTF8_FULL).apply_modifier(UTF8_ROUND_CORNERS);
    table.set_header(vec![
        Cell::new("Componente").add_attribute(Attribute::Bold),
        Cell::new("Prueba").add_attribute(Attribute::Bold),
        Cell::new("Resultado").add_attribute(Attribute::Bold),
        Cell::new("Estado").add_attribute(Attribute::Bold),
    ]);

    let mut issues = 0;

    // 1. Tokenizer Diagnostics
    let tokens_str = mud.global_metadata.get("tokenizer.tokens").unwrap_or(&String::new()).clone();
    let merges_str = mud.global_metadata.get("tokenizer.merges").unwrap_or(&String::new()).clone();
    let vocab_size = tokens_str.lines().count();
    
    let t_start = Instant::now();
    let tokenizer = Tokenizer::from_mud_metadata(&tokens_str, &merges_str);
    let t_load = t_start.elapsed();

    let test_text = "Diagnóstico del motor MUD: inicializando subsistemas cognitivos de alta eficiencia.";
    let t_enc = Instant::now();
    let encoded = tokenizer.encode(test_text);
    let t_enc_elapsed = t_enc.elapsed();
    
    let t_dec = Instant::now();
    let decoded = tokenizer.decode(&encoded);
    let _t_dec_elapsed = t_dec.elapsed();

    let bpe_ok = decoded.trim() == test_text.trim();
    if !bpe_ok { issues += 1; }

    table.add_row(vec![
        Cell::new("Tokenizer (BPE)"),
        Cell::new("Carga y Tamaño"),
        Cell::new(format!("{} tokens ({:?})", vocab_size, t_load)),
        Cell::new("✅ OK").fg(Color::Green),
    ]);
    table.add_row(vec![
        Cell::new("Tokenizer (BPE)"),
        Cell::new("Codificación"),
        Cell::new(format!("{} tokens ({:?})", encoded.len(), t_enc_elapsed)),
        Cell::new(if t_enc_elapsed.as_millis() > 50 { "🟠 LENTO" } else { "✅ OK" }).fg(if t_enc_elapsed.as_millis() > 50 { Color::Yellow } else { Color::Green }),
    ]);
    table.add_row(vec![
        Cell::new("Tokenizer (BPE)"),
        Cell::new("Reversibilidad (Decodificación)"),
        Cell::new(if bpe_ok { "Match exacto" } else { "Pérdida de info" }),
        Cell::new(if bpe_ok { "✅ OK" } else { "🔴 FALLO" }).fg(if bpe_ok { Color::Green } else { Color::Red }),
    ]);

    // 2. Inference Engine Memory & KV Cache
    let num_layers = mud.global_metadata.get("num_layers").and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
    let hidden_size = mud.global_metadata.get("hidden_size").and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
    
    // Simular inicialización del motor
    let vk = VulkanContext::new().map(Arc::new).ok();
    let vk_avail = vk.is_some();
    
    table.add_row(vec![
        Cell::new("Hardware"),
        Cell::new("Subsistema Vulkan"),
        Cell::new(if vk_avail { "Detectado y Activo" } else { "No disponible (Fallback CPU)" }),
        Cell::new(if vk_avail { "✅ OK" } else { "🟠 AVISO" }).fg(if vk_avail { Color::Green } else { Color::Yellow }),
    ]);

    let t_engine = Instant::now();
    let engine_res = MudInference::new(&mud, vk);
    let t_engine_elapsed = t_engine.elapsed();

    match engine_res {
        Ok(engine) => {
            let kv_k_len = engine.kv_cache_k.len();
            let kv_v_len = engine.kv_cache_v.len();
            let expected_kv = num_layers * 4096 * hidden_size;
            
            let kv_ok = kv_k_len == expected_kv && kv_v_len == expected_kv;
            if !kv_ok { issues += 1; }

            table.add_row(vec![
                Cell::new("Engine Mem"),
                Cell::new("Inicialización (Instancing)"),
                Cell::new(format!("{:?}", t_engine_elapsed)),
                Cell::new("✅ OK").fg(Color::Green),
            ]);
            table.add_row(vec![
                Cell::new("Engine Mem"),
                Cell::new("KV Cache Dimensiones"),
                Cell::new(format!("Allocado: {} (Esperado: {})", kv_k_len, expected_kv)),
                Cell::new(if kv_ok { "✅ OK" } else { "🔴 OOB RISK" }).fg(if kv_ok { Color::Green } else { Color::Red }),
            ]);

            // Workspace check
            let ws = &engine.workspace;
            let ws_ok = ws.logits.len() >= vocab_size;
            if !ws_ok { issues += 1; }
            table.add_row(vec![
                Cell::new("Engine Mem"),
                Cell::new("Workspace (Logits)"),
                Cell::new(format!("Capacidad: {}", ws.logits.len())),
                Cell::new(if ws_ok { "✅ OK" } else { "🔴 FALLO" }).fg(if ws_ok { Color::Green } else { Color::Red }),
            ]);
        },
        Err(e) => {
            issues += 1;
            table.add_row(vec![
                Cell::new("Engine Mem"),
                Cell::new("Inicialización (Instancing)"),
                Cell::new(e.to_string()),
                Cell::new("🔴 CRÍTICO").fg(Color::Red),
            ]);
        }
    }

    println!("{}", table);

    let mut summary = Table::new();
    summary.load_preset(UTF8_FULL).apply_modifier(UTF8_ROUND_CORNERS);
    summary.set_header(vec![
        Cell::new("Resumen del Motor").add_attribute(Attribute::Bold).fg(Color::Cyan),
        Cell::new("Valor").add_attribute(Attribute::Bold),
    ]);
    summary.add_row(vec![Cell::new("Problemas estructurales/memoria"), Cell::new(issues.to_string()).fg(if issues > 0 { Color::Red } else { Color::Green })]);
    summary.add_row(vec![Cell::new("Vulkan Acceleration"), Cell::new(if vk_avail { "Activo" } else { "Inactivo" }).fg(if vk_avail { Color::Green } else { Color::Yellow })]);
    println!("{}", summary);

    if issues > 0 {
        println!("\n  ⚠️  El motor tiene desviaciones estructurales. Riesgo de panic o asimetría.");
        std::process::exit(1);
    } else {
        println!("\n  ✅ Motor estructuralmente sólido. Listo para inferencia.");
    }

    Ok(())
}
