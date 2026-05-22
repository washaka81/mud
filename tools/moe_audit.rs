/// moe_audit.rs — Auditor MoE para 256 Micro-Expertos en 16 Clústeres Funcionales
/// ==================================================================================
/// Verifica salud, balance de carga, diversidad de activación y coherencia ternaria
/// del sistema MoE hiper-granular del SLIME ENGINE.
use forge_llm::mud::MudFile;
use forge_llm::mud::inference::MudInference;
use forge_llm::vulkan::VulkanContext;
use std::sync::Arc;

// ── Tabla Maestra de los 16 Clústeres Funcionales ──────────────────────────────
const CLUSTER_NAMES: [&str; 16] = [
    "Planificación & CoT",      // E001-E016
    "Lógica Formal & Simbólica",// E017-E032
    "El Evaluador Interno",     // E033-E048
    "Razonamiento Difuso",      // E049-E064
    "Gramática & Sintaxis AST", // E065-E080
    "Optimización & Bajo Nivel",// E081-E096
    "Algoritmia Avanzada",      // E097-E112
    "Álgebra Lineal Comp.",     // E113-E128
    "Cálculo & Sistemas Din.",  // E129-E144
    "Análisis Estadístico",     // E145-E160
    "Física Cuántica",          // E161-E176
    "Mecánica Clásica & Termo.",// E177-E192
    "Química Molecular",        // E193-E208
    "Bioinformática & Genética",// E209-E224
    "Sistemas Complejos",       // E225-E240
    "Taxonomías & Datos Fact.", // E241-E256
];

const NUM_EXPERTS: usize = 256;
const CLUSTER_SIZE: usize = 16;
const NUM_CLUSTERS: usize = 16;

fn main() -> anyhow::Result<()> {
    println!("\x1b[1;35m");
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║     MUD SLIME ENGINE — MOE AUDIT: 256 MICRO-EXPERTOS / 16 CLÚSTERES ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\x1b[0m");

    let mud_path = std::env::args().nth(1).unwrap_or_else(|| "models/core_skills.mud".to_string());
    let vk = Arc::new(VulkanContext::new()?);
    let mud_file = MudFile::load(&mud_path)?;
    let mut engine = MudInference::new(&mud_file, vk)?;

    let n_layers  = engine.model.layers.len();
    let n_experts = engine.model.num_experts;
    let hidden    = engine.model.hidden_size;

    println!("\n\x1b[1;34m[CONFIG]\x1b[0m Modelo: {} | Capas: {} | Expertos registrados: {} | Hidden: {}",
             mud_path, n_layers, n_experts, hidden);

    if n_experts != NUM_EXPERTS {
        println!("\x1b[1;33m  ⚠ NOTA: El modelo tiene {} expertos (esperados {}). \
                  La tabla de clústeres se mapea proporcionalmente.\x1b[0m",
                 n_experts, NUM_EXPERTS);
    }

    // ── TEST 1: Anatomía por Clúster (magnitud de pesos) ─────────────────────
    println!("\n\x1b[1;34m[TEST 1] Anatomía de Pesos por Clúster Funcional\x1b[0m");
    println!("{:<3} {:<28} {:>6} {:>6} {:>6} {:>8}  {:<12}",
             "ID", "Clúster", "W1-Mag", "W2-Mag", "W3-Mag", "Estado", "Expertos");
    println!("{}", "─".repeat(80));

    let core = mud_file.skills.get("core").expect("No core skill");
    let mut dead_clusters   = 0usize;
    let mut weak_clusters   = 0usize;
    let mut healthy_clusters = 0usize;

    for c in 0..NUM_CLUSTERS {
        let expert_start = c * CLUSTER_SIZE;
        let expert_end   = (expert_start + CLUSTER_SIZE).min(n_experts);
        if expert_start >= n_experts { break; }

        let mut w1_sum = 0.0f32;
        let mut w2_sum = 0.0f32;
        let mut w3_sum = 0.0f32;
        let mut counted = 0;

        for e in expert_start..expert_end {
            for l in 0..n_layers {
                let m1 = magnitude(core, &format!("blk.{}.expert.{}.w1.weight", l, e));
                let m2 = magnitude(core, &format!("blk.{}.expert.{}.w2.weight", l, e));
                let m3 = magnitude(core, &format!("blk.{}.expert.{}.w3.weight", l, e));
                if m1 > 0.0 || m2 > 0.0 || m3 > 0.0 {
                    w1_sum += m1; w2_sum += m2; w3_sum += m3;
                    counted += 1;
                }
            }
        }

        let (m1, m2, m3) = if counted > 0 {
            (w1_sum / counted as f32, w2_sum / counted as f32, w3_sum / counted as f32)
        } else {
            (0.0, 0.0, 0.0)
        };
        let avg = (m1 + m2 + m3) / 3.0;

        let (status, color) = if avg < 0.02 {
            dead_clusters += 1;
            ("💀 MUERTO ", "\x1b[1;31m")
        } else if avg < 0.10 {
            weak_clusters += 1;
            ("💤 DÉBIL  ", "\x1b[1;33m")
        } else if avg < 0.25 {
            healthy_clusters += 1;
            ("✅ ACTIVO ", "\x1b[0;32m")
        } else {
            healthy_clusters += 1;
            ("🔥 ÓPTIMO ", "\x1b[1;32m")
        };

        let n_loaded = if counted > 0 { expert_end - expert_start } else { 0 };
        println!("{}{:>2}  {:<28} {:>6.4} {:>6.4} {:>6.4}  {}{}  ({}/{})\x1b[0m",
                 color, c + 1, CLUSTER_NAMES[c], m1, m2, m3,
                 color, status, n_loaded, CLUSTER_SIZE);
    }

    println!("\n\x1b[1;34m[RESUMEN ANATÓMICO]\x1b[0m");
    println!("  🔥 Clústeres óptimos/activos: {}", healthy_clusters);
    println!("  💤 Clústeres débiles:         {}", weak_clusters);
    println!("  💀 Clústeres muertos:         {}", dead_clusters);

    // ── TEST 2: Balance de Carga — Diversidad de Enrutamiento ───────────────
    println!("\n\x1b[1;34m[TEST 2] Balance de Carga del Router (Top-K Diversity)\x1b[0m");

    let test_prompts = vec![
        // Dominio: Código / Optimización
        "fn main() { let x = optimize_shader(); }",
        "optimize rust allocator parallel threads avx2",
        // Dominio: Matemáticas / Lógica
        "integral diferencial gradiente tensor álgebra lineal",
        "boolean algebra silogism deductive proof theorem",
        // Dominio: Ciencias
        "mecánica cuántica función de onda densidad de probabilidad",
        "DNA sequence alignment protein folding bioinformatics",
        // Dominio: Planificación
        "plan step backtrack decision tree meta-cognition",
        // Dominio: General
        "hola mundo que tal estás hoy",
    ];

    let mut cluster_activations = vec![0usize; NUM_CLUSTERS];
    let mut total_activations   = 0usize;

    for prompt in &test_prompts {
        let tokens = engine.tokenizer.encode(prompt);
        if tokens.is_empty() { continue; }

        let mut x = vec![0.0f32; hidden];
        engine.embed_token(tokens[0], &mut x);

        // Simular el gate de la primera capa
        if let Some(layer) = engine.model.layers.first() {
            // Calcular gate logits manualmente usando el campo expuesto
            let ws_gate = &mut engine.workspace.gate_logits;
            ws_gate.fill(0.0);

            MudInference::gemv_vulkan_or_cpu(
                &*engine.vulkan_ctx,
                "audit_gate",
                hidden,
                engine.model.num_experts,
                &x,
                layer.gate_w,
                1.0,
                ws_gate,
            );

            let routing = layer.router.route(ws_gate);
            for (expert_id, _prob) in &routing {
                let cluster_id = (expert_id * NUM_CLUSTERS) / n_experts;
                let cluster_id = cluster_id.min(NUM_CLUSTERS - 1);
                cluster_activations[cluster_id] += 1;
                total_activations += 1;
            }
        }
    }

    println!("{:<3} {:<28} {:>8} {:>8}  {}", "ID", "Clúster", "Activac.", "% Total", "Balance");
    println!("{}", "─".repeat(70));
    let ideal_pct = 100.0 / NUM_CLUSTERS as f32;

    let mut max_dev = 0.0f32;
    for c in 0..NUM_CLUSTERS {
        let count = cluster_activations[c];
        let pct   = if total_activations > 0 { count as f32 * 100.0 / total_activations as f32 } else { 0.0 };
        let dev   = (pct - ideal_pct).abs();
        if dev > max_dev { max_dev = dev; }
        let bar_len = (pct as usize).min(30);
        let bar = "█".repeat(bar_len) + &"░".repeat(30usize.saturating_sub(bar_len));
        let color = if dev < 5.0 { "\x1b[32m" } else if dev < 15.0 { "\x1b[33m" } else { "\x1b[31m" };
        println!("{}{:>2}  {:<28} {:>8} {:>7.1}%  [{}]\x1b[0m",
                 color, c + 1, CLUSTER_NAMES[c], count, pct, bar);
    }

    let balance_score = 100.0 - max_dev.min(100.0);
    println!("\n  📊 Score de Balance: {:.1}% (desviación máxima: {:.1}%)", balance_score, max_dev);
    if balance_score > 85.0 {
        println!("  \x1b[1;32m✅ Balance ÓPTIMO — el router distribuye correctamente entre clústeres\x1b[0m");
    } else if balance_score > 60.0 {
        println!("  \x1b[1;33m⚠ Balance DÉBIL — considera aumentar el coeficiente de auxiliary_loss\x1b[0m");
    } else {
        println!("  \x1b[1;31m❌ Balance CRÍTICO — modo colapso de expertos detectado\x1b[0m");
    }

    // ── TEST 3: Estabilidad Vulkan bajo stress ───────────────────────────────
    println!("\n\x1b[1;34m[TEST 3] Stress-Test Vulkan (50 pasos continuos)\x1b[0m");
    let mut x_stress  = vec![0.1f32; hidden];
    let mut conv_pos  = 0usize;
    let mut nan_count = 0usize;

    for i in 0..50 {
        engine.step(&mut x_stress, "stress", &[], conv_pos);
        conv_pos += 1;
        let has_nan = x_stress.iter().any(|v| !v.is_finite());
        if has_nan { nan_count += 1; }
        if i % 10 == 0 {
            let l2: f32 = x_stress.iter().map(|v| v * v).sum::<f32>().sqrt();
            println!("  Step {:>3} | L2-norm={:.4} | NaN={}", i, l2, has_nan);
        }
    }

    if nan_count == 0 {
        println!("  \x1b[1;32m✅ Estabilidad numérica: PERFECTA (0 NaN en 50 pasos)\x1b[0m");
    } else {
        println!("  \x1b[1;31m❌ Inestabilidad: {} pasos con NaN/Inf detectados\x1b[0m", nan_count);
    }

    // ── TEST 4: Coherencia del cuantizador ternario ───────────────────────────
    println!("\n\x1b[1;34m[TEST 4] Coherencia del Cuantizador Ternario {{-1, 0, 1}}\x1b[0m");
    let mut total_pos = 0u64; let mut total_zero = 0u64; let mut total_neg = 0u64;

    for (name, tensor) in &core.tensors {
        if !name.contains("weight") || tensor.t_type != forge_llm::mud::MudTensorType::Ternary2Bit {
            continue;
        }
        let n_elems: usize = tensor.shape.iter().product();
        let n_u32 = n_elems.div_ceil(16);
        let data = unsafe { std::slice::from_raw_parts(tensor.data_ptr as *const u32, n_u32) };
        for &val in data {
            for j in 0..16 {
                let bits = (val >> (j * 2)) & 3;
                match bits { 1 => total_pos += 1, 2 => total_neg += 1, _ => total_zero += 1 }
            }
        }
    }

    let total = (total_pos + total_neg + total_zero).max(1) as f64;
    let pct_pos  = total_pos  as f64 / total * 100.0;
    let pct_neg  = total_neg  as f64 / total * 100.0;
    let pct_zero = total_zero as f64 / total * 100.0;
    let symmetry = (pct_pos - pct_neg).abs();

    println!("  +1: {:.2}%  |  0: {:.2}%  |  -1: {:.2}%", pct_pos, pct_zero, pct_neg);
    println!("  Asimetría (+/-): {:.2}%", symmetry);
    if pct_zero > 0.7 * 100.0 {
        println!("  \x1b[1;31m❌ EXCESO DE CEROS — peso muerto (sparsity > 70%)\x1b[0m");
    } else if symmetry < 5.0 {
        println!("  \x1b[1;32m✅ Distribución ternaria SIMÉTRICA — cuantizador saludable\x1b[0m");
    } else {
        println!("  \x1b[1;33m⚠ Asimetría detectada — revisar bias de inicialización\x1b[0m");
    }

    // ── RESUMEN FINAL ────────────────────────────────────────────────────────
    println!("\n\x1b[1;35m╔══════════════════════════════════════════════════════╗");
    println!("║              REPORTE FINAL DE AUDITORÍA MOE          ║");
    println!("╚══════════════════════════════════════════════════════╝\x1b[0m");
    println!("  Capas:       {}", n_layers);
    println!("  Expertos:    {} → {} clústeres × {} micro-expertos", n_experts, NUM_CLUSTERS, CLUSTER_SIZE);
    println!("  Saludables:  {} clústeres  |  Débiles: {}  |  Muertos: {}",
             healthy_clusters, weak_clusters, dead_clusters);
    println!("  Balance:     {:.1}%", balance_score);
    println!("  NaN stress:  {}", if nan_count == 0 { "NINGUNO ✅" } else { "DETECTADOS ❌" });
    println!("  Simetría:    {:.2}%", symmetry);

    Ok(())
}

fn magnitude(skill: &forge_llm::mud::MudSkill, name: &str) -> f32 {
    let Some(t) = skill.tensors.get(name) else { return 0.0 };
    let n_elems: usize = t.shape.iter().product();
    if n_elems == 0 { return 0.0; }
    let n_u32 = n_elems.div_ceil(16);
    let data = unsafe { std::slice::from_raw_parts(t.data_ptr as *const u32, n_u32) };
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for &val in data {
        for j in 0..16 {
            if count >= n_elems { break; }
            let bits = (val >> (j * 2)) & 3;
            sum += if bits == 1 { 1.0 } else if bits == 2 { 1.0 } else { 0.0 };
            count += 1;
        }
    }
    sum / n_elems as f32
}
