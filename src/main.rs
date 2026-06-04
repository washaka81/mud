use crossterm::{
    cursor, execute,
    style::{Color, Print, ResetColor, SetForegroundColor},
    terminal::{self, Clear, ClearType},
};
use forge_llm::mud::inference::MudInference;
use forge_llm::mud::MudFile;
use forge_llm::vulkan::VulkanContext;
use std::io::{self, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use sysinfo::System;

// --- Global Atomic Flags for Concurrent UI & Signal Handling ---
pub static SHOULD_TERMINATE_CHAT: AtomicBool = AtomicBool::new(false);
pub static LAST_TPS: AtomicUsize = AtomicUsize::new(0);
const C_WARN: Color = Color::Rgb {
    r: 255,
    g: 180,
    b: 0,
};

const MAX_RESPONSE_TOKENS: usize = 256;

fn print_banner(_stdout: &mut io::Stdout) -> anyhow::Result<()> {
    println!("\x1b[38;5;201m   ███╗   ███╗ ██╗   ██╗ ██████╗ \x1b[0m");
    println!("\x1b[38;5;165m   ████╗ ████║ ██║   ██║ ██╔══██╗\x1b[0m");
    println!("\x1b[38;5;129m   ██╔████╔██║ ██║   ██║ ██║  ██║\x1b[0m");
    println!("\x1b[38;5;93m   ██║╚██╔╝██║ ██║   ██║ ██║  ██║\x1b[0m");
    println!("\x1b[38;5;57m   ██║ ╚═╝ ██║ ╚██████╔╝ ██████╔╝\x1b[0m");
    println!("\x1b[38;5;45m   ╚═╝     ╚═╝  ╚═════╝  ╚═════╝ \x1b[0m");
    Ok(())
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let mut stdout = io::stdout();

    // --- Terminal UI Initial Setup ---
    let (_, mut rows) = terminal::size().unwrap_or((80, 24));
    if rows > 1 {
        print!("\x1b[1;{}r", rows - 1);
        let _ = io::stdout().flush();
    }

    execute!(stdout, Clear(ClearType::All), cursor::MoveTo(0, 0))?;
    print_banner(&mut stdout)?;

    let args: Vec<String> = std::env::args().collect();
    let mud_path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("models/core_skills.mud");

    // HW-01: Detect Hardware Profile and optimize Thread Pool
    let hw = forge_llm::hardware::HardwareProfile::detect();
    if rayon::ThreadPoolBuilder::new()
        .num_threads(hw.preferred_threads)
        .build_global()
        .is_err()
    {
        // Global pool already initialized, ignoring
    }

    let use_vlk = std::env::var("MUD_USE_VULKAN").unwrap_or("1".to_string());
    let use_vlk_bool = use_vlk != "0" && use_vlk.to_lowercase() != "false";
    let vk = if use_vlk_bool {
        VulkanContext::new().map(Arc::new).ok()
    } else {
        None
    };
    if vk.is_none() && use_vlk_bool {
        execute!(
            stdout,
            SetForegroundColor(C_WARN),
            Print("  ⚠️  Vulkan falló. Usando fallback CPU.\n"),
            ResetColor
        )?;
    }

    if !Path::new(mud_path).exists() {
        execute!(
            stdout,
            SetForegroundColor(C_WARN),
            Print(format!("  ❌ Model '{}' not found.\n", mud_path)),
            ResetColor
        )?;
        return Ok(());
    }

    let mud_file = MudFile::load(mud_path)?;
    let mut engine = MudInference::new(&mud_file, vk)?;

    ctrlc::set_handler(move || {
        SHOULD_TERMINATE_CHAT.store(true, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    println!("\x1b[1;35m╭────────────────────────────────────────────────────────────╮");
    println!(
        "│  \x1b[1;36m🧠 MODELO:\x1b[0m  {:<45} │",
        format!(
            "{} (Ternary MoE Transform)",
            mud_path.split('/').next_back().unwrap_or(mud_path)
        )
    );
    let accel_str = if engine.vulkan_ctx.is_some() {
        "Vulkan iGPU (Zero-Copy Unified Memory)"
    } else {
        "CPU Fallback (SIMD AVX2 Optimized)"
    };
    println!("│  \x1b[1;36m🎮 ACEL:\x1b[0m    {:<45} │", accel_str);
    println!("╰────────────────────────────────────────────────────────────╯\x1b[0m");

    println!("  \x1b[1;32m✨ MUD Engine Initialized. Type /help for commands.\x1b[0m\n");

    let total_exp = engine.model.num_experts;
    let vlk_available = engine.vulkan_ctx.is_some();

    let mut conversation_pos = 0usize;

    let mut last_rows = rows;

    loop {
        if SHOULD_TERMINATE_CHAT.load(Ordering::SeqCst) {
            break;
        }

        // Render the physical bottom status bar statically once per loop turn (zero duplication, zero blink)
        if let Ok((new_cols, new_rows)) = terminal::size() {
            let cols = new_cols;
            // Update scrolling region if terminal was resized
            if new_rows != last_rows && new_rows > 1 {
                print!("\x1b[1;{}r", new_rows - 1);
                let _ = io::stdout().flush();
                last_rows = new_rows;
                rows = new_rows;
            }

            let mut sys_bg = System::new_all();
            sys_bg.refresh_memory();
            let used_mem = sys_bg.used_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
            let total_mem = sys_bg.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
            let tps_val = LAST_TPS.load(Ordering::Relaxed);
            let accel_str = if vlk_available {
                "iGPU (VLK)"
            } else {
                "CPU (AVX2)"
            };
            let moe_str = if total_exp > 1 {
                format!("{} (MoE)", total_exp)
            } else {
                "1 (Dense)".to_string()
            };

            let bar_text = format!(
                " ⚡ MUD Engine │ Experts: {} │ Speed: {} t/s │ Mem: {:.1}/{:.1}G │ Accel: {} ",
                moe_str,
                if tps_val > 0 {
                    format!("{:.1}", tps_val as f32 / 10.0)
                } else {
                    "──".to_string()
                },
                used_mem,
                total_mem,
                accel_str
            );

            let _ = execute!(
                stdout,
                cursor::SavePosition,
                cursor::MoveTo(0, rows - 1),
                crossterm::style::SetBackgroundColor(Color::DarkGrey),
                SetForegroundColor(Color::White),
                Print(format!("{:<width$}", bar_text, width = cols as usize)),
                ResetColor,
                cursor::RestorePosition,
            );
            let _ = stdout.flush();
        }

        print!("YOU ❯ ");
        stdout.flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let trimmed = input.trim();

        if trimmed.is_empty() {
            continue;
        }
        if trimmed == "/exit" || trimmed == "/quit" {
            break;
        }

        if trimmed == "/help" {
            println!("\x1b[1;35m╭──────────────────────────────────────────────────╮");
            println!("│ 💬 MUD INTERACTIVE CHAT COMMANDS                 │");
            println!("├──────────────────────────────────────────────────┤");
            println!("│  /help           - Show this help menu           │");
            println!("│  /exit or /quit  - Terminate session             │");
            println!("╰──────────────────────────────────────────────────╯\x1b[0m");
            continue;
        }

        print!("\nMUD ❯ ");
        stdout.flush()?;

        let mut current_x = vec![0.0f32; engine.model.hidden_size];
        let start_gen = Instant::now();

        engine.prompt(trimmed, &mut current_x, &mut conversation_pos);

        let mut response_tokens = Vec::new();
        let (full_results, _) = engine.generate(
            &current_x,
            MAX_RESPONSE_TOKENS,
            trimmed,
            &mut conversation_pos,
            0,
            |token_id, text| {
                response_tokens.push(token_id);
                print!("{}", text);
                let _ = io::stdout().flush();
            },
        );

        let elapsed = start_gen.elapsed().as_secs_f32();
        let tps = if full_results.is_empty() {
            0.0
        } else {
            full_results.len() as f32 / elapsed
        };
        LAST_TPS.store((tps * 10.0) as usize, Ordering::Relaxed);
        println!("\n");

        if conversation_pos >= 1024 {
            let _ = execute!(
                stdout,
                SetForegroundColor(Color::Cyan),
                Print("  💤 [MUD] Límite de Atención alcanzado (1024 tokens). Ejecutando Sleep & Fold...\n"),
                ResetColor
            );
            engine.sleep_and_fold();
            conversation_pos = 0; // Reset KV-Cache index, Mamba maintains the long-term context
            let _ = execute!(
                stdout,
                SetForegroundColor(Color::Green),
                Print("  ✅ Contexto asimilado en estado Mamba O(1).\n\n"),
                ResetColor
            );
        }
    }

    execute!(stdout, cursor::Show)?;
    // Reset scrolling region to full screen
    print!("\x1b[r");
    let _ = io::stdout().flush();
    Ok(())
}

// Typewriter removed in favor of native streaming

// Telemetry and inline status bar rendered inline after each generation step.
