use forge_llm::vulkan::VulkanContext;
use forge_llm::mud::MudFile;
use forge_llm::mud::inference::MudInference;
use sysinfo::System;
use std::io::{self, Write};
use std::time::{Duration, Instant};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use crossterm::{
    execute,
    cursor,
    terminal::{self, Clear, ClearType},
    style::{Color, Print, ResetColor, SetForegroundColor},
};
use std::path::Path;

// --- Global Atomic Flags for Concurrent UI & Signal Handling ---
pub static SHOULD_TERMINATE_CHAT: AtomicBool = AtomicBool::new(false);
pub static IS_TYPING: AtomicBool = AtomicBool::new(false);
pub static LAST_TPS: AtomicUsize = AtomicUsize::new(0);

const C_ACCENT: Color = Color::Rgb { r: 150, g: 100, b: 255 };
const C_DIM: Color = Color::Rgb { r: 120, g: 120, b: 120 };
const C_WARN: Color = Color::Rgb { r: 255, g: 180, b: 0 };

fn print_banner(_stdout: &mut io::Stdout) -> anyhow::Result<()> {
    // Run the professional banner tool
    let output = std::process::Command::new("target/release/model_banner").output()?;
    println!("{}", String::from_utf8_lossy(&output.stdout));
    Ok(())
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let hw_profile = forge_llm::hardware::HardwareProfile::detect();
    let _ = rayon::ThreadPoolBuilder::new().num_threads(hw_profile.preferred_threads).build_global();
    let mut sys = System::new_all();
    let mut stdout = io::stdout();

    execute!(stdout, Clear(ClearType::All), cursor::MoveTo(0, 0))?;
    print_banner(&mut stdout)?;
    
    let args: Vec<String> = std::env::args().collect();
    let mud_path = args.get(1).map(|s| s.as_str()).unwrap_or("models/core_skills.mud");
    
    let vk = VulkanContext::new().map(Arc::new).ok();
    if vk.is_none() {
        let use_vlk = std::env::var("MUD_USE_VULKAN").unwrap_or("1".to_string());
        if use_vlk != "0" && use_vlk.to_lowercase() != "false" {
            execute!(stdout, SetForegroundColor(C_WARN), Print("  ⚠️  Vulkan falló/deshabilitado. Usando fallback CPU.\n"), ResetColor)?;
        }
    }

    if !Path::new(mud_path).exists() {
        execute!(stdout, SetForegroundColor(C_WARN), Print(format!("  ❌ Model '{}' not found.\n", mud_path)), ResetColor)?;
        return Ok(());
    }

    let mud_file = MudFile::load(mud_path)?;
    let mut engine = MudInference::new(&mud_file, vk)?;

    ctrlc::set_handler(move || {
        SHOULD_TERMINATE_CHAT.store(true, Ordering::SeqCst);
        forge_llm::mud::auto_trainer::SHOULD_TERMINATE.store(true, Ordering::SeqCst);
    }).expect("Error setting Ctrl-C handler");

    let model_iq: f32 = mud_file.global_metadata.get("iq.score")
        .and_then(|v| v.parse::<f32>().ok()).unwrap_or(8.87);

    let (iq_label, _iq_color) = if model_iq < 15.0 {
        ("COGNICIÓN FRAGMENTADA", C_WARN)
    } else if model_iq < 100.0 {
        ("ASISTENTE FUNCIONAL", C_ACCENT)
    } else {
        ("RAZONAMIENTO MAESTRO", Color::Rgb { r: 120, g: 255, b: 120 })
    };

    // Run the professional IQ box tool
    let iq_output = std::process::Command::new("target/release/iq_box")
        .arg(format!("{:.2}", model_iq))
        .arg(iq_label)
        .output()?;
    println!("{}", String::from_utf8_lossy(&iq_output.stdout));

    println!("  ✨  MUD Engine Initialized. Type /help for commands.");

    let active_exp_ref = engine.active_experts.clone();
    let total_exp = engine.model.num_experts;
    let vlk_available = engine.vulkan_ctx.is_some();
    
    std::thread::spawn(move || {
        let mut sys_bg = System::new_all();
        while !SHOULD_TERMINATE_CHAT.load(Ordering::Relaxed) {
            if !IS_TYPING.load(Ordering::Relaxed) {
                let mut out = io::stdout();
                let active = active_exp_ref.load(Ordering::Relaxed);
                let _ = update_status_bar(&mut out, active, total_exp, vlk_available, &mut sys_bg);
            }
            std::thread::sleep(Duration::from_millis(2500));
        }
    });

    let mut conversation_pos = 0usize;

    loop {
        if SHOULD_TERMINATE_CHAT.load(Ordering::SeqCst) { break; }
        
        print!("\nYOU ❯ ");
        stdout.flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let trimmed = input.trim();
        
        if trimmed.is_empty() { continue; }
        if trimmed == "/exit" || trimmed == "/quit" { break; }
        
        IS_TYPING.store(true, Ordering::Relaxed);
        
        print!("\nMUD ❯ ");
        stdout.flush()?;

        let mut current_x = vec![0.0f32; engine.model.hidden_size];
        let start_gen = Instant::now();

        engine.prompt(trimmed, &mut current_x, &mut conversation_pos);
        
        let response_tuple = engine.generate(&mut current_x, 256, trimmed, &mut conversation_pos);
        let response_tokens = response_tuple.0;
        let decoded = engine.tokenizer.decode(&response_tokens);
        
        type_writer(&decoded, &mut stdout)?;
        
        let elapsed = start_gen.elapsed().as_secs_f32();
        let tps = response_tokens.len() as f32 / elapsed;
        LAST_TPS.store(tps as usize, Ordering::Relaxed);

        println!("\n");
        IS_TYPING.store(false, Ordering::Relaxed);
    }

    execute!(stdout, cursor::Show)?;
    Ok(())
}

fn type_writer(text: &str, stdout: &mut io::Stdout) -> anyhow::Result<()> {
    for c in text.chars() {
        print!("{}", c);
        stdout.flush()?;
        std::thread::sleep(Duration::from_millis(15));
    }
    Ok(())
}

fn update_status_bar(stdout: &mut io::Stdout, active: usize, total: usize, vlk: bool, sys: &mut System) -> anyhow::Result<()> {
    sys.refresh_memory();
    let used_mem = sys.used_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
    let total_mem = sys.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
    let tps = LAST_TPS.load(Ordering::Relaxed);

    let vlk_str = if vlk { "VLK ✓" } else { "CPU" };
    
    let (_, rows) = terminal::size()?;
    execute!(
        stdout,
        cursor::SavePosition,
        cursor::MoveTo(0, rows - 1),
        SetForegroundColor(Color::Black),
        crossterm::style::SetBackgroundColor(C_ACCENT),
        Print(format!(" MUD-V1.5 │ Exp {}/{} │ {} t/s │ Mem {:.1}/{:.1}G │ {} │ IQ {:.1} ", 
                      active, total, if tps > 0 { tps.to_string() } else { "──".to_string() },
                      used_mem, total_mem, vlk_str, 8.9)),
        ResetColor,
        cursor::RestorePosition,
    )?;
    stdout.flush()?;
    Ok(())
}
