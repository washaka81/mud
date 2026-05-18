use forge_llm::vulkan::VulkanContext;
use forge_llm::mud::MudFile;
use forge_llm::mud::inference::MudInference;
use std::sync::Arc;
use std::path::Path;
use std::time::Duration;
use std::thread;
use std::io::{self, Write};
use crossterm::{
    execute,
    style::{Color, Print, ResetColor, SetForegroundColor, Attribute, SetAttribute, SetBackgroundColor},
    terminal::{self, Clear, ClearType},
    cursor::{self, MoveToColumn},
};
use sysinfo::System;

// --- Modern Visual Styles ---
const MUD_PRIMARY: Color = Color::Cyan;
const MUD_SECONDARY: Color = Color::Green;
const USER_PROMPT: &str = "❯";

fn main() -> anyhow::Result<()> {
    env_logger::init();
    
    // Initialize System Monitor
    let mut sys = System::new_all();
    
    let mut stdout = io::stdout();
    execute!(stdout, Clear(ClearType::All), cursor::MoveTo(0, 0))?;
    
    print_banner(&mut stdout)?;

    let mud_path = "models/core_skills.mud";
    let vk = Arc::new(VulkanContext::new().unwrap());

    if !Path::new(mud_path).exists() {
        println!("Error: Model file {} not found.", mud_path);
        return Ok(());
    }

    let mud_file = MudFile::load(mud_path)?;
    let engine = MudInference::new(&mud_file, vk)?;

    let mut conversation_pos = 0;
    let mut last_tps = 0.0;

    // --- Terminal UI Initialization ---
    let (_cols, rows) = terminal::size().unwrap_or((80, 24));
    // 1. Set scrolling region to leave the last line free for the status bar (resets cursor to 1,1)
    execute!(stdout, Print(format!("\x1B[1;{}r", rows - 1)))?;
    // 2. Clear screen and move to home
    execute!(stdout, terminal::Clear(terminal::ClearType::All), cursor::MoveTo(0, 0))?;
    // 3. Print ASCII Banner inside scrolling region
    print_banner(&mut stdout)?;
    
    execute!(
        stdout,
        SetForegroundColor(MUD_PRIMARY),
        SetAttribute(Attribute::Bold),
        Print("\n [MUD Engine Initialized] "),
        ResetColor,
        SetAttribute(Attribute::Reset),
        Print("Architecture: Ternary 1.58-bit MoE. Ready.\n")
    )?;
    stdout.flush()?;

    let mut input = String::new();
    loop {
        // --- Refresh Hardware Stats ---
        sys.refresh_cpu_all();
        sys.refresh_memory();

        let (_cols, current_rows) = terminal::size().unwrap_or((80, 24));
        let mmap_mb = mud_file.mmap.as_ref().unwrap().len() as f64 / 1024.0 / 1024.0;
        let graph_nodes = engine.model.knowledge_graph.read().unwrap().nodes.len();
        let cpu_load = sys.global_cpu_usage();
        let mem_used = sys.used_memory() / 1024 / 1024;
        let mem_total = sys.total_memory() / 1024 / 1024;

        // --- Draw Sticky Footer ---
        execute!(
            stdout,
            cursor::SavePosition,
            cursor::MoveTo(0, current_rows - 1),
            terminal::Clear(terminal::ClearType::CurrentLine),
            SetBackgroundColor(Color::Rgb { r: 35, g: 35, b: 85 }),
            SetForegroundColor(Color::White),
            SetAttribute(Attribute::Bold),
            Print(format!(
                " MUD v1.58b | CPU: {:.1}% | RAM: {}/{} MB | Model: {:.1} MB | Experts: {} | Knowledge: {} nodes | {:.1} t/s | HW: Iris Xe ",
                cpu_load, mem_used, mem_total, mmap_mb, engine.model.num_experts, graph_nodes, last_tps
            )),
            SetBackgroundColor(Color::Reset),
            ResetColor,
            SetAttribute(Attribute::Reset),
            cursor::RestorePosition,
        )?;
        stdout.flush()?;

        execute!(
            stdout,
            SetForegroundColor(Color::Yellow),
            SetAttribute(Attribute::Bold),
            Print(format!("\n {}  ", USER_PROMPT)),
            ResetColor,
            SetAttribute(Attribute::Reset),
        )?;
        stdout.flush()?;

        input.clear();
        if io::stdin().read_line(&mut input)? == 0 { break; }
        let trimmed = input.trim();

        if trimmed == "/exit" || trimmed == "\u{11}" { break; }
        if trimmed.is_empty() { continue; }

        if trimmed.starts_with("/ingest ") {
            let path = &trimmed[8..];
            match forge_llm::mud::ingester::MudIngester::ingest(path, &engine) {
                Ok(n) => println!("  ✅ Success: Ingested {} knowledge chunks.", n),
                Err(e) => println!("  ❌ Error during ingestion: {}", e),
            }
            continue;
        }

        let tokens = engine.tokenizer.encode(trimmed);
        if tokens.is_empty() {
            println!("  ⚠️ Error: Word not understood.");
            continue; 
        }
        
        let mut x = vec![0.0f32; engine.model.hidden_size];
        print_thinking(&mut stdout, true)?;
        
        for &token in tokens.iter() {
            engine.embed_token(token, &mut x);
            engine.step(&mut x, trimmed, &[], conversation_pos); 
            conversation_pos += 1;
        }
        
        let start_gen = std::time::Instant::now();
        let response_tokens = engine.generate(&x, 48, trimmed, &mut conversation_pos);
        let duration = start_gen.elapsed();
        last_tps = if response_tokens.is_empty() { 0.0 } else { response_tokens.len() as f64 / duration.as_secs_f64() };

        let mut response = decode_tokens(&engine.tokenizer, &response_tokens);
        if response.trim().is_empty() {
            response = "I am processing your intent, but my vocabulary is still evolving.".to_string();
        }

        engine.format_text(&mut response);
        print_thinking(&mut stdout, false)?;
        
        execute!(
            stdout,
            SetForegroundColor(MUD_SECONDARY),
            SetAttribute(Attribute::Bold),
            Print(" MUD "),
            SetAttribute(Attribute::Italic),
            Print("❯ "),
            ResetColor,
            SetAttribute(Attribute::Reset),
        )?;
        
        type_writer(&response, Duration::from_millis(3))?;
        println!();
    }

    // Reset scrolling region before exit
    execute!(stdout, Print("\x1B[r"))?;
    println!("\nInference session closed. Goodbye.");
    Ok(())
}

fn print_banner(stdout: &mut io::Stdout) -> io::Result<()> {
    execute!(
        stdout,
        SetForegroundColor(Color::Rgb { r: 0, g: 255, b: 255 }),
        SetAttribute(Attribute::Bold),
        Print("   __  __ _   _ ____  \n"),
        Print("  |  \\/  | | | |  _ \\ \n"),
        Print("  | |\\/| | | | | | | |\n"),
        Print("  | |  | | |_| | |_| |\n"),
        Print("  |_|  |_|\\___/|____/ \n"),
        SetAttribute(Attribute::Reset),
        SetForegroundColor(Color::DarkGrey),
        Print("  Modular Understanding Dynamics v1.58b\n"),
        ResetColor,
        Print("-----------------------------------------------\n")
    )
}

fn print_thinking(stdout: &mut io::Stdout, active: bool) -> io::Result<()> {
    if active {
        execute!(
            stdout,
            SetForegroundColor(Color::Magenta),
            SetAttribute(Attribute::Italic),
            Print(" ... MUD is thinking ..."),
            ResetColor,
            SetAttribute(Attribute::Reset),
            MoveToColumn(0),
        )
    } else {
        execute!(stdout, Clear(ClearType::CurrentLine), MoveToColumn(0))
    }
}

fn type_writer(text: &str, delay: Duration) -> io::Result<()> {
    let mut in_ansi = false;
    
    // We check for tags and change color accordingly
    let mut i = 0;
    let chars: Vec<char> = text.chars().collect();
    
    while i < chars.len() {
        let remaining: String = chars[i..].iter().collect();
        
        if remaining.starts_with("<thinking>") {
            execute!(io::stdout(), 
                SetForegroundColor(Color::Cyan), 
                SetAttribute(Attribute::Italic),
                SetAttribute(Attribute::Dim)
            )?;
            i += 10;
            continue;
        } else if remaining.starts_with("</thinking>") {
            execute!(io::stdout(), ResetColor, SetAttribute(Attribute::Reset))?;
            i += 11;
            println!(); // Line break after thinking
            continue;
        } else if remaining.starts_with("<answer>") {
            execute!(io::stdout(), ResetColor, SetAttribute(Attribute::Bold))?;
            i += 8;
            continue;
        } else if remaining.starts_with("</answer>") {
            execute!(io::stdout(), ResetColor, SetAttribute(Attribute::Reset))?;
            i += 9;
            continue;
        }

        let c = chars[i];
        if c == '\x1b' { in_ansi = true; }
        print!("{}", c);
        io::stdout().flush()?;
        if in_ansi {
            if c == 'm' { in_ansi = false; }
        } else {
            thread::sleep(delay);
        }
        i += 1;
    }
    execute!(io::stdout(), ResetColor, SetAttribute(Attribute::Reset))?;
    Ok(())
}

fn decode_tokens(tokenizer: &forge_llm::model::tokenizer::Tokenizer, ids: &[u32]) -> String {
    let mut response = String::new();
    for &id in ids {
        let text = tokenizer.decode(&[id]);
        response.push_str(&text);
        response.push(' ');
    }
    response
}
