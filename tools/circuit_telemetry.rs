use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    widgets::{Block, Borders, Paragraph, Wrap},
    Terminal,
};
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

use forge_llm::mud::corpus_trainer::MudCorpusTrainer;

/// Live TUI for the seed-driven Training Circuit (`--circuit`).
///
/// Shows, in real time:
///   - Header: current phase (FASE), seed, battery remaining.
///   - Juez / system events (honors, rollbacks, match scores, benchmarks).
///   - Jugador A (Alpha) / Profesor and Jugador B (Beta) / Alumno panels.
///   - A scrolling event log of every telemetry line.
fn main() -> anyhow::Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/smollm2.mud".to_string());

    let corpus_dir = "training/corpus".to_string();
    println!("Loading MUD Model and corpus cache. Please wait...");
    let mut trainer = match MudCorpusTrainer::new(model_path, corpus_dir) {
        Ok(t) => t,
        Err(e) => {
            println!("Failed to load trainer: {}", e);
            return Ok(());
        }
    };

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let running = Arc::new(AtomicBool::new(true));
    let abort = Arc::new(AtomicBool::new(false));
    let r2 = running.clone();
    let a2 = abort.clone();
    ctrlc::set_handler(move || {
        r2.store(false, Ordering::SeqCst);
        a2.store(true, Ordering::SeqCst);
    })?;

    let (tx, rx) = mpsc::channel::<String>();

    let stop_flag = abort.clone();
    // The circuit pushes every event (phase, judge, A/B, professor/student) to `tx`.
    std::env::set_var("MUD_CIRCUIT_TUI", "1");
    std::thread::spawn(move || {
        let _ = trainer.run_training_circuit(Some(tx.clone()), stop_flag);
        let _ = tx.send("Circuit finished.".to_string());
    });

    let mut last_a = String::new();
    let mut last_b = String::new();
    let mut current_stats = String::from("Aguardando telemetría JEPA...");
    let mut current_turn = String::from("Pre-Circuit");
    let mut professor_mode = false;
    let mut event_log: Vec<String> =
        vec!["[JUEZ] Inicializando circuito de entrenamiento...".to_string()];

    let mut rpg_hp = 100.0f32;
    let mut rpg_max_hp = 100.0f32;
    let mut rpg_gen = 1u32;
    let mut rpg_winrate = 0.0f32;
    
    let mut battle_hp_a = 100.0f32;
    let mut battle_hp_b = 100.0f32;
    let mut name_a = "Jugador A (Evolutivo)".to_string();
    let mut name_b = "Jugador B (Baseline)".to_string();

    while running.load(Ordering::SeqCst) {
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.code == KeyCode::Char('q') {
                    abort.store(true, Ordering::SeqCst);
                    break;
                }
            }
        }

        while let Ok(msg) = rx.try_recv() {
            route_message(
                &msg,
                &mut last_a,
                &mut last_b,
                &mut current_stats,
                &mut current_turn,
                &mut professor_mode,
                &mut event_log,
                &mut rpg_hp,
                &mut rpg_max_hp,
                &mut rpg_gen,
                &mut rpg_winrate,
                &mut battle_hp_a,
                &mut battle_hp_b,
                &mut name_a,
                &mut name_b,
            );
        }

        terminal.draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .margin(1)
                .constraints(
                    [
                        Constraint::Length(3),
                        Constraint::Length(3),
                        Constraint::Percentage(45),
                        Constraint::Percentage(45),
                    ]
                    .as_ref(),
                )
                .split(f.size());

            let header_text = format!(
                "  MUD Training Circuit ({}) | [q] Quit / Save",
                current_turn
            );
            let header = Paragraph::new(header_text)
                .style(
                    Style::default()
                        .fg(Color::White)
                        .bg(Color::DarkGray)
                        .add_modifier(Modifier::BOLD),
                )
                .block(Block::default().borders(Borders::ALL));
            f.render_widget(header, chunks[0]);

            let hp_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
                .split(chunks[1]);

            let ratio_a = (battle_hp_a / 100.0).clamp(0.0, 1.0) as f64;
            
            let gauge_a = ratatui::widgets::Gauge::default()
                .block(Block::default().title(format!(" {} ", name_a)).borders(Borders::ALL).style(Style::default().fg(Color::LightBlue)))
                .gauge_style(Style::default().fg(Color::White).bg(Color::DarkGray))
                .ratio(ratio_a)
                .label(format!("Batalla: {:.1} | Global: {:.1}/{:.1} (Gen {})", battle_hp_a, rpg_hp, rpg_max_hp, rpg_gen));
            f.render_widget(gauge_a, hp_chunks[0]);

            let ratio_b = (battle_hp_b / 100.0).clamp(0.0, 1.0) as f64;
            let gauge_b = ratatui::widgets::Gauge::default()
                .block(Block::default().title(format!(" {} ", name_b)).borders(Borders::ALL).style(Style::default().fg(Color::LightMagenta)))
                .gauge_style(Style::default().fg(Color::White).bg(Color::DarkGray))
                .ratio(ratio_b)
                .label(format!("Batalla: {:.1} | Fijo", battle_hp_b));
            f.render_widget(gauge_b, hp_chunks[1]);

            let arena_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
                .split(chunks[2]);

            let (title_a, col_a) = if professor_mode {
                (format!(" Profesor ({}) ", name_a), Color::LightBlue)
            } else {
                (format!(" Doppelgänger A ({}) ", name_a), Color::LightBlue)
            };
            let (title_b, col_b) = if professor_mode {
                (format!(" Alumno ({}) ", name_b), Color::LightMagenta)
            } else {
                (format!(" Doppelgänger B ({}) ", name_b), Color::LightMagenta)
            };

            let alpha_block = Paragraph::new(last_a.clone())
                .wrap(Wrap { trim: true })
                .block(
                    Block::default()
                        .title(title_a)
                        .borders(Borders::ALL)
                        .style(Style::default().fg(col_a)),
                );
            f.render_widget(alpha_block, arena_chunks[0]);

            let beta_block = Paragraph::new(last_b.clone())
                .wrap(Wrap { trim: true })
                .block(
                    Block::default()
                        .title(title_b)
                        .borders(Borders::ALL)
                        .style(Style::default().fg(col_b)),
                );
            f.render_widget(beta_block, arena_chunks[1]);

            let combined = event_log.join("\n");
            let judge_text = format!("{}\n\n{}", current_stats, combined);
            
            // Calculate pseudo scroll based on rough wrapped lines to keep at bottom
            let width = chunks[3].width.saturating_sub(2) as usize;
            let mut total_lines = 0;
            for line in judge_text.lines() {
                total_lines += 1 + line.len() / width.max(1);
            }
            let scroll_offset = total_lines.saturating_sub(chunks[3].height.saturating_sub(2) as usize) as u16;

            let judge = Paragraph::new(judge_text).wrap(Wrap { trim: true }).scroll((scroll_offset, 0)).block(
                Block::default()
                    .title("  MUD Circuit Events ")
                    .borders(Borders::ALL)
                    .style(Style::default().fg(Color::White)),
            );
            f.render_widget(judge, chunks[3]);
        })?;
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn route_message(
    msg: &str,
    last_a: &mut String,
    last_b: &mut String,
    current_stats: &mut String,
    current_turn: &mut String,
    professor_mode: &mut bool,
    event_log: &mut Vec<String>,
    rpg_hp: &mut f32,
    rpg_max_hp: &mut f32,
    rpg_gen: &mut u32,
    rpg_winrate: &mut f32,
    battle_hp_a: &mut f32,
    battle_hp_b: &mut f32,
    name_a: &mut String,
    name_b: &mut String,
) {
    if msg.starts_with("STATS|") {
        let parts: Vec<&str> = msg.split('|').collect();
        if parts.len() == 3 {
            *current_stats = format!(
                "📊 Estabilidad JEPA -> VarH: {} | VarJ: {}",
                parts[1], parts[2]
            );
        }
    } else if msg.starts_with("Player A:") {
        *last_a = msg.replace("Player A:", "").trim().to_string();
    } else if msg.starts_with("Player B:") {
        *last_b = msg.replace("Player B:", "").trim().to_string();
    } else if msg.starts_with("Professor:") {
        *last_a = msg.replace("Professor:", "").trim().to_string();
    } else if msg.starts_with("Student:") {
        *last_b = msg.replace("Student:", "").trim().to_string();
    } else if msg.starts_with("=== INICIANDO ARENA DE JUEGO:") {
        *battle_hp_a = 100.0;
        *battle_hp_b = 100.0;
        *professor_mode = msg.contains("Professor-Student");
        *current_turn = msg
            .replace("=== INICIANDO ARENA DE JUEGO:", "")
            .trim()
            .to_string();
        push_event(event_log, &format!("[JUEZ] {}", msg));
    } else if msg.starts_with("[thinking]") {
        *current_turn = msg.replace("[thinking]", "").trim().to_string();
    } else if msg.contains("FASE=") {
        *battle_hp_a = 100.0;
        *battle_hp_b = 100.0;
        *last_a = String::new();
        *last_b = String::new();
        push_event(event_log, msg);
    } else if msg.starts_with("[JUEZ]") {
        push_event(event_log, msg);
    } else if msg.starts_with("REWARD|") {
        *current_stats = format!("🏆 {}", msg);
        let parts: Vec<&str> = msg.split('|').collect();
        for p in parts {
            if let Some(rest) = p.strip_prefix("A:") {
                if let Ok(v) = rest.parse::<f32>() {
                    if v < 0.0 { *battle_hp_a += v; }
                }
            } else if let Some(rest) = p.strip_prefix("B:") {
                if let Ok(v) = rest.parse::<f32>() {
                    if v < 0.0 { *battle_hp_b += v; }
                }
            }
        }
        if *battle_hp_a < 0.0 { *battle_hp_a = 0.0; }
        if *battle_hp_b < 0.0 { *battle_hp_b = 0.0; }
    } else if msg.contains("RPG Stats: HP") {
        // format: "circuit 🛡️  RPG Stats: HP 100.0/100.0 | Gen 1 | WinRate 0.00 | Ciclos 0"
        let parts: Vec<&str> = msg.split('|').collect();
        if parts.len() >= 3 {
            let hp_part = parts[0].split("HP ").nth(1).unwrap_or("100.0/100.0");
            let hp_vals: Vec<&str> = hp_part.trim().split('/').collect();
            if hp_vals.len() == 2 {
                if let Ok(v) = hp_vals[0].parse::<f32>() { *rpg_hp = v; }
                if let Ok(v) = hp_vals[1].parse::<f32>() { *rpg_max_hp = v; }
            }
            let gen_part = parts[1].replace("Gen", "").trim().to_string();
            if let Ok(g) = gen_part.parse::<u32>() { *rpg_gen = g; }
            let win_part = parts[2].replace("WinRate", "").trim().to_string();
            if let Ok(w) = win_part.parse::<f32>() { *rpg_winrate = w; }
        }
        if parts.len() >= 6 {
            let n_a = parts[4].replace("A:", "").trim().to_string();
            if !n_a.is_empty() { *name_a = n_a; }
            let n_b = parts[5].replace("B:", "").trim().to_string();
            if !n_b.is_empty() { *name_b = n_b; }
        }
        push_event(event_log, msg);
    } else {
        push_event(event_log, msg);
    }
}

fn push_event(log: &mut Vec<String>, msg: &str) {
    log.push(format!("[LOG] {}", msg));
    if log.len() > 50 {
        log.remove(0);
    }
}
