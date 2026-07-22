use std::io;
use std::time::Duration;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::mpsc;
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Style, Modifier},
    widgets::{Block, Borders, Paragraph},
    Terminal,
};

use forge_llm::mud::corpus_trainer::MudCorpusTrainer;

fn main() -> anyhow::Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let running = Arc::new(AtomicBool::new(true));
    let r2 = running.clone();
    ctrlc::set_handler(move || {
        r2.store(false, Ordering::SeqCst);
    })?;

    // We'd ideally hook an MPSC channel into DebateArena. 
    // For now, since DebateArena just uses println!, we'd capture it, or just 
    // run it in a thread and let it print? No, ratatui needs events.
    // I need to add an MPSC sender to DebateArena if I want it to send events.
    // For now, I will just spawn the thread, but we'll modify DebateArena to accept an optional sender.
    
    // Instead of doing all of that right now, let's just make debate_telemetry a shell that
    // triggers run_trainer via standard output redirection, OR we add a channel.
    // Let's add a simple channel.
    let (tx, rx) = mpsc::channel::<String>();
    let stop_flag = running.clone();

    let model_path = std::env::args().nth(1).unwrap_or_else(|| "models/core_skills.mud".to_string());
    std::thread::spawn(move || {
        let corpus_dir = "training/corpus".to_string();
        if let Ok(mut trainer) = MudCorpusTrainer::new(model_path, corpus_dir) {
            let _ = tx.send("Starting MUD Debate Arena Session...".to_string());
            let _ = trainer.run_debate_session(Some(tx.clone()), stop_flag.clone());
            let _ = tx.send("Debate finished.".to_string());
        }
    });

    let mut last_msg_a = String::new();
    let mut last_msg_b = String::new();
    let mut stats_text = String::from("Telemetría JEPA y Termodinámica en tiempo real irán aquí...");

    while running.load(Ordering::SeqCst) {
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.code == KeyCode::Char('q') {
                    break;
                }
            }
        }
        
        while let Ok(msg) = rx.try_recv() {
            if msg.starts_with("STATS|") {
                let parts: Vec<&str> = msg.split('|').collect();
                if parts.len() == 3 {
                    stats_text = format!("VarH: {} | VarJ: {}", parts[1], parts[2]);
                }
            } else if msg.starts_with("Player A:") {
                last_msg_a = msg;
            } else if msg.starts_with("Player B:") {
                last_msg_b = msg;
            } else {
                stats_text = msg; // System messages
            }
        }

        terminal.draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .margin(1)
                .constraints([Constraint::Length(3), Constraint::Percentage(70), Constraint::Percentage(30)].as_ref())
                .split(f.size());

            let header = Paragraph::new("⚔️  Forge LLM: Debate Arena (Live) | [q] Quit".to_string())
                .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
                .block(Block::default().borders(Borders::ALL));
            f.render_widget(header, chunks[0]);

            let arena_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
                .split(chunks[1]);

            let alpha_block = Paragraph::new(last_msg_a.clone())
                .block(Block::default().title(" Doppelgänger A (Alpha) ").borders(Borders::ALL).style(Style::default().fg(Color::Cyan)));
            f.render_widget(alpha_block, arena_chunks[0]);

            let beta_block = Paragraph::new(last_msg_b.clone())
                .block(Block::default().title(" Doppelgänger B (Beta) ").borders(Borders::ALL).style(Style::default().fg(Color::Red)));
            f.render_widget(beta_block, arena_chunks[1]);

            let stats = Paragraph::new(stats_text.clone())
                .block(Block::default().title(" Juez / JEPA Diagnostics ").borders(Borders::ALL));
            f.render_widget(stats, chunks[2]);
        })?;
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}
