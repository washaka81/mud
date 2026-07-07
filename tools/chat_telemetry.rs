use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::{Backend, CrosstermBackend},
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    symbols,
    text::Span,
    widgets::{Axis, Block, Borders, Chart, Dataset, GraphType},
    Terminal,
};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Seek, SeekFrom};
use std::time::Duration;

struct App {
    h_ent_data: Vec<(f64, f64)>,
    varh_data: Vec<(f64, f64)>,
    t_softmx_data: Vec<(f64, f64)>,
    tok_s_data: Vec<(f64, f64)>,
    step_count: f64,
    window_size: f64,
    scroll_offset: f64,
}

impl App {
    fn new() -> Self {
        Self {
            h_ent_data: Vec::new(),
            varh_data: Vec::new(),
            t_softmx_data: Vec::new(),
            tok_s_data: Vec::new(),
            step_count: 0.0,
            window_size: 500.0,
            scroll_offset: 0.0,
        }
    }
}

fn main() -> anyhow::Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new();

    let file_path = "mud_metrics.log";
    let file = File::open(file_path);

    let mut reader = if let Ok(f) = file {
        Some(BufReader::new(f))
    } else {
        None
    };

    let res = run_app(&mut terminal, &mut app, &mut reader, file_path);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        println!("{:?}", err)
    }

    Ok(())
}

fn run_app<B: Backend>(
    terminal: &mut Terminal<B>,
    app: &mut App,
    reader: &mut Option<BufReader<File>>,
    file_path: &str,
) -> io::Result<()> {
    let mut last_pos = 0;

    loop {
        // Try opening the file if it wasn't open
        if reader.is_none() {
            if let Ok(f) = File::open(file_path) {
                *reader = Some(BufReader::new(f));
            }
        }

        // Read new lines
        if let Some(r) = reader.as_mut() {
            if r.seek(SeekFrom::Start(last_pos)).is_ok() {
                let mut line = String::new();
                while let Ok(len) = r.read_line(&mut line) {
                    if len == 0 {
                        break;
                    }

                    
                    if !line.contains("Pos") && !line.contains("Step") && !line.contains("===") {
                        let parts: Vec<&str> = line.split(|c: char| c == '|' || c.is_whitespace()).filter(|s| !s.is_empty()).collect();
                        if parts.len() >= 18 {
                            let step = parts[0].parse::<f64>().unwrap_or(app.step_count + 1.0);
                            app.step_count = step;
                            
                            if let Ok(varh) = parts[6].parse::<f64>() {
                                app.varh_data.push((step, varh));
                            }
                            if let Ok(t_softmx) = parts[10].parse::<f64>() {
                                app.t_softmx_data.push((step, t_softmx));
                            }
                            if let Some(h_val) = parts[17].strip_prefix("H:") {
                                if let Ok(h_ent) = h_val.parse::<f64>() {
                                    app.h_ent_data.push((step, h_ent));
                                }
                            }
                            if let Some(tok_val) = parts.get(18).and_then(|p| p.strip_suffix("t/s")) {
                                if let Ok(tok_s) = tok_val.parse::<f64>() {
                                    app.tok_s_data.push((step, tok_s));
                                }
                            }
                        }
                    }
                    line.clear();
                }
                if let Ok(pos) = r.stream_position() {
                    last_pos = pos;
                }
            }
        }

        terminal.draw(|f| ui(f, app))?;

        if event::poll(Duration::from_millis(500))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                    KeyCode::Left => app.scroll_offset += 100.0,
                    KeyCode::Right => {
                        app.scroll_offset -= 100.0;
                        if app.scroll_offset < 0.0 {
                            app.scroll_offset = 0.0;
                        }
                    }
                    KeyCode::Up => app.window_size += 100.0,
                    KeyCode::Down => {
                        app.window_size -= 100.0;
                        if app.window_size < 100.0 {
                            app.window_size = 100.0;
                        }
                    }
                    KeyCode::Char('r') => {
                        app.scroll_offset = 0.0;
                        app.window_size = 500.0;
                    }
                    _ => {}
                }
            }
        }
    }
}

fn ui(f: &mut ratatui::Frame, app: &App) {
    let vertical_chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
        .split(f.size());
        
    let top_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
        .split(vertical_chunks[0]);
        
    let bottom_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
        .split(vertical_chunks[1]);

    let mut max_step = app.step_count.max(10.0) - app.scroll_offset;
    if max_step < app.window_size {
        max_step = app.window_size;
    }
    let min_step = (max_step - app.window_size).max(0.0);

    // Filter data for the window
    let display_h_ent: Vec<(f64, f64)> = app
        .h_ent_data
        .iter()
        .filter(|(x, _)| *x >= min_step)
        .copied()
        .collect();
    let display_varh: Vec<(f64, f64)> = app
        .varh_data
        .iter()
        .filter(|(x, _)| *x >= min_step)
        .copied()
        .collect();

    let display_t_softmx: Vec<(f64, f64)> = app
        .t_softmx_data
        .iter()
        .filter(|(x, _)| *x >= min_step)
        .copied()
        .collect();
    let display_tok_s: Vec<(f64, f64)> = app
        .tok_s_data
        .iter()
        .filter(|(x, _)| *x >= min_step)
        .copied()
        .collect();

    let max_h_ent = display_h_ent
        .iter()
        .map(|(_, y)| *y)
        .fold(0.0f64, f64::max)
        .max(0.1);
    let min_h_ent = display_h_ent
        .iter()
        .map(|(_, y)| *y)
        .fold(max_h_ent, f64::min)
        .min(0.0);

    let max_varh = display_varh
        .iter()
        .map(|(_, y)| *y)
        .fold(0.0f64, f64::max)
        .max(0.1);
    let min_varh = display_varh
        .iter()
        .map(|(_, y)| *y)
        .fold(max_varh, f64::min)
        .min(0.0);

    let max_t_softmx = display_t_softmx
        .iter()
        .map(|(_, y)| *y)
        .fold(0.0f64, f64::max)
        .max(0.1);
    let min_t_softmx = display_t_softmx
        .iter()
        .map(|(_, y)| *y)
        .fold(max_t_softmx, f64::min)
        .min(-0.1);

    let max_tok_s = display_tok_s
        .iter()
        .map(|(_, y)| *y)
        .fold(0.0f64, f64::max)
        .max(0.1);
    let min_tok_s = display_tok_s
        .iter()
        .map(|(_, y)| *y)
        .fold(max_tok_s, f64::min)
        .min(-0.1);

    let datasets_h_ent = vec![Dataset::default()
        .name("H_Ent (Regression)")
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(Color::Cyan))
        .data(&display_h_ent)];

    let chart_h_ent = Chart::new(datasets_h_ent)
        .block(
            Block::default()
                .title(Span::styled(
                    "Inference Entropy Regression (H_Ent)",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ))
                .borders(Borders::ALL),
        )
        .x_axis(
            Axis::default()
                .title("Steps")
                .style(Style::default().fg(Color::Gray))
                .bounds([min_step, max_step])
                .labels(vec![
                    Span::raw(format!("{}", min_step)),
                    Span::raw(format!("{}", max_step)),
                ]),
        )
        .y_axis(
            Axis::default()
                .title("Entropy")
                .style(Style::default().fg(Color::Gray))
                .bounds([min_h_ent, max_h_ent])
                .labels(vec![
                    Span::raw(format!("{:.2}", min_h_ent)),
                    Span::raw(format!("{:.2}", (min_h_ent + max_h_ent) / 2.0)),
                    Span::raw(format!("{:.2}", max_h_ent)),
                ]),
        );

    f.render_widget(chart_h_ent, top_chunks[0]);

    let datasets_varh = vec![Dataset::default()
        .name("VarH (Gradient Direction/Dispersion)")
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(Color::Yellow))
        .data(&display_varh)];

    let chart_varh = Chart::new(datasets_varh)
        .block(
            Block::default()
                .title(Span::styled(
                    "Hidden Variance (VarH)",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ))
                .borders(Borders::ALL),
        )
        .x_axis(
            Axis::default()
                .title("Steps")
                .style(Style::default().fg(Color::Gray))
                .bounds([min_step, max_step])
                .labels(vec![
                    Span::raw(format!("{}", min_step)),
                    Span::raw(format!("{}", max_step)),
                ]),
        )
        .y_axis(
            Axis::default()
                .title("VarH")
                .style(Style::default().fg(Color::Gray))
                .bounds([min_varh, max_varh])
                .labels(vec![
                    Span::raw(format!("{:.2}", min_varh)),
                    Span::raw(format!("{:.2}", (min_varh + max_varh) / 2.0)),
                    Span::raw(format!("{:.2}", max_varh)),
                ]),
        );

    f.render_widget(chart_varh, top_chunks[1]);

    let datasets_t_softmx = vec![Dataset::default()
        .name("Softmax Temp (T_Est)")
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(Color::Magenta))
        .data(&display_t_softmx)];

    let chart_t_softmx = Chart::new(datasets_t_softmx)
        .block(
            Block::default()
                .title(Span::styled(
                    "Softmax Est. Temperature",
                    Style::default()
                        .fg(Color::Magenta)
                        .add_modifier(Modifier::BOLD),
                ))
                .borders(Borders::ALL),
        )
        .x_axis(
            Axis::default()
                .title("Steps")
                .style(Style::default().fg(Color::Gray))
                .bounds([min_step, max_step])
                .labels(vec![
                    Span::raw(format!("{}", min_step)),
                    Span::raw(format!("{}", max_step)),
                ]),
        )
        .y_axis(
            Axis::default()
                .title("Temp")
                .style(Style::default().fg(Color::Gray))
                .bounds([min_t_softmx, max_t_softmx])
                .labels(vec![
                    Span::raw(format!("{:.2}", min_t_softmx)),
                    Span::raw(format!("{:.2}", (min_t_softmx + max_t_softmx) / 2.0)),
                    Span::raw(format!("{:.2}", max_t_softmx)),
                ]),
        );

    f.render_widget(chart_t_softmx, bottom_chunks[0]);

    let datasets_tok_s = vec![Dataset::default()
        .name("Tokens per Second")
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(Color::Green))
        .data(&display_tok_s)];

    let chart_tok_s = Chart::new(datasets_tok_s)
        .block(
            Block::default()
                .title(Span::styled(
                    "Inference Speed (tok/s)",
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ))
                .borders(Borders::ALL),
        )
        .x_axis(
            Axis::default()
                .title("Steps")
                .style(Style::default().fg(Color::Gray))
                .bounds([min_step, max_step])
                .labels(vec![
                    Span::raw(format!("{}", min_step)),
                    Span::raw(format!("{}", max_step)),
                ]),
        )
        .y_axis(
            Axis::default()
                .title("Derivative")
                .style(Style::default().fg(Color::Gray))
                .bounds([min_tok_s, max_tok_s])
                .labels(vec![
                    Span::raw(format!("{:.2}", min_tok_s)),
                    Span::raw(format!("{:.2}", (min_tok_s + max_tok_s) / 2.0)),
                    Span::raw(format!("{:.2}", max_tok_s)),
                ]),
        );

    f.render_widget(chart_tok_s, bottom_chunks[1]);
}
