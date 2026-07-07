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
    loss_data: Vec<(f64, f64)>, // (step, loss)
    varj_data: Vec<(f64, f64)>, // (step, varj)
    e_jepa_data: Vec<(f64, f64)>, // (step, e_jepa)
    delta_u_data: Vec<(f64, f64)>, // (step, delta_u)
    step_count: f64,
    window_size: f64,
    scroll_offset: f64,
}

impl App {
    fn new() -> Self {
        Self {
            loss_data: Vec::new(),
            varj_data: Vec::new(),
            e_jepa_data: Vec::new(),
            delta_u_data: Vec::new(),
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

    let file_path = "mud_train_metrics.log";
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

                    if !line.contains("Pos") && !line.contains("---") {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        // Epoch Batch AvgLoss Perplexity LrnRate LossVel VarLoss SatMode Z_Entrop T_Softmx Align(T) Integral σ(v)% Cognitive dE/dt
                        if parts.len() >= 14 {
                            if let Ok(e_jepa) = parts[11].parse::<f64>() { // Integral
                                app.e_jepa_data.push((app.step_count, e_jepa));
                            }
                            if let Ok(varj) = parts[3].parse::<f64>() { // Perplexity
                                app.varj_data.push((app.step_count, varj));
                            }
                            if let Ok(delta_u) = parts[13].parse::<f64>() { // Cognitive Derivative
                                app.delta_u_data.push((app.step_count, delta_u));
                            }
                            if let Ok(loss) = parts[2].parse::<f64>() { // AvgLoss
                                app.step_count += 1.0;
                                app.loss_data.push((app.step_count, loss));
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
    let display_loss: Vec<(f64, f64)> = app
        .loss_data
        .iter()
        .filter(|(x, _)| *x >= min_step)
        .copied()
        .collect();
    let display_varj: Vec<(f64, f64)> = app
        .varj_data
        .iter()
        .filter(|(x, _)| *x >= min_step)
        .copied()
        .collect();

    let display_e_jepa: Vec<(f64, f64)> = app
        .e_jepa_data
        .iter()
        .filter(|(x, _)| *x >= min_step)
        .copied()
        .collect();
    let display_delta_u: Vec<(f64, f64)> = app
        .delta_u_data
        .iter()
        .filter(|(x, _)| *x >= min_step)
        .copied()
        .collect();

    let max_loss = display_loss
        .iter()
        .map(|(_, y)| *y)
        .fold(0.0f64, f64::max)
        .max(5.0);
    let min_loss = display_loss
        .iter()
        .map(|(_, y)| *y)
        .fold(max_loss, f64::min)
        .min(0.0);

    let max_varj = display_varj
        .iter()
        .map(|(_, y)| *y)
        .fold(0.0f64, f64::max)
        .max(1.0);
    let min_varj = display_varj
        .iter()
        .map(|(_, y)| *y)
        .fold(max_varj, f64::min)
        .min(0.0);

    let max_e_jepa = display_e_jepa
        .iter()
        .map(|(_, y)| *y)
        .fold(0.0f64, f64::max)
        .max(1.0);
    let min_e_jepa = display_e_jepa
        .iter()
        .map(|(_, y)| *y)
        .fold(max_e_jepa, f64::min)
        .min(-1.0);

    let max_delta_u = display_delta_u
        .iter()
        .map(|(_, y)| *y)
        .fold(0.0f64, f64::max)
        .max(1.0);
    let min_delta_u = display_delta_u
        .iter()
        .map(|(_, y)| *y)
        .fold(max_delta_u, f64::min)
        .min(-1.0);

    let datasets_loss = vec![Dataset::default()
        .name("Avg Loss")
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(Color::Cyan))
        .data(&display_loss)];

    let chart_loss = Chart::new(datasets_loss)
        .block(
            Block::default()
                .title(Span::styled(
                    "Real-Time PosLoss Regression",
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
                .title("Loss")
                .style(Style::default().fg(Color::Gray))
                .bounds([min_loss, max_loss])
                .labels(vec![
                    Span::raw(format!("{:.2}", min_loss)),
                    Span::raw(format!("{:.2}", (min_loss + max_loss) / 2.0)),
                    Span::raw(format!("{:.2}", max_loss)),
                ]),
        );

    f.render_widget(chart_loss, top_chunks[0]);

    let datasets_varj = vec![Dataset::default()
        .name("Perplexity")
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(Color::Yellow))
        .data(&display_varj)];

    let chart_varj = Chart::new(datasets_varj)
        .block(
            Block::default()
                .title(Span::styled(
                    "Perplexity (Distribution Sharpness)",
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
                .title("Perplexity")
                .style(Style::default().fg(Color::Gray))
                .bounds([min_varj, max_varj])
                .labels(vec![
                    Span::raw(format!("{:.2}", min_varj)),
                    Span::raw(format!("{:.2}", (min_varj + max_varj) / 2.0)),
                    Span::raw(format!("{:.2}", max_varj)),
                ]),
        );

    f.render_widget(chart_varj, top_chunks[1]);

    let datasets_e_jepa = vec![Dataset::default()
        .name("JEPA Integral")
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(Color::Magenta))
        .data(&display_e_jepa)];

    let chart_e_jepa = Chart::new(datasets_e_jepa)
        .block(
            Block::default()
                .title(Span::styled(
                    "JEPA Attractor (Integral I)",
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
                .title("Integral")
                .style(Style::default().fg(Color::Gray))
                .bounds([min_e_jepa, max_e_jepa])
                .labels(vec![
                    Span::raw(format!("{:.2}", min_e_jepa)),
                    Span::raw(format!("{:.2}", (min_e_jepa + max_e_jepa) / 2.0)),
                    Span::raw(format!("{:.2}", max_e_jepa)),
                ]),
        );

    f.render_widget(chart_e_jepa, bottom_chunks[0]);

    let datasets_delta_u = vec![Dataset::default()
        .name("Conciencia (Cognitive)")
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(Color::Green))
        .data(&display_delta_u)];

    let chart_delta_u = Chart::new(datasets_delta_u)
        .block(
            Block::default()
                .title(Span::styled(
                    "Conciencia (Cognitive Derivative)",
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
                .title("Conciencia")
                .style(Style::default().fg(Color::Gray))
                .bounds([min_delta_u, max_delta_u])
                .labels(vec![
                    Span::raw(format!("{:.2}", min_delta_u)),
                    Span::raw(format!("{:.2}", (min_delta_u + max_delta_u) / 2.0)),
                    Span::raw(format!("{:.2}", max_delta_u)),
                ]),
        );

    f.render_widget(chart_delta_u, bottom_chunks[1]);
}
