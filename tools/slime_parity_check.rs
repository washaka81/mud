use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use forge_llm::mud::slime::{float_to_half_bits, half_to_float_bits};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::Color,
    widgets::{
        canvas::{Canvas, Line, Points},
        Block, Borders,
    },
    Terminal,
};
use std::io::{self, stdout};

fn main() -> io::Result<()> {
    // 1. Generate normal ranges of FP32 for activations and JEPA state.
    // Typical ranges for LLM activations: -10.0 to 10.0, mostly clustered around 0.

    let mut f32_inputs = Vec::new();
    let mut f16_outputs = Vec::new();
    let mut i16_outputs = Vec::new();

    let n_points = 500;

    // iscale for matmul accum simulation: typically 128 / 32767 = 0.0039
    let iscale = 128.0 / 32767.0;

    let mut sum_x = 0.0;
    let mut sum_y_f16 = 0.0;
    let mut sum_xy_f16 = 0.0;
    let mut sum_x2 = 0.0;

    for i in 0..n_points {
        let val: f32 = -5.0 + (10.0 * (i as f32) / (n_points as f32));

        // 16-bit JEPA packing (f16)
        let jepa_packed = float_to_half_bits(val);
        let unpacked_f16 = half_to_float_bits(jepa_packed);

        // 16-bit i16 accum packing
        let accum = (val / iscale).clamp(-32767.0, 32767.0) as i16;
        let unpacked_i16 = (accum as f32) * iscale;

        f32_inputs.push(val);
        f16_outputs.push(unpacked_f16);
        i16_outputs.push(unpacked_i16);

        // For linear regression of f16
        sum_x += val;
        sum_y_f16 += unpacked_f16;
        sum_xy_f16 += val * unpacked_f16;
        sum_x2 += val * val;
    }

    // Linear regression: y = m*x + b
    let n = n_points as f32;
    let m_f16 = (n * sum_xy_f16 - sum_x * sum_y_f16) / (n * sum_x2 - sum_x * sum_x);
    let b_f16 = (sum_y_f16 - m_f16 * sum_x) / n;

    // Coherence Validation (R^2 and MSE)
    let mut ss_tot = 0.0;
    let mut ss_res = 0.0;
    let mut mse_f16 = 0.0;
    let mut mse_i16 = 0.0;
    let y_mean = sum_y_f16 / n;

    for i in 0..n_points {
        let x = f32_inputs[i];
        let y_f16 = f16_outputs[i];
        let y_i16 = i16_outputs[i];
        let y_pred = m_f16 * x + b_f16;

        ss_tot += (y_f16 - y_mean).powi(2);
        ss_res += (y_f16 - y_pred).powi(2);

        // Error cuadrático medio para verificar la deformación de precisión
        mse_f16 += (x - y_f16).powi(2);
        mse_i16 += (x - y_i16).powi(2);
    }

    let r_squared_f16 = 1.0 - (ss_res / ss_tot);
    mse_f16 /= n;
    mse_i16 /= n;

    // Output stats to terminal first
    println!("SlimeRegister Parity Check:");
    println!("Total Points: {}", n_points);
    println!("JEPA f16 Regression: y = {:.6}x + {:.6}", m_f16, b_f16);
    println!("JEPA f16 R² (Coherencia): {:.6}", r_squared_f16);
    println!("JEPA f16 MSE (Error de cuantización): {:.8}", mse_f16);
    println!("Accum i16 MSE (Error de discretización): {:.8}\n", mse_i16);

    // Assertions de Homeostasis Matemática (P-17 Fail-fast)
    if (m_f16 - 1.0).abs() > 0.01 {
        eprintln!("CRITICAL ERROR: Slope m is diverging from 1.0 (Value: {}). Ternary/JEPA packing is failing.", m_f16);
        std::process::exit(1);
    }
    if r_squared_f16 < 0.999 {
        eprintln!(
            "CRITICAL ERROR: Regresión incoherente. R² cayó por debajo de 0.999 (Value: {}).",
            r_squared_f16
        );
        std::process::exit(1);
    }
    if mse_f16 > 1e-4 {
        eprintln!(
            "CRITICAL ERROR: MSE f16 inaceptable ({}). Dispersión extrema en el cast de bits.",
            mse_f16
        );
        std::process::exit(1);
    }

    println!("✅ Coherencia verificada: Paridad estable y matemáticamente sana dentro de SlimeRegister.\n");
    println!("Presiona cualquier tecla para visualizar las gráficas...");
    let mut user_input = String::new();
    io::stdin().read_line(&mut user_input)?;

    // Setup Ratatui terminal
    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Simulate JEPA Homeostasis (Lexical Resonance convergence)
    let mut jepa_orbit: Vec<(f64, f64)> = Vec::new();
    let mut current_z = 5.0; // Initial Lexical Resonance energy at layer 0
    let alpha = 0.05;
    for step in 0..100 {
        // Simulate ternary noise fading as QAT STE learns
        let noise = (step as f32 * 0.8).sin() * 3.0 * (1.0 - (step as f32 / 100.0));
        let error = current_z + noise;
        let correction = alpha * error;
        current_z -= correction;
        jepa_orbit.push((step as f64, current_z as f64));
    }

    loop {
        terminal.draw(|f| {
            let main_chunks = Layout::default()
                .direction(Direction::Vertical)
                .margin(1)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
                .split(f.size());

            let top_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
                .split(main_chunks[0]);

            let points_f16: Vec<(f64, f64)> = f32_inputs
                .iter()
                .zip(f16_outputs.iter())
                .map(|(&x, &y)| (x as f64, y as f64))
                .collect();

            let points_i16: Vec<(f64, f64)> = f32_inputs
                .iter()
                .zip(i16_outputs.iter())
                .map(|(&x, &y)| (x as f64, y as f64))
                .collect();

            // Canvas for f16 JEPA
            let canvas_f16 = Canvas::default()
                .block(
                    Block::default()
                        .title("JEPA f16 Packing (f16 vs f32)")
                        .borders(Borders::ALL),
                )
                .x_bounds([-5.0, 5.0])
                .y_bounds([-5.0, 5.0])
                .paint(|ctx| {
                    ctx.draw(&Points {
                        coords: &points_f16,
                        color: Color::Cyan,
                    });
                    // Regression line
                    ctx.draw(&Line {
                        x1: -5.0,
                        y1: m_f16 as f64 * -5.0 + b_f16 as f64,
                        x2: 5.0,
                        y2: m_f16 as f64 * 5.0 + b_f16 as f64,
                        color: Color::Red,
                    });

                    // Leyenda
                    ctx.print(
                        -4.5,
                        4.0,
                        ratatui::text::Span::styled(
                            "● Dispersión (f16)",
                            ratatui::style::Style::default().fg(Color::Cyan),
                        ),
                    );
                    ctx.print(
                        -4.5,
                        3.2,
                        ratatui::text::Span::styled(
                            "― Regresión y=mx+b",
                            ratatui::style::Style::default().fg(Color::Red),
                        ),
                    );
                });

            // Canvas for i16 accum
            let canvas_i16 = Canvas::default()
                .block(
                    Block::default()
                        .title("Accum i16 Packing (i16 vs f32)")
                        .borders(Borders::ALL),
                )
                .x_bounds([-5.0, 5.0])
                .y_bounds([-5.0, 5.0])
                .paint(|ctx| {
                    ctx.draw(&Points {
                        coords: &points_i16,
                        color: Color::Green,
                    });
                    ctx.draw(&Line {
                        x1: -5.0,
                        y1: -5.0,
                        x2: 5.0,
                        y2: 5.0,
                        color: Color::Red,
                    });

                    // Leyenda
                    ctx.print(
                        -4.5,
                        4.0,
                        ratatui::text::Span::styled(
                            "● Dispersión (i16)",
                            ratatui::style::Style::default().fg(Color::Green),
                        ),
                    );
                    ctx.print(
                        -4.5,
                        3.2,
                        ratatui::text::Span::styled(
                            "― Regresión Ideal",
                            ratatui::style::Style::default().fg(Color::Red),
                        ),
                    );
                });

            // Canvas for JEPA Orbit
            let canvas_orbit = Canvas::default()
                .block(
                    Block::default()
                        .title("JEPA Attractor Convergence (Δz -> 0)")
                        .borders(Borders::ALL),
                )
                .x_bounds([0.0, 100.0])
                .y_bounds([-4.0, 6.0])
                .paint(|ctx| {
                    ctx.draw(&Line {
                        x1: 0.0,
                        y1: 0.0,
                        x2: 100.0,
                        y2: 0.0,
                        color: Color::DarkGray,
                    });
                    for i in 0..jepa_orbit.len().saturating_sub(1) {
                        ctx.draw(&Line {
                            x1: jepa_orbit[i].0,
                            y1: jepa_orbit[i].1,
                            x2: jepa_orbit[i + 1].0,
                            y2: jepa_orbit[i + 1].1,
                            color: Color::Yellow,
                        });
                    }
                    ctx.print(
                        2.0,
                        5.0,
                        ratatui::text::Span::styled(
                            "~ Resonancia Léxica (Capa 0)",
                            ratatui::style::Style::default().fg(Color::Yellow),
                        ),
                    );
                    ctx.print(
                        60.0,
                        1.5,
                        ratatui::text::Span::styled(
                            "~ Homeostasis Matemática",
                            ratatui::style::Style::default().fg(Color::Yellow),
                        ),
                    );
                    ctx.print(
                        80.0,
                        4.0,
                        ratatui::text::Span::styled(
                            "― Línea 0 (Equilibrio)",
                            ratatui::style::Style::default().fg(Color::DarkGray),
                        ),
                    );
                });

            f.render_widget(canvas_f16, top_chunks[0]);
            f.render_widget(canvas_i16, top_chunks[1]);
            f.render_widget(canvas_orbit, main_chunks[1]);
        })?;

        if let Event::Key(key) = event::read()? {
            if key.code == KeyCode::Char('q') || key.code == KeyCode::Esc {
                break;
            }
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    Ok(())
}
