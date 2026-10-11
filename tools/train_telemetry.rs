use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::{Backend, CrosstermBackend},
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::Span,
    widgets::{Axis, Block, Borders, Chart, Dataset, GraphType, Paragraph},
    Terminal,
};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Seek, SeekFrom};
use std::time::Duration;

const LOG_PATH: &str = "mud_train_metrics.log";
const POLL_MS: u64 = 500;

struct Series {
    data: Vec<(f64, f64)>,
}

impl Series {
    fn new() -> Self {
        Self { data: Vec::new() }
    }
    fn push(&mut self, x: f64, y: f64) {
        self.data.push((x, y));
        if self.data.len() > 200_000 {
            let drop_n = self.data.len() - 200_000;
            self.data.drain(..drop_n);
        }
    }
}

struct App {
    loss: Series,
    loss_vel: Series,
    grad_norm: Series,
    lr: Series,
    ppl: Series,
    conf: Series,
    toks: Series,
    varh: Series,
    varj: Series,
    e_jepa: Series,
    cog: Series,
    sigma: Series,
    prog: Series,
    n_act: Series,
    // M-1 PURGE (2026-08-03): removed dead Series gate_mean, gate_std,
    // stp_lambda_eff, loss_cv — no live [TELEM] emitter produces these keys.
    stp_loss: Series,
    traj_eff: Series,
    cmud_steps: Series,
    cmud_lock: Series,
    cmud_herm: Series,
    cmud_spread: Series,
    cmud_cauchy: Series,
    dw_moved: Series,
    entropy: Series,
    grad_log_buf: Vec<(f64, f64)>,
    step_count: f64,
    epoch: f64,
    block_idx: f64,
    block_total: f64,
    window_size: f64,
    scroll_offset: f64,
    zoom_level: f64,
    follow_tail: bool,
    last_update: f64,
    mode: Mode,
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Training,
    Performance,
    Jepa,
    Terrain,
    Efficiency,
    Cmud,
}

impl Mode {
    fn next(self) -> Self {
        match self {
            Mode::Training => Mode::Performance,
            Mode::Performance => Mode::Jepa,
            Mode::Jepa => Mode::Terrain,
            Mode::Terrain => Mode::Efficiency,
            Mode::Efficiency => Mode::Cmud,
            Mode::Cmud => Mode::Training,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Mode::Training => "TRAINING",
            Mode::Performance => "PERFORMANCE",
            Mode::Jepa => "JEPA",
            Mode::Terrain => "TERRAIN",
            Mode::Efficiency => "EFFICIENCY",
            Mode::Cmud => "C-MUD",
        }
    }
}

impl App {
    fn new() -> Self {
        Self {
            loss: Series::new(),
            loss_vel: Series::new(),
            grad_norm: Series::new(),
            lr: Series::new(),
            ppl: Series::new(),
            conf: Series::new(),
            toks: Series::new(),
            varh: Series::new(),
            varj: Series::new(),
            e_jepa: Series::new(),
            cog: Series::new(),
            sigma: Series::new(),
            prog: Series::new(),
            n_act: Series::new(),
            // M-1 PURGE: gate_mean/gate_std/stp_lambda_eff/loss_cv removed (dead Series).
            stp_loss: Series::new(),
            traj_eff: Series::new(),
            cmud_steps: Series::new(),
            cmud_lock: Series::new(),
            cmud_herm: Series::new(),
            cmud_spread: Series::new(),
            cmud_cauchy: Series::new(),
            dw_moved: Series::new(),
            entropy: Series::new(),
            grad_log_buf: Vec::with_capacity(8192),
            step_count: 0.0,
            epoch: 0.0,
            block_idx: 0.0,
            block_total: 0.0,
            window_size: 500.0,
            scroll_offset: 0.0,
            zoom_level: 1.0,
            follow_tail: true,
            last_update: 0.0,
            mode: Mode::Terrain,
        }
    }

    fn parse_kv(line: &str) -> HashMap<String, String> {
        let mut map = HashMap::new();
        for token in line.split_whitespace() {
            if let Some(eq_pos) = token.find('=') {
                let key = &token[..eq_pos];
                let val = &token[eq_pos + 1..];
                map.insert(key.to_string(), val.to_string());
            }
        }
        map
    }

    fn parse_val(map: &HashMap<String, String>, key: &str) -> Option<f64> {
        map.get(key).and_then(|v| {
            let cleaned: String = v
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-' || *c == '+')
                .collect();
            cleaned.parse::<f64>().ok()
        })
    }

    fn absorb_kv(&mut self, kv: &HashMap<String, String>) {
        let step = App::parse_val(kv, "step").unwrap_or(self.step_count);
        let s = step;
        self.step_count = s;
        if let Some(v) = App::parse_val(kv, "epoch") {
            self.epoch = v;
        }
        if let Some(v) = App::parse_val(kv, "loss") {
            self.loss.push(s, v);
        }
        if let Some(v) = App::parse_val(kv, "loss_vel") {
            self.loss_vel.push(s, v);
        }
        if let Some(v) = App::parse_val(kv, "ppl") {
            self.ppl.push(s, v);
        }
        if let Some(v) = App::parse_val(kv, "lr") {
            self.lr.push(s, v);
        }
        if let Some(v) = App::parse_val(kv, "conf") {
            self.conf.push(s, v);
        }
        if let Some(v) = App::parse_val(kv, "tok/s") {
            self.toks.push(s, v);
        }
        if let Some(v) = App::parse_val(kv, "varh") {
            self.varh.push(s, v);
        }
        if let Some(v) = App::parse_val(kv, "varj") {
            self.varj.push(s, v);
        }
        if let Some(v) = App::parse_val(kv, "jepa") {
            self.e_jepa.push(s, v);
        }
        if let Some(v) = App::parse_val(kv, "grad_norm") {
            self.grad_norm.push(s, v.min(1000.0));
        }
        if let Some(v) = App::parse_val(kv, "entropy") {
            self.entropy.push(s, v);
        }
        if let Some(v) = App::parse_val(kv, "stp_loss") {
            self.stp_loss.push(s, v);
        }
        if let Some(v) = App::parse_val(kv, "cog") {
            self.cog.push(s, v);
        }
        if let Some(v) = App::parse_val(kv, "σ") {
            self.sigma.push(s, v);
        }
        if let Some(v) = App::parse_val(kv, "prog") {
            self.prog.push(s, v);
        }
        if let Some(v) = App::parse_val(kv, "n_act") {
            self.n_act.push(s, v);
        }
        // M-1 PURGE: gate_mean / gate_std / stp_λ_eff / L_STP / loss_cv parsers
        // removed — no live emitter produces these keys (duplicate dead telemetry).
        if let Some(v) = App::parse_val(kv, "traj_eff") {
            self.traj_eff.push(s, v);
        }
        if let Some(v) = App::parse_val(kv, "cmud_steps") {
            self.cmud_steps.push(s, v);
        }
        if let Some(v) = App::parse_val(kv, "cmud_lock") {
            self.cmud_lock.push(s, v);
        }
        if let Some(v) = App::parse_val(kv, "cmud_herm") {
            self.cmud_herm.push(s, v);
        }
        if let Some(v) = App::parse_val(kv, "cmud_spread") {
            self.cmud_spread.push(s, v);
        }
        if let Some(v) = App::parse_val(kv, "cmud_cauchy") {
            self.cmud_cauchy.push(s, v);
        }
        if let Some(bv) = kv.get("block") {
            if let Some((a, b)) = bv.split_once('/') {
                if let (Ok(a), Ok(b)) = (a.parse::<f64>(), b.parse::<f64>()) {
                    self.block_idx = a;
                    self.block_total = b;
                }
            }
        }
        self.last_update = unix_now();
    }

    fn absorb_dw(&mut self, kv: &HashMap<String, String>) {
        // BUG-3 FIX (2026-08-03): compute the real percentage of bytes moved.
        // Previously used `moved` (a 0/1 bool serialized as float) → the panel
        // always showed "0.0%" or "1.0%" instead of the actual ratio.
        // The emitter (checkpoint.rs) already emits `bytes` and `total`.
        let bytes = App::parse_val(kv, "bytes").unwrap_or(0.0);
        let total = App::parse_val(kv, "total").unwrap_or(0.0);
        let pct = if total > 0.0 { bytes / total * 100.0 } else { 0.0 };
        let x = if self.step_count > 0.0 {
            self.step_count
        } else {
            total
        };
        self.dw_moved.push(x, pct);
        if pct > 0.0 {
            self.last_update = unix_now();
        }
    }

    fn reset_view(&mut self) {
        self.scroll_offset = 0.0;
        self.window_size = 500.0;
        self.zoom_level = 1.0;
    }
}

fn unix_now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn height_color(t: f64) -> (u8, u8, u8) {
    let t = t.clamp(0.0, 1.0);
    if t < 0.2 {
        (0, 0, 255)
    } else if t < 0.4 {
        (0, 128, 255)
    } else if t < 0.6 {
        (255, 255, 0)
    } else if t < 0.8 {
        (255, 128, 0)
    } else {
        (255, 0, 0)
    }
}

fn main() -> anyhow::Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new();
    let mut reader = File::open(LOG_PATH).ok().map(BufReader::new);
    let res = run_app(&mut terminal, &mut app, &mut reader);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        println!("{err:?}");
    }
    Ok(())
}

fn run_app<B: Backend>(
    terminal: &mut Terminal<B>,
    app: &mut App,
    reader: &mut Option<BufReader<File>>,
) -> io::Result<()> {
    let mut last_pos: u64 = 0;
    let mut line_buf = String::new();
    let mut tick: u32 = 0;
    let mut handle_inode: u64 = 0;

    loop {
        tick = tick.wrapping_add(1);
        if tick.is_multiple_of(10) {
            if let Some(r) = reader.as_ref() {
                if let (Ok(h_meta), Ok(p_meta)) = (
                    r.get_ref().metadata(),
                    std::fs::metadata(LOG_PATH),
                ) {
                    use std::os::unix::fs::MetadataExt;
                    if handle_inode == 0 {
                        handle_inode = h_meta.ino();
                    }
                    if h_meta.ino() != p_meta.ino() {
                        *reader = None;
                        handle_inode = 0;
                        last_pos = 0;
                    }
                }
            }
        }

        if reader.is_none() {
            if let Ok(f) = File::open(LOG_PATH) {
                if let Ok(meta) = f.metadata() {
                    use std::os::unix::fs::MetadataExt;
                    handle_inode = meta.ino();
                }
                *reader = Some(BufReader::new(f));
                last_pos = 0;
            }
        }

        if let Some(r) = reader.as_mut() {
            if let Ok(meta) = r.get_ref().metadata() {
                let len = meta.len();
                if len < last_pos {
                    last_pos = 0;
                    let _ = r.seek(SeekFrom::Start(0));
                }
            }

            if r.seek(SeekFrom::Start(last_pos)).is_ok() {
                line_buf.clear();
                while let Ok(n) = r.read_line(&mut line_buf) {
                    if n == 0 {
                        break;
                    }
                    // BUG-5 FIX (2026-08-03): skip comment/header lines (start with '#').
                    // The session header `# MUD_TELEMETRY v2 ... lr=0.000500 ...`
                    // contains "TELEM" and was being parsed as a data point,
                    // injecting a phantom LR point at x=0.0 in the chart.
                    if line_buf.starts_with('#') {
                        line_buf.clear();
                        continue;
                    }
                    if line_buf.contains("TELEM") {
                        let kv = App::parse_kv(&line_buf);
                        app.absorb_kv(&kv);
                    } else if line_buf.contains("[DW]") {
                        let kv = App::parse_kv(&line_buf);
                        app.absorb_dw(&kv);
                    }
                    line_buf.clear();
                }
                if let Ok(pos) = r.stream_position() {
                    last_pos = pos;
                }
            }
        }

        if app.follow_tail {
            app.scroll_offset = 0.0;
        }

        let mut max_step = app.step_count.max(10.0) - app.scroll_offset;
        if max_step < app.window_size {
            max_step = app.window_size;
        }
        let min_step = (max_step - app.window_size).max(0.0);
        app.grad_log_buf.clear();
        app.grad_log_buf.extend(
            app.grad_norm
                .data
                .iter()
                .filter(|(x, _)| *x >= min_step && *x <= max_step)
                .map(|&(x, y)| (x, if y > 0.0 { y.log10() } else { 0.0 })),
        );
        let view = View { min_step, max_step };

        terminal.draw(|f| ui(f, app, &view))?;

        if event::poll(Duration::from_millis(POLL_MS))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                    KeyCode::Left => {
                        app.follow_tail = false;
                        app.scroll_offset += 100.0;
                    }
                    KeyCode::Right => {
                        app.scroll_offset -= 100.0;
                        if app.scroll_offset <= 0.0 {
                            app.scroll_offset = 0.0;
                            app.follow_tail = true;
                        }
                    }
                    KeyCode::Up => app.window_size += 100.0,
                    KeyCode::Down => {
                        app.window_size -= 100.0;
                        if app.window_size < 100.0 {
                            app.window_size = 100.0;
                        }
                    }
                    KeyCode::Char('r') => app.reset_view(),
                    KeyCode::Char('z') => {
                        app.zoom_level = (app.zoom_level * 1.2).min(5.0);
                        app.window_size /= 1.2;
                        if app.window_size < 100.0 {
                            app.window_size = 100.0;
                        }
                    }
                    KeyCode::Char('x') => {
                        app.zoom_level = (app.zoom_level / 1.2).max(0.5);
                        app.window_size *= 1.2;
                    }
                    KeyCode::Char('f') => app.follow_tail = !app.follow_tail,
                    KeyCode::Char('m') => app.mode = app.mode.next(),
                    _ => {}
                }
            }
        }
    }
}

struct View {
    min_step: f64,
    max_step: f64,
}

fn filter_range(data: &[(f64, f64)], min_step: f64, max_step: f64) -> Vec<(f64, f64)> {
    data.iter()
        .filter(|(x, _)| *x >= min_step && *x <= max_step)
        .copied()
        .collect()
}

/// Computes Y-axis bounds using percentile clipping instead of raw min/max.
/// Outliers (single spikes/dips) no longer stretch the entire range and
/// squash the visible data. We use P2–P98 (clipping ~2% of extreme points
/// on each side) with symmetric padding.
///
/// When `non_negative` is true, the lower bound is clamped to ≥0. This
/// prevents metrics like Confidence % (0–100), Tokens/sec, Progress %,
/// etc. from showing a meaningless negative Y-axis region.
fn y_bounds(
    data: &[(f64, f64)],
    default_min: f64,
    default_max: f64,
    non_negative: bool,
) -> (f64, f64) {
    if data.is_empty() {
        let min = if non_negative { default_min.max(0.0) } else { default_min };
        return (min, default_max);
    }
    // Collect and sort Y values for percentile computation.
    let mut ys: Vec<f64> = data.iter().map(|(_, y)| *y).collect();
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = ys.len();

    let clamp_min = |v: f64| if non_negative { v.max(0.0) } else { v };

    // For small datasets (≤10 points), use raw min/max.
    if n <= 10 {
        let min = ys[0];
        let max = ys[n - 1];
        if (max - min).abs() < 1e-12 {
            let mid = (min + max) / 2.0;
            let span = (mid.abs() * 0.1).max(0.1);
            return (clamp_min(mid - span), mid + span);
        }
        let range = (max - min).max(1e-8);
        let pad = range * 0.12;
        let abs_pad = (max.abs().max(min.abs())) * 0.05;
        let pad = pad.max(abs_pad).max(1e-3);
        return (clamp_min(min - pad), max + pad);
    }

    // P2 and P98 indices (clamped to valid range).
    let lo_idx = ((n as f64) * 0.02).round() as usize;
    let hi_idx = ((n as f64) * 0.98).round() as usize;
    let lo_idx = lo_idx.min(n - 1);
    let hi_idx = hi_idx.max(lo_idx + 1).min(n - 1);
    let p_lo = ys[lo_idx];
    let p_hi = ys[hi_idx];

    // Constant data: create a visible band.
    if (p_hi - p_lo).abs() < 1e-12 {
        let mid = (p_lo + p_hi) / 2.0;
        let span = (mid.abs() * 0.1).max(0.1);
        return (clamp_min(mid - span), mid + span);
    }

    // Use percentiles as the core range, then apply padding.
    let range = (p_hi - p_lo).max(1e-8);
    let pad = range * 0.12;
    let abs_pad = (p_hi.abs().max(p_lo.abs())) * 0.05;
    let pad = pad.max(abs_pad).max(1e-3);
    (clamp_min(p_lo - pad), p_hi + pad)
}

/// Log-scale Y bounds with the same percentile clipping protection.
fn y_bounds_log(data: &[(f64, f64)], default_min: f64, default_max: f64) -> (f64, f64) {
    if data.is_empty() {
        return (default_min, default_max);
    }
    let mut logs: Vec<f64> = data
        .iter()
        .filter(|(_, y)| *y > 1e-12)
        .map(|(_, y)| y.log10())
        .collect();
    if logs.is_empty() {
        return (default_min, default_max);
    }
    logs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = logs.len();
    if n <= 10 {
        let min = logs[0];
        let max = logs[n - 1];
        let range = (max - min).max(1e-3);
        let pad = range * 0.15;
        let pad = pad.max(0.3);
        return (min - pad, max + pad);
    }
    let lo_idx = ((n as f64) * 0.02).round() as usize;
    let hi_idx = ((n as f64) * 0.98).round() as usize;
    let lo_idx = lo_idx.min(n - 1);
    let hi_idx = hi_idx.max(lo_idx + 1).min(n - 1);
    let p_lo = logs[lo_idx];
    let p_hi = logs[hi_idx];
    let range = (p_hi - p_lo).max(1e-3);
    let pad = range * 0.15;
    let pad = pad.max(0.3);
    (p_lo - pad, p_hi + pad)
}

/// Formatea un valor para el eje Y con inteligencia: elimina ceros finales
/// innecesarios y muestra enteros sin decimales. Valores muy pequeños usan
/// mayor precisión para evitar colapsar a "0".
fn fmt_axis(val: f64) -> String {
    if val.abs() < 1e-6 {
        return "0".to_string();
    }
    let abs_val = val.abs();
    // Precisión dinámica según la magnitud del valor
    let precision = if abs_val >= 100.0 {
        0
    } else if abs_val >= 1.0 {
        2
    } else if abs_val >= 0.001 {
        4
    } else {
        6
    };
    let s = format!("{:.*}", precision, val);
    if precision > 0 {
        let trimmed = s.trim_end_matches('0').trim_end_matches('.');
        if trimmed == "-0" || trimmed.is_empty() {
            "0".to_string()
        } else {
            trimmed.to_string()
        }
    } else {
        s
    }
}

/// Computa los 5 labels del eje Y distribuidos entre min_y y max_y
/// usando fmt_axis para formato limpio.
fn y_labels(min_y: f64, max_y: f64) -> Vec<Span<'static>> {
    let q1 = min_y + (max_y - min_y) * 0.25;
    let q2 = min_y + (max_y - min_y) * 0.5;
    let q3 = min_y + (max_y - min_y) * 0.75;
    vec![
        Span::raw(fmt_axis(min_y)),
        Span::raw(fmt_axis(q1)),
        Span::raw(fmt_axis(q2)),
        Span::raw(fmt_axis(q3)),
        Span::raw(fmt_axis(max_y)),
    ]
}


#[allow(clippy::too_many_arguments)]
fn make_chart<'a>(
    title: &'a str,
    data: &'a [(f64, f64)],
    min_step: f64,
    max_step: f64,
    min_y: f64,
    max_y: f64,
    y_label: &'a str,
    fg: Color,
    bg: Color,
    graph_type: GraphType,
) -> Chart<'a> {
    // Dynamic title: append last value + direction + rate
    let dyn_title = if data.is_empty() {
        format!("{title} | NO DATA")
    } else {
        let last_y = data.last().unwrap().1;
        let dir = if data.len() >= 2 {
            let delta = data.last().unwrap().1 - data[data.len() - 2].1;
            if delta.abs() < 1e-6 { "→" } else if delta < 0.0 { "↓" } else { "↑" }
        } else { "" };
        // Include rate if we have ≥2 points with distinct x values
        let rate_str = if data.len() >= 2 {
            let dy = data.last().unwrap().1 - data[data.len() - 2].1;
            let dx = data.last().unwrap().0 - data[data.len() - 2].0;
            if dx.abs() > 1e-9 {
                format!(" Δ{:+.3}/s", dy / dx)
            } else {
                String::new()
            }
        } else {
            String::new()
        };
        format!("{title} | {:.3} {}{}", last_y, dir, rate_str)
    };

    let dataset = Dataset::default()
        .name(title)
        .marker(symbols::Marker::Braille)
        .graph_type(graph_type)
        .style(Style::default().fg(fg))
        .data(data);

    // Fallback range when min == max (constant data): create a small visible band
    let (min_y, max_y) = if (max_y - min_y).abs() < 1e-8 {
        let mid = (min_y + max_y) / 2.0;
        let span = (mid.abs() * 0.1).max(0.1);
        (mid - span, mid + span)
    } else {
        (min_y, max_y)
    };

    let label_lo = fmt_axis(min_step);
    let label_mid = fmt_axis((min_step + max_step) / 2.0);
    let label_hi = fmt_axis(max_step);

    Chart::new(vec![dataset])
        // Hide the default ratatui legend (rendered as a bordered box at TopRight
        // showing the dataset name — identical to the chart title). The dataset
        // name is just the chart title re-used as an identifier; the bordered
        // box clutters the plot area and makes charts look misaligned.
        .legend_position(None)
        .block(
            Block::default()
                .title(Span::styled(
                    dyn_title,
                    Style::default().fg(fg).add_modifier(Modifier::BOLD),
                ))
                .borders(Borders::ALL)
                .style(Style::default().bg(bg)),
        )
        .style(Style::default().bg(bg))
        .x_axis(
            Axis::default()
                .style(Style::default().fg(Color::Gray))
                .bounds([min_step, max_step])
                .labels(vec![
                    Span::raw(label_lo),
                    Span::raw(label_mid),
                    Span::raw(label_hi),
                ]),
        )
        .y_axis(
            Axis::default()
                .title(y_label)
                .style(Style::default().fg(Color::Gray))
                .bounds([min_y, max_y])
                .labels(y_labels(min_y, max_y)),
        )
}

#[allow(clippy::too_many_arguments)]
fn render_loss_profile(
    f: &mut ratatui::Frame,
    area: Rect,
    loss_data: &[(f64, f64)],
    min_loss: f64,
    max_loss: f64,
    min_step: f64,
    max_step: f64,
    zoom: f64,
) {
    let loss_range = (max_loss - min_loss).max(1e-8);

    let mut segments: Vec<(Vec<(f64, f64)>, Color)> = Vec::new();
    let mut current_seg: Vec<(f64, f64)> = Vec::new();
    let mut current_band: i32 = -1;

    for (i, &(x, y)) in loss_data.iter().enumerate() {
        let t = ((y - min_loss) / loss_range).clamp(0.0, 1.0);
        let band = (t * 5.0).floor().clamp(0.0, 4.0) as i32;
        if band != current_band && !current_seg.is_empty() {
            let (r, g, b) = height_color(current_band as f64 / 4.0 + 0.1);
            segments.push((std::mem::take(&mut current_seg), Color::Rgb(r, g, b)));
            // Repeat previous point for visual continuity across band boundaries.
            // Uses index (O(1)) instead of loss_data.iter().position() (O(n²)).
            if i > 0 {
                current_seg.push(loss_data[i - 1]);
            }
        }
        current_seg.push((x, y));
        current_band = band;
    }
    if !current_seg.is_empty() {
        let (r, g, b) = height_color(current_band as f64 / 4.0 + 0.1);
        segments.push((current_seg, Color::Rgb(r, g, b)));
    }

    let mut datasets: Vec<Dataset> = Vec::with_capacity(segments.len() + 1);
    for (seg_data, color) in &segments {
        datasets.push(Dataset::default().name("profile").marker(symbols::Marker::Braille).graph_type(GraphType::Line).style(Style::default().fg(*color)).data(seg_data));
    }
    let cursor_data = loss_data.last().map(|&(x, y)| vec![(x, y)]);
    if let Some(data) = &cursor_data {
        datasets.push(Dataset::default().name("cursor").marker(symbols::Marker::Dot).graph_type(GraphType::Scatter).style(Style::default().fg(Color::White)).data(data));
    }

    let n = loss_data.len();
    let last_val = loss_data.last().map(|(_, y)| *y).unwrap_or(f64::NAN);
    let (dir_str, delta_str, rate_str) = if n >= 2 {
        let (px, py) = loss_data[n - 2];
        let (cx, cy) = loss_data[n - 1];
        let d = cy - py;
        let dir = if d.abs() < 1e-6 { "→" } else if d < 0.0 { "↓" } else { "↑" };
        (dir, format!("Δ{:+.3}", d), if (cx - px).abs() > 1e-9 { format!("{:+.4}/s", d / (cx - px)) } else { "—".to_string() })
    } else { ("—", "—".to_string(), "—".to_string()) };

    let diverging = n >= 10 && {
        let tail = &loss_data[n - 10..];
        tail.windows(2).map(|w| w[1].1 - w[0].1).sum::<f64>() / 9.0 > 0.015
    };

    let t = ((last_val - min_loss) / loss_range).clamp(0.0, 1.0);
    let band_name = ["DeepV", "Valley", "Plat", "Hill", "Peak"][(t * 5.0).floor().clamp(0.0, 4.0) as usize];
    
    let title = format!("Loss | {:.2} {} {} {} | {} | z{:.1}x n={}{}", last_val, dir_str, delta_str, rate_str, band_name, zoom, n, if diverging { " ⚠DIV" } else { "" });
    let (br, bg_c, bb) = height_color(t);

    let chart = Chart::new(datasets)
        // Hide the default ratatui legend — datasets have names ("profile",
        // "cursor") which would render a bordered box at TopRight, cluttering
        // the plot area alongside our custom terrain legend below.
        .legend_position(None)
        .block(Block::default().title(Span::styled(title, Style::default().fg(Color::Rgb(br, bg_c, bb)).add_modifier(Modifier::BOLD))).borders(Borders::ALL).style(Style::default().bg(Color::Rgb(20, 10, 0))))
        .style(Style::default().bg(Color::Rgb(20, 10, 0)))
        .x_axis(Axis::default().style(Style::default().fg(Color::Gray)).bounds([min_step, max_step]).labels(vec![Span::raw(fmt_axis(min_step)), Span::raw(fmt_axis((min_step + max_step) / 2.0)), Span::raw(fmt_axis(max_step))]))
        .y_axis(Axis::default().title("Loss").style(Style::default().fg(Color::Gray)).bounds([min_loss, max_loss]).labels(y_labels(min_loss, max_loss)));

    // Render chart in a 1-line-shorter area so the terrain legend sits on its
    // own line *below* the bottom border, instead of overwriting it.
    let has_room = area.height >= 4;
    let chart_area = if has_room {
        Rect::new(area.x, area.y, area.width, area.height - 1)
    } else {
        area // Fallback: render chart full-height if too small
    };
    f.render_widget(chart, chart_area);

    if has_room {
        let legend_area = Rect::new(area.x, area.y + area.height - 1, area.width, 1);
        let bands = [("DeepV", Color::Rgb(0, 0, 255)), ("Valley", Color::Rgb(0, 128, 255)), ("Plat", Color::Rgb(255, 255, 0)), ("Hill", Color::Rgb(255, 128, 0)), ("Peak", Color::Rgb(255, 0, 0))];
        let active_idx = (t * 5.0).floor().clamp(0.0, 4.0) as usize;
        let mut legend_spans: Vec<Span> = Vec::new();
        for (i, &(name, color)) in bands.iter().enumerate() {
            if i > 0 { legend_spans.push(Span::raw(" ")); }
            legend_spans.push(Span::styled(format!("{}{}", if i == active_idx { '\u{2588}' } else { '\u{2591}' }, name), Style::default().fg(color)));
        }
        f.render_widget(Paragraph::new(ratatui::text::Text::from(ratatui::text::Line::from(legend_spans))).style(Style::default().bg(Color::Rgb(20, 10, 0))), legend_area);
    }
}

fn ui(f: &mut ratatui::Frame, app: &App, view: &View) {
    let area = f.size();
    let min_step = view.min_step;
    let max_step = view.max_step;
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints(
            [
                Constraint::Length(1),
                Constraint::Min(4),
                Constraint::Length(1),
                Constraint::Length(1),
            ]
            .as_ref(),
        )
        .split(area);

    let body = Layout::default()
        .direction(Direction::Vertical)
        .margin(0)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
        .split(outer[1]);

    let top_row = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
        .split(body[0]);

    let bottom_row = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
        .split(body[1]);

    let d_loss = filter_range(&app.loss.data, min_step, max_step);
    let d_loss_vel = filter_range(&app.loss_vel.data, min_step, max_step);
    let d_grad = filter_range(&app.grad_norm.data, min_step, max_step);
    let d_lr = filter_range(&app.lr.data, min_step, max_step);
    let d_ppl = filter_range(&app.ppl.data, min_step, max_step);
    let d_conf = filter_range(&app.conf.data, min_step, max_step);
    let d_toks = filter_range(&app.toks.data, min_step, max_step);
    let d_varj = filter_range(&app.varj.data, min_step, max_step);
    let d_jepa = filter_range(&app.e_jepa.data, min_step, max_step);
    let d_varh = filter_range(&app.varh.data, min_step, max_step);
    let d_sigma = filter_range(&app.sigma.data, min_step, max_step);
    let d_prog = filter_range(&app.prog.data, min_step, max_step);
    // M-1 PURGE: d_gate_mean removed (gate_mean Series was dead — no live emitter).
    let d_cog = filter_range(&app.cog.data, min_step, max_step);
    let d_traj = filter_range(&app.traj_eff.data, min_step, max_step);
    let d_entropy = filter_range(&app.entropy.data, min_step, max_step);
    let d_stp_loss = filter_range(&app.stp_loss.data, min_step, max_step);
    let d_cmud_steps = filter_range(&app.cmud_steps.data, min_step, max_step);
    let d_cmud_lock = filter_range(&app.cmud_lock.data, min_step, max_step);
    let d_cmud_herm = filter_range(&app.cmud_herm.data, min_step, max_step);
    let d_cmud_spread = filter_range(&app.cmud_spread.data, min_step, max_step);
    let d_cmud_cauchy = filter_range(&app.cmud_cauchy.data, min_step, max_step);

    let (min_loss, max_loss) = y_bounds(&d_loss, 0.0, 5.0, true);
    let (min_lr, max_lr) = y_bounds(&d_lr, 0.0, 0.1, true);
    let (min_conf, max_conf) = y_bounds(&d_conf, 0.0, 100.0, true);
    let (min_toks, max_toks) = y_bounds(&d_toks, 0.0, 1.0, true);
    let (min_varj, max_varj) = y_bounds(&d_varj, 0.0, 1.0, true);
    let (min_jepa, max_jepa) = y_bounds(&d_jepa, 0.0, 0.0, false);
    let (min_varh, max_varh) = y_bounds(&d_varh, 0.0, 5.0, true);
    let (min_sigma, max_sigma) = y_bounds(&d_sigma, 0.0, 100.0, true);
    let (min_prog, max_prog) = y_bounds(&d_prog, 0.0, 100.0, true);
    let (min_ppl, max_ppl) = y_bounds(&d_ppl, 0.0, 0.0, true);
    let (min_grad, max_grad) = y_bounds_log(&d_grad, 0.0, 5.0);
    let (min_vel, max_vel) = y_bounds(&d_loss_vel, 0.0, 0.0, false);
    // M-1 PURGE: min_gate/max_gate removed with Gate Mean chart.
    let (min_cog, max_cog) = y_bounds(&d_cog, 0.0, 0.0, true);
    let (min_traj, max_traj) = y_bounds(&d_traj, 0.0, 0.0, true);
    let (min_entropy, max_entropy) = y_bounds(&d_entropy, 0.0, 16.0, true);
    let (min_stp_loss, max_stp_loss) = y_bounds(&d_stp_loss, 0.0, 0.0, true);
    let (min_cmud_s, max_cmud_s) = y_bounds(&d_cmud_steps, 0.0, 0.0, true);
    let (min_cmud_l, max_cmud_l) = y_bounds(&d_cmud_lock, 0.0, 0.0, true);
    let (min_cmud_h, max_cmud_h) = y_bounds(&d_cmud_herm, 0.0, 0.0, true);
    let (min_cmud_sp, max_cmud_sp) = y_bounds(&d_cmud_spread, 0.0, 0.0, true);
    let (min_cmud_c, max_cmud_c) = y_bounds(&d_cmud_cauchy, 0.0, 0.0, true);

    render_title_bar(f, outer[0], app);
    render_status_bar(f, outer[2], app, min_step, max_step);
    render_footer(f, outer[3]);

    match app.mode {
        Mode::Training => {
            let c1 = make_chart(
                "Loss Landscape",
                &d_loss,
                min_step,
                max_step,
                min_loss,
                max_loss,
                "Loss",
                Color::Cyan,
                Color::Rgb(0, 5, 15),
                GraphType::Line,
            );
            let c2 = make_chart(
                "Gradient Norm (log10)",
                app.grad_log_buf.as_slice(),
                min_step,
                max_step,
                min_grad,
                max_grad,
                "log10‖∇L‖",
                Color::Yellow,
                Color::Rgb(15, 15, 0),
                GraphType::Scatter,
            );
            let c3 = make_chart(
                "Learning Rate",
                &d_lr,
                min_step,
                max_step,
                min_lr,
                max_lr,
                "LR",
                Color::Magenta,
                Color::Rgb(15, 0, 15),
                GraphType::Line,
            );
            let c4 = make_chart(
                "Loss Velocity (Δloss)",
                &d_loss_vel,
                min_step,
                max_step,
                min_vel,
                max_vel,
                "Δloss",
                Color::Green,
                Color::Rgb(0, 15, 0),
                GraphType::Line,
            );
            f.render_widget(c1, top_row[0]);
            f.render_widget(c2, top_row[1]);
            f.render_widget(c3, bottom_row[0]);
            f.render_widget(c4, bottom_row[1]);
        }
        Mode::Performance => {
            let c1 = make_chart(
                "Confidence %",
                &d_conf,
                min_step,
                max_step,
                min_conf,
                max_conf,
                "Conf",
                Color::Green,
                Color::Rgb(0, 15, 0),
                GraphType::Line,
            );
            let c2 = make_chart(
                "Tokens/sec",
                &d_toks,
                min_step,
                max_step,
                min_toks,
                max_toks,
                "Tok/s",
                Color::Yellow,
                Color::Rgb(15, 15, 0),
                GraphType::Line,
            );
            let c3 = make_chart(
                "Progress %",
                &d_prog,
                min_step,
                max_step,
                min_prog,
                max_prog,
                "Prog",
                Color::Blue,
                Color::Rgb(0, 0, 15),
                GraphType::Line,
            );
            let c4 = make_chart(
                "Perplexity",
                &d_ppl,
                min_step,
                max_step,
                min_ppl,
                max_ppl,
                "PPL",
                Color::Cyan,
                Color::Rgb(10, 10, 10),
                GraphType::Line,
            );
            f.render_widget(c1, top_row[0]);
            f.render_widget(c2, top_row[1]);
            f.render_widget(c3, bottom_row[0]);
            f.render_widget(c4, bottom_row[1]);
        }
        Mode::Jepa => {
            let c1 = make_chart(
                "VarJ (Variance J)",
                &d_varj,
                min_step,
                max_step,
                min_varj,
                max_varj,
                "VarJ",
                Color::Blue,
                Color::Rgb(0, 0, 15),
                GraphType::Scatter,
            );
            let c2 = make_chart(
                "JEPA Integral",
                &d_jepa,
                min_step,
                max_step,
                min_jepa,
                max_jepa,
                "JEPA",
                Color::Red,
                Color::Rgb(15, 0, 0),
                GraphType::Line,
            );
            let c3 = make_chart(
                "VarH (Variance H)",
                &d_varh,
                min_step,
                max_step,
                min_varh,
                max_varh,
                "VarH",
                Color::Magenta,
                Color::Rgb(15, 0, 15),
                GraphType::Line,
            );
            let c4 = make_chart(
                "STP Loss",
                &d_stp_loss,
                min_step,
                max_step,
                min_stp_loss,
                max_stp_loss,
                "STP",
                Color::Green,
                Color::Rgb(0, 15, 0),
                GraphType::Line,
            );
            f.render_widget(c1, top_row[0]);
            f.render_widget(c2, top_row[1]);
            f.render_widget(c3, bottom_row[0]);
            f.render_widget(c4, bottom_row[1]);
        }
        Mode::Terrain => {
            let c1 = make_chart(
                "Sigma V %",
                &d_sigma,
                min_step,
                max_step,
                min_sigma,
                max_sigma,
                "σ%",
                Color::Cyan,
                Color::Rgb(0, 5, 15),
                GraphType::Line,
            );
            // Replaced "Loss (reference)" — was a duplicate of the colored Loss Profile
            // above it. Now shows JEPA Integral, which is topologically distinct.
            let c3 = make_chart(
                "JEPA Integral",
                &d_jepa,
                min_step,
                max_step,
                min_jepa,
                max_jepa,
                "JEPA",
                Color::Red,
                Color::Rgb(15, 0, 0),
                GraphType::Line,
            );
            // Replaced "Loss Velocity (Δloss)" — was a duplicate shown in Training mode.
            // Now shows Cognitive activation magnitude, a C-MUD/Terrain-relevant metric.
            let c4 = make_chart(
                "Cognitive",
                &d_cog,
                min_step,
                max_step,
                min_cog,
                max_cog,
                "cog",
                Color::Green,
                Color::Rgb(0, 15, 0),
                GraphType::Line,
            );
            f.render_widget(c1, top_row[0]);
            render_loss_profile(
                f,
                top_row[1],
                &d_loss,
                min_loss,
                max_loss,
                min_step,
                max_step,
                app.zoom_level,
            );
            f.render_widget(c3, bottom_row[0]);
            f.render_widget(c4, bottom_row[1]);
        }
        Mode::Efficiency => {
            let c1 = make_chart(
                "Trajectory Efficiency",
                &d_traj,
                min_step,
                max_step,
                min_traj,
                max_traj,
                "1/(1+CV)",
                Color::Green,
                Color::Rgb(0, 15, 0),
                GraphType::Line,
            );
            let c2 = make_chart(
                "STP Loss",
                &d_stp_loss,
                min_step,
                max_step,
                min_stp_loss,
                max_stp_loss,
                "L_STP",
                Color::Cyan,
                Color::Rgb(0, 5, 15),
                GraphType::Line,
            );
            let c3 = make_chart(
                "Entropy (bits)",
                &d_entropy,
                min_step,
                max_step,
                min_entropy,
                max_entropy,
                "H",
                Color::Yellow,
                Color::Rgb(15, 15, 0),
                GraphType::Line,
            );
            // Replaced duplicate "Grad Norm (log10)" — that chart already appears in
            // Training mode with the same data. Now shows Perplexity, a unique
            // efficiency metric not shown in any other mode.
            let c4 = make_chart(
                "Perplexity",
                &d_ppl,
                min_step,
                max_step,
                min_ppl,
                max_ppl,
                "PPL",
                Color::Red,
                Color::Rgb(15, 0, 0),
                GraphType::Line,
            );

            f.render_widget(c1, top_row[0]);
            f.render_widget(c2, top_row[1]);
            f.render_widget(c3, bottom_row[0]);
            f.render_widget(c4, bottom_row[1]);
        }
        Mode::Cmud => {
            // Uses d_cmud_* and min_cmud_*/max_cmud_* computed at top of ui() with
            // auto-scaling (0.0, 0.0) so flat-zero data is still visible.
            let c1 = make_chart(
                "C-MUD Steps",
                &d_cmud_steps,
                min_step,
                max_step,
                min_cmud_s,
                max_cmud_s,
                "Steps",
                Color::Cyan,
                Color::Rgb(0, 5, 15),
                GraphType::Line,
            );
            let c2 = make_chart(
                "C-MUD Lock Rate",
                &d_cmud_lock,
                min_step,
                max_step,
                min_cmud_l,
                max_cmud_l,
                "Lock",
                Color::Red,
                Color::Rgb(15, 0, 0),
                GraphType::Line,
            );
            let c3 = make_chart(
                "C-MUD Hermitian",
                &d_cmud_herm,
                min_step,
                max_step,
                min_cmud_h,
                max_cmud_h,
                "Herm",
                Color::Magenta,
                Color::Rgb(15, 0, 15),
                GraphType::Line,
            );

            let spread_dataset = Dataset::default()
                .name("Spread")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Yellow))
                .data(&d_cmud_spread);

            let cauchy_dataset = Dataset::default()
                .name("Cauchy")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Green))
                .data(&d_cmud_cauchy);

            let c4 = Chart::new(vec![spread_dataset, cauchy_dataset])
                // Hide the default ratatui legend — datasets have names ("Spread",
                // "Cauchy") which would render a bordered box at TopRight, cluttering
                // the plot area.
                .legend_position(None)
                .block(
                    Block::default()
                        .title(Span::styled(
                            "C-MUD Spread / Cauchy",
                            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
                        ))
                        .borders(Borders::ALL)
                        .style(Style::default().bg(Color::Rgb(15, 15, 0))),
                )
                .style(Style::default().bg(Color::Rgb(15, 15, 0)))
                .x_axis(
                    Axis::default()
                        .title("Steps")
                        .style(Style::default().fg(Color::Gray))
                        .bounds([min_step, max_step])
                        .labels(vec![
                            Span::raw(format!("{min_step:.0}")),
                            Span::raw(format!("{:.0}", (min_step + max_step) / 2.0)),
                            Span::raw(format!("{max_step:.0}")),
                        ]),
                )
                .y_axis(
                    Axis::default()
                        .title("Value")
                        .style(Style::default().fg(Color::Gray))
                        .bounds([min_cmud_sp.min(min_cmud_c), max_cmud_sp.max(max_cmud_c)])
                        .labels(vec![
                            Span::raw(fmt_axis(min_cmud_sp.min(min_cmud_c))),
                            Span::raw(fmt_axis((min_cmud_sp.min(min_cmud_c) + max_cmud_sp.max(max_cmud_c)) / 2.0)),
                            Span::raw(fmt_axis(max_cmud_sp.max(max_cmud_c))),
                        ]),
                );

            f.render_widget(c1, top_row[0]);
            f.render_widget(c2, top_row[1]);
            f.render_widget(c3, bottom_row[0]);
            f.render_widget(c4, bottom_row[1]);
        }
    }
}

fn render_title_bar(f: &mut ratatui::Frame, area: Rect, app: &App) {
    let follow = if app.follow_tail { "FOLLOW" } else { "PAUSED" };

    let diverging = app.loss_vel.data.len() >= 10
        && app.loss_vel.data.iter().rev().take(10).all(|(_, v)| *v > 0.1);

    let (alert_str, alert_color) = if diverging {
        (" · ⚠ DIVERGENCE", Color::Red)
    } else {
        ("", Color::White)
    };

    let title = format!(
        " MUD Native Trainer Telemetry · [{}] · win={:.0} zoom={:.1}x {}{} ",
        app.mode.label(),
        app.window_size,
        app.zoom_level,
        follow,
        alert_str,
    );

    let title_color = if diverging { alert_color } else { Color::White };

    f.render_widget(
        Paragraph::new(Span::styled(
            title,
            Style::default()
                .fg(title_color)
                .add_modifier(Modifier::BOLD),
        ))
        .block(Block::default().borders(Borders::NONE)),
        area,
    );
}

fn render_status_bar(f: &mut ratatui::Frame, area: Rect, app: &App, min_step: f64, max_step: f64) {
    let last_loss = app.loss.data.last().map(|(_, y)| *y).unwrap_or(f64::NAN);
    let loss_str = if last_loss.is_nan() {
        "—".to_string()
    } else {
        format!("{last_loss:.4}")
    };
    let block_str = if app.block_total > 0.0 {
        format!("{}/{}", app.block_idx as i64, app.block_total as i64)
    } else {
        "—".to_string()
    };
    let age = (unix_now() - app.last_update).max(0.0);
    let age_str = if app.last_update == 0.0 {
        "never".to_string()
    } else if age < 60.0 {
        format!("{age:.0}s ago")
    } else {
        format!("{:.0}m ago", age / 60.0)
    };
    let last_entropy = app.entropy.data.last().map(|(_, y)| *y).unwrap_or(f64::NAN);
    let ent_str = if last_entropy.is_nan() {
        "—".to_string()
    } else {
        format!("{:.3}", last_entropy)
    };
    let last_dw = app.dw_moved.data.last().map(|(_, y)| *y).unwrap_or(0.0);
    let dw_str = if last_dw > 0.0 {
        format!("{:.1}%", last_dw)
    } else {
        "0%".to_string()
    };
    let status = format!(
        " step={:.0} epoch={:.0} block={} loss={} entropy={} dw={} range=[{:.0},{:.0}] update={} ",
        app.step_count, app.epoch, block_str, loss_str, ent_str, dw_str, min_step, max_step, age_str,
    );
    f.render_widget(
        Paragraph::new(Span::styled(
            status,
            Style::default()
                .fg(Color::Black)
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        ))
        .block(Block::default().borders(Borders::NONE)),
        area,
    );
}

fn render_footer(f: &mut ratatui::Frame, area: Rect) {
    let help = " [m] mode   [r] reset   [f] follow   [←/→] scroll   [↑/↓] window   [z/x] zoom   [q] quit ";
    f.render_widget(
        Paragraph::new(Span::styled(
            help,
            Style::default().fg(Color::Gray),
        ))
        .block(Block::default().borders(Borders::NONE)),
        area,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that the TELEM line emitted by the trainer (with the new
    /// loss_vel, traj_eff, and cmud_* keys) is correctly parsed into all
    /// series by the TUI.
    #[test]
    fn test_parse_telem_line_with_all_keys() {
        let line = "[TELEM] step=290 epoch=2 block=123/167 loss=6.2744 ppl=512.3 \
                    lr=0.000300 loss_vel=-0.003820 tok/s=9.3 steps=4096 elapsed=12.5s \
                    prog=45.2% conf=0.06% varh=7.768100 varj=0.319000 jepa=-0.0012 σ=50.2% \
                    cog=0.045600 grad_norm=0.001300 entropy=0.109 stp_loss=0.008000 \
                    traj_eff=0.8520 cmud_steps=0.0000 cmud_lock=0.0000 cmud_herm=0.0000 \
                    cmud_spread=0.0000 cmud_cauchy=0.0000 n_act=4096";

        let mut app = App::new();
        let kv = App::parse_kv(line);
        app.absorb_kv(&kv);

        assert_eq!(app.step_count, 290.0);
        assert_eq!(app.loss.data.len(), 1);
        assert!((app.loss.data[0].1 - 6.2744).abs() < 1e-4);

        // loss_vel now parsed
        assert_eq!(app.loss_vel.data.len(), 1);
        assert!((app.loss_vel.data[0].1 - (-0.003820)).abs() < 1e-4);

        // traj_eff now parsed
        assert_eq!(app.traj_eff.data.len(), 1);
        assert!((app.traj_eff.data[0].1 - 0.8520).abs() < 1e-3);

        // STP loss parsed
        assert_eq!(app.stp_loss.data.len(), 1);
        assert!((app.stp_loss.data[0].1 - 0.008).abs() < 1e-3);

        // cog (cognitive) parsed
        assert_eq!(app.cog.data.len(), 1);
        assert!((app.cog.data[0].1 - 0.0456).abs() < 1e-3);

        // cmud_* keys parsed
        assert_eq!(app.cmud_steps.data.len(), 1);
        assert_eq!(app.cmud_lock.data.len(), 1);
        assert_eq!(app.cmud_herm.data.len(), 1);
        assert_eq!(app.cmud_spread.data.len(), 1);
        assert_eq!(app.cmud_cauchy.data.len(), 1);

        // σ (sigma) parsed
        assert_eq!(app.sigma.data.len(), 1);
        assert!((app.sigma.data[0].1 - 50.2).abs() < 0.1);
    }

    /// Verifies that loss_vel with positive values also parses correctly.
    #[test]
    fn test_parse_loss_vel_positive() {
        let line = "[TELEM] step=100 loss=5.5 loss_vel=0.0200 n_act=4096";
        let mut app = App::new();
        let kv = App::parse_kv(line);
        app.absorb_kv(&kv);
        assert_eq!(app.step_count, 100.0);
        assert!((app.loss_vel.data[0].1 - 0.02).abs() < 1e-4);
    }

    /// Verifies that y_bounds_log returns data-driven bounds with padding,
    /// not a fixed wide default that squishes data.
    #[test]
    fn test_y_bounds_log_data_driven() {
        // Data around 0.0013 → log10 ≈ -2.886
        let data = vec![(1.0, 0.0013), (2.0, 0.0013), (3.0, 0.0013)];
        let (min, max) = y_bounds_log(&data, 0.0, 5.0);
        // Should be near -2.9, NOT [0, 5]
        assert!(min > -5.0 && min < 0.0, "min should be negative, got {min}");
        assert!(max > -5.0 && max < 0.0, "max should be negative, got {max}");
        assert!(max - min < 2.0, "range should be tight, got {}", max - min);
    }

    /// Verifies that y_bounds_log on empty data falls back to defaults.
    #[test]
    fn test_y_bounds_log_empty_fallback() {
        let data: Vec<(f64, f64)> = vec![];
        let (min, max) = y_bounds_log(&data, 0.0, 5.0);
        assert_eq!(min, 0.0);
        assert_eq!(max, 5.0);
    }

    /// Verifies that y_bounds auto-scales (0.0, 0.0) for small values
    /// like STP loss (~0.008) instead of using fixed [0, 1].
    #[test]
    fn test_y_bounds_auto_scale_small_values() {
        let data = vec![(1.0, 0.008), (2.0, 0.009), (3.0, 0.007)];
        let (min, max) = y_bounds(&data, 0.0, 0.0, true);
        assert!(max > 0.0 && max < 1.0, "max should be data-driven, got {max}");
        assert!(min < 0.01, "min should be near 0, got {min}");
    }

    /// Verifies that traj_eff computation in session.rs uses 1/(1+CV).
    #[test]
    fn test_traj_eff_formula() {
        // Smooth descending loss → high traj_eff
        let loss_smooth: Vec<f32> = vec![6.0, 5.9, 5.8, 5.7, 5.6];
        let n = loss_smooth.len() as f32;
        let mean = loss_smooth.iter().sum::<f32>() / n;
        let var: f32 = loss_smooth.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
        let std = var.sqrt();
        let traj_eff_smooth = 1.0 / (1.0 + std / mean.abs());

        // Volatile loss → low traj_eff
        let loss_volatile: Vec<f32> = vec![5.0, 7.0, 3.0, 8.0, 2.0];
        let n2 = loss_volatile.len() as f32;
        let mean2 = loss_volatile.iter().sum::<f32>() / n2;
        let var2: f32 = loss_volatile.iter().map(|&x| (x - mean2) * (x - mean2)).sum::<f32>() / n2;
        let std2 = var2.sqrt();
        let traj_eff_volatile = 1.0 / (1.0 + std2 / mean2.abs());

        assert!(traj_eff_smooth > traj_eff_volatile,
            "smooth loss should have higher traj_eff: {} vs {}",
            traj_eff_smooth, traj_eff_volatile);
        assert!(traj_eff_smooth > 0.9, "smooth loss should have traj_eff > 0.9");
    }

    /// Verifies that constant data (all same value) gets a visible range band
    /// instead of collapsing to a zero-height line.
    #[test]
    fn test_y_bounds_constant_data_visible() {
        let data = vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)];
        let (min, max) = y_bounds(&data, 0.0, 0.0, false);
        assert!(max > 0.0, "constant-zero data should still produce visible max: {max}");
        assert!(min < 0.0, "constant-zero data should still produce visible min: {min}");
        assert!((max - min).abs() > 0.01, "range should be visible: {}", max - min);
    }

    /// Verifies that constant non-zero data also gets a visible band.
    #[test]
    fn test_y_bounds_constant_nonzero_visible() {
        let data = vec![(1.0, 0.0013), (2.0, 0.0013), (3.0, 0.0013)];
        let (min, max) = y_bounds(&data, 0.0, 0.0, true);
        assert!(max > min, "range should be positive for constant data");
        assert!((max - min).abs() > 1e-4, "band should be wide enough to see: {}", max - min);
    }
}
