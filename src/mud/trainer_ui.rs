//! Unified, homogeneous console presentation for the MUD corpus trainer.
//!
//! Borderless, compact layout — uniform tab stops (no box-drawing characters)
//! to maximize horizontal real-estate on 80-col terminals. All helpers are
//! allocation-light (no per-line heap churn in the hot path — setup/teardown only).
//!
//! Layout convention:
//!   TAG:    10 chars left-aligned (cyan, bold)
//!   Value:  right-padded to uniform column
//!   Group:  2-space indent, section headers in magenta
//!
//! Single source of truth for terminal presentation so the trainer output reads
//! as one coherent, professional session instead of mixed emoji banners.

pub const W: usize = 80;
pub const INNER: usize = W - 2; // 78

const BOLD: &str = "\x1b[1m";
const NC: &str = "\x1b[0m";
const CYAN: &str = "\x1b[1;36m";
const YEL: &str = "\x1b[1;33m";
const GRN: &str = "\x1b[1;32m";
const MAG: &str = "\x1b[1;35m";
const RED: &str = "\x1b[1;31m";

/// A compact section header (no box border): magenta, bold.
pub fn section(label: &str) -> String {
    format!("{BOLD}{MAG}▸ {label}{NC}")
}

/// A key/value pair with uniform tab alignment:
///   `  key:    value`
/// Key is cyan bold, value plain.  The key field is padded to 12 chars.
pub fn kv(key: &str, val: &str) -> String {
    format!("  {CYAN}{key:12}{NC}  {val}")
}

/// A flat note line: `  [tag] message` with color-coded tag.
pub fn note(kind: &str, msg: &str) -> String {
    let (tag, col) = match kind {
        "ok" => ("[ok]", GRN),
        "ram" => ("[ram]", CYAN),
        "stp" => ("[stp]", MAG),
        "warn" => ("[warn]", YEL),
        "err" => ("[err]", RED),
        _ => ("[..]", CYAN),
    };
    format!("  {col}{BOLD}{tag}{NC} {msg}")
}

/// Phase/step markers between major stages.
pub fn phase(idx: &str, label: &str) -> String {
    format!("{YEL}{idx}{NC} {label}")
}

/// Pretty single-line TELEM report for stderr (human-readable, borderless, color-coded).
/// The single-line machine-readable `[TELEM] key=val ...` is still written
/// to `mud_train_metrics.log` for the TUI parser.
pub struct TelemSnapshot {
    pub step: usize,
    pub epoch: usize,
    pub epochs: usize,
    pub block: usize,
    pub blocks: usize,
    pub loss: f32,
    pub lr: f32,
    pub tps: f32,
    pub prog: f32,
    pub grad_norm: f32,
    pub entropy: f32,
    pub varh: f32,
    pub varj: f32,
    pub jepa: f32,
    pub cog: f32,
    pub sigma: f32,
}

pub fn format_telem_pretty(s: &TelemSnapshot) -> String {
    format!(
"  {CYAN}Step{NC}: {step:>6}  {MAG}Epoch{NC}: {epoch:>02}/{epochs:<02}  {YEL}Block{NC}: {block:>05}/{blocks}  \
{RED}Loss{NC}: {loss:>8.4}  {YEL}LR{NC}: {lr:.2e}  {CYAN}Speed{NC}: {tps:5.1} tk/s  {GRN}Prog{NC}: {prog:5.1}%  \
{MAG}VarJ{NC}: {varj:>8.6}  {CYAN}VarH{NC}: {varh:>8.6}  {YEL}JEPA{NC}: {jepa:>8.4}  {RED}σ{NC}: {sigma:.1}%  \
{RED}Grad‖{NC}: {grad_norm:.4}  {CYAN}H{NC}: {entropy:.3}b  {MAG}Cog{NC}: {cog:.2}",
        step=s.step, epoch=s.epoch, epochs=s.epochs, block=s.block, blocks=s.blocks,
        loss=s.loss, lr=s.lr, tps=s.tps, prog=s.prog,
        varh=s.varh, varj=s.varj, jepa=s.jepa, cog=s.cog, sigma=s.sigma,
        grad_norm=s.grad_norm, entropy=s.entropy,
    )
}

/// Append a circuit/telemetry event to `logs/circuit.log` (one line, UTC-ish
/// local timestamp) and return the same line for live stdout. The file is created
/// on first use; failures are swallowed (telemetry must never crash training).
pub fn circuit_event(kind: &str, msg: &str) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (h, m, s) = ((now / 3600) % 24, (now / 60) % 60, now % 60);
    let ts = format!("{:02}:{:02}:{:02}", h, m, s);
    let line = format!("[{}] {} {}", ts, kind, msg);
    // RIESGO-4 FIX (2026-08-03): ensure `logs/` dir exists; previously a missing dir
    // silently swallowed the OpenOptions failure (telemetry lost). Failures are
    // logged, not panicked (P-17: telemetry must never crash training).
    if std::fs::create_dir_all("logs").is_err() {
        log::error!("[circuit_event] failed to create logs/ directory");
    }
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("logs/circuit.log")
    {
        use std::io::Write;
        let _ = writeln!(f, "{}", line);
    } else {
        log::error!("[circuit_event] failed to open logs/circuit.log");
    }
    line
}
