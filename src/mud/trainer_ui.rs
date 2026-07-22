//! Unified, homogeneous console presentation for the MUD corpus trainer.
//!
//! Single source of truth for box geometry and tag styling so the end-to-end
//! trainer output reads as one coherent, professional session instead of a
//! patchwork of boxes, emoji banners, and mixed stderr/stdout notes.
//!
//! Box geometry: total width W = 80, interior INNER = 78 (even for clean
//! centering). All helpers are `pub` and allocation-light (no per-line heap
//! churn in the hot path — note prints happen at setup/teardown, not per step).

pub const W: usize = 80;
pub const INNER: usize = W - 2; // 78

const BOLD: &str = "\x1b[1m";
const NC: &str = "\x1b[0m";
const CYAN: &str = "\x1b[1;36m";
const YEL: &str = "\x1b[1;33m";
const GRN: &str = "\x1b[1;32m";
const MAG: &str = "\x1b[1;35m";
const RED: &str = "\x1b[1;31m";

#[inline]
fn rule() -> String {
    "─".repeat(INNER)
}

/// Top/bottom of the box: `╭─────╮` / `╰─────╯`.
pub fn box_top() -> String {
    format!("{BOLD}╭{}╮{NC}", rule())
}
pub fn box_bottom() -> String {
    format!("{BOLD}╰{}╯{NC}", rule())
}

use unicode_width::UnicodeWidthStr;

/// A centered single-line title (e.g. the tool name).
pub fn box_title(text: &str) -> String {
    let w = text.width();
    let total_pad = INNER.saturating_sub(w);
    let left_pad = total_pad / 2;
    let right_pad = total_pad - left_pad;
    format!("{BOLD}│{}{}{}{NC}", " ".repeat(left_pad), text, " ".repeat(right_pad))
}

/// Section divider: `├─ Label ─────────────────────────┤`.
pub fn box_section(label: &str) -> String {
    let prefix = format!("─ {label} ");
    let w = prefix.width();
    let dashes = "─".repeat(INNER.saturating_sub(w));
    format!("{BOLD}├{prefix}{dashes}┤{NC}")
}

/// An indented key/value line inside the box.
/// `key` is colorized; `val` is plain.
pub fn box_kv(key: &str, val: &str) -> String {
    format!("  {CYAN}{key}:{NC}  {val}")
}

/// A flat (no box) indented note line for setup/teardown chatter.
/// `kind` renders in a fixed tag color; keeps a single consistent look.
pub fn note(kind: &str, msg: &str) -> String {
    let (tag, col) = match kind {
        "ok" => ("[ok]", GRN),
        "ram" => ("[ram]", CYAN),
        "stp" => ("[stp]", MAG),
        "warn" => ("[warn]", YEL),
        "err" => ("[err]", RED),
        _ => ("[..]", CYAN),
    };
    format!("  {col}{tag}{NC} {msg}")
}

/// A phase/step marker placed between major stages (replaces emoji banners).
pub fn phase(idx: &str, label: &str) -> String {
    format!("  {YEL}{idx}{NC} {label}")
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
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("logs/circuit.log")
    {
        use std::io::Write;
        let _ = writeln!(f, "{}", line);
    }
    line
}
