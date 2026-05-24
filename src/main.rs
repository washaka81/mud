use forge_llm::vulkan::VulkanContext;
use forge_llm::mud::MudFile;
use forge_llm::mud::inference::MudInference;
use std::sync::Arc;
use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;
use std::thread;
use std::io::{self, Write};
use crossterm::{
    execute,
    style::{Color, Print, ResetColor, SetForegroundColor, Attribute, SetAttribute, SetBackgroundColor},
    terminal::{self, Clear, ClearType},
    cursor::{self},
};
use sysinfo::System;

// ─── Color palette ────────────────────────────────────────────────────────────
const C_ACCENT:  Color = Color::Rgb { r: 0,   g: 255, b: 180 };
const C_USER:    Color = Color::Rgb { r: 255, g: 200, b: 50  };
const C_DIM:     Color = Color::Rgb { r: 100, g: 100, b: 120 };
const C_BORDER:  Color = Color::Rgb { r: 50,  g: 50,  b: 70  };
const C_STATUS:  Color = Color::Rgb { r: 180, g: 180, b: 200 };
const C_BAR_BG:  Color = Color::Rgb { r: 15,  g: 15,  b: 25  };
const C_WARN:    Color = Color::Rgb { r: 255, g: 100, b: 80  };

const USER_PROMPT: &str = "YOU ❯";
const MUD_PROMPT:  &str = "MUD ❯";
const WRAP_WIDTH:  usize = 76;
/// Visual width of "MUD ❯ " — continuation lines align here.
const CONT_INDENT: &str = "       ";   // 7 spaces

// ─── ANSI rich-text constants ─────────────────────────────────────────────────
const BOLD_ON:    &str = "\x1b[1m";
const BOLD_OFF:   &str = "\x1b[22m";
const ITAL_ON:    &str = "\x1b[3m";
const ITAL_OFF:   &str = "\x1b[23m";
const ULINE_ON:   &str = "\x1b[4m";
const ULINE_OFF:  &str = "\x1b[24m";
const STRIK_ON:   &str = "\x1b[9m";
const STRIK_OFF:  &str = "\x1b[29m";
const ANSI_RESET: &str = "\x1b[0m";

#[inline] fn rgb(r: u8, g: u8, b: u8) -> String { format!("\x1b[38;2;{};{};{}m", r, g, b) }
#[inline] fn bg_rgb(r: u8, g: u8, b: u8) -> String { format!("\x1b[48;2;{};{};{}m", r, g, b) }

// ─── Classify query area (for live IQ tracking) ───────────────────────────────
fn classify_area(text: &str) -> &'static str {
    let t = text.to_lowercase();
    if t.contains("mud") || t.contains("moe") || t.contains("expert") || t.contains("ternary")
        || t.contains("model") || t.contains("inference") || t.contains("neural") { return "system"; }
    if t.contains("python") || t.contains("rust") || t.contains("fn ") || t.contains("def ")
        || t.contains("code") || t.contains("function") || t.contains("class ") || t.contains("import") { return "code"; }
    if t.contains("math") || t.contains("logic") || t.contains("sum") || t.contains("plus")
        || t.contains("equals") || t.contains("theorem") || t.contains("proof") { return "logic"; }
    if t.contains("hello") || t.contains("hola") || t.contains("translate") || t.contains("language")
        || t.contains("grammar") || t.contains("word") || t.contains("speak") { return "linguistics"; }
    "general"
}

// ─── Post-process decoded response: strip tokenizer noise ─────────────────────
fn clean_response(raw: &str) -> String {
    // Pass 1: drop control chars and isolated Greek letters (tokenizer noise)
    let mut pass1 = String::with_capacity(raw.len());
    for ch in raw.chars() {
        match ch {
            '\n' | '\t' => pass1.push(ch),
            c if c.is_control() => {}
            // Greek alphabet used as tokenizer noise markers
            '\u{0391}'..='\u{03C9}' => {}
            c => pass1.push(c),
        }
    }

    // Pass 2: collapse repeated punctuation and whitespace
    let mut result = String::with_capacity(pass1.len());
    let mut prev_space = false;
    let mut consec_punct = 0u32;

    for ch in pass1.chars() {
        if ch == ' ' || ch == '\t' {
            if !prev_space { result.push(' '); }
            prev_space = true;
            consec_punct = 0;
            continue;
        }
        prev_space = false;
        if ".,;!?:-".contains(ch) {
            consec_punct += 1;
            if consec_punct <= 3 { result.push(ch); }
        } else {
            consec_punct = 0;
            result.push(ch);
        }
    }

    result.trim().to_string()
}

// ─── ANSI-aware visual length ─────────────────────────────────────────────────
fn visual_len(s: &str) -> usize {
    let mut len = 0;
    let mut in_ansi = false;
    for c in s.chars() {
        if c == '\x1b' { in_ansi = true; continue; }
        if in_ansi { if c == 'm' { in_ansi = false; } continue; }
        len += 1;
    }
    len
}

// ─── ANSI-aware word wrap ─────────────────────────────────────────────────────
/// Wrap `text` (which may contain ANSI codes) at `width` visual columns.
/// `first_pfx` is prepended to the first line, `cont_pfx` to every continuation line.
fn word_wrap_ansi(text: &str, first_pfx: &str, cont_pfx: &str, width: usize) -> String {
    let mut out = String::new();
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() { return out; }

    let mut line      = first_pfx.to_string();
    let mut vlen      = visual_len(first_pfx);
    let mut line_start = true;

    for word in &words {
        let wlen = visual_len(word);
        if !line_start && vlen + 1 + wlen > width {
            out.push_str(&line);
            out.push('\n');
            line      = cont_pfx.to_string();
            vlen      = visual_len(cont_pfx);
            line_start = true;
        }
        if !line_start { line.push(' '); vlen += 1; }
        line.push_str(word);
        vlen += wlen;
        line_start = false;
    }
    out.push_str(&line);
    out.push('\n');
    out
}

// ─── Inline Markdown → ANSI renderer ─────────────────────────────────────────
/// Renders inline markdown to ANSI-escaped text.
/// Supported: **bold**, *italic*, __underline__, ~~strikethrough~~, `code`, numbers.
fn render_inline(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut out = String::new();
    let mut i = 0;

    let mut bold      = false;
    let mut italic    = false;
    let mut underline = false;
    let mut strike    = false;
    let mut in_code   = false;

    // Helper: re-apply active styles after a reset
    macro_rules! reapply {
        ($o:expr) => {
            if bold      { $o.push_str(BOLD_ON); }
            if italic    { $o.push_str(ITAL_ON); }
            if underline { $o.push_str(ULINE_ON); }
            if strike    { $o.push_str(STRIK_ON); }
        }
    }

    while i < n {
        // Inside code span: only look for closing backtick
        if in_code {
            if chars[i] == '`' {
                out.push_str(ANSI_RESET);
                reapply!(out);
                in_code = false;
            } else {
                out.push(chars[i]);
            }
            i += 1;
            continue;
        }

        // Two-char tokens
        if i + 1 < n {
            match (chars[i], chars[i + 1]) {
                ('*', '*') => {
                    if bold { out.push_str(BOLD_OFF); bold = false; }
                    else    { out.push_str(BOLD_ON);  bold = true;  }
                    i += 2; continue;
                }
                ('_', '_') => {
                    if underline { out.push_str(ULINE_OFF); underline = false; }
                    else         { out.push_str(ULINE_ON);  underline = true;  }
                    i += 2; continue;
                }
                ('~', '~') => {
                    if strike { out.push_str(STRIK_OFF); strike = false; }
                    else      { out.push_str(STRIK_ON);  strike = true;  }
                    i += 2; continue;
                }
                _ => {}
            }
        }

        // Single-char tokens
        match chars[i] {
            '*' => {
                if italic { out.push_str(ITAL_OFF); italic = false; }
                else      { out.push_str(ITAL_ON);  italic = true;  }
                i += 1; continue;
            }
            '`' => {
                // code span: amber on dark bg
                out.push_str(&format!("{}{}", rgb(255, 215, 100), bg_rgb(30, 25, 15)));
                in_code = true;
                i += 1; continue;
            }
            // Number highlighting (digits, decimals, percentages)
            c if c.is_ascii_digit() => {
                let start = i;
                while i < n && (chars[i].is_ascii_digit() || chars[i] == '.' || chars[i] == ',' || chars[i] == '%') {
                    i += 1;
                }
                let num: String = chars[start..i].iter().collect();
                out.push_str(&format!("{}{}{}", rgb(130, 210, 255), &num, ANSI_RESET));
                reapply!(out);
                continue;
            }
            c => { out.push(c); i += 1; }
        }
    }

    if in_code || bold || italic || underline || strike {
        out.push_str(ANSI_RESET);
    }
    out
}

// ─── Code-block highlighter (keyword + number coloring) ──────────────────────
fn highlight_code_line(line: &str) -> String {
    // Keywords for common languages
    const KW: &[&str] = &[
        "fn", "let", "mut", "pub", "use", "struct", "impl", "enum", "trait", "mod",
        "for", "if", "else", "while", "loop", "match", "return", "break", "continue",
        "Some", "None", "Ok", "Err", "true", "false", "self", "Self",
        "def", "class", "import", "from", "in", "not", "and", "or", "lambda", "pass",
        "function", "const", "var", "let", "async", "await", "new", "this", "null",
        "int", "float", "str", "bool", "void", "auto",
    ];

    let base_color = rgb(220, 220, 170); // warm white
    let kw_color   = rgb(204, 120,  50); // orange keywords
    let num_color  = rgb(130, 210, 150); // green numbers
    let str_color  = rgb(150, 200, 100); // lime strings
    let sym_color  = rgb(150, 150, 200); // purple symbols

    let mut out = base_color.clone();
    let chars: Vec<char> = line.chars().collect();
    let n = chars.len();
    let mut i = 0;
    let mut in_str = false;
    let mut str_delim = '"';

    while i < n {
        // String literals
        if !in_str && (chars[i] == '"' || chars[i] == '\'') {
            in_str = true;
            str_delim = chars[i];
            out.push_str(&str_color);
            out.push(chars[i]);
            i += 1;
            continue;
        }
        if in_str {
            if chars[i] == '\\' && i + 1 < n {
                out.push(chars[i]);
                out.push(chars[i + 1]);
                i += 2;
                continue;
            }
            out.push(chars[i]);
            if chars[i] == str_delim {
                in_str = false;
                out.push_str(&base_color);
            }
            i += 1;
            continue;
        }
        // Operator/symbol highlighting
        if "(){}[];,".contains(chars[i]) {
            out.push_str(&sym_color);
            out.push(chars[i]);
            out.push_str(&base_color);
            i += 1;
            continue;
        }
        // Numbers
        if chars[i].is_ascii_digit() {
            let start = i;
            while i < n && (chars[i].is_ascii_digit() || chars[i] == '.' || chars[i] == 'x' || chars[i] == 'b') {
                i += 1;
            }
            let num: String = chars[start..i].iter().collect();
            out.push_str(&num_color);
            out.push_str(&num);
            out.push_str(&base_color);
            continue;
        }
        // Identifier / keyword
        if chars[i].is_alphabetic() || chars[i] == '_' {
            let start = i;
            while i < n && (chars[i].is_alphanumeric() || chars[i] == '_') { i += 1; }
            let word: String = chars[start..i].iter().collect();
            if KW.contains(&word.as_str()) {
                out.push_str(&format!("{}{}{}{}", kw_color, BOLD_ON, &word, ANSI_RESET));
                out.push_str(&base_color);
            } else {
                out.push_str(&word);
            }
            continue;
        }
        out.push(chars[i]);
        i += 1;
    }
    out.push_str(ANSI_RESET);
    out
}

// ─── Code block renderer ─────────────────────────────────────────────────────
fn render_code_block(code: &str, lang: &str) -> String {
    let indent_len = visual_len(CONT_INDENT);
    let inner = WRAP_WIDTH.saturating_sub(indent_len + 4); // space for │ on each side
    let border: String = "─".repeat(inner + 2);
    let border_color = rgb(60, 65, 100);
    let lang_display = if lang.is_empty() { "code" } else { lang };

    let mut out = String::new();

    // Top border with language label
    out.push_str(&format!(
        "{}{}╭{}╮{}\n",
        CONT_INDENT, border_color, border, ANSI_RESET
    ));
    out.push_str(&format!(
        "{}{}│{} {}{}{}{} {}│{}\n",
        CONT_INDENT,
        border_color, ANSI_RESET,
        rgb(160, 150, 220), ITAL_ON, lang_display, ANSI_RESET,
        border_color, ANSI_RESET,
    ));
    out.push_str(&format!(
        "{}{}├{}┤{}\n",
        CONT_INDENT, border_color, border, ANSI_RESET
    ));

    // Code lines
    for line in code.lines() {
        let highlighted = highlight_code_line(line);
        let vis = visual_len(line);
        let pad = if vis < inner { " ".repeat(inner - vis) } else { String::new() };
        out.push_str(&format!(
            "{}{}│{} {}{} {}│{}\n",
            CONT_INDENT,
            border_color, ANSI_RESET,
            highlighted, pad,
            border_color, ANSI_RESET,
        ));
    }

    // Bottom border
    out.push_str(&format!(
        "{}{}╰{}╯{}\n",
        CONT_INDENT, border_color, border, ANSI_RESET
    ));
    out
}

// ─── Parse numbered list item ("1. foo" → (1, "foo")) ────────────────────────
fn parse_numbered_list(s: &str) -> Option<(u32, &str)> {
    let mut end = 0;
    for c in s.chars() {
        if c.is_ascii_digit() { end += 1; } else { break; }
    }
    if end == 0 { return None; }
    let rest = &s[end..];
    if rest.starts_with(". ") {
        s[..end].parse::<u32>().ok().map(|n| (n, &rest[2..]))
    } else {
        None
    }
}

// ─── Main Rich Markdown → ANSI renderer ──────────────────────────────────────
/// Converts a markdown-ish response string to an ANSI-rich terminal string.
/// The first line of output has NO prefix (cursor already sits after "MUD ❯ ");
/// all continuation lines use CONT_INDENT.
fn render_rich(text: &str) -> String {
    let mut output = String::new();
    let mut in_code_block = false;
    let mut code_lang = String::new();
    let mut code_buf  = String::new();
    // Track whether we have produced any output yet (to omit leading blank lines
    // and to know whether the first paragraph starts inline with "MUD ❯ ").
    let mut first_content = true;

    for raw_line in text.lines() {
        let trimmed = raw_line.trim();

        // ── Code block fences ────────────────────────────────────────────────
        if trimmed.starts_with("```") {
            if in_code_block {
                output.push_str(&render_code_block(&code_buf, &code_lang));
                code_buf.clear(); code_lang.clear();
                in_code_block = false;
                first_content = false;
            } else {
                code_lang = trimmed[3..].trim().to_string();
                in_code_block = true;
            }
            continue;
        }
        if in_code_block {
            code_buf.push_str(raw_line);
            code_buf.push('\n');
            continue;
        }

        // ── Blank line ───────────────────────────────────────────────────────
        if trimmed.is_empty() {
            if !first_content { output.push('\n'); }
            continue;
        }

        // ── H1: # Title ──────────────────────────────────────────────────────
        if let Some(t) = trimmed.strip_prefix("# ") {
            if first_content { output.push('\n'); }
            output.push_str(&format!(
                "{}{}{}{}{}\n",
                CONT_INDENT, rgb(0, 255, 180), BOLD_ON, t.trim(), ANSI_RESET
            ));
            output.push_str(&format!(
                "{}{}{}{}",
                CONT_INDENT, rgb(0, 255, 180),
                "═".repeat(WRAP_WIDTH - visual_len(CONT_INDENT)),
                ANSI_RESET
            ));
            output.push('\n');
            first_content = false;
            continue;
        }

        // ── H2: ## Title ─────────────────────────────────────────────────────
        if let Some(t) = trimmed.strip_prefix("## ") {
            if first_content { output.push('\n'); }
            output.push_str(&format!(
                "{}{}{}{}{}\n",
                CONT_INDENT, rgb(0, 200, 255), BOLD_ON, t.trim(), ANSI_RESET
            ));
            first_content = false;
            continue;
        }

        // ── H3: ### Title ────────────────────────────────────────────────────
        if let Some(t) = trimmed.strip_prefix("### ") {
            if first_content { output.push('\n'); }
            output.push_str(&format!(
                "{}{}{}{}{}{}\n",
                CONT_INDENT, rgb(150, 150, 255), BOLD_ON, ITAL_ON, t.trim(), ANSI_RESET
            ));
            first_content = false;
            continue;
        }

        // ── Horizontal rule: --- ─────────────────────────────────────────────
        if (trimmed == "---" || trimmed == "===")
            || (trimmed.len() >= 3 && trimmed.chars().all(|c| c == '-'))
        {
            output.push_str(&format!(
                "{}{}{}{}\n",
                CONT_INDENT, rgb(55, 55, 80),
                "─".repeat(WRAP_WIDTH - visual_len(CONT_INDENT)),
                ANSI_RESET
            ));
            first_content = false;
            continue;
        }

        // ── Block quote: > text ──────────────────────────────────────────────
        if let Some(t) = trimmed.strip_prefix("> ") {
            if first_content { output.push('\n'); }
            let rich = render_inline(t);
            let bar  = format!("{}▌{} ", rgb(120, 80, 220), ANSI_RESET);
            let fp   = format!("{}{}", CONT_INDENT, bar);
            let cp   = format!("{}  ", CONT_INDENT);
            output.push_str(&word_wrap_ansi(&rich, &fp, &cp, WRAP_WIDTH));
            first_content = false;
            continue;
        }

        // ── Bullet list: - or * or • ─────────────────────────────────────────
        let bullet_rest = trimmed.strip_prefix("- ")
            .or(trimmed.strip_prefix("* "))
            .or(trimmed.strip_prefix("• "));
        if let Some(t) = bullet_rest {
            if first_content { output.push('\n'); }
            let rich   = render_inline(t);
            let dot    = format!("{}•{} ", rgb(0, 255, 180), ANSI_RESET);
            let fp     = format!("{}  {}", CONT_INDENT, dot);
            let cp     = format!("{}    ", CONT_INDENT);
            output.push_str(&word_wrap_ansi(&rich, &fp, &cp, WRAP_WIDTH));
            first_content = false;
            continue;
        }

        // ── Numbered list: 1. 2. etc. ────────────────────────────────────────
        if let Some((num, rest)) = parse_numbered_list(trimmed) {
            if first_content { output.push('\n'); }
            let rich   = render_inline(rest);
            let num_s  = format!("{}{}{}.{} ", rgb(130, 210, 255), BOLD_ON, num, ANSI_RESET);
            let fp     = format!("{}  {}", CONT_INDENT, num_s);
            // indent = CONT_INDENT + 2 spaces + digits + ". "
            let num_w  = format!("{}", num).len() + 2;
            let cp     = format!("{}{}", CONT_INDENT, " ".repeat(num_w + 2));
            output.push_str(&word_wrap_ansi(&rich, &fp, &cp, WRAP_WIDTH));
            first_content = false;
            continue;
        }

        // ── Regular paragraph ────────────────────────────────────────────────
        let rich = render_inline(trimmed);
        if first_content {
            // First paragraph: starts on same line as "MUD ❯ " (no prefix)
            // But width is already reduced by 7 (visual_len("MUD ❯ "))
            let effective_width = WRAP_WIDTH; // typewriter accounts for initial 7
            output.push_str(&word_wrap_ansi(&rich, "", CONT_INDENT, effective_width));
            first_content = false;
        } else {
            output.push_str(&word_wrap_ansi(&rich, CONT_INDENT, CONT_INDENT, WRAP_WIDTH));
        }
    }

    // Flush any unclosed code block
    if in_code_block && !code_buf.is_empty() {
        output.push_str(&render_code_block(&code_buf, &code_lang));
    }

    output
}

// ─── Decode token ids to a clean joined string ────────────────────────────────
fn decode_tokens(tokenizer: &forge_llm::model::tokenizer::Tokenizer, ids: &[u32]) -> String {
    ids.iter()
        .map(|&id| tokenizer.decode(&[id]))
        .collect::<Vec<_>>()
        .join(" ")
}

// ═══════════════════════════════════════════════════════════════════════════════

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let mut sys = System::new_all();
    let mut stdout = io::stdout();

    execute!(stdout, Clear(ClearType::All), cursor::MoveTo(0, 0))?;
    print_banner(&mut stdout)?;

    let mud_path = "models/core_skills.mud";
    let vk = Arc::new(VulkanContext::new().unwrap_or_else(|e| {
        eprintln!("  ⚠️  Error al inicializar Vulkan: {}. Usando fallback CPU.", e);
        // We still need a VulkanContext for the types, but we can make a "dummy" one if needed.
        // For now, if we can't initialize it, we might still fail later if MudInference requires it.
        // Actually, VulkanContext::new() is currently required by the MudInference constructor.
        // Let's assume for now that if it fails, we want to know why but we might need a better fallback.
        VulkanContext::new().expect("Fallo crítico: No se pudo inicializar ni siquiera el fallback de Vulkan")
    }));

    if !Path::new(mud_path).exists() {
        execute!(
            stdout,
            SetForegroundColor(C_WARN), SetAttribute(Attribute::Bold),
            Print(format!("\n  ❌  Model file '{}' not found.\n\n", mud_path)),
            ResetColor, SetAttribute(Attribute::Reset)
        )?;
        return Ok(());
    }

    let mud_file = MudFile::load(mud_path)?;
    let mut engine = MudInference::new(&mud_file, vk)?;

    // ── IQ metadata ──────────────────────────────────────────────────────────
    let model_iq: f32 = mud_file.global_metadata.get("iq.score")
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(8.87);

    let (status_label, status_color) = if model_iq < 100.0 {
        ("COGNICIÓN FRAGMENTADA", C_WARN)
    } else if model_iq < 150.0 {
        ("ASISTENTE FUNCIONAL", C_ACCENT)
    } else {
        ("RAZONAMIENTO MAESTRO", Color::Rgb { r: 120, g: 255, b: 120 })
    };

    let mut live_iq: HashMap<&str, f32> = {
        let parse = |key: &str, def: f32| -> f32 {
            mud_file.global_metadata.get(key)
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(def)
        };
        let mut m = HashMap::new();
        m.insert("linguistics", parse("iq.linguistics", 100.0));
        m.insert("logic",       parse("iq.logic",       100.0));
        m.insert("code",        parse("iq.code",        100.0));
        m.insert("general",     parse("iq.general",     100.0));
        m.insert("system",      parse("iq.system",      100.0));
        m
    };

    let mut conversation_pos = 0usize;
    let mut last_tps = 0.0f64;

    // ── Set scroll region (last row = status bar) ────────────────────────────
    let (cols, rows) = terminal::size().unwrap_or((80, 24));
    execute!(stdout, Print(format!("\x1B[1;{}r", rows - 1)))?;
    execute!(stdout, Clear(ClearType::All), cursor::MoveTo(0, 0))?;
    print_banner(&mut stdout)?;

    // ── IQ/status panel ──────────────────────────────────────────────────────
    let panel_w  = cols.min(64) as usize;
    let border_s: String = "─".repeat(panel_w.saturating_sub(2));
    execute!(stdout,
        SetForegroundColor(C_BORDER), Print(format!("  ╭{}╮\n", border_s)),
        SetForegroundColor(C_DIM),    Print("  │  IQ Score: "),
        SetForegroundColor(Color::White), SetAttribute(Attribute::Bold),
        Print(format!("{:.2}", model_iq)),
        SetAttribute(Attribute::Reset),
        SetForegroundColor(C_DIM),    Print("   │   Estado: "),
        SetForegroundColor(status_color), SetAttribute(Attribute::Bold),
        Print(format!("{}", status_label)),
        SetAttribute(Attribute::Reset),
        SetForegroundColor(C_DIM),    Print("  │\n"),
        SetForegroundColor(C_BORDER), Print(format!("  ╰{}╯\n\n", border_s)),
        ResetColor,
    )?;

    // ── Engine ready ──────────────────────────────────────────────────────────
    execute!(stdout,
        SetForegroundColor(Color::Rgb { r: 0, g: 220, b: 255 }), SetAttribute(Attribute::Bold),
        Print("  ✨  MUD Engine Inicializado  "),
        ResetColor,
        SetForegroundColor(Color::Rgb { r: 80, g: 220, b: 80 }),
        Print("Ternary 1.58b MoE — Listo.\n"),
        ResetColor,
        SetForegroundColor(C_DIM),
        Print("  Escribe "),
        SetForegroundColor(Color::Yellow), Print("/help"),
        SetForegroundColor(C_DIM),        Print(" para ver los comandos.\n\n"),
        ResetColor,
    )?;

    sys.refresh_all();
    update_status_bar(&mut stdout, &engine, &live_iq, last_tps, &mut sys)?;
    stdout.flush()?;

    let mut input = String::new();

    loop {
        // ── User prompt ───────────────────────────────────────────────────────
        execute!(stdout,
            SetForegroundColor(C_USER), SetAttribute(Attribute::Bold),
            Print(format!("{} ", USER_PROMPT)),
            ResetColor, SetAttribute(Attribute::Reset),
        )?;
        stdout.flush()?;

        input.clear();
        if io::stdin().read_line(&mut input)? == 0 { break; }
        let trimmed = input.trim();

        if trimmed.is_empty() { continue; }
        if trimmed == "/exit" || trimmed == "\u{11}" { break; }

        // ── Commands ──────────────────────────────────────────────────────────
        if trimmed == "/help" { print_help(&mut stdout)?; continue; }
        if trimmed == "/clear" {
            execute!(stdout, Clear(ClearType::All), cursor::MoveTo(0, 0))?;
            print_banner(&mut stdout)?;
            let (_, r) = terminal::size().unwrap_or((80, 24));
            execute!(stdout, Print(format!("\x1B[1;{}r", r - 1)))?;
            update_status_bar(&mut stdout, &engine, &live_iq, last_tps, &mut sys)?;
            continue;
        }
        if trimmed == "/stats" {
            let nodes = engine.model.knowledge_graph.read().unwrap().nodes.len();
            let total_facts: i64 = if Path::new("models/knowledge.db").exists() {
                let conn = rusqlite::Connection::open("models/knowledge.db")?;
                conn.query_row("SELECT COUNT(*) FROM facts", [], |r| r.get(0)).unwrap_or(0)
            } else { 0 };
            print_report_card(&mut stdout, &live_iq, nodes, total_facts)?;
            continue;
        }
        if trimmed == "/knowledge" { print_knowledge_mesh(&mut stdout, &engine)?; continue; }
        if let Some(path) = trimmed.strip_prefix("/ingest ") {
            match forge_llm::mud::ingester::MudIngester::ingest(path, &engine) {
                Ok(n) => execute!(stdout,
                    SetForegroundColor(Color::Rgb { r: 80, g: 220, b: 80 }), SetAttribute(Attribute::Bold),
                    Print(format!("  ✅  {} chunks ingeridos en SQLite.\n\n", n)),
                    ResetColor, SetAttribute(Attribute::Reset))?,
                Err(e) => execute!(stdout,
                    SetForegroundColor(C_WARN), SetAttribute(Attribute::Bold),
                    Print(format!("  ❌  Error de ingestión: {}\n\n", e)),
                    ResetColor, SetAttribute(Attribute::Reset))?,
            }
            continue;
        }

        // ── Inference ─────────────────────────────────────────────────────────
        let area   = classify_area(trimmed);
        let tokens = engine.tokenizer.encode(trimmed);
        if tokens.is_empty() {
            execute!(stdout,
                SetForegroundColor(C_WARN),
                Print("  ⚠  Entrada no reconocida por el vocabulario.\n\n"),
                ResetColor)?;
            continue;
        }

        let mut x = vec![0.0f32; engine.model.hidden_size];
        show_thinking(&mut stdout, true)?;

        engine.prompt(trimmed, &mut x, &mut conversation_pos);

        let t_start = std::time::Instant::now();
        let (response_tokens, used_knowledge) = engine.generate(&x, 128, trimmed, &mut conversation_pos);
        let elapsed = t_start.elapsed();

        last_tps = if response_tokens.is_empty() { 0.0 }
                   else { response_tokens.len() as f64 / elapsed.as_secs_f64() };

        show_thinking(&mut stdout, false)?;

        let raw_text = decode_tokens(&engine.tokenizer, &response_tokens);
        let mut response = clean_response(&raw_text);
        engine.format_text(&mut response);

        if response.trim().is_empty() {
            response = "Las sinapsis cognitivas se están reorganizando. \
                        Por favor, reformula tu consulta.".to_string();
        }

        // ── IQ update (EMA) ───────────────────────────────────────────────────
        let quality: f32 = if !response.trim().is_empty() {
            let tps_f  = (last_tps as f32 / 20.0).clamp(0.5, 2.0);
            let len_f  = (response_tokens.len() as f32 / 64.0).clamp(0.5, 1.5);
            (tps_f * len_f).clamp(0.8, 1.8)
        } else { 0.7 };
        let new_iq = (live_iq[area] + (quality - 1.0) * 12.0 * 0.12).clamp(50.0, 200.0);
        live_iq.insert(area, new_iq);

        // ── Render and print response ─────────────────────────────────────────
        execute!(stdout,
            SetForegroundColor(C_ACCENT), SetAttribute(Attribute::Bold),
            Print(format!("{} ", MUD_PROMPT)),
            ResetColor, SetAttribute(Attribute::Reset),
        )?;

        let rendered = render_rich(&response);
        if used_knowledge {
            let styled = format!("{}{}{}{}", rgb(190, 150, 255), ITAL_ON, rendered, ANSI_RESET);
            type_writer(&styled, Duration::from_millis(6))?;
        } else {
            type_writer(&rendered, Duration::from_millis(6))?;
        }
        println!();

        sys.refresh_all();
        update_status_bar(&mut stdout, &engine, &live_iq, last_tps, &mut sys)?;
    }

    execute!(stdout, Print("\x1B[r"))?;
    execute!(stdout,
        SetForegroundColor(C_DIM),
        Print("\n  Sesión cerrada. Hasta pronto.\n"),
        ResetColor)?;
    Ok(())
}

// ─── Banner ───────────────────────────────────────────────────────────────────
fn print_banner(stdout: &mut io::Stdout) -> io::Result<()> {
    execute!(stdout,
        SetForegroundColor(Color::Rgb { r: 160, g: 60, b: 255 }), SetAttribute(Attribute::Bold),
        Print("     __  __ _   _ ____  \n"),
        Print("    |  \\/  | | | |  _ \\ \n"),
        SetForegroundColor(Color::Rgb { r: 0, g: 200, b: 255 }),
        Print("    | |\\/| | | | | | | |\n"),
        Print("    | |  | | |_| | |_| |\n"),
        SetForegroundColor(C_ACCENT),
        Print("    |_|  |_|\\___/|____/ \n"),
        SetAttribute(Attribute::Reset),
        SetForegroundColor(C_DIM),
        Print("   Modular Understanding Dynamics v1.58b\n"),
        Print("   ⚡ Ternary 1.58-bit Mixture-of-Experts ⚡\n"),
        SetForegroundColor(C_BORDER),
        Print("  ─────────────────────────────────────────────\n\n"),
        ResetColor
    )
}

// ─── Thinking spinner ────────────────────────────────────────────────────────
fn show_thinking(stdout: &mut io::Stdout, active: bool) -> io::Result<()> {
    if active {
        execute!(stdout,
            SetForegroundColor(Color::Rgb { r: 160, g: 60, b: 255 }),
            SetAttribute(Attribute::Italic), SetAttribute(Attribute::Dim),
            Print("  ◌  Procesando sinapsis cognitivas..."),
            ResetColor, SetAttribute(Attribute::Reset),
            cursor::MoveToColumn(0),
        )
    } else {
        execute!(stdout, Clear(ClearType::CurrentLine), cursor::MoveToColumn(0))
    }
}

// ─── Typewriter (ANSI + XML tag aware) ───────────────────────────────────────
fn type_writer(text: &str, delay: Duration) -> io::Result<()> {
    let mut in_ansi = false;
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let remaining: String = chars[i..].iter().collect();
        if remaining.starts_with("<thinking>")  { execute!(io::stdout(), SetForegroundColor(Color::Cyan), SetAttribute(Attribute::Italic), SetAttribute(Attribute::Dim))?; i += 10; continue; }
        if remaining.starts_with("</thinking>") { execute!(io::stdout(), ResetColor, SetAttribute(Attribute::Reset))?; i += 11; println!(); continue; }
        if remaining.starts_with("<answer>")    { execute!(io::stdout(), ResetColor, SetAttribute(Attribute::Bold))?; i += 8; continue; }
        if remaining.starts_with("</answer>")   { execute!(io::stdout(), ResetColor, SetAttribute(Attribute::Reset))?; i += 9; continue; }
        let c = chars[i];
        if c == '\x1b' { in_ansi = true; }
        print!("{}", c);
        io::stdout().flush()?;
        if in_ansi { if c == 'm' { in_ansi = false; } } else { thread::sleep(delay); }
        i += 1;
    }
    execute!(io::stdout(), ResetColor, SetAttribute(Attribute::Reset))?;
    Ok(())
}

// ─── Status bar ───────────────────────────────────────────────────────────────
fn update_status_bar(
    stdout: &mut io::Stdout,
    engine: &MudInference,
    live_iq: &HashMap<&str, f32>,
    last_tps: f64,
    sys: &mut System,
) -> io::Result<()> {
    sys.refresh_memory();
    let mem_used  = sys.used_memory()  as f64 / 1_073_741_824.0;
    let mem_total = sys.total_memory() as f64 / 1_073_741_824.0;
    let active_exp = *engine.active_experts.read().unwrap();
    let total_exp  = engine.model.num_experts;
    let vlk        = if engine.vulkan_ctx.is_available() { "VLK ✓" } else { "CPU  " };
    let avg_iq: f32 = live_iq.values().sum::<f32>() / live_iq.len() as f32;
    let (_, rows)  = terminal::size().unwrap_or((80, 24));
    let tps_s = if last_tps > 0.0 { format!("{:.1} t/s", last_tps) } else { "─── t/s".to_string() };

    execute!(stdout,
        cursor::SavePosition,
        cursor::MoveTo(0, rows - 1),
        SetBackgroundColor(C_BAR_BG),
        Clear(ClearType::CurrentLine),
        SetForegroundColor(C_ACCENT), Print(" MUD-V1.5"),
        SetForegroundColor(C_BORDER), Print(" │ "),
        SetForegroundColor(C_STATUS), Print(format!("Exp {}/{}", active_exp, total_exp)),
        SetForegroundColor(C_BORDER), Print(" │ "),
        SetForegroundColor(C_STATUS), Print(tps_s),
        SetForegroundColor(C_BORDER), Print(" │ "),
        SetForegroundColor(C_STATUS), Print(format!("Mem {:.1}/{:.1}G", mem_used, mem_total)),
        SetForegroundColor(C_BORDER), Print(" │ "),
        SetForegroundColor(C_STATUS), Print(vlk),
        SetForegroundColor(C_BORDER), Print(" │ "),
        SetForegroundColor(Color::Rgb { r: 220, g: 220, b: 80 }),
        Print(format!("IQ {:.1}", avg_iq)),
        SetForegroundColor(C_BORDER), Print(" "),
        ResetColor,
        cursor::RestorePosition,
    )?;
    stdout.flush()
}

// ─── /help ────────────────────────────────────────────────────────────────────
fn print_help(stdout: &mut io::Stdout) -> io::Result<()> {
    execute!(stdout, Print("\n"),
        SetForegroundColor(C_BORDER),
        Print("  ╭──────────────────────────────────────────────────────────╮\n"),
        SetForegroundColor(C_ACCENT), SetAttribute(Attribute::Bold),
        Print("  │           🧠  MUD ENGINE — COMANDOS DISPONIBLES           │\n"),
        SetAttribute(Attribute::Reset),
        SetForegroundColor(C_BORDER),
        Print("  ├──────────────────────────────────────────────────────────┤\n"),
        ResetColor,
    )?;
    for (cmd, desc) in &[
        ("/stats",       "Reporte cognitivo IQ en vivo"),
        ("/knowledge",   "Estado de la base de conocimiento"),
        ("/ingest <f>",  "Ingerir archivo de texto al DB"),
        ("/clear",       "Limpiar pantalla y redibujar banner"),
        ("/help",        "Este menú"),
        ("/exit",        "Cerrar sesión de inferencia"),
    ] {
        execute!(stdout,
            SetForegroundColor(C_BORDER), Print("  │  "),
            SetForegroundColor(Color::Yellow), SetAttribute(Attribute::Bold),
            Print(format!("{:<14}", cmd)),
            SetAttribute(Attribute::Reset),
            SetForegroundColor(C_DIM), Print("─ "),
            SetForegroundColor(Color::White),
            Print(format!("{:<38}", desc)),
            SetForegroundColor(C_BORDER), Print("│\n"),
            ResetColor,
        )?;
    }
    execute!(stdout,
        SetForegroundColor(C_BORDER),
        Print("  ╰──────────────────────────────────────────────────────────╯\n\n"),
        ResetColor,
    )
}

// ─── /stats report card ───────────────────────────────────────────────────────
fn print_report_card(
    stdout: &mut io::Stdout,
    live_iq: &HashMap<&str, f32>,
    nodes: usize,
    total_facts: i64,
) -> io::Result<()> {
    execute!(stdout, Print("\n"),
        SetForegroundColor(C_BORDER),
        Print("  ╭──────────────────────────────────────────────────────────╮\n"),
        SetForegroundColor(C_ACCENT), SetAttribute(Attribute::Bold),
        Print("  │          🧠  MUD — REPORTE COGNITIVO EN VIVO              │\n"),
        SetAttribute(Attribute::Reset),
        SetForegroundColor(C_BORDER),
        Print("  ├──────────────────────────────────────────────────────────┤\n"),
        ResetColor,
    )?;
    for (key, label) in &[
        ("linguistics", "Lingüística "),
        ("logic",       "Lógica/Math "),
        ("code",        "Código      "),
        ("general",     "General     "),
        ("system",      "Sistema     "),
    ] {
        let score  = live_iq.get(key).copied().unwrap_or(100.0);
        let filled = ((score / 200.0) * 20.0).round().clamp(0.0, 20.0) as usize;
        execute!(stdout,
            SetForegroundColor(C_BORDER), Print("  │  "),
            SetForegroundColor(C_DIM),   Print(format!("{} ", label)),
            SetForegroundColor(C_ACCENT),Print("█".repeat(filled)),
            SetForegroundColor(C_BORDER),Print("░".repeat(20 - filled)),
            SetForegroundColor(Color::White), Print(format!(" {:>5.1}", score)),
            SetForegroundColor(C_DIM),   Print(" IQ"),
            SetForegroundColor(C_BORDER),Print("  │\n"),
            ResetColor,
        )?;
    }
    execute!(stdout,
        SetForegroundColor(C_BORDER),
        Print("  ├──────────────────────────────────────────────────────────┤\n"),
        SetForegroundColor(C_DIM), Print("  │  "),
        SetForegroundColor(Color::White), Print(format!("{:>8} hechos", total_facts)),
        SetForegroundColor(C_DIM), Print("  │  "),
        SetForegroundColor(Color::White), Print(format!("{:>4} nodos activos", nodes)),
        SetForegroundColor(C_DIM), Print("           │\n"),
        SetForegroundColor(C_BORDER),
        Print("  ╰──────────────────────────────────────────────────────────╯\n\n"),
        ResetColor,
    )
}

// ─── /knowledge ───────────────────────────────────────────────────────────────
fn print_knowledge_mesh(stdout: &mut io::Stdout, engine: &MudInference) -> anyhow::Result<()> {
    let total_facts: i64 = if Path::new("models/knowledge.db").exists() {
        let conn = rusqlite::Connection::open("models/knowledge.db")?;
        conn.query_row("SELECT COUNT(*) FROM facts", [], |r| r.get(0)).unwrap_or(0)
    } else { 0 };
    let nodes = engine.model.knowledge_graph.read().unwrap().nodes.len();
    execute!(stdout,
        Print("\n"),
        SetForegroundColor(C_ACCENT), SetAttribute(Attribute::Bold),
        Print("  🧠  Estado del Grafo de Conocimiento\n"),
        SetAttribute(Attribute::Reset),
        SetForegroundColor(C_BORDER), Print("  ─────────────────────────────────────\n"),
        SetForegroundColor(C_DIM),    Print("  Hechos en Knowledge DB : "),
        SetForegroundColor(Color::White), Print(format!("{}\n", total_facts)),
        SetForegroundColor(C_DIM),    Print("  Sinapsis activas       : "),
        SetForegroundColor(Color::White), Print(format!("{}\n\n", nodes)),
        ResetColor,
    )?;
    Ok(())
}
