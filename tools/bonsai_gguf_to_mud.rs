//! RAM-first importer: PrismML Ternary Bonsai GGUF Q2_0 (g128) → MUD ELUT + PRQ.
//!
//! Design constraints (15 GiB laptop, i7-class):
//! - **Never** load unpacked FP16 safetensors (~3.4 GB for 1.7B).
//! - Input is **mmap-only** (OS pages the 442 MB GGUF on demand).
//! - Convert **row-by-row**: no full-tensor FP32 expand (1.7B×4 ≈ 6.8 GB forbidden).
//! - Output via [`StreamingMudWriter`] — tensor data goes to disk immediately.
//! - Peak process RSS target: mmap page cache + one row scratch (~ few MiB) + small headers.
//!
//! Q2_0 g128 wire format (PrismML / custom GGUF type id 42):
//! ```text
//! block = [scale: f16 LE][qs: 32 bytes]  // 128 weights × 2-bit
//! w_i = (q_i - 1) * scale,  q ∈ {0,1,2} → {-1,0,+1}
//! ```
//!
//! Path A (today's GEMV): expand group scales → **per-row PRQ** + pack exact trits to ELUT 4-bit.
//! No re-thresholding (native ternary — preserve {-1,0,+1} codes).

use anyhow::{bail, Context, Result};
use forge_llm::mud::{MudTensorType, StreamingMudWriter};
use memmap2::Mmap;
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

const Q2_0_G128: u32 = 42;
const GROUP: usize = 128;
const BLOCK_BYTES: usize = 34; // 2 (f16 scale) + 32 (packed 2-bit)

#[derive(Clone)]
struct GgufTensor {
    name: String,
    /// GGML order: dims[0] = innermost (contiguous) = n_in for 2D weights.
    dims: Vec<usize>,
    t_type: u32,
    /// Offset relative to data section start.
    offset: usize,
}

struct GgufView {
    mmap: Mmap,
    data_start: usize,
    tensors: Vec<GgufTensor>,
    meta: HashMap<String, String>,
}

fn read_str(mmap: &[u8], pos: &mut usize) -> Result<String> {
    if *pos + 8 > mmap.len() {
        bail!("GGUF string length OOB at {}", *pos);
    }
    let n = u64::from_le_bytes(mmap[*pos..*pos + 8].try_into()?) as usize;
    *pos += 8;
    if n > 64 * 1024 * 1024 {
        bail!("GGUF string absurdly long ({n}) at {}", *pos);
    }
    if *pos + n > mmap.len() {
        bail!("GGUF string body OOB");
    }
    let s = String::from_utf8_lossy(&mmap[*pos..*pos + n]).into_owned();
    *pos += n;
    Ok(s)
}

fn skip_val(mmap: &[u8], pos: &mut usize, t: u32) -> Result<()> {
    match t {
        0 | 1 | 7 => {
            *pos += 1;
        }
        2 | 3 => {
            *pos += 2;
        }
        4..=6 => {
            *pos += 4;
        }
        8 => {
            let _ = read_str(mmap, pos)?;
        }
        9 => {
            if *pos + 12 > mmap.len() {
                bail!("array header OOB");
            }
            let it = u32::from_le_bytes(mmap[*pos..*pos + 4].try_into()?);
            *pos += 4;
            let n = u64::from_le_bytes(mmap[*pos..*pos + 8].try_into()?) as usize;
            *pos += 8;
            for _ in 0..n {
                skip_val(mmap, pos, it)?;
            }
        }
        10..=12 => {
            *pos += 8;
        }
        _ => bail!("unknown GGUF value type {t}"),
    }
    Ok(())
}

fn read_val_stringish(mmap: &[u8], pos: &mut usize, t: u32) -> Result<Option<String>> {
    match t {
        4 => {
            let v = u32::from_le_bytes(mmap[*pos..*pos + 4].try_into()?);
            *pos += 4;
            Ok(Some(v.to_string()))
        }
        5 => {
            let v = i32::from_le_bytes(mmap[*pos..*pos + 4].try_into()?);
            *pos += 4;
            Ok(Some(v.to_string()))
        }
        6 => {
            let v = f32::from_le_bytes(mmap[*pos..*pos + 4].try_into()?);
            *pos += 4;
            Ok(Some(v.to_string()))
        }
        7 => {
            let v = mmap[*pos] != 0;
            *pos += 1;
            Ok(Some(v.to_string()))
        }
        8 => Ok(Some(read_str(mmap, pos)?)),
        10 => {
            let v = u64::from_le_bytes(mmap[*pos..*pos + 8].try_into()?);
            *pos += 8;
            Ok(Some(v.to_string()))
        }
        _ => {
            skip_val(mmap, pos, t)?;
            Ok(None)
        }
    }
}

fn load_gguf(path: &str) -> Result<GgufView> {
    let file = File::open(path).with_context(|| format!("open {path}"))?;
    let mmap = unsafe { Mmap::map(&file)? };
    if mmap.len() < 24 || &mmap[0..4] != b"GGUF" {
        bail!("not a GGUF file");
    }
    let mut pos = 4usize;
    let _ver = u32::from_le_bytes(mmap[pos..pos + 4].try_into()?);
    pos += 4;
    let n_tensors = u64::from_le_bytes(mmap[pos..pos + 8].try_into()?) as usize;
    pos += 8;
    let n_kv = u64::from_le_bytes(mmap[pos..pos + 8].try_into()?) as usize;
    pos += 8;

    let mut meta = HashMap::new();
    // Keep only small scalar/string KVs — skip huge tokenizer arrays (use sidecar tokenizer.json).
    let keep = [
        "general.architecture",
        "general.basename",
        "general.size_label",
        "general.finetune",
        "qwen3.block_count",
        "qwen3.context_length",
        "qwen3.embedding_length",
        "qwen3.feed_forward_length",
        "qwen3.attention.head_count",
        "qwen3.attention.head_count_kv",
        "qwen3.attention.key_length",
        "qwen3.attention.value_length",
        "qwen3.rope.freq_base",
        "qwen3.attention.layer_norm_rms_epsilon",
        "qwen3.rope.scaling.type",
        "qwen3.rope.scaling.factor",
        "qwen3.rope.scaling.original_context_length",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.padding_token_id",
        "tokenizer.ggml.bos_token_id",
        "general.file_type",
    ];
    for _ in 0..n_kv {
        let key = read_str(&mmap, &mut pos)?;
        let t = u32::from_le_bytes(mmap[pos..pos + 4].try_into()?);
        pos += 4;
        if keep.iter().any(|k| *k == key) {
            if let Some(v) = read_val_stringish(&mmap, &mut pos, t)? {
                meta.insert(key, v);
            }
        } else {
            skip_val(&mmap, &mut pos, t)?;
        }
    }

    let mut tensors = Vec::with_capacity(n_tensors);
    for _ in 0..n_tensors {
        let name = read_str(&mmap, &mut pos)?;
        let n_dims = u32::from_le_bytes(mmap[pos..pos + 4].try_into()?) as usize;
        pos += 4;
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(u64::from_le_bytes(mmap[pos..pos + 8].try_into()?) as usize);
            pos += 8;
        }
        let t_type = u32::from_le_bytes(mmap[pos..pos + 4].try_into()?);
        pos += 4;
        let offset = u64::from_le_bytes(mmap[pos..pos + 8].try_into()?) as usize;
        pos += 8;
        tensors.push(GgufTensor {
            name,
            dims,
            t_type,
            offset,
        });
    }

    let align = 32usize;
    let data_start = (pos + align - 1) & !(align - 1);

    Ok(GgufView {
        mmap,
        data_start,
        tensors,
        meta,
    })
}

#[inline]
fn elut_bits(trit: i8) -> u32 {
    match trit {
        1 => 0x1,
        -1 => 0xF,
        _ => 0x0,
    }
}

/// Convert one Q2_0 g128 matrix to ELUT bytes + per-row PRQ scales without full FP expand.
///
/// GGUF layout: `dims = [n_in, n_out]` (ne0 contiguous). MUD wants `[n_out, n_in]`.
/// Returns `(elut_le_bytes, scales_f32_le_bytes, mud_shape)`.
fn q2_0_to_elut_prq(
    raw: &[u8],
    n_in: usize,
    n_out: usize,
) -> Result<(Vec<u8>, Vec<u8>, Vec<usize>)> {
    if n_in == 0 || n_out == 0 {
        bail!("empty tensor dims");
    }
    if !n_in.is_multiple_of(GROUP) {
        bail!("n_in={n_in} not multiple of {GROUP}");
    }
    let groups_per_row = n_in / GROUP;
    let expected = n_out * groups_per_row * BLOCK_BYTES;
    if raw.len() < expected {
        bail!(
            "Q2_0 payload short: have {} need {} (n_out={n_out} n_in={n_in})",
            raw.len(),
            expected
        );
    }

    let u32s_per_row = n_in.div_ceil(8);
    let mut packed = vec![0u32; n_out * u32s_per_row];
    let mut scales = vec![0.0f32; n_out];

    for row in 0..n_out {
        let row_base = row * groups_per_row * BLOCK_BYTES;
        let mut abs_sum = 0.0f32;
        let pack_row = &mut packed[row * u32s_per_row..(row + 1) * u32s_per_row];

        for g in 0..groups_per_row {
            let b = row_base + g * BLOCK_BYTES;
            let scale = half::f16::from_le_bytes([raw[b], raw[b + 1]]).to_f32();
            let qs = &raw[b + 2..b + BLOCK_BYTES];
            let col0 = g * GROUP;
            for i in 0..GROUP {
                let byte = qs[i / 4];
                let shift = (i % 4) * 2;
                let q = (byte >> shift) & 0x3;
                // q=3 reserved; map to 0 for safety
                let trit: i8 = match q {
                    0 => -1,
                    1 => 0,
                    2 => 1,
                    _ => 0,
                };
                if trit != 0 {
                    abs_sum += scale.abs();
                }
                let col = col0 + i;
                let bits = elut_bits(trit);
                let u32_idx = col / 8;
                let nshift = (col % 8) * 4;
                pack_row[u32_idx] |= bits << nshift;
            }
        }
        // Native ternary: PRQ = row absmean of reconstructed |s*t| (no 0.707 re-dampen).
        scales[row] = (abs_sum / n_in as f32).max(1.1e-8);
    }

    let mut out_bytes = Vec::with_capacity(packed.len() * 4);
    for p in &packed {
        out_bytes.extend_from_slice(&p.to_le_bytes());
    }
    let scale_bytes: Vec<u8> = scales.iter().flat_map(|s| s.to_le_bytes()).collect();
    Ok((out_bytes, scale_bytes, vec![n_out, n_in]))
}

fn tensor_bytes<'a>(view: &'a GgufView, t: &GgufTensor) -> Result<&'a [u8]> {
    let start = view.data_start + t.offset;
    // size from next tensor or EOF
    let end = view
        .tensors
        .iter()
        .filter(|o| o.offset > t.offset)
        .map(|o| view.data_start + o.offset)
        .min()
        .unwrap_or(view.mmap.len());
    if start >= view.mmap.len() || end > view.mmap.len() || end < start {
        bail!("tensor {} data range invalid", t.name);
    }
    Ok(&view.mmap[start..end])
}

fn load_sidecar_tokenizer(dir: &Path, meta: &mut HashMap<String, String>) {
    // Prefer compact files already on disk (avoid GGUF tokenizer.ggml.tokens array).
    let vocab_txt = dir.join("vocab.json");
    let merges = dir.join("merges.txt");
    let added = dir.join("added_tokens.json");
    if merges.exists() {
        if let Ok(s) = std::fs::read_to_string(&merges) {
            // Drop GPT-2 "#version: 0.2" header if present — rank starts at first merge.
            let cleaned: String = s
                .lines()
                .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
                .collect::<Vec<_>>()
                .join("\n");
            meta.insert("tokenizer.merges".into(), cleaned);
        }
    }
    // vocab as id→token lines; merge added_tokens specials (Qwen chat markers).
    if vocab_txt.exists() {
        if let Ok(s) = std::fs::read_to_string(&vocab_txt) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
                if let Some(obj) = v.as_object() {
                    let mut max_id = 0usize;
                    for (_tok, idv) in obj {
                        if let Some(id) = idv.as_u64() {
                            max_id = max_id.max(id as usize);
                        }
                    }
                    // Overlay added_tokens.json (may extend past base vocab)
                    let mut extras: Vec<(String, usize)> = Vec::new();
                    if added.exists() {
                        if let Ok(as_) = std::fs::read_to_string(&added) {
                            if let Ok(av) = serde_json::from_str::<serde_json::Value>(&as_) {
                                if let Some(aobj) = av.as_object() {
                                    for (tok, idv) in aobj {
                                        if let Some(id) = idv.as_u64() {
                                            let id = id as usize;
                                            max_id = max_id.max(id);
                                            extras.push((tok.clone(), id));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // Prefer explicit vocab_size from metadata if already set
                    if let Some(vs) = meta.get("vocab_size").and_then(|s| s.parse::<usize>().ok()) {
                        max_id = max_id.max(vs.saturating_sub(1));
                    }
                    let mut id_to = vec![String::new(); max_id + 1];
                    for (tok, idv) in obj {
                        if let Some(id) = idv.as_u64() {
                            id_to[id as usize] = tok.clone();
                        }
                    }
                    for (tok, id) in extras {
                        id_to[id] = tok;
                    }
                    for (i, slot) in id_to.iter_mut().enumerate() {
                        if slot.is_empty() {
                            *slot = format!("<dummy_{i}>");
                        }
                    }
                    meta.insert("tokenizer.tokens".into(), id_to.join("\n"));
                    meta.insert("vocab_size".into(), id_to.len().to_string());
                }
            }
        }
    }
    let cfg = dir.join("config.json");
    if cfg.exists() {
        if let Ok(s) = std::fs::read_to_string(&cfg) {
            meta.insert("raw_config_json".into(), s);
        }
    }
}

fn build_global_metadata(view: &GgufView, model_dir: &Path) -> HashMap<String, String> {
    let mut m = HashMap::new();
    m.insert("arch".into(), "mud-ternary-qwen3-bonsai".into());
    m.insert("source".into(), "prism-ml/Ternary-Bonsai-Q2_0-g128".into());
    m.insert(
        "quant_path".into(),
        "g128→ELUT+PRQ (streaming, no FP expand)".into(),
    );

    let get = |k: &str| view.meta.get(k).cloned();
    if let Some(v) = get("qwen3.block_count") {
        m.insert("num_hidden_layers".into(), v.clone());
        m.insert("num_layers".into(), v);
    }
    if let Some(v) = get("qwen3.embedding_length") {
        m.insert("hidden_size".into(), v);
    }
    if let Some(v) = get("qwen3.feed_forward_length") {
        m.insert("intermediate_size".into(), v.clone());
        m.insert("ffn_hidden".into(), v);
    }
    if let Some(v) = get("qwen3.attention.head_count") {
        m.insert("num_attention_heads".into(), v.clone());
        m.insert("num_heads".into(), v);
    }
    if let Some(v) = get("qwen3.attention.head_count_kv") {
        m.insert("num_key_value_heads".into(), v.clone());
        m.insert("num_kv_heads".into(), v);
    }
    if let Some(v) = get("qwen3.attention.key_length") {
        m.insert("head_dim".into(), v);
    }
    if let Some(v) = get("qwen3.context_length") {
        // Cap training context later; store native max
        m.insert("max_position_embeddings".into(), v);
    }
    if let Some(v) = get("qwen3.rope.freq_base") {
        m.insert("rope_theta".into(), v.clone());
        m.insert("rope.freq_base".into(), v);
    }
    if let Some(v) = get("qwen3.attention.layer_norm_rms_epsilon") {
        m.insert("rms_norm_eps".into(), v);
    }
    if let Some(v) = get("tokenizer.ggml.eos_token_id") {
        m.insert("eos_token_id".into(), v);
    }
    if let Some(v) = get("tokenizer.ggml.padding_token_id") {
        m.insert("pad_token_id".into(), v);
    }
    // Qwen chat often has no BOS; pad as fallback
    m.insert(
        "bos_token_id".into(),
        get("tokenizer.ggml.bos_token_id")
            .or_else(|| get("tokenizer.ggml.padding_token_id"))
            .unwrap_or_else(|| "151643".into()),
    );
    m.insert("hidden_act".into(), "silu".into());
    m.insert("tie_word_embeddings".into(), "true".into());
    m.insert("num_experts".into(), "1".into());

    // vocab_size from emb tensor if present: GGUF [n_embd, n_vocab]
    if let Some(emb) = view.tensors.iter().find(|t| t.name == "token_embd.weight") {
        if emb.dims.len() == 2 {
            m.insert("vocab_size".into(), emb.dims[1].to_string());
        }
    }

    load_sidecar_tokenizer(model_dir, &mut m);

    // Drop bulky duplicate: tokens+merges are enough for Tokenizer::from_mud_metadata
    m.remove("tokenizer_json");

    if let Err(e) = forge_llm::mud::p13::ensure_canonical_metadata_aliases(&mut m) {
        eprintln!("[P-13] warning: could not fully normalize metadata: {e}");
    }
    m
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <Ternary-Bonsai-*-Q2_0.gguf> <output.mud> [--no-tokenizer-sidecar]",
            args[0]
        );
        eprintln!("RAM-first: mmap GGUF + stream ELUT rows; never expands full FP32.");
        std::process::exit(2);
    }
    let gguf_path = &args[1];
    let out_path = &args[2];
    let skip_tok = args.iter().any(|a| a == "--no-tokenizer-sidecar");

    eprintln!("🎋 Bonsai GGUF Q2_0 g128 → MUD ELUT (streaming, RAM-first)");
    eprintln!("  in : {gguf_path}");
    eprintln!("  out: {out_path}");

    let view = load_gguf(gguf_path)?;
    let model_dir = Path::new(gguf_path)
        .parent()
        .unwrap_or_else(|| Path::new("."));
    let mut global = build_global_metadata(&view, model_dir);
    if skip_tok {
        global.remove("tokenizer_json");
        global.remove("tokenizer.tokens");
        global.remove("tokenizer.merges");
    }

    // Plan MUD tensors in stable write order: for each GGUF tensor, emit weight(+scale) or F32.
    #[derive(Clone)]
    enum PlanItem {
        /// Copy F32 as-is (norms).
        F32 {
            name: String,
            shape: Vec<usize>,
            gguf_idx: usize,
        },
        /// Q2_0 → ELUT weight + PRQ scale (two writer slots).
        Q2 {
            weight_name: String,
            scale_name: String,
            n_in: usize,
            n_out: usize,
            gguf_idx: usize,
        },
    }

    let mut plan: Vec<PlanItem> = Vec::new();
    let mut type_counts = HashMap::new();
    for (i, t) in view.tensors.iter().enumerate() {
        *type_counts.entry(t.t_type).or_insert(0usize) += 1;
        match t.t_type {
            0 => {
                // F32
                plan.push(PlanItem::F32 {
                    name: t.name.clone(),
                    shape: t.dims.clone(),
                    gguf_idx: i,
                });
            }
            Q2_0_G128 => {
                if t.dims.len() != 2 {
                    bail!("{}: expected 2D Q2_0, got dims={:?}", t.name, t.dims);
                }
                let n_in = t.dims[0];
                let n_out = t.dims[1];
                let weight_name = t.name.clone();
                let scale_name = if weight_name.ends_with(".weight") {
                    weight_name.replace(".weight", ".prq_scale")
                } else {
                    format!("{weight_name}.prq_scale")
                };
                plan.push(PlanItem::Q2 {
                    weight_name,
                    scale_name,
                    n_in,
                    n_out,
                    gguf_idx: i,
                });
            }
            other => bail!("unsupported GGUF type {other} on tensor {}", t.name),
        }
    }
    eprintln!(
        "  GGUF tensors: {} | types: {:?}",
        view.tensors.len(),
        type_counts
    );

    // Flatten writer meta: weight then scale for Q2 items
    let mut tensors_meta: Vec<(String, MudTensorType, Vec<usize>)> = Vec::new();
    for item in &plan {
        match item {
            PlanItem::F32 { name, shape, .. } => {
                tensors_meta.push((name.clone(), MudTensorType::Float32, shape.clone()));
            }
            PlanItem::Q2 {
                weight_name,
                scale_name,
                n_in,
                n_out,
                ..
            } => {
                tensors_meta.push((
                    weight_name.clone(),
                    MudTensorType::Ternary2Bit,
                    vec![*n_out, *n_in],
                ));
                tensors_meta.push((scale_name.clone(), MudTensorType::Float32, vec![*n_out]));
            }
        }
    }

    // Estimate disk
    let mut est = 0usize;
    for (_n, ty, sh) in &tensors_meta {
        let ne: usize = sh.iter().product();
        est += match ty {
            MudTensorType::Ternary2Bit => ne.div_ceil(8) * 4,
            MudTensorType::Float32 => ne * 4,
            _ => ne,
        };
    }
    eprintln!(
        "  planned MUD tensors: {} | est. payload ≈ {:.1} MiB",
        tensors_meta.len(),
        est as f64 / (1024.0 * 1024.0)
    );

    let mut writer = StreamingMudWriter::create(out_path, &global, &tensors_meta)?;
    let mut done = 0usize;
    let total = plan.len();
    let t0 = std::time::Instant::now();

    for item in &plan {
        match item {
            PlanItem::F32 { gguf_idx, .. } => {
                let t = &view.tensors[*gguf_idx];
                let raw = tensor_bytes(&view, t)?;
                let ne: usize = t.dims.iter().product();
                let need = ne * 4;
                if raw.len() < need {
                    bail!("{} F32 short", t.name);
                }
                writer.write_tensor_data(&raw[..need])?;
            }
            PlanItem::Q2 {
                weight_name,
                n_in,
                n_out,
                gguf_idx,
                ..
            } => {
                let t = &view.tensors[*gguf_idx];
                let raw = tensor_bytes(&view, t)?;
                let (elut, scales, _shape) = q2_0_to_elut_prq(raw, *n_in, *n_out)
                    .with_context(|| format!("convert {weight_name}"))?;
                writer.write_tensor_data(&elut)?;
                writer.write_tensor_data(&scales)?;
            }
        }
        done += 1;
        if done.is_multiple_of(20) || done == total {
            eprint!(
                "\r  convert {}/{} ({:.0}%)  elapsed {:.1}s   ",
                done,
                total,
                100.0 * done as f32 / total as f32,
                t0.elapsed().as_secs_f32()
            );
            let _ = io::stderr().flush();
        }
    }
    eprintln!();
    writer.close(out_path)?;

    let out_sz = std::fs::metadata(out_path)?.len();
    eprintln!(
        "✅ wrote {out_path} ({:.1} MiB) in {:.1}s — no full FP expand",
        out_sz as f64 / (1024.0 * 1024.0),
        t0.elapsed().as_secs_f32()
    );
    eprintln!(
        "  next: FREEZE_EMB=1 LAST_N=2 seating; materialize_for_ste_train keeps frozen on mmap"
    );
    Ok(())
}
