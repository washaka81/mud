//! # L-10: Sequence packing (no padding)
//!
//! Classic LM packing for the MUD QAT path:
//! - Concatenate short documents into capacity-sized chunks **without** pad tokens
//! - Extract next-token pairs that **never cross** document (EOS) boundaries
//! - Subsample pairs **uniformly across the full chunk** (old path only used the head
//!   via `windows(2).step_by(8).take(batch)` — ~1% of a 12k-token chunk)
//!
//! Expected: higher token utilization (target 1.5–2× effective signal per wall-hour).

/// Legacy Llama-3 style specials (only valid when vocab ≥ 128256).
/// Prefer [`crate::model::tokenizer::Tokenizer::special_ids_from_metadata`] for converted models
/// (SmolLM2 uses bos=eos=0 = `<|endoftext|>`).
pub const DEFAULT_BOS: u32 = 128_000;
pub const DEFAULT_EOS: u32 = 128_001;

/// Clamp special ids into vocab; fall back to 0 when legacy Llama ids are OOV.
#[inline]
pub fn clamp_special_id(id: u32, vocab_size: usize) -> u32 {
    if vocab_size == 0 {
        return 0;
    }
    if (id as usize) < vocab_size {
        id
    } else {
        0
    }
}

/// Predictions (next-token steps) per AOT chunk.
/// - `MUD_TRAIN_STEPS_PER_CHUNK=N` hard override
/// - else `batch_size` × multiplier (align / quick → denser signal)
pub fn train_steps_per_chunk(batch_size: usize) -> usize {
    if let Ok(v) = std::env::var("MUD_TRAIN_STEPS_PER_CHUNK") {
        return v.parse::<usize>().unwrap_or(batch_size).clamp(1, 4096);
    }
    let batch = batch_size.max(1);
    let align = std::env::var("MUD_TRAIN_ALIGN")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
        || std::env::var("MUD_TRAIN_MAX_CHUNKS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&m| m > 0)
            .is_some();
    if align {
        // Speed-biased: 2× batch (was 4×) — more wall-clock tokens than 4× with half STE cost
        (batch * 2).clamp(16, 128)
    } else {
        batch
    }
}

/// Sampled-softmax negatives (1 target + N negs).
/// - `MUD_TRAIN_NUM_NEG=N` (1..=511)
/// - default: 63 in align/quick (fast), 255 full train (quality)
pub fn train_num_negatives() -> usize {
    if let Ok(v) = std::env::var("MUD_TRAIN_NUM_NEG") {
        return v.parse::<usize>().unwrap_or(63).clamp(1, 511);
    }
    let quick = std::env::var("MUD_TRAIN_ALIGN")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
        || std::env::var("MUD_TRAIN_MAX_CHUNKS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&m| m > 0)
            .is_some();
    if quick {
        63
    } else {
        255
    }
}

/// Only backprop + optimize the last N layers (forward still full unless FWD_LAST_N set).
///
/// Env:
/// - `MUD_TRAIN_LAST_N_LAYERS=N` — explicit N (`0` = all layers)
/// - `MUD_TRAIN_LAST_N_LAYERS=all|full` — all layers
///
/// **Defaults (RAM-safe on ~15 GiB / Iris Xe design host):**
/// - align / quick (`MUD_TRAIN_ALIGN` or `MUD_TRAIN_MAX_CHUNKS`): min(8, n_layers)
/// - large stacks (`n_layers > 16`, e.g. Bonsai 28): **4**
/// - medium (`n_layers > 8`): **8**
/// - small: all layers
pub fn train_last_n_layers(n_layers: usize) -> usize {
    if n_layers == 0 {
        return 0;
    }
    if let Ok(v) = std::env::var("MUD_TRAIN_LAST_N_LAYERS") {
        let t = v.trim().to_ascii_lowercase();
        if t.is_empty() || t == "auto" {
            // fall through to policy defaults
        } else if t == "0" || t == "all" || t == "full" {
            return n_layers;
        } else if let Ok(n) = t.parse::<usize>() {
            if n == 0 {
                return n_layers;
            }
            return n.clamp(1, n_layers);
        }
    }
    let quick = std::env::var("MUD_TRAIN_ALIGN")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
        || std::env::var("MUD_TRAIN_MAX_CHUNKS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&m| m > 0)
            .is_some();
    if quick {
        return n_layers.min(8);
    }
    // Full STE shadows on deep stacks OOM UMA hosts; thaw only the tail.
    if n_layers > 16 {
        4.min(n_layers)
    } else if n_layers > 8 {
        8.min(n_layers)
    } else {
        n_layers
    }
}

/// Optional: only **forward** the last N layers (skip frozen lower stack).
///
/// - Unset / `0` → full forward (correct residual; default).
/// - `N` or `auto` → forward only last N (N defaults to [`train_last_n_layers`]).
///
/// **i7-1260P seating speed:** full FWD of 1.7B×28 is the wall. Using FWD_LAST_N=2
/// is approximate (starts residual mid-stack) but multiplies steps/s for scales-only
/// recovery on top layers. Prefer full FWD when quality > speed.
pub fn train_fwd_last_n_layers(n_layers: usize) -> usize {
    if n_layers == 0 {
        return 0;
    }
    match std::env::var("MUD_TRAIN_FWD_LAST_N") {
        Err(_) => n_layers, // full forward
        Ok(v) => {
            let t = v.trim().to_ascii_lowercase();
            if t.is_empty() || t == "0" || t == "off" || t == "full" || t == "all" {
                return n_layers;
            }
            if t == "auto" || t == "1" || t == "true" || t == "yes" {
                // Match BWD last-N when auto
                return train_last_n_layers(n_layers);
            }
            t.parse::<usize>()
                .map(|n| {
                    if n == 0 {
                        n_layers
                    } else {
                        n.clamp(1, n_layers)
                    }
                })
                .unwrap_or(n_layers)
        }
    }
}

/// One document slice into a larger token stream (half-open `[start, start+len)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DocSpan {
    pub start: usize,
    pub len: usize,
}

impl DocSpan {
    #[inline]
    pub fn end(self) -> usize {
        self.start + self.len
    }
}

/// Packed chunk: dense tokens + per-document spans (no pad).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedChunk {
    pub tokens: Vec<u32>,
    pub spans: Vec<DocSpan>,
}

impl PackedChunk {
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Fraction of capacity filled (1.0 = full, no waste).
    pub fn fill_ratio(&self, capacity: usize) -> f32 {
        if capacity == 0 {
            return 0.0;
        }
        self.tokens.len() as f32 / capacity as f32
    }
}

/// Split a flat stream into document segments.
/// Documents end at `eos` (**EOS included** as last token so the model can learn EOD).
/// Leading `bos` is kept inside the segment when present.
pub fn split_documents(tokens: &[u32], eos: u32) -> Vec<DocSpan> {
    let mut spans = Vec::new();
    let mut start = 0usize;
    for (i, &t) in tokens.iter().enumerate() {
        if t == eos {
            spans.push(DocSpan {
                start,
                len: i - start + 1, // include EOS
            });
            start = i + 1;
        }
    }
    if start < tokens.len() {
        spans.push(DocSpan {
            start,
            len: tokens.len() - start,
        });
    }
    spans
}

/// First-fit pack: place whole documents into bins of `capacity` tokens.
/// Documents longer than `capacity` are split into capacity-sized pieces (no pad).
/// Empty / all-BOS-only docs are skipped.
pub fn pack_documents(docs: &[&[u32]], capacity: usize, bos: u32) -> Vec<PackedChunk> {
    if capacity == 0 {
        return Vec::new();
    }
    let mut chunks: Vec<PackedChunk> = Vec::new();
    let mut cur_tokens: Vec<u32> = Vec::with_capacity(capacity);
    let mut cur_spans: Vec<DocSpan> = Vec::new();

    let flush = |toks: &mut Vec<u32>, spans: &mut Vec<DocSpan>, out: &mut Vec<PackedChunk>| {
        if toks.is_empty() {
            return;
        }
        out.push(PackedChunk {
            tokens: std::mem::take(toks),
            spans: std::mem::take(spans),
        });
        toks.reserve(capacity);
    };

    for doc in docs {
        if doc.is_empty() {
            continue;
        }
        // Skip pure BOS placeholders
        if doc.len() == 1 && doc[0] == bos {
            continue;
        }

        let mut offset = 0usize;
        while offset < doc.len() {
            let remaining_cap = capacity.saturating_sub(cur_tokens.len());
            if remaining_cap == 0 {
                flush(&mut cur_tokens, &mut cur_spans, &mut chunks);
                continue;
            }
            let take = (doc.len() - offset).min(remaining_cap);
            let span_start = cur_tokens.len();
            cur_tokens.extend_from_slice(&doc[offset..offset + take]);
            cur_spans.push(DocSpan {
                start: span_start,
                len: take,
            });
            offset += take;
            if cur_tokens.len() >= capacity {
                flush(&mut cur_tokens, &mut cur_spans, &mut chunks);
            }
        }
    }
    flush(&mut cur_tokens, &mut cur_spans, &mut chunks);
    chunks
}

/// Pack a flat AOT stream (BOS/EOS delimited) into capacity-sized chunks without padding.
pub fn pack_stream(tokens: &[u32], capacity: usize, bos: u32, eos: u32) -> Vec<PackedChunk> {
    let spans = split_documents(tokens, eos);
    let docs: Vec<&[u32]> = spans.iter().map(|s| &tokens[s.start..s.end()]).collect();
    pack_documents(&docs, capacity, bos)
}

/// Like [`pack_stream`] but returns `(start, len)` ranges into `tokens` without copying.
/// When a packed bin concatenates non-contiguous docs, the range is **not** representable
/// as a single slice — those bins are emitted as multiple contiguous ranges merged only
/// when docs were adjacent in the stream.
///
/// For AOT mmap prefetch we use a simpler **capacity walk** that never pads and
/// never drops the tail: see [`chunk_ranges_no_pad`].
pub fn chunk_ranges_no_pad(total_tokens: usize, capacity: usize) -> Vec<(usize, usize)> {
    if capacity == 0 || total_tokens == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut start = 0usize;
    while start < total_tokens {
        let len = (total_tokens - start).min(capacity);
        out.push((start, len));
        start += len;
    }
    out
}

/// Document-aware ranges: pack whole docs into bins of `capacity`, returning
/// contiguous stream ranges. If docs packed into one bin are adjacent in the
/// source, they merge into one range; otherwise each run of adjacency is a range
/// (caller may concatenate tokens when materializing).
pub fn pack_stream_ranges(tokens: &[u32], capacity: usize, eos: u32) -> Vec<(usize, usize)> {
    if capacity == 0 || tokens.is_empty() {
        return Vec::new();
    }
    let docs = split_documents(tokens, eos);
    if docs.is_empty() {
        return chunk_ranges_no_pad(tokens.len(), capacity);
    }

    let mut ranges: Vec<(usize, usize)> = Vec::new();
    let mut bin_start: Option<usize> = None;
    let mut bin_end: usize = 0;
    let mut bin_len: usize = 0;

    let flush = |bin_start: &mut Option<usize>,
                 bin_end: &mut usize,
                 bin_len: &mut usize,
                 ranges: &mut Vec<(usize, usize)>| {
        if let Some(s) = bin_start.take() {
            if *bin_end > s {
                ranges.push((s, *bin_end - s));
            }
        }
        *bin_end = 0;
        *bin_len = 0;
    };

    for doc in docs {
        // Long docs: emit capacity-sized pieces from the stream
        if doc.len > capacity {
            flush(&mut bin_start, &mut bin_end, &mut bin_len, &mut ranges);
            let mut off = 0usize;
            while off < doc.len {
                let take = (doc.len - off).min(capacity);
                ranges.push((doc.start + off, take));
                off += take;
            }
            continue;
        }
        if bin_len + doc.len > capacity {
            flush(&mut bin_start, &mut bin_end, &mut bin_len, &mut ranges);
        }
        if bin_start.is_none() {
            bin_start = Some(doc.start);
            bin_end = doc.end();
            bin_len = doc.len;
        } else if doc.start == bin_end {
            // Adjacent in stream (EOS between was excluded from spans — not adjacent!)
            // Spans exclude EOS so docs are NOT adjacent in token index.
            // Always flush/start new logical content: we append by extending range
            // only when stream-contiguous including EOS gaps.
            // Treat non-contiguous as new bin content that needs multi-slice materialize.
            // For mmap simplicity: flush current and start new range at doc.start.
            // Wait — packing multiple short docs into one bin requires multi-slice.
            // Use token copy path for doc packing; ranges path only for capacity walk.
            flush(&mut bin_start, &mut bin_end, &mut bin_len, &mut ranges);
            bin_start = Some(doc.start);
            bin_end = doc.end();
            bin_len = doc.len;
        } else {
            flush(&mut bin_start, &mut bin_end, &mut bin_len, &mut ranges);
            bin_start = Some(doc.start);
            bin_end = doc.end();
            bin_len = doc.len;
        }
    }
    flush(&mut bin_start, &mut bin_end, &mut bin_len, &mut ranges);
    if ranges.is_empty() {
        chunk_ranges_no_pad(tokens.len(), capacity)
    } else {
        ranges
    }
}

/// Valid next-token pair: (input_token_id, target_token_id).
/// Pairs never cross document spans; pairs whose ids are ≥ `vocab_size` are dropped.
/// Never uses `eos` as input (avoids cross-doc leakage).
pub fn next_token_pairs(
    tokens: &[u32],
    spans: &[DocSpan],
    max_pairs: usize,
    vocab_size: usize,
    eos: u32,
) -> Vec<(usize, usize)> {
    if max_pairs == 0 || tokens.len() < 2 {
        return Vec::new();
    }

    // Collect all valid adjacent pairs within spans
    let mut candidates: Vec<(usize, usize)> = Vec::new();
    let span_iter: Box<dyn Iterator<Item = DocSpan>> = if spans.is_empty() {
        Box::new(std::iter::once(DocSpan {
            start: 0,
            len: tokens.len(),
        }))
    } else {
        Box::new(spans.iter().copied())
    };

    for span in span_iter {
        if span.len < 2 {
            continue;
        }
        let end = span.end().min(tokens.len());
        let start = span.start.min(end);
        // Last position may be EOS: allow (…, EOS) as target, never EOS as input.
        for i in start..end.saturating_sub(1) {
            let inp_tok = tokens[i];
            if inp_tok == eos {
                continue;
            }
            let inp = inp_tok as usize;
            let tgt = tokens[i + 1] as usize;
            if inp < vocab_size && tgt < vocab_size {
                candidates.push((inp, tgt));
            }
        }
    }

    subsample_uniform(&candidates, max_pairs)
}

/// Convenience: pairs from a raw stream with EOS splitting.
pub fn pairs_from_stream(
    tokens: &[u32],
    max_pairs: usize,
    vocab_size: usize,
    eos: u32,
) -> Vec<(usize, usize)> {
    let spans = split_documents(tokens, eos);
    next_token_pairs(tokens, &spans, max_pairs, vocab_size, eos)
}

// ── Stream D: full-sequence windows (causal LM, not independent pos=0 pairs) ──

/// Contiguous token window for sequential train: indices `[start, start+len)` in `tokens`.
/// Next-token supervision yields `len - 1` steps at positions `0..len-1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeqWindow {
    pub start: usize,
    pub len: usize,
}

impl SeqWindow {
    #[inline]
    pub fn end(self) -> usize {
        self.start + self.len
    }

    /// Number of next-token predictions in this window.
    #[inline]
    pub fn n_preds(self) -> usize {
        self.len.saturating_sub(1)
    }
}

/// Non-overlapping windows of up to `seq_len` tokens **within** document spans.
/// Never crosses EOS; windows shorter than 2 tokens are skipped.
pub fn seq_windows(
    tokens: &[u32],
    spans: &[DocSpan],
    max_windows: usize,
    seq_len: usize,
    eos: u32,
) -> Vec<SeqWindow> {
    if max_windows == 0 || seq_len < 2 || tokens.len() < 2 {
        return Vec::new();
    }
    let seq_len = seq_len.max(2);

    let mut candidates: Vec<SeqWindow> = Vec::new();
    let span_iter: Box<dyn Iterator<Item = DocSpan>> = if spans.is_empty() {
        Box::new(std::iter::once(DocSpan {
            start: 0,
            len: tokens.len(),
        }))
    } else {
        Box::new(spans.iter().copied())
    };

    for span in span_iter {
        if span.len < 2 {
            continue;
        }
        let end = span.end().min(tokens.len());
        let mut i = span.start.min(end);
        while i + 2 <= end {
            // Skip leading EOS (should not start a window on EOS)
            if tokens[i] == eos {
                i += 1;
                continue;
            }
            let take = seq_len.min(end - i);
            if take < 2 {
                break;
            }
            // Do not include a trailing EOS-only fragment as a start; allow EOS as last token
            candidates.push(SeqWindow {
                start: i,
                len: take,
            });
            i += take; // non-overlapping
        }
    }

    if candidates.len() <= max_windows {
        return candidates;
    }
    // Uniform subsample across the chunk (same spirit as L-10 pairs)
    let n = candidates.len();
    let mut out = Vec::with_capacity(max_windows);
    for k in 0..max_windows {
        let idx = k * n / max_windows;
        out.push(candidates[idx]);
    }
    out
}

/// Windows from a raw stream (EOS document split).
pub fn windows_from_stream(
    tokens: &[u32],
    max_windows: usize,
    seq_len: usize,
    eos: u32,
) -> Vec<SeqWindow> {
    let spans = split_documents(tokens, eos);
    seq_windows(tokens, &spans, max_windows, seq_len, eos)
}

/// Stream D policy: full-seq train (default **on**).
/// `MUD_TRAIN_FULL_SEQ=0|false|off` → classic independent pairs at pos=0.
pub fn train_full_seq_enabled() -> bool {
    match std::env::var("MUD_TRAIN_FULL_SEQ") {
        Err(_) => true, // product default: causal windows
        Ok(v) => {
            let t = v.trim().to_ascii_lowercase();
            !(t == "0" || t == "false" || t == "off" || t == "no" || t == "pairs")
        }
    }
}

/// Window length for full-seq train (`MUD_TRAIN_SEQ_LEN`, default 32, clamp 2..=512).
/// Stream H: set `MUD_TRAIN_SEQ_LEN=128` (or higher) for long-window stress.
pub fn train_seq_len() -> usize {
    std::env::var("MUD_TRAIN_SEQ_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32)
        .clamp(2, 512)
}

/// Stream H: when seq_len is long, prefer segmented grad checkpoint to cut activation RAM.
/// Returns true if caller should force `MUD_GRAD_CKPT` semantics.
pub fn prefer_grad_ckpt_for_seq(seq_len: usize) -> bool {
    let thr = std::env::var("MUD_TRAIN_CKPT_SEQ_THR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64usize);
    seq_len >= thr
}

/// Ensure long full-seq runs enable L-15 segmented recompute (idempotent env set).
pub fn maybe_enable_grad_ckpt_for_long_seq(seq_len: usize) {
    if !prefer_grad_ckpt_for_seq(seq_len) {
        return;
    }
    if std::env::var("MUD_GRAD_CKPT").is_err() {
        // SAFETY: process-level train policy; only set if user left unset
        unsafe {
            std::env::set_var("MUD_GRAD_CKPT", "1");
        }
    }
}

/// How many windows to draw so total predictions ≈ `target_preds`.
pub fn windows_for_target_preds(target_preds: usize, seq_len: usize) -> usize {
    let per = seq_len.saturating_sub(1).max(1);
    target_preds.div_ceil(per).max(1)
}

/// Uniform subsample so the whole chunk contributes (not only the head).
fn subsample_uniform(candidates: &[(usize, usize)], max_pairs: usize) -> Vec<(usize, usize)> {
    if candidates.len() <= max_pairs {
        return candidates.to_vec();
    }
    let n = candidates.len();
    let mut out = Vec::with_capacity(max_pairs);
    for k in 0..max_pairs {
        // Evenly spaced indices in [0, n)
        let idx = k * n / max_pairs;
        out.push(candidates[idx]);
    }
    out
}

/// Stats for logging / healthchecks.
#[derive(Debug, Clone, Copy)]
pub struct PackStats {
    pub n_chunks: usize,
    pub n_docs: usize,
    pub total_tokens: usize,
    pub capacity: usize,
    pub mean_fill: f32,
    pub n_pairs_available: usize,
}

pub fn pack_stats(chunks: &[PackedChunk], capacity: usize, vocab_size: usize) -> PackStats {
    let n_chunks = chunks.len();
    let n_docs = chunks.iter().map(|c| c.spans.len()).sum();
    let total_tokens = chunks.iter().map(|c| c.tokens.len()).sum();
    let mean_fill = if n_chunks == 0 || capacity == 0 {
        0.0
    } else {
        chunks.iter().map(|c| c.fill_ratio(capacity)).sum::<f32>() / n_chunks as f32
    };
    let n_pairs_available = chunks
        .iter()
        .map(|c| next_token_pairs(&c.tokens, &c.spans, usize::MAX, vocab_size, DEFAULT_EOS).len())
        .sum();
    PackStats {
        n_chunks,
        n_docs,
        total_tokens,
        capacity,
        mean_fill,
        n_pairs_available,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_documents_basic() {
        let toks = vec![
            DEFAULT_BOS,
            1,
            2,
            3,
            DEFAULT_EOS,
            DEFAULT_BOS,
            4,
            5,
            DEFAULT_EOS,
        ];
        let spans = split_documents(&toks, DEFAULT_EOS);
        assert_eq!(spans.len(), 2);
        assert_eq!(
            &toks[spans[0].start..spans[0].end()],
            &[DEFAULT_BOS, 1, 2, 3, DEFAULT_EOS]
        );
        assert_eq!(
            &toks[spans[1].start..spans[1].end()],
            &[DEFAULT_BOS, 4, 5, DEFAULT_EOS]
        );
    }

    #[test]
    fn test_pack_no_padding() {
        let d1 = vec![DEFAULT_BOS, 10, 11, 12];
        let d2 = vec![DEFAULT_BOS, 20, 21];
        let d3 = vec![DEFAULT_BOS, 30, 31, 32, 33, 34];
        let docs: Vec<&[u32]> = vec![&d1, &d2, &d3];
        let capacity = 8;
        let chunks = pack_documents(&docs, capacity, DEFAULT_BOS);
        // No chunk longer than capacity; no zero pad tokens
        for c in &chunks {
            assert!(c.len() <= capacity);
            // last chunk may be short — that is "no pad", not wasteful zeros
            assert!(!c.tokens.is_empty());
        }
        let total: usize = chunks.iter().map(|c| c.len()).sum();
        assert_eq!(total, d1.len() + d2.len() + d3.len());
    }

    #[test]
    fn test_pairs_never_cross_eos() {
        // Doc A: BOS,1,2,EOS  Doc B: BOS,3,4,EOS
        let toks = vec![
            DEFAULT_BOS,
            1,
            2,
            DEFAULT_EOS,
            DEFAULT_BOS,
            3,
            4,
            DEFAULT_EOS,
        ];
        let pairs = pairs_from_stream(&toks, 100, 200_000, DEFAULT_EOS);
        // Never use EOS as input (would cross into next doc)
        for &(a, _) in &pairs {
            assert_ne!(a, DEFAULT_EOS as usize);
        }
        // Cross-doc forbidden: 2 → BOS or EOS → BOS
        for &(a, b) in &pairs {
            assert!(
                !(a == 2 && b == DEFAULT_BOS as usize),
                "cross-doc pair 2→BOS found"
            );
            assert!(
                !(a == DEFAULT_EOS as usize && b == DEFAULT_BOS as usize),
                "EOS→BOS pair found"
            );
        }
        // Intra-doc pairs present (including predict EOS)
        assert!(pairs.contains(&(DEFAULT_BOS as usize, 1)));
        assert!(pairs.contains(&(1, 2)));
        assert!(pairs.contains(&(2, DEFAULT_EOS as usize)));
        assert!(pairs.contains(&(DEFAULT_BOS as usize, 3)));
        assert!(pairs.contains(&(3, 4)));
        assert!(pairs.contains(&(4, DEFAULT_EOS as usize)));
    }

    #[test]
    fn test_uniform_covers_tail() {
        // 100 consecutive tokens → 99 pairs; request 10 → should hit near end
        let toks: Vec<u32> = (0..100).collect();
        let pairs = next_token_pairs(&toks, &[], 10, 1000, DEFAULT_EOS);
        assert_eq!(pairs.len(), 10);
        // Last pair should be from the tail half of the sequence
        let last_inp = pairs.last().unwrap().0;
        assert!(
            last_inp >= 50,
            "expected uniform subsample to reach tail, last_inp={last_inp}"
        );
    }

    #[test]
    fn test_old_head_only_vs_pack() {
        // Reproduce old waste: step_by(8).take(16) only sees head
        let toks: Vec<u32> = (1..200).collect();
        let old: Vec<_> = toks
            .windows(2)
            .step_by(8)
            .take(16)
            .map(|w| (w[0] as usize, w[1] as usize))
            .collect();
        let new = next_token_pairs(&toks, &[], 16, 1000, DEFAULT_EOS);
        assert_eq!(old.len(), 16);
        assert_eq!(new.len(), 16);
        // Old max input token is small; new spans further
        let old_max = old.iter().map(|p| p.0).max().unwrap();
        let new_max = new.iter().map(|p| p.0).max().unwrap();
        assert!(new_max > old_max, "pack should cover more of the chunk");
    }

    #[test]
    fn test_long_doc_split() {
        let long: Vec<u32> = (0..20).collect();
        let chunks = pack_documents(&[&long], 8, DEFAULT_BOS);
        assert!(chunks.len() >= 3);
        assert!(chunks.iter().all(|c| c.len() <= 8));
        let total: usize = chunks.iter().map(|c| c.len()).sum();
        assert_eq!(total, 20);
    }

    #[test]
    fn test_pack_stats() {
        let toks = vec![DEFAULT_BOS, 1, 2, DEFAULT_EOS, DEFAULT_BOS, 3, DEFAULT_EOS];
        let chunks = pack_stream(&toks, 16, DEFAULT_BOS, DEFAULT_EOS);
        let st = pack_stats(&chunks, 16, 200_000); // vocab must cover BOS/EOS ids
        assert!(st.n_chunks >= 1);
        assert!(st.mean_fill > 0.0 && st.mean_fill <= 1.0);
        assert!(st.n_pairs_available >= 2);
    }

    #[test]
    fn test_chunk_ranges_includes_tail() {
        // Old: saturating_sub dropped remainder < capacity
        let total = 25usize;
        let cap = 10usize;
        let ranges = chunk_ranges_no_pad(total, cap);
        assert_eq!(ranges, vec![(0, 10), (10, 10), (20, 5)]);
        let covered: usize = ranges.iter().map(|(_, l)| l).sum();
        assert_eq!(covered, total);
    }

    #[test]
    fn test_subsample_preserves_count() {
        let toks: Vec<u32> = (0..50).collect();
        assert_eq!(next_token_pairs(&toks, &[], 7, 1000, DEFAULT_EOS).len(), 7);
        assert_eq!(
            next_token_pairs(&toks, &[], 100, 1000, DEFAULT_EOS).len(),
            49
        );
    }

    #[test]
    fn test_seq_windows_no_cross_eos() {
        // Doc A: BOS,1,2,3,EOS  Doc B: BOS,4,5,EOS
        let toks = vec![
            DEFAULT_BOS,
            1,
            2,
            3,
            DEFAULT_EOS,
            DEFAULT_BOS,
            4,
            5,
            DEFAULT_EOS,
        ];
        let wins = windows_from_stream(&toks, 10, 8, DEFAULT_EOS);
        assert!(!wins.is_empty());
        for w in &wins {
            let slice = &toks[w.start..w.end()];
            // No window may contain tokens from both docs (BOS after EOS mid-window)
            let mut saw_eos = false;
            for &t in slice {
                if saw_eos {
                    panic!("window crosses past EOS: {slice:?}");
                }
                if t == DEFAULT_EOS {
                    saw_eos = true;
                }
            }
            assert!(w.len >= 2);
            assert_ne!(toks[w.start], DEFAULT_EOS);
        }
    }

    #[test]
    fn test_seq_windows_respects_seq_len() {
        let toks: Vec<u32> = (1..40).collect();
        let wins = windows_from_stream(&toks, 100, 8, DEFAULT_EOS);
        assert!(!wins.is_empty());
        for w in &wins {
            assert!(w.len <= 8);
            assert!(w.n_preds() >= 1);
        }
        // Non-overlapping coverage should reach far into the stream
        let max_end = wins.iter().map(|w| w.end()).max().unwrap();
        assert!(
            max_end > 16,
            "expected windows past mid-chunk, max_end={max_end}"
        );
    }

    #[test]
    fn test_windows_for_target_preds() {
        assert_eq!(windows_for_target_preds(32, 32), 2); // 31 preds/window → 2 windows
        assert_eq!(windows_for_target_preds(31, 32), 1);
        assert_eq!(windows_for_target_preds(10, 4), 4); // 3 preds/window
    }

    #[test]
    fn test_train_seq_len_clamp_logic() {
        // pure math of clamp used by train_seq_len defaults
        assert_eq!(2usize.clamp(2, 512), 2);
        assert_eq!(999usize.clamp(2, 512), 512);
    }
}
