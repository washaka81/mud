use crate::mud::slime::SlimeWorkspace;
use crate::mud::slime_forward::evaluate_slime_block;
use crate::mud::slime_forward::SlimeLayer;

/// DSpark Speculative Drafter (Priority 39)
/// Proposes K token candidates using a lightweight ternary model (e.g. first 2 layers).
pub struct SlimeDrafter<'a> {
    layers: Vec<&'a SlimeLayer>,
    output_weight: *const f32,
    hidden: usize,
    vocab_size: usize,
    // P-01: Zero-allocation buffers
    regs_buf: Vec<f32>,
    logits_buf: Vec<f32>,
}

impl<'a> SlimeDrafter<'a> {
    pub fn new(
        all_layers: &'a [SlimeLayer],
        num_draft_layers: usize,
        output_weight: *const f32,
        hidden: usize,
        vocab_size: usize,
    ) -> Self {
        let layers: Vec<&'a SlimeLayer> = all_layers.iter().take(num_draft_layers).collect();

        Self {
            layers,
            output_weight,
            hidden,
            vocab_size,
            regs_buf: vec![0.0; hidden],
            logits_buf: vec![0.0; vocab_size],
        }
    }

    /// Proposes K tokens.
    /// The drafter uses the main workspace, temporarily advancing the KV cache.
    /// The main model will overwrite these positions during verification.
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn propose_tokens(
        &mut self,
        ws: &mut SlimeWorkspace,
        start_pos: usize,
        k: usize,
        eps: f32,
        current_token: u32,
        token_embd_ptr: *const f32,
    ) -> Vec<u32> {
        let mut proposals = Vec::with_capacity(k);
        let mut tok = current_token;

        if self.output_weight.is_null() || token_embd_ptr.is_null() {
            return proposals;
        }

        let pool = crate::mud::pcore_pool::get_pool();
        let rows_per_task = ((self.vocab_size / 8) / 4 * 4).max(4);
        let out_w_p = self.output_weight as usize;

        for step in 0..k {
            let pos = start_pos + step;

            // 1. Embed token
            let row_start = tok as usize * self.hidden;
            for h in 0..self.hidden {
                let emb_val = unsafe { *token_embd_ptr.add(row_start + h) };
                crate::mud::slime::SlimeRegister::init_from_embed(
                    &mut ws.registers[h],
                    &mut ws.jepa_z,
                    h,
                    ws.hidden_size,
                    ws.num_layers,
                    emb_val,
                    pos == 0,
                );
            }

            // 2. Run draft layers
            for (l_idx, layer) in self.layers.iter().enumerate() {
                evaluate_slime_block(layer, l_idx, ws, pos, eps, None);
            }

            // 3. Simple Greedy LM Head (Parallelized & Zero-Allocation)
            for h in 0..self.hidden {
                self.regs_buf[h] = ws.registers[h].matmul_accum;
            }

            let regs_p = self.regs_buf.as_ptr() as usize;
            let logits_p = self.logits_buf.as_mut_ptr() as usize;
            let hidden_dim = self.hidden;
            let vocab = self.vocab_size;

            for i in 0..8 {
                let start_row = i * rows_per_task;
                let end_row = if i == 7 {
                    vocab
                } else {
                    start_row + rows_per_task
                };
                if start_row >= end_row {
                    break;
                }

                pool.execute(move || {
                    let r_ptr = regs_p as *const f32;
                    let l_ptr = logits_p as *mut f32;
                    let w_ptr = out_w_p as *const f32;
                    let rows = end_row - start_row;

                    unsafe {
                        // Compute sgemm for a subset of the vocabulary (subset of rows of output_weight)
                        // sgemm_abt computes: out[i] = dot(A[0], B[i])
                        // A is 1xHidden, B is VocabXHidden.
                        crate::asm::sgemm_abt(
                            1,
                            rows,
                            hidden_dim,
                            r_ptr,
                            w_ptr.add(start_row * hidden_dim),
                            l_ptr.add(start_row),
                        );
                    }
                });
            }
            pool.wait_all();

            // Find max logit
            let mut best_id = 0;
            let mut max_val = f32::NEG_INFINITY;
            for (i, &l) in self.logits_buf.iter().enumerate() {
                if l > max_val {
                    max_val = l;
                    best_id = i;
                }
            }

            proposals.push(best_id as u32);
            tok = best_id as u32;
        }

        proposals
    }
}

/// DSPARK (DeepSeek, 2026 — open-source MIT via DeepSpec): confidence-scheduled
/// speculative decoding. Given a drafter `confidence` in `[0,1]` (a prefix-survival
/// estimate, e.g. from a confidence head) and the current engine `load` in `[0,1]`
/// (PCorePool utilization), choose how many draft tokens `gamma` to verify this
/// round — the **hardware-aware prefix scheduler**.
///
/// - High confidence + low load → verify a longer block (lower per-user latency).
/// - Low confidence or high load → verify only the high-confidence prefix, avoiding
///   wasted batch capacity on tokens the target will reject (prevents the throughput
///   cliff under strict SLAs). Reuses `gemv_policy`'s load signal at the speculative
///   level. `base_gamma` is the configured maximum draft length.
pub fn schedule_draft_length(base_gamma: usize, confidence: f32, load: f32) -> usize {
    if base_gamma == 0 {
        return 0;
    }
    let confidence = confidence.clamp(0.0, 1.0);
    let load = load.clamp(0.0, 1.0);
    let factor = (confidence * (1.0 - 0.5 * load)).clamp(0.25, 1.0);
    let gamma = (base_gamma as f32 * factor).round() as usize;
    gamma.clamp(1, base_gamma)
}

/// DSP-2 + DSP-5: derive the confidence-scheduled draft length directly from a
/// draft/target hidden-state pair (spherical-normalized cosine, see `confidence_spherical`),
/// then run the hardware-aware prefix scheduler. Connects the confidence head (DSP-2)
/// with the spherical-norm alignment (DSP-5).
pub fn schedule_draft_from_hidden(
    base_gamma: usize,
    draft_hidden: &[f32],
    target_hidden: &[f32],
    load: f32,
) -> usize {
    let conf = confidence_spherical(draft_hidden, target_hidden);
    schedule_draft_length(base_gamma, conf, load)
}

/// DSP-1: semi-autoregressive (first-order Markov) draft loop.
/// Produces up to `k` tokens, feeding each drafted token back as the previous
/// token for the next step (sequential conditioning). `next_logits` returns the
/// next-position logits for the given prefix; `markov_scale>0` applies a
/// sequential bias from the previous token (the lightweight Markov head that
/// mitigates suffix decay). The model-backed instance of this loop is
/// `SlimeDrafter::propose_tokens`.
pub fn sequential_draft<F>(
    prefix: &[u32],
    k: usize,
    markov_scale: f32,
    mut next_logits: F,
) -> Vec<u32>
where
    F: FnMut(&[u32]) -> Vec<f32>,
{
    let mut seq = prefix.to_vec();
    let mut out = Vec::with_capacity(k);
    let mut prev: u32 = prefix.last().copied().unwrap_or(0);
    for _ in 0..k {
        let mut logits = next_logits(&seq);
        if logits.is_empty() {
            break;
        }
        if markov_scale != 0.0 {
            markov_bias(prev, &mut logits, markov_scale);
        }
        let tok = argmax(&logits);
        out.push(tok);
        seq.push(tok);
        prev = tok;
    }
    out
}

/// DSP-1: lightweight sequential (Markov) bias — shift the next-token logits by a
/// deterministic per-previous-token offset so the draft conditions on the accepted
/// prior token. Weight-free stub; training replaces this with a learned 1-gram table.
pub fn markov_bias(prev_token: u32, logits: &mut [f32], scale: f32) {
    if logits.is_empty() || scale == 0.0 {
        return;
    }
    let bias = (((prev_token as usize) + 1).wrapping_mul(2654435761)) % logits.len();
    logits[bias] += scale;
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut mv = f32::NEG_INFINITY;
    for (i, &l) in logits.iter().enumerate() {
        if l > mv {
            mv = l;
            best = i;
        }
    }
    best as u32
}

/// DSP-3: anchor-bounded packing — split a packed sequence into non-overlapping
/// anchor spans of length <= `anchor_stride`, covering the whole sequence with
/// NO padding (token-level attention indices, not 2D masks). Mirrors DSpark's
/// anchor-bounded training packing over MUD's L-10 `sequence_pack`.
#[derive(Debug, PartialEq)]
pub struct AnchorSpan {
    pub start: usize,
    pub end: usize,
}

pub fn anchor_boundaries(seq_len: usize, anchor_stride: usize) -> Vec<AnchorSpan> {
    if seq_len == 0 || anchor_stride == 0 {
        return Vec::new();
    }
    let stride = anchor_stride.max(1);
    let mut spans = Vec::new();
    let mut s = 0;
    while s < seq_len {
        let e = (s + stride).min(seq_len);
        spans.push(AnchorSpan { start: s, end: e });
        s = e;
    }
    spans
}

/// Flatten anchor spans into token-level (global) attention indices — length ==
/// `seq_len` exactly (zero padding), preserving each anchor's local causal order.
pub fn anchor_attention_indices(spans: &[AnchorSpan]) -> Vec<usize> {
    let mut idx = Vec::new();
    for sp in spans {
        for i in sp.start..sp.end {
            idx.push(i);
        }
    }
    idx
}

/// DSP-4: hidden-state communication O(d). Instead of shipping full vocabulary
/// logits `O(V)` between drafter/target workers, ship only a `d`-dim hidden-state
/// summary (the activation just before the LM head). `project_hidden_to_d`
/// deterministically projects `H -> d` with a seeded LCG matrix (weight-free stub
/// standing in for a learned low-rank projector). The shipped payload is `O(d)`.
pub fn project_hidden_to_d(hidden: &[f32], d: usize, seed: u64) -> Vec<f32> {
    if d == 0 || hidden.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0.0f32; d];
    let mut rng = seed;
    for slot in out.iter_mut() {
        let mut acc = 0.0f32;
        for &x in hidden.iter() {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let w = ((rng >> 40) as f32 / (1u64 << 24) as f32) - 0.5; // ~[-0.5, 0.5]
            acc += w * x;
        }
        *slot = acc;
    }
    out
}

/// Bytes transferred for `O(d)` hidden comm vs `O(V)` logits comm (f32).
pub fn hidden_comm_bytes(hidden_dim: usize, vocab: usize, d: usize) -> (usize, usize) {
    (hidden_dim.max(d) * 4, vocab * 4)
}

/// DSP-5: spherical (L2) normalization for draft-target alignment. Keeps the
/// confidence head scale-invariant (RFC DeepSpec #52) and avoids the rejection
/// cascade when draft/target manifolds drift. Zero vector -> zeros (no NaN).
pub fn spherical_norm(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm <= 1e-12 {
        return vec![0.0; v.len()];
    }
    v.iter().map(|x| x / norm).collect()
}

/// Confidence from a draft hidden-state vs target hidden-state: cosine similarity
/// after spherical normalization, clamped to `[0,1]`. Scale-invariant.
pub fn confidence_spherical(draft: &[f32], target: &[f32]) -> f32 {
    if draft.len() != target.len() || draft.is_empty() {
        return 0.0;
    }
    let a = spherical_norm(draft);
    let b = spherical_norm(target);
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    dot.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drafter_initialization() {
        // Just verify we can instantiate it cleanly
        let layers = vec![];
        let drafter = SlimeDrafter::new(&layers, 2, std::ptr::null(), 256, 1000);
        assert_eq!(drafter.layers.len(), 0);
    }

    #[test]
    fn test_schedule_draft_length() {
        // base case: full confidence, idle engine → verify the whole block.
        assert_eq!(schedule_draft_length(5, 1.0, 0.0), 5);
        // low confidence under load → shrink toward the minimum.
        let g = schedule_draft_length(5, 0.1, 0.9);
        assert!(
            g < 5,
            "low confidence+high load must shrink gamma (got {g})"
        );
        assert!(g >= 1, "gamma must stay >=1 (got {g})");
        // high load alone (confident) still shrinks somewhat.
        let g2 = schedule_draft_length(5, 0.9, 0.9);
        assert!((1..=5).contains(&g2));
        // zero base → zero.
        assert_eq!(schedule_draft_length(0, 1.0, 0.0), 0);
        // clamps inputs.
        assert_eq!(schedule_draft_length(5, 2.0, -1.0), 5);
    }

    #[test]
    fn test_schedule_draft_from_hidden() {
        // parallel (high confidence) + idle -> full block; orthogonal -> shrink.
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];
        assert_eq!(schedule_draft_from_hidden(5, &a, &b, 0.0), 5);
        let g = schedule_draft_from_hidden(5, &a, &c, 0.0);
        assert!(g < 5, "orthogonal draft/target must shrink gamma (got {g})");
    }

    #[test]
    fn test_sequential_draft_length_and_markov() {
        // mock: constant logits -> all drafts equal to argmax (index 0 on zeros).
        let draft = sequential_draft(&[], 4, 0.0, |_| vec![0.0; 8]);
        assert_eq!(draft.len(), 4);
        assert_eq!(draft, vec![0, 0, 0, 0]);

        // with Markov bias, the previous-token offset shifts the argmax and the
        // chain diverges from the unbiased path.
        let draft_biased = sequential_draft(&[], 4, 1.0, |prefix| {
            let mut l = vec![0.0; 8];
            // bias index depends on last token so the chain evolves.
            if let Some(&p) = prefix.last() {
                markov_bias(p, &mut l, 1.0);
            }
            l
        });
        assert_eq!(draft_biased.len(), 4);
        // first token: prev=0 -> bias index 1 -> argmax 1 (not 0).
        assert_eq!(draft_biased[0], 1u32);
    }

    #[test]
    fn test_anchor_boundaries_padding_free() {
        // seq_len=10, stride=4 -> 3 spans, last is remainder, full coverage, no pad.
        let spans = anchor_boundaries(10, 4);
        assert_eq!(
            spans,
            vec![
                AnchorSpan { start: 0, end: 4 },
                AnchorSpan { start: 4, end: 8 },
                AnchorSpan { start: 8, end: 10 },
            ]
        );
        let idx = anchor_attention_indices(&spans);
        assert_eq!(idx.len(), 10);
        assert_eq!(idx, (0..10).collect::<Vec<_>>());
        // no overlap: each start equals previous end.
        for w in spans.windows(2) {
            assert_eq!(w[0].end, w[1].start);
        }

        // edge: zero seq -> empty (no padding allocated).
        assert!(anchor_boundaries(0, 4).is_empty());
        // exact multiple: single span.
        assert_eq!(
            anchor_boundaries(4, 4),
            vec![AnchorSpan { start: 0, end: 4 }]
        );
        // stride 0 guard.
        assert!(anchor_boundaries(10, 0).is_empty());
    }

    #[test]
    fn test_project_hidden_to_d_od() {
        let h = vec![1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32];
        let p1 = project_hidden_to_d(&h, 2, 12345);
        assert_eq!(p1.len(), 2);
        // determinism.
        assert_eq!(p1, project_hidden_to_d(&h, 2, 12345));
        // different seed -> different projection.
        assert_ne!(p1, project_hidden_to_d(&h, 2, 99999));
        // linearity: project(c*h) == c*project(h).
        let scaled = h.iter().map(|x| x * 2.0_f32).collect::<Vec<_>>();
        let p2 = project_hidden_to_d(&scaled, 2, 12345);
        for (a, b) in p1.iter().zip(p2.iter()) {
            assert!((a * 2.0_f32 - b).abs() < 1e-5_f32);
        }
        // O(d) comm is far smaller than O(V) when d << vocab.
        let (od, ov) = hidden_comm_bytes(576, 128000, 64);
        assert!(od < ov);
    }

    #[test]
    fn test_spherical_norm_and_confidence() {
        // unit vector stays unit.
        let n = spherical_norm(&[3.0_f32, 4.0_f32]);
        assert!((n[0] - 0.6_f32).abs() < 1e-6_f32);
        assert!((n[1] - 0.8_f32).abs() < 1e-6_f32);
        // zero vector -> zeros, no NaN.
        let z = spherical_norm(&[0.0_f32, 0.0_f32, 0.0_f32]);
        assert!(z.iter().all(|&x| x == 0.0_f32));

        // parallel -> ~1.0, orthogonal -> ~0.0.
        let d = vec![1.0_f32, 1.0_f32, 1.0_f32];
        let t = vec![1.0_f32, 1.0_f32, 1.0_f32];
        let o = vec![1.0_f32, -1.0_f32, 0.0_f32];
        assert!((confidence_spherical(&d, &t) - 1.0_f32).abs() < 1e-6_f32);
        assert!(confidence_spherical(&d, &o) <= 1e-6_f32 + 1e-6_f32);
        // scale invariance: scaling inputs must not change confidence.
        let d2 = d.iter().map(|x| x * 7.0_f32).collect::<Vec<_>>();
        let t2 = t.iter().map(|x| x * 0.3_f32).collect::<Vec<_>>();
        assert!((confidence_spherical(&d, &t) - confidence_spherical(&d2, &t2)).abs() < 1e-6_f32);
    }
}
