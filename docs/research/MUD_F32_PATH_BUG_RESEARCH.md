# Research: F32 Inference Path Incoherence

## Symptom
Models with `mud:weight_type=ternary2bit` but `token_embd.weight: Float32` (Q2_0→F32 converted embedding, tied LM head) run without crash after the `k_f32`→`ffn_out_f32` buffer fix, but produce incoherent output (success 0%, `coherence_healthy=FAIL`, high entropy ~4.5 bits). Pure-ternary models (`ale.mud`) are also incoherent, so the defect is not F32-exclusive but the F32 path amplifies it.

## Candidate root causes
1. **Embedding scale mismatch** — F32 embedding is used raw (scale_up=1.0) while ternary weights are PRQ-scaled. The residual stream magnitude may be off by ~1/emb_rms, causing saturation in attention/norms. Ternary path applies per-row scale_up; F32 path does not rescale.
2. **RMSNorm numerical drift** — `apply_output_norm` GPU path now fixed, but per-layer `rms_norm` (CPU) on F32 residuals may accumulate error; the ternary path uses i8-quantized activations with different rounding.
3. **RoPE not applied / wrong freq** — if `rope_theta`/`rope_inv_freq` mismatch the converted model's config, attention positions are wrong. Check `rope.freq_base` vs `rope_theta`.
4. **Attention softmax / GQA** — F32 path uses the same SDPA as ternary; unlikely but check head_dim and n_kv_heads parsing.
5. **SiLU/activation** — `dense_ffn_swiglu` F32 path submits up+gate then waits; if the wait barrier is missed, gate uses stale data.
6. **LM head stride** — `f32_gemv_rowwise` assumes row-major `[n_out, n_in]`. Converter stores `output.weight` as `[vocab, hidden]` (row-major). If the mmap slice is transposed or misaligned, logits are garbage. Verify with a tiny reference GEMV.
7. **Tied head aliasing** — `out_tensor = output.weight.unwrap_or(emb_tensor)`; when tied, `head_w = take(emb_f32)` moves the embedding out. If `emb_f32` was already consumed/modified earlier, the head is corrupt. Verify `emb_slice` borrow ends before the take (it does, per PERF-7).

## Proposed solutions (ranked)
A. **Reference comparison** (highest value): implement a slow, scalar, reference forward for a single layer on both a known-good ternary model and the F32 model; diff intermediate activations (post-norm, Q, attention out, FFN out) to locate the first divergence. Add a `--debug-layer N` flag.
B. **Scale calibration**: set `MUD_INFER_SCALE_UP=auto` (clamp 1/emb_rms to [1,8]) for F32 embeddings and measure coherence; if improves, make it the default for F32 emb.
C. **Verify RoPE**: print `rope_theta`, `rope_inv_freq[0]`, compare to HF config; fix parsing if needed.
D. **Verify LM-head stride**: add a unit test computing one logit with scalar dot product vs `f32_gemv_rowwise`; fail if mismatch.
E. **Fallback to ternary conversion**: re-convert the Q2_0 model keeping weights ternary (not F32) so it uses the well-tested ternary path; accept larger conversion effort. This sidesteps the F32 path entirely.
F. **Activation barrier check**: in `dense_ffn_swiglu` F32 branch, ensure `wait_all()` is called before SiLU reads `ffn_gate_f32`.

## Immediate next step
Implement (D) and (A) first — they are cheap and will localize the defect before any risky change.

## Live diagnostic (2026-10-28)
Instrumented `inference.rs` with `MUD_DEBUG_LAYERS=1` on the Kaggle F32 model (896 hidden, 24 layers, single token). Measured hidden-norm (L2 of registers) per layer:

| layer | pre-norm | post-layer |
|-------|----------|------------|
| 0     | 0.53     | 34.06      |
| 1     | 34.06    | 44.90      |
| 2     | 44.90    | 44.90      |

The explosion happens inside layer 0 (0.53 → 34.06), i.e. the very first transformer block. Subsequent layers saturate. This points to the attention or FFN residual of layer 0, not to depth accumulation.

## Next concrete steps
1. Instrument `evaluate_slime_block` (gated by `MUD_DEBUG_LAYERS`) to print per-sub-block norms for layer 0: post-attn-norm, Q/K/V norms, attention-out norm, post-FFN-norm, residual. This localizes whether the blow-up is in attention (Q/K/V magnitude) or FFN (up/gate/down).
2. Compare with a pure-ternary model (`ale.mud`) running the same instrumentation: if ternary also explodes, the defect is in shared code (RMSNorm/attention); if only F32 explodes, it is in the F32 GEMV or scale path.
3. Candidate fix based on (1)/(2):
   - If Q/K/V norms are >>1: add per-row output scaling or RMSNorm on Q/K (Qwen2 has `attn_q_norm`/`attn_k_norm` — verify these tensors are loaded and applied in the F32 path; `q_norm_w`/`k_norm_w` may be null for Qwen2.5-Coder).
   - If FFN-out explodes: check SiLU gate magnitude and down-projection scale; possibly the down weight is not normalized.
   - If residual accumulates: verify the residual add uses `read_accum`/`write_accum` with correct f16 dual packing (P-02) and that `act_scale` is applied.

## Status
- Per-layer norm diagnostic: DONE (finding documented above).
- Per-sub-block diagnostic: PENDING.
- Fix: PENDING (depends on (1)).
