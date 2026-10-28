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
