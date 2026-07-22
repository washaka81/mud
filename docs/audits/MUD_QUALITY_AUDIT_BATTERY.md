# MUD Quality Battery — Final Audit

**Date:** 2026-07-16  
**Model:** SmolLM2 → `models/smollm2.mud`  
**Scope:** Full solution battery (convert + align + infer + audit)

---

## 1. Solutions applied

| # | Solution | Status |
|---|----------|--------|
| A1 | Convert **without** `--ternarize-emb` (emb FP32) | ✅ Done |
| A2 | **Tied embeddings** (no separate `output.weight` when `tie_word_embeddings=true`) | ✅ Done |
| A3 | Optional `--untie-emb` / `--ternarize-emb` | ✅ Done |
| A4 | Output `1/√H` before ternary pack (when untied ternary) | ✅ Code path present |
| B5 | **`MUD_TRAIN_FREEZE_EMB=1`** (default on `--align`) | ✅ Done |
| B6 | Last-N layers BWD (`LAST_N=8`) | ✅ Done |
| B7 | Medium bilingual corpus + tile | ✅ Present |
| B8 | AWAKE-01 (4 iters) on quality pass | ✅ Done |
| C9 | Greedy infer + `MAX_TOKENS` | ✅ Done |
| C10 | **Skip peak boost** when max logit &lt; 0.05; target peak 8 | ✅ Done |
| C11 | Default `scale_up=1` for FP32 emb | ✅ Done |
| D12 | Live VarH/VarJ pre-BWD + `DEAD_ACT` flag | ✅ Done |
| D13 | FFN w1/w3 mapping unit tests | ✅ 7/7 pass |
| D14 | Converter auditor | ✅ 100% operational |

---

## 2. Convert results

```
[convert] tied embeddings — no separate output.weight (use token_embd)
ECC parity: 210 ternary tensors (layers only; emb not ternary)
token_embd.weight: Float32 [49152, 576]
tie_word_embeddings: true
converter_auditor: 🟢 100% OPERATIONAL CERTIFICATION
  Note: 90 orphan mHC tensors (injected; non-blocking)
```

FFN mapping tests (`moe_load`):
- `w3=up`, `w1=gate`, `w2=down` ✅
- multi-expert discover ✅

---

## 3. Align results (quality battery)

| Param | Value |
|-------|--------|
| Wall time | **~131 s** |
| Chunks | 16 × 24 steps |
| Layers BWD | last 8 / 30 |
| Emb | **frozen** |
| Softmax negs | 63 |
| AWAKE | on (4 iters) |
| Optimizer | SGD · lr=5e-4 |

### Telemetry (live VarH/VarJ)

| Metric | Value |
|--------|--------|
| Loss first → last | 6.03 → 5.61 |
| Loss min / max | 4.31 / 8.28 |
| **VarH mean** | **1.66** (alive; dead_count=0) |
| **VarJ mean** | **0.033** (alive) |
| **Cognitive mean** | **10.0** (alive) |
| DEAD_ACT flags | 0 |

Artifacts:
- `models/smollm2.mud`
- `weights/checkpoints/model_latest_checkpoint.mud`

---

## 4. Inference results

```
emb_rms=0.130  scale_up=1.000  logit_scale=1.000  tied=true  ternary_emb=false
Logit max ≈ 15–17 (healthy dynamic range after DC remove)
```

| Prompt | Result |
|--------|--------|
| Hello | Non-empty generation, **not coherent English** |
| What is 2+2? | Same — token salad |
| The capital of France is | Same |

**Engine status:** load / forward / sample / stop — **OK**  
**Language quality:** still **failed** for this ternary body + short STE align.

---

## 5. Root-cause assessment

| Layer | Finding |
|-------|---------|
| Pipeline | Trainer, telemetry, BOS/EOS, tied FP32 emb, freeze emb — **healthy** |
| Activations | VarH~1.6 during train — **not dead** |
| Convert body | Ternary Q/K/V/O/FFN is aggressive; residual language capacity limited without bitnet-native or long distillation |
| LM-head | FP32 tied emb helps dynamic range vs ternary head (SQNR was ~3 dB) |
| Align budget | 16×24 steps on last 8 layers is a **smoke** recover, not full recovery |

---

## 6. Residual gaps / next hard steps

1. **BitNet-native weights** (if available as U8 ternary) → `repack_bitnet_to_mud` path  
2. **Longer recover**: `LAST_N=0`, 1k+ chunks, emb frozen, corpus from `_stash_full`  
3. **Layer-0 GEMV parity** vs safetensors BF16 (quant error certificate)  
4. **Distill / knowledge transfer** from FP teacher for top layers  
5. **Infer knobs**: try `MUD_INFER_SCALE_UP=auto` only if peaks collapse  

---

## 7. Certification matrix

| Check | Result |
|-------|--------|
| Convert builds | ✅ |
| Emb FP32 + tied | ✅ |
| Auditor operational | ✅ |
| FFN name tests | ✅ 7/7 |
| Align completes | ✅ ~2.2 min |
| Telemetry VarH/VarJ live | ✅ |
| Freeze emb | ✅ |
| Infer runs without crash | ✅ |
| Coherent text | ❌ Not yet |

**Overall:** *Operational certification* for convert/train/infer **infrastructure**.  
*Language certification* for ternary SmolLM2 body: **not granted** after this battery; continue with residual gaps §6.

---

## 8. Reproduce

```bash
# Convert (quality)
cargo run --release --bin universal_converter -- \
  models/smollm2/model.safetensors models/smollm2.mud

# Align
MUD_TRAIN_FREEZE_EMB=1 MUD_TRAIN_LAST_N_LAYERS=8 MUD_TRAIN_SKIP_AWAKE=0 \
  cargo run --release --bin run_trainer -- --align models/smollm2.mud

# Infer
MUD_INFER_GREEDY=1 MUD_INFER_MAX_TOKENS=64 \
  cargo run --release --bin forge_llm -- weights/checkpoints/model_latest_checkpoint.mud

# Audit
cargo run --release --bin converter_auditor -- models/smollm2.mud
cargo run --release --bin train_telemetry
```
