# INFORME DE LOGROS vs DEUDA — Forge LLM

**Fecha:** 2026-07-16 (actualizado post streams A–E)  
**Base:** `GEMINI.md` (SSOT) + verificación en código

---

## 1. LOGROS RECIENTES

### 2026-07-16 — Depth streams A–E

| Stream | Item | Estado |
|--------|------|--------|
| A | Adam / SparseAdam moments | ✅ |
| B | MoE `.mud` load + train-expert | ✅ |
| C | GPU GEMV auto policy | ✅ |
| D | Full-seq packed train | ✅ |
| E | CSA top-k lightning indexer | ✅ |

Detalle: `docs/sessions/MUD_SESSION_REPORT_2026-07-16_STREAMS_AE.md` · backlog F+: `docs/research/MUD_IMPROVEMENTS_POST_AE.md`.

### 2026-07-15/16 — Ledger y polish

| # | Item | Estado |
|---|------|--------|
| 1 | Document hierarchy GEMINI / AGENTS / VISION | ✅ |
| 2 | ASM polish + NaN guards | ✅ |
| 3 | `lm_head_logits_avx2` wired | ✅ |
| 4 | L-01…L-15 + Phase B/B+ | ✅ |
| 5 | Compute stack doc | ✅ |

---

## 2. RUNTIME TRUTH (do not contradict)

| Subsystem | Live path | Not live / next |
|-----------|-----------|-----------------|
| Forward GEMV | AVX2×8 + **auto ash** (`gemv_policy`) | QKV multi-matrix one CB (**F**) |
| Accum dtype | **f32** | dual-f16 / i8-act GEMV |
| Weight storage | ELUT 4-bit + PRQ | — |
| QAT step | Muon/GaLore/Chunked + **Adam moments** + STE pack | Multi-expert STE joint (**G**) |
| Train sequence | **Full-seq windows** (default) | Joint multi-token BPTT (**H**) |
| Attention history | HCA + dense + **CSA top-k** (infer) | CSA v2 LSH / W_compress (**J**) |
| KV storage | f32 ring + f32 HCA | bf16/quant (**I**) |
| LM head | `lm_head_logits_avx2` | — |

---

## 3. DEUDA ABIERTA

**Ledger L-01…L-15:** all **DONE** (see `GEMINI.md` §6).

**Open product backlog:** streams **F–L** in `docs/research/MUD_IMPROVEMENTS_POST_AE.md`.

---

## 4. SUPERSEDED CLAIMS (correct if you see old text)

| Old claim | Correction |
|-----------|------------|
| Adam moments still stub / missing | **LIVE** stream A |
| GPU GEMV API only / not in forward | **LIVE** auto + force-on |
| Train only pairs @ pos=0 | **Default full-seq** (D); pairs if `MUD_TRAIN_FULL_SEQ=0` |
| MoE ExpertBus only, no `.mud` load | **LIVE** `moe_load` (B) |
| CSA/HCA = mean-pool only | **+ top-k indexer** (E) |
| Muon only “planned” | L-01 LIVE |
| Next work = L-01 | **Wrong** — next = F+ improvements |

---

## 5. TOOLING

- `./mud.sh ci` · `audit-full` · `health`
- `mud_full_audit` sections 3b CSA, 6a GEMV, 6b MoE, 8 full-seq
- `gemv_auto_bench` — break-even on device
- `training_healthcheck` — LIVE policy lines

---

## 6. HANDOFF (siguiente)

1. **T-0 GO** — foundation shippable (`docs/manuals/LAUNCH_COUNTDOWN.md`)  
2. **Orbit F–L foundations DONE** · remaining: router STE BPTT, residual-bank wire, nightly e2e cert  
3. Keep green: `./mud.sh ci` · `audit-full` · `gemv_auto_bench`

*Update this file when closing F+ items; do not invent parallel ledgers.*
