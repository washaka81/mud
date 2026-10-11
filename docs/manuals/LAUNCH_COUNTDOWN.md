# LAUNCH COUNTDOWN — Forge LLM / MUD

**Mission:** despegue de producto usable (inferencia + train local ternario).  
**Clock started:** 2026-07-16 · **Reiniciado:** 2026-10-11  
**SSOT status:** `GEMINI.md` §0.3 · **PRIORIDAD 0:** [`docs/planning/MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md`](../planning/MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md) (`R-##`)

---

## ⛔ T-0 "GO" (2026-07-16) — RETIRADO

El board de abajo certificó el despegue **por inspección**. La auditoría del 2026-10-11 ejecutó el proyecto y encontró que varios gates eran falsos o vacíos:

| Gate original | Por qué se retira |
|---|---|
| G2 "Tests lib verdes" | Sólo en **debug**; `cargo test --lib --release` **falla** (UB en `fenwick.rs:307-312`) |
| G4 "Audit CERTIFIED" / G5 "Health preflight" | Dependen de `models/smollm2.mud`, **ausente** en el árbol ⇒ pasan vacíos |
| G6 "`./mud.sh ci`" | Corre sólo debug + `clippy --lib`; el CI remoto es **inválido** y nunca se ejecuta |
| G7 "Inferencia smoke" | Carga y responde, pero la salida es **incoherente** (~46 tok/s medidos) y el trainer **no entrena la cabeza** que usa la inferencia |
| "Orbit F–L DONE" | Varias filas retiradas (Fenwick batch, wormholes −50 %, "0 OOB", conteos de tests) |

**Consecuencia:** el despegue se re-gatea por las **olas del plan maestro**, no por esta lista.

---

## 🥇 Countdown vigente — RE-LAUNCH (gates = olas `R-##`)

| T- | Ola / Gate | Criterio de verificación | Estado |
|----|------------|--------------------------|--------|
| **R0** | **Ola 0 — Verdad y CI** (`R-01…R-06`) | `actionlint .github/workflows/ci.yml` limpio · `cargo test --lib --release` verde · `git status` limpio · `./mud.sh env-audit` = 0 divergencias | 🔴 pendiente |
| **R1** | **Ola 1 — Entrenamiento** (`R-10…R-20`) | `./mud.sh cert-loss <modelo>` con loss **cayendo** ≥40 chunks · `output.weight` cambia entre base y checkpoint · finite-diff SiLU/mHC verde | 🔴 pendiente |
| **R2** | **Ola 2 — Seguridad de entrada** (`R-30…R-38`) | proptest metadata↔shape ⇒ `Err` (no panic/OOB) · GGUF con cotas · `mud.sh unsafe-audit` = 0 | 🔴 pendiente |
| **R3** | **Ola 3 — Eficiencia** (`R-40…R-48`) | `./mud.sh bench-infer` ≥3× tok/s vs baseline (46 tok/s) · RSS < 40 MB en `ale.mud` | 🔴 pendiente |
| **R4** | **Ola 4 — Conversión** (`R-50…R-55`) | gate de paridad ppl/logits vs referencia en CI · unpack I2_S con vector de test | 🔴 pendiente |
| **R5** | **Ola 5 — Limpieza** (`R-60…R-68`) | `cargo build --all-targets` sin huérfanos · `mud.sh tools` sin comandos roto · raíz P-18 | 🔴 pendiente |
| **R6** | **Ola 6 — Verificación** (`R-70…R-74`) | tests de los gates (fallan con fixture degradado) · `bench-energy` J/token documentado | 🔴 pendiente |
| **T-0′** | **DESPEGUE** | Olas 0–1 cerradas + `./mud.sh ci` verde en **release** | ⛔ bloqueado |

> **Regla:** ninguna fila se marca ✅ sin `(comando, salida, fecha)`. Ver `R-73`.

---

## Countdown board histórico (2026-07-16, retirado)

| T- | Gate | Estado | Notas |
|----|------|--------|-------|
| **T-10** | Housekeeping P-18 (root md) | ✅ | Debt → `docs/`; root = 6 SSOT only |
| **T-9** | TREE + docs index sync | ✅ | `TREE.md`, `docs/README.md` |
| **T-8** | `cargo test --lib` | ✅ | **259** passed, 2 ignored (debug) |
| **T-7** | clippy `-D warnings` | ✅ | lib + mud_full_audit + forge_llm |
| **T-6** | `mud_full_audit` CERTIFIED | ⚠️ | válido sólo con modelo presente |
| **T-5** | `training_healthcheck` | ⚠️ | idem |
| **T-4** | `./mud.sh ci` | ⚠️ | debug-only |
| **T-3** | `cargo build --release` | ✅ | ~45s clean |
| **T-2** | Inferencia smoke | ⚠️ | carga; salida incoherente |
| **T-1** | Handoff + knobs freeze note | ✅ | See below |
| **T-0** | ~~DESPEGUE~~ | ⛔ **RETIRADO 2026-10-11** | foundation shippable *por inspección* |

---

## T-1 env freeze (recommended)

Safe defaults (laptop Iris Xe class):

```bash
# unset = auto policies (product defaults)
# MUD_GPU_GEMV=auto
# MUD_TRAIN_FULL_SEQ=1   # full-seq windows
# MUD_CSA=1              # top-k HCA when large
# MUD_USE_VULKAN=1       # if ash stack OK
```

Force-safe if GPU flaky:

```bash
export MUD_GPU_GEMV=0
export MUD_USE_VULKAN=0
```

---

## T-0 — What “despegue” means

**GO for foundation / local product spine:**

- Ternary inference CLI on ship model
- STE QAT train path with live optimizers + full-seq
- MoE load / train-expert knobs
- HCA + CSA + GEMV auto
- CI + structural audit green
- Docs hierarchy clean (P-18)

**Orbit (post-launch backlog):**

| Stream | Item | Status |
|--------|------|--------|
| F | QKV multi-matrix one CB | ✅ DONE |
| K | Loss certification CI gate | ✅ DONE |
| G | Multi-expert STE (round-robin) | ✅ DONE |
| H | Long full-seq + residual API | ✅ DONE |
| I | KV f16 packs | ✅ DONE |
| J | CSA LSH prefilter | ✅ DONE |
| L | Converter P-13 aliases | ✅ DONE |

→ `docs/research/MUD_IMPROVEMENTS_POST_AE.md`

---

## Operator quickstart (post-launch)

```bash
./mud.sh chat                    # or: cargo run --release -- models/smollm2.mud
./mud.sh health models/smollm2.mud
./mud.sh audit-full models/smollm2.mud
./mud.sh train                   # corpus trainer when corpus ready
cargo run --release --bin gemv_auto_bench
```

---

## Abort / scrub

| Symptom | Action |
|---------|--------|
| audit critical | Fix before any tag |
| GEMV first-token hang | `MUD_GPU_GEMV=0` |
| train OOM | `MUD_TRAIN_SEQ_LEN=16` / `MUD_GRAD_CKPT=1` |

---

## Changelog

| Date | Event |
|------|-------|
| 2026-07-16 | Countdown created; T-10…T-0 executed → **GO**. |
| 2026-07-16 | Stream **F** QKV one-CB landed in orbit. |
| 2026-07-16 | Stream **K** loss cert unit tests + `./mud.sh cert-loss`. |
| 2026-07-16 | Residual-bank train wire + MoE hash train (G+/H finish). |
| 2026-07-20 | Telemetry TUI fix (`[TELEM]`→log + key parser + ΔW panel, TLM); pointer hot-loop opt (P-00/P-01); debate writeback hash check; STE deadzone caveat (default LR no-op on converged base). |
| 2026-07-23 | Q2_0→F32 conversion fix; Optimization Sprint P0–P5 (AVX2 batch, AlignedArray<T>, JEPA SIMD batch 8, zero-alloc LM head batch); G+ SlimeX dynamic expert unpacking. |
| 2026-07-24 | Sprint 5 (5.1–5.4: zero-alloc MoE, AtomicU8, LUT backward, branchless pack); Fenwick trees (802 LOC, 25 tests, 19.5× speedup); Sprints 6-7 security docs (`THREAT_MODEL.md`, `HARDENING.md`, `INCIDENT_RESPONSE.md`); 344 lib tests; clippy 0. |
| 2026-07-26 | **Sprint 8 — Batch A/B/C/D-partial.** F1 Fenwick nucleus en `debate_trainer.rs` (patrón `main.rs:438`); F5b `CertMode::Percentile` + `percentile_loss` via `FenwickU32` (6 tests); Muon/GaLore in-place + scratch reuse (P-01 ~4 alloc/NS-call → 0); SEC-H1 caps `MudFile::load` (`MAX_META_ENTRIES=4096` etc) + 3 tests; SEC-H8 `MUD_CIRCUIT_MAX_CYCLES` cap; SEC-H7 `COLLECT_FILES_MAX_DEPTH=32` recursion cap + 1 test; PERF-7 tied-output zero copy (~295 MB saved smollm2 inference); run_trainer absolute paths + `MUD_TRAIN_CKPT_DIR`/`MUD_TRAIN_CORPUS_DIR` env override; `MUD_KV_COMPRESS_CELLV5` env reservado con sync warning. **407 lib tests ✅ | clippy `--all-targets` 0 ✅ | release build verde ✅.** Documentación: `docs/sessions/MUD_SESSION_REPORT_2026-07-26_PENDIENTES.md`. Deferred: 5.1/5.2 CellV5 Fases 2/3 (sesión dedicada bit-exact/lossless) + BATCH E Vulkan D5 elementwise (HW sign-off Iris Xe + dL/dX shader). |
