# LAUNCH COUNTDOWN — Forge LLM / MUD

**Mission:** despegue de producto usable (inferencia + train local ternario).  
**Clock started:** 2026-07-16  
**SSOT status:** `GEMINI.md` · backlog: `docs/research/MUD_IMPROVEMENTS_POST_AE.md`

---

## Definition of GO (despegue)

| # | Criterio | Cómo verificar |
|---|----------|----------------|
| G1 | Build release limpio | `cargo build --release` |
| G2 | Tests lib verdes | `cargo test --lib -- --test-threads=2` |
| G3 | Clippy clean (lib + bins clave) | `cargo clippy --lib --bin mud_full_audit --bin forge_llm -- -D warnings` |
| G4 | Audit estructural CERTIFIED | `./mud.sh audit-full models/smollm2.mud` |
| G5 | Health preflight | `./mud.sh health models/smollm2.mud` |
| G6 | CI script | `./mud.sh ci` |
| G7 | Inferencia smoke | load model + quit |
| G8 | Docs raíz P-18 | Solo SSOT en `*.md` raíz |
| G9 | Ops knobs documentados | Gap + improvements + this manual |
| G10 | Model path known | `models/smollm2.mud` (ship sample) |

---

## Countdown board — **T-0 GO** 🚀

| T- | Gate | Estado | Notas |
|----|------|--------|-------|
| **T-10** | Housekeeping P-18 (root md) | ✅ | Debt → `docs/`; root = 6 SSOT only |
| **T-9** | TREE + docs index sync | ✅ | `TREE.md`, `docs/README.md` |
| **T-8** | `cargo test --lib` | ✅ | **186** passed, 2 ignored |
| **T-7** | clippy `-D warnings` | ✅ | lib + mud_full_audit + forge_llm |
| **T-6** | `mud_full_audit` CERTIFIED | ✅ | 0 critical, 0 warnings |
| **T-5** | `training_healthcheck` | ✅ | CERTIFIED smollm2 |
| **T-4** | `./mud.sh ci` | ✅ | L-12 health battery complete |
| **T-3** | `cargo build --release` | ✅ | ~45s clean |
| **T-2** | Inferencia smoke | ✅ | load smollm2 + quit |
| **T-1** | Handoff + knobs freeze note | ✅ | See below |
| **T-0** | **DESPEGUE** | ✅ **GO** | 2026-07-16 — foundation shippable |

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
