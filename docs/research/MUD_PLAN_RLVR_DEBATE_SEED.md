# MUD Plan: RLVR Debate por Supervivencia de Semilla (Recompensa/Castigo + Juez + Aprendizaje)

**Date:** 2026-07-17
**Author:** agent (research-backed)
**Status:** IMPLEMENTED + VALIDATED (2026-07-17). Phase 1/2/3 + infinite-mode + Professor-Student mode + seed-driven Training Circuit done. **Honors-mode eval gate corrected (2026-07-18):** structural integrity + win-rate benchmark vs baseline; false-collapse/null-ptr bugs fixed.
**Supersedes:** ad-hoc reward stubs in `debate_trainer.rs` (`compute_jepa_reward`, `apply_learning`).
**Constraint (2026-07-17):** NO external API on any judge. Local-only scoring
(self-embedding cosine + `rustc`), P-07 compliant.

### Implementation notes (2026-07-17)
- `src/mud/debate_trainer.rs`: `generate_agent_response` now mirrors the
  `main.rs` inference path exactly (was the root cause of numeric-collapse
  output): `SlimeRegister::init_from_embed` (not `write_accum`), DC-bias
  removal, thermodynamic peak rescale (~8.0), repetition penalty (1.25), then
  Doppler temp + top-p. Without these the debate emitted a single number/token.
  `bos_id` + `max_new_tokens` are env-tunable (`MUD_DEBATE_BOS`,
  `MUD_DEBATE_MAX_NEW_TOKENS`, default 32).
- `src/mud/arena_judge.rs` (NEW): `Judge` trait, `VerifiableJudge` (uses
  `game.winner()`), `RustJudge` (wraps `rlvr::RlvrCritic`), `TextJudge` (local
  claim-extraction + self-embedding cosine, no API). 3 unit tests pass.
- `src/mud/debate_trainer.rs`: `run_game` rewritten as **infinite loop** until
  `stop_flag`/Ctrl-C/`MUD_DEBATE_MAX_TIME`. Each finished match is scored by the
  judge; `DocumentDebate` uses `TextJudge` on the last pair of responses;
  reward/penalty asymmetric (`R_WIN=1.0`/`R_LOSE=0.7`, env-tunable); JEPA aux
  (`MUD_DEBATE_JEPB_LAMBDA`, default 0.05); degenerate-play penalty; neural-kick
  jitter on `var_h<1e-3`. `DebateArena` gained `stop_flag: Arc<AtomicBool>`.
- `src/mud/arena_games.rs`: `DocumentDebate` now stores `last_a`/`last_b` +
  `last_responses()` + `max_turns()`; `winner()` stays `None` (resolved by
  `TextJudge` in the arena loop).
- `src/mud/corpus_trainer.rs`: `run_debate_session(sender, stop_flag)`; passes a
  `game_factory` to `run_game`; Fase-3 writeback gated by `MUD_DEBATE_LEARN`
  (default OFF) via the existing in-place `sync_tensor` + `mud.save` path.
- `tools/run_trainer.rs` + `tools/debate_telemetry.rs`: pass `stop_flag`
  (`Arc<AtomicBool>`); TUI Ctrl-C / `q` stops the infinite arena.
- **Validation:** `cargo clippy --lib` 0 warnings on new code; `cargo test --lib
  -- --test-threads=1` 221 passed; debate runs live (infinite, detenable by
  SIGINT / `MUD_DEBATE_MAX_TIME`).
- **Open decisions resolved:** (2) asymmetric penalty CONFIRMED; (3) `LEARN`
  default OFF CONFIRMED; (4) `K=8` deferred — Fase 2 (population) not yet wired
  to `run_game` (single Alpha/Beta pair retained; tournament loop is the next
  step if multi-seed survival is required).

### Implementation notes (2026-07-18) — honors-mode evaluation gate fix

The seed-driven Training Circuit (`--circuit`) decides the save via a post-phase
**honors evaluation** (`circuit_eval_integrity` + `circuit_benchmark_games`). The
first implementation had two defects that made every phase roll back:

1. **Null `data_ptr` after `MudFile::load`.** The `.mud` loader leaves raw
   `data_ptr`s pointing into the mmap; weight tensors only get a writable
   backing buffer after `materialize_writable()`. The integrity check read
   `data_ptr` directly → 90/210 tensors reported `null` → always "missing".
   *Fix:* call `mud.materialize_writable()` before inspecting.
2. **Wrong tensor keys.** The model is MoE, so FFN weights live at
   `blk.N.expert.0.w{1,2,3}.weight`, not `blk.N.ffn_{up,gate,down}.weight`
   (attention `q/k/v/output` were correct). *Fix:* enumerate the MoE expert
   keys alongside the attention keys (210 total for a 30-layer model).
3. **False collapse flag.** The original check flagged a tensor as "collapsed"
   if >99% of its nibbles were identical. Healthy ternary weights are naturally
   skewed toward `0x0`, so a good model false-flagged. *Fix:* integrity is now
   **structural** — every STE-writable tensor must be present, non-empty and
   have a valid backing buffer (`data_ptr` or `owned_data`). Per-weight skew is
   no longer a failure; the **quality regression gate** is the verifiable-game
   win-rate benchmark vs the circuit-start baseline.

After the fix the circuit logs e.g.
`🏅 HONORES ✓ integridad: integrity ok (210 weight tensors present) | calidad
win_rate=0.00 (baseline 0.00, 0 matches) → se guarda`.

**CPU caveat:** on the i7-1260P a single verifiable match (≈3 turns × N tokens)
can exceed the phase time-box, so the benchmark may report `0 matches`. Then
`win_rate == baseline == 0.00` and the quality gate cannot regress — integrity
stays the effective guard. Rollback still fires on a real structural failure or
a quality drop once matches complete. Validated: `cargo clippy --all-targets -D
warnings` 0/0, `cargo test --lib` 221 passed, live circuit run confirms
honors/save flow.

> Goal: turn the existing `DebateArena` (two `Doppelganger` clones of one `.mud`,
> `src/mud/debate_trainer.rs`) into a **seed-survival RLVR loop** with a
> *verifiable judge*, a *reward/penalty signal*, and *real weight learning* —
> all on commodity CPU (i7-1260P, ~15 GiB, Iris Xe), zero Python (P-07),
> raw-pointer hot path (P-00), zero-alloc in the learn step (P-01), clippy 0/0 (P-06).

---

## 0. Why (research grounding)

| Mechanism | Literature | Measured benefit |
|-----------|-----------|------------------|
| RLVR (verifiable reward) | DeepSeek-R1 / "RLVR beats SFT on reasoning" (2025) | reward on *checkable* outcome (win/compile/correct) >> imitation; no reward-model needed |
| Self-play population | AlphaGo / Evolution strategies | survival-of-fittest select pressure converges faster than fixed-opponent |
| Seed-survival (elitism) | OpenES / CMA-ES | keep top-K genomes, mutate rest → no catastrophic forgetting across games |
| JEPA as intrinsic critic | LLM-JEPA (arXiv:2509.14252) | `varj`/representation-collapse signal as *auxiliary* reward, not the objective |

Key facts shaping this plan:
1. **Debate has no judge today.** `DocumentDebate.winner()` returns `None` (`arena_games.rs:237`) — decided implicitly. We add an explicit, *verifiable* judge.
2. **Reward is coupled to learning but weak.** `apply_learning` (`debate_trainer.rs:129`) uses `loss = -reward` only if `|loss| > 0.01`; `compute_jepa_reward` (`:115`) is VarH/VarJ-based. We keep JEPA as *aux* and add a *task/verifiable* reward.
3. **No seed concept.** `run_game` (`debate_trainer.rs:410`) plays one game Alpha vs Beta (same model). We add a **population of K seeds** (independent `Doppelganger` clones + RNG streams) and let survivors breed.
4. **`RlvrCritic` exists but is unwired** (`rlvr.rs`, compiles Rust → ±1). Reuse it as the verifiable judge for code/math games; add an LLM-as-judge text scorer for `DocumentDebate`.
5. **Low-resource budget.** No extra forward passes for the critic beyond what the game already computes. Judge runs on CPU; keep it O(turns), not O(tokens²).

---

## 1. Scope & Non-Goals

**In scope**
- Phase 1: explicit **verifiable judge** + reward/penalty signal wired into `apply_learning`.
- Phase 2: **seed-survival population** (K seeds, elitism, mutation) replacing the single Alpha/Beta game.
- Phase 3: **learning integration** — gradient from judge reward flows to the shared `.mud` (STE pack), with JEPA aux reward + anti-collapse.

**Non-goals**
- No external reward model / LLM API call (P-07: no Python; judge is local + verifiable).
- No change to ELUT wire format, GEMV kernels, Vulkan path.
- No infinite-horizon games; cap turns (already `turn_count < 50`).
- No PBT across full model params — only last-N layers get STE updates (RAM-safe, like corpus_trainer).

---

## 2. Current truth (verified in code)

- `DebateArena` (`debate_trainer.rs:21`): two `Doppelganger { workspace, bw_ws, tapes, gradients, rng }` over shared `layers`/`shadow_layers`.
- `run_game<G: ArenaGame>` (`:410`): loop `while !is_terminal() && turn<50`; each turn `generate_agent_response` (`:288`, top-p + Doppler temp) → `game.apply_move` → reward `f32`.
- `apply_learning` (`:129`): backward with `loss = -reward`, clipped `|loss|>0.01`.
- `compute_jepa_reward` (`:115`): returns VarH/VarJ-based scalar (intrinsic only).
- `arena_games.rs`: `ArenaGame` trait (`is_terminal`, `winner`, `apply_move→f32`, `get_state_prompt`); `DocumentDebate.winner()` → `None`; `MathChallenge` has real winner.
- `rlvr.rs`: `RlvrCritic::evaluate_rust_code` → `rustc --emit=metadata` → +1/−1. **Unwired to debate.**
- `self_play.rs`: `apply_gradient_jitter`, `is_sequence_confident`, entropy — anti-collapse helpers, unused in debate.
- Entry: `./mud.sh debate` → `debate_telemetry` TUI → `run_debate_session(Some(tx))`.

---

## 3. Phase 1 — Verifiable Judge + Reward/Penalty (no new infra)

**Rationale:** today's reward is VarJ-coupled (weak, not task-linked). Add a
*deterministic, checkable* judge so the winner actually earns reward and the
loser is penalized. This is exactly RLVR: reward only on verifiable outcome.

### 3.1 Judge contract
```rust
// src/mud/arena_judge.rs (new, tiny, P-06 clean)
pub trait Judge {
    /// Returns reward in [-1, +1] for a finished game from the POV of `player`.
    /// +1 win, -1 loss, 0 draw/timeout. Aux signals (clarity, coherence) optional.
    fn score(&self, game: &dyn ArenaGame, player: Player) -> f32;
}
```
- `VerifiableJudge` for `MathChallenge`/`GrammarChallenge`/`TicTacToe`: use existing
  `game.winner()` (already correct for these) → ±1.
- `RustJudge` for code games: wrap `RlvrCritic::evaluate_rust_code` (reuse `rlvr.rs`)
  → ±1 on compile/spec.
- `TextJudge` for `DocumentDebate`: local, no-API heuristic — **verifiable** proxy:
  claim overlap vs opponent (F1 of asserted propositions), contradiction count,
  length/structure floor. Not an LLM call (P-07). Returns `[-1, +1]`.
  *Decision:* ship `TextJudge` as a structured claim-scorer (extraction of
  assertions + entailment-via-embedding-cosine on the local `.mud` emb), not an
  external model.

### 3.2 Reward/penalty wiring
- Winner gets `+R_WIN` (1.0); loser `-R_LOSE` (0.7, asymmetric so draws don't
  dominate); timeout/draw 0.0. Tunable via `MUD_DEBATE_RWIN` / `MUD_DEBATE_RLOSE`.
- Aux JEPA intrinsic: `reward_intrinsic = λ_j * compute_jepa_reward(var_h, var_j)`
  (`λ_j ≈ 0.05`, like STP weight) — keeps representations alive, anti-collapse.
- Total per-turn reward fed to `apply_learning` as `loss = -(R_verifiable + reward_intrinsic)`.
- **Penalty for degenerate play:** if `is_sequence_confident` (self_play.rs) says
  entropy collapse OR `var_h < 1e-3` (DEAD_ACT) → extra `-R_DEGEN` penalty.
  Reuse `self_play::apply_gradient_jitter` after the step to break symmetry.

### 3.3 Changes
1. `src/mud/arena_judge.rs`: `Judge` trait + `VerifiableJudge`/`RustJudge`/`TextJudge`.
2. `run_game` (`debate_trainer.rs:410`): after `game.is_terminal()`, call
   `judge.score(game, player)` for both players; pass winner reward into
   `apply_learning` (replace the bare `compute_jepa_reward` path).
3. `DocumentDebate` (`arena_games.rs:174`): implement `winner()` via `TextJudge`
   (so `is_terminal` + `winner` are consistent; drop the `None` fallback).
4. Keep `compute_jepa_reward` as the intrinsic aux term (renamed `jepa_aux`).

### 3.4 Acceptance (Phase 1)
- Unit test: `VerifiableJudge` on `TicTacToe` known terminal → ±1 correct sign.
- Unit test: `TextJudge` prefers a reply that asserts the opponent's claim (higher
  F1) over a non-sequitur (verifiable, deterministic).
- 1-game smoke (`MathChallenge`): winner's `apply_learning` loss sign == −reward;
  loser penalized; no NaN; `var_h` stays > 1e-3 (anti-collapse).
- `cargo clippy --all-targets` 0/0; `cargo test --lib` green.

---

## 4. Phase 2 — Seed-Survival Population (tournament)

**Rationale:** single Alpha/Beta is a fixed opponent → overfits one strategy.
A K-seed population with elitism + mutation gives selection pressure that
converges (OpenES/CMA-ES lesson) and avoids forgetting between games.

### 4.1 Population model
- `K` seeds (default `MUD_DEBATE_SEEDS=8`): each seed = a `Doppelganger` clone with
  its **own RNG stream** (`seed_from_u64(i)`, reuse `corpus_trainer.rs:2282` LCG)
  and an independent last-N shadow copy.
- Tournament: round-robin or Swiss, `MUD_DEBATE_ROUNDS` games/seed/cycle.
- After each cycle: rank by cumulative judge score. **Elitism:** top `E`
  (default 2) seeds survive unchanged; bottom `K-E` are **respawned from a
  mutated survivor** (`apply_gradient_jitter` on their shadow, self_play.rs).
- "Supervivencia de semilla" = only winners' weights persist into the next cycle;
  losers' mutations are discarded (no STE write for them).

### 4.2 Changes
1. `DebateArena`: add `seeds: Vec<Doppelganger>` + `rng_streams: Vec<u64>`.
   `run_tournament(game_factory, layers, ...)` loops cycles.
2. `run_debate_session` (`corpus_trainer.rs:634`): build `K` seeds, call
   `run_tournament` instead of one `run_game`.
3. `apply_learning` called **only for surviving seeds** (elitism gate). Losers'
   gradients zeroed before STE pack.
4. `MUD_DEBATE_SEEDS`, `MUD_DEBATE_ROUNDS`, `MUD_DEBATE_ELITE` env flags.

### 4.3 Acceptance (Phase 2)
- K=4 smoke: after 2 cycles, elite seed score strictly > respawning seed mean
  (selection pressure observable); no NaN; RAM bounded (last-N only).
- Determinism: same `seed_from_u64` → identical tournament result (reproducible).
- `./mud.sh debate` runs a full tournament and prints per-seed scoreboard
  (via `trainer_ui::note`, not emoji).

---

## 5. Phase 3 — Learning Integration (STE writeback + JEPA aux)

**Rationale:** reward must reach the shared `.mud` or it's decoration. Mirror the
proven F1 path: dense last-N STE pack, JEPA as aux, zero inference cost.

### 5.1 Writeback
- Survivors' `shadow_layers` → STE pack into the in-memory `.mud` tensors
  (reuse `corpus_trainer.rs::sync_shadow_to_mud` / `apply_optimizer_cpu_step_and_pack`).
- Only last-N layers mutated (RAM-safe, ~15 GiB). Clamp like mHC (`[0,4]` spirit).
- Save checkpoint after each tournament cycle (`save_checkpoint`, existing).

### 5.2 JEPA aux + anti-collapse
- `reward_intrinsic = λ_j * jepa_aux` kept through Phase 3.
- If `var_h < 1e-3` on any seed → `apply_gradient_jitter` (self_play.rs) before pack.
- Keep `MUD_DEBATE_JEPB_LAMBDA` (default 0.05).

### 5.3 Changes
1. `run_tournament` → after elite selection, pack survivors via shared
   `sync_shadow_to_mud` + checkpoint.
2. Add telemetry: per-cycle elite score, mean score, `var_h` spread (reuse
   `ActStatsAccum` pattern from corpus_trainer).
3. Gate behind `MUD_DEBATE_LEARN=1` (default OFF first, AWAKE-01 discipline).

### 5.4 Acceptance (Phase 3)
- K=4, 2 cycles, `MUD_DEBATE_LEARN=1`: shared `.mud` `cmp` DIFFERs post-tournament;
  elite score up vs cycle 1; `var_h` alive; no NaN; clippy 0/0; tests green.
- Inference cost: **0** (debate is train-only, like STP).

---

## 6. Order, risk, rollback

| Step | Deliverable | Risk | Inference cost | Rollback |
|------|-------------|------|----------------|----------|
| 1 | Verifiable judge + reward/penalty | Low | 0 | env flag / unwired trait |
| 2 | Seed-survival population | Med | 0 | single-game fallback |
| 3 | STE writeback + JEPA aux | Med | 0 | `MUD_DEBATE_LEARN` default OFF |

- Each phase = own commit + tests. Don't proceed until acceptance met.
- All new behavior behind env flags first; flip default only after proof.

## 7. What we expect to gain

- **Training signal that means something:** verifiable win/loss instead of
  VarJ-coupled noise → faster, spike-free convergence of the debate policy.
- **No forgetting between games:** elitism keeps top seeds; mutation explores.
- **Zero inference cost:** debate + reward are train-only; the deployed `.mud`
  is unchanged at inference time.
- **Aligned with MUD thesis:** a *verifiable* judge = "Modular Understanding
  Dynamics" made operational (the model must *defend* a coherent position to win).

## 8. Files to touch

| Phase | Files |
|-------|-------|
| 1 | `src/mud/arena_judge.rs` (new), `debate_trainer.rs` (`judge.score` in `run_game`), `arena_games.rs` (`DocumentDebate::winner` via TextJudge), `rlvr.rs` (reuse), `self_play.rs` (reuse jitter) |
| 2 | `debate_trainer.rs` (`DebateArena::seeds`, `run_tournament`), `corpus_trainer.rs` (`run_debate_session` builds K seeds) |
| 3 | `corpus_trainer.rs` (`sync_shadow_to_mud` reuse, checkpoint), `debate_trainer.rs` (elite pack) |

## 9. Open decisions to confirm before coding

> **DECIDED 2026-07-17: NO EXTERNAL API ANYWHERE.** All judges (`VerifiableJudge`,
> `RustJudge`, `TextJudge`) run locally in the Rust binary on the i7-1260P using
> the already-loaded `.mud` weights (embeddings) + `rustc` for code. No network,
> no LLM service, no Python (P-07). `TextJudge` = local claim-extraction +
> self-embedding cosine, deterministic.

1. ~~`TextJudge` as local claim-F1 scorer (no API)~~ — CONFIRMED local-only.
2. ~~Asymmetric penalty `R_WIN=1.0 / R_LOSE=0.7`~~ — CONFIRMED (env-tunable).
3. ~~Default `MUD_DEBATE_LEARN=0`~~ — CONFIRMED (Fase 3 writeback gated OFF).
4. **DEFERRED:** K-seed population (Fase 2). Current `run_game` keeps the single
   Alpha/Beta pair and plays infinite matches; the tournament/elitism loop is the
   next step if multi-seed survival is required. Not wired to avoid 8× shadow RAM
   on 15 GiB before proving single-pair learning converges.

## 10. Usage (as implemented)

```bash
# Infinite debate, detenable by Ctrl-C / 'q' in TUI, or MUD_DEBATE_MAX_TIME:
./mud.sh debate                      # TUI (debate_telemetry), Ctrl-C or 'q' to stop
cargo run --release --bin run_trainer -- --debate   # headless; stops on MUD_DEBATE_MAX_TIME

# Tuning env vars:
MUD_DEBATE_MAX_TIME=600              # seconds before auto-stop (infinite if unset/large)
MUD_DEBATE_RWIN=1.0                  # winner reward
MUD_DEBATE_RLOSE=0.7                 # loser penalty (asymmetric)
MUD_DEBATE_JEPB_LAMBDA=0.05          # JEPA intrinsic aux weight (anti-collapse)
MUD_DEBATE_LEARN=1                   # persist shadow→MUD after the session (default OFF)
MUD_DEBATE_MODE=professor            # Professor-Student loop (default: debate)
MUD_DEBATE_MAX_NEW_TOKENS=24         # per-response cap; auto by free RAM if unset
```

### Professor-Student mode (`MUD_DEBATE_MODE=professor`)

The arena runs an infinite **professor → student → professor-grades** loop with a
local-no-API `ProfessorJudge` (`arena_judge.rs`). Each match has 3 phases
(`arena_games.rs::ProfessorStudent`):

1. **phase 0 (professor/A):** re-states the exercise from a local pool
   (`professor_exercises()` — grammar/syntax/coherence/pragmatism, ES, no network).
2. **phase 1 (student/B):** the model answers; this text is graded.
3. **phase 2 (professor/A):** the model emits a correction; triggers `ProfessorJudge::grade`.

`ProfessorJudge::grade` returns a 4-dim rubrik in `[0,1]`
`[grammar, syntax, coherence, pragmatism]` plus a signed reward in `[-1,+1]`
(strong on grammar+syntax, moderate on coherence+pragmatism). The rubrik is logged
per match (`rubrik gram=.. syn=.. coh=.. prag=.. -> R=..`). Reward split:
**A=professor gets `0.3×`** (learns to pose+grade), **B=student gets `1.0×`**
(learns to answer well). Exercises rotate via `ex_idx` (`Arc<AtomicUsize>`) across
matches so the student keeps practicing new tasks until `quit`/Ctrl-C.

```bash
# Headless professor-student loop (auto-stops after 120s):
MUD_DEBATE_MODE=professor MUD_DEBATE_MAX_TIME=120 \
  cargo run --release --bin run_trainer -- --debate
```

### Training Circuit (`--circuit` / `./mud.sh circuit`) — seed-driven batteries

A single infinite command that rotates **training batteries** until `quit` / Ctrl-C.
Each **seed** mints a distinct **battery** = a shuffled ordering of the four phases,
so the schedule is never a fixed monotonic `align→debate→games→professor`; every
seed explores the phases in a different order. When a battery is exhausted a fresh
seed produces a new battery, keeping training non-repetitive.

Phases (all local, no-API, P-07):
- **align** — one corpus alignment epoch (STE QAT); persists to `.mud` on exit.
- **debate** — RLVR document debate (`TextJudge`).
- **games** — verifiable seed-survival games (`MathChallenge` / `TicTacToe`,
  `VerifiableJudge` on `winner()`); added via `MUD_DEBATE_MODE=games`.
- **professor** — professor→student→grade loop (`ProfessorJudge` rubrik).

Shuffling uses a tiny deterministic LCG (mulberry mix) — no external RNG crate,
P-07 friendly. The seed derives from a monotonic counter mixed with wall-clock
nanos, so each run and each new battery differs. Each phase is time-boxed by
`MUD_CIRCUIT_MAX_PER_MODE` (default 120s) so the loop never freezes on a slow
phase; `quit`/Ctrl-C stops the loop and the last phase leaves the `.mud` saved
(honors-mode persistence `MUD_DEBATE_LEARN` defaults ON in the circuit).

#### Unified telemetry + log

The circuit prints **live telemetry** on every event and appends the same lines
to **`logs/circuit.log`** (timestamped, `[HH:MM:SS] circuit <msg>`). Each
event tells you exactly what the circuit is doing:

```
[04:55:26] circuit 🌱 Semilla 753… · batería: games → professor → debate → align
[04:55:26] circuit ▶ Ciclo #1 · Semilla 753… · FASE=games · batería restante: [professor, debate, align]
[04:55:48] circuit ✓ Ciclo #1 · games completada [OK] en 21.9s · total circuito: 21.9s
[04:56:27] circuit ✓ Ciclo #2 · professor completada [OK] en 39.6s · total circuito: 61.5s
[04:56:27] circuit ▶ Ciclo #3 · Semilla 753… · FASE=debate · batería restante: [align]
```

Live telemetry shows: current **seed**, current **phase (FASE)**, the **remaining
battery**, the **phase duration** and the **cumulative circuit time** — so the
loop is never a silent black box. Per-phase metrics (loss, VarH/VarJ, rubrik)
are emitted by the underlying sessions (`run_alignment_session` /
`run_debate_session`) and likewise land in the log.

```bash
# Infinite seed-driven circuit (Ctrl-C / 'q' to stop and save):
./mud.sh circuit
# or headless:
cargo run --release --bin run_trainer -- models/smollm2.mud --circuit

# Tuning:
MUD_CIRCUIT_MAX_PER_MODE=120     # seconds per phase (time-box; never freezes)
MUD_CIRCUIT_EPOCHS=1             # alignment epochs per align phase
MUD_CIRCUIT_BATCH=16             # alignment batch size
MUD_DEBATE_LEARN=1               # persist each phase (default ON in circuit)
MUD_DEBATE_MAX_NEW_TOKENS=24     # per-response cap for debate/games/professor
MUD_CIRCUIT_EVAL=1               # honors-mode post-phase evaluation (default 1)
```

#### Honors-mode evaluation (the circuit decides the save)

After every phase the circuit runs `circuit_eval_integrity(path)` + a quality
benchmark and only then decides whether to **keep** the phase's writeback
(honors) or **roll back** to the previous `.mud` (a `.bak_circuit` snapshot is
taken before each phase). Two gates:

1. **Integrity (structural).** The `.mud` reloads and `materialize_writable()`
   is called; every STE-writable weight tensor must be present, non-empty and
   have a valid backing buffer (`data_ptr` or `owned_data`). A degenerate
   writeback (failed STE pack / dropped tensor) fails this gate. *Per-weight
   byte-skew is deliberately NOT flagged* — healthy ternary weights are naturally
   skewed toward `0x0`, so a naive "all-identical nibble" check false-flags a
   good model. The regression gate for capacity is the quality benchmark.
2. **Quality (verifiable-game win rate).** `circuit_benchmark_games` replays a
   few `MathChallenge` / `TicTacToe` matches (player A = model, `VerifiableJudge`
   on `winner()`) reusing `run_debate_session` in `games` mode over an internal
   channel. `quality_score` = model win rate vs the **baseline** captured at
   circuit start. A phase is honored only if integrity holds **and**
   `win_rate >= baseline - 0.15`.

> **CPU caveat:** on slow commodity silicon a single verifiable match
> (≈3 turns × N tokens) may exceed the phase time-box, so the benchmark can
> report `0 matches` — in that case `win_rate == baseline == 0.00` and the
> quality gate cannot regress, leaving integrity as the effective guard. The
> logic still triggers rollback the moment a phase produces either a structural
> failure or a real quality drop once matches complete.

**Inference cost:** 0. Debate + judge + reward + circuit are train-only; the
deployed `.mud` is untouched at inference unless `MUD_DEBATE_LEARN=1`.

### Implementation notes (2026-07-18) — SIGSEGV en save + panic hardening

El circuito (`--circuit`) moría con `Violación de segmento` (core dump) al
finalizar la fase `align` en `mud.save`. Dos causas y sus fixes:

1. **SIGSEGV en `MudFile::save` (root cause).** El segundo pase de escritura
   leía los datos de un tensor sin `owned_data` vía `tensor.data_ptr`
   (`src/mud/mod.rs` ~L342). Pero `ecc_verify_all` deja `data_ptr = null` en los
   tensores `.ecc`, y `materialize_writable` puede dejar `data_ptr` colgante tras
   dropear el mmap global. **Fix:** `mud.save` ahora lee desde `tensor.mmap`
   (su propio `Arc<Mmap>`, offset alineado a 32) cuando existe; si no hay
   `owned_data` ni `mmap`, escribe ceros en vez de dereferenciar null/dangling.
   Verificado: `run_alignment_session` llega a `alignment session completed.`
   sin crash.
2. **Panic "Dead RMSNorm" mata el proceso.** Si el modelo colapsa (todas las
   activaciones ≈0, `act_scale ≈ EPSILON_FLOOR/127`), `slime_forward.rs:797`
   hace `panic!` (fail-fast P-17). Ese panic ocurre **dentro de un worker thread
   del PCorePool** (el GEMV AVX2×8 corre ahí), así que `catch_unwind` en el hilo
   `main` (añadido en `run_training_circuit` para cada fase y para el baseline)
   **no lo captura** y el proceso muere. **Fix parcial:** el loop ahora envuelve
   align/debate/games/professor y el baseline benchmark en `catch_unwind`, y
   hace rollback al `.bak_circuit` si la fase falla/panic. Pero el panic en
   worker thread sigue sin capturarse — requiere captura en `PCorePool` o un
   modelo de partida sano.

> **Diagnóstico pendiente:** `models/smollm2.mud` hace forward normal sin
> panic, pero en el path de debate/juegos del circuito el primer forward da
> `Dead RMSNorm L0: act_scale=7.87e-11` (== `EPSILON_FLOOR/127`), lo que indica
> `attn_norm_w` apuntando a pesos ~0 o registros de entrada muertos en ese path.
> El checkpoint `weights/checkpoints/model_latest_checkpoint.mud` venía ya
> colapsado (`varh≈0.0129`, `cog≈0.0005` en el `circuit.log`). El circuito debe
> partir de un `.mud` sano; un modelo muerto no es entrenable. Siguiente paso:
> investigar por qué `generate_agent_response` / el baseline de juegos deja los
> registros en ~0 (¿norm weights no materializados? ¿setup de prompt?) o
> capturar panics en `PCorePool`.

### TUI del circuito (2026-07-18) — `./mud.sh circuit` despliega TUI

El comando `./mud.sh circuit` ahora lanza `tools/circuit_telemetry.rs` (TUI
ratatui/crossterm) en vez de `run_trainer --circuit` headless. El TUI:

- Spawnea `run_training_circuit(Some(tx), stop_flag)` en un thread y consume el
  `mpsc::channel` de eventos (el mismo `announce`/`sender` que ya emitía
  `run_debate_session` y el loop del circuito).
- Muestra 3 paneles verticales: header (fase/arena actual), arena
  (Jugador A / Jugador B, o **Profesor / Alumno** si detecta
  `=== INICIANDO ARENA DE JUEGO: Professor-Student`), y Juez / JEPA / event-log.
- `Ctrl-C` / `q` detiene y guarda (via `stop_flag` → el circuito hace rollback
  a `.bak_circuit` si corresponde). La telemetría sigue escribiéndose a
  `logs/circuit.log` (porque `circuit_event` escribe al archivo antes de
  devolver la línea al `announce`).

> **Limitación:** si el `.mud` de partida está colapsado (pesos = 0), el
> `panic!` de `slime_rmsnorm_i8` ocurre en un **worker thread del PCorePool**
> y mata el proceso (no capturado por `catch_unwind` en el hilo main). El TUI
> requiere un modelo sano de partida para sobrevivir; ver nota anterior.


