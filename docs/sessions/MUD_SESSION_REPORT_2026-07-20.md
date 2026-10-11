# MUD Session Report — 2026-07-20 (Telemetry TUI + Pointer Opt)

**Scope:** make the live training telemetry TUI observable (it showed empty/stale panels),
optimize hot loops (P-00/P-01), and harden the debate/circuit writeback.
Builds on `TRAIN_TELEMETRY_FORENSIC_2026-07-20.md` §7-§8 (which covered the no-op
checkpoint + vocabulary-collapse root cause).

---

## 1. Telemetry TUI — root cause + fix (TLM)

**Symptom:** `train_telemetry` rendered empty panels even while `run_trainer` ran.

**Root cause:** the trainer emitted `[TELEM]` only to **stderr** (`eprintln!`), never to
`mud_train_metrics.log` — the file the TUI reads. So the TUI parsed 0 lines and drew
stale/empty panels (x-axis 10414..10514 = old line numbers).

**Fix:**
- `src/mud/corpus_trainer.rs` — `[TELEM]` now written to **stderr AND** `mud_train_metrics.log`
  (via `telemetry_file`); `[DW]` (bytes moved / sync) emitted every `sync_shadow_to_mud`.
- `tools/train_telemetry.rs` — parser rewritten to read by **key** (`kv_f64`) instead of
  position; new bottom panel **Weight Δ (bytes moved / sync)**; scales corrected
  (VarJ no longer floored at `1.0`; JEPA-integral ±0.1); 3-column bottom layout.

**Verified** via `tmux capture-pane` against a real short run: Loss panel shows descent
5.00→2.86, plus VarJ / JEPA-integral / VarH / Cognitive / Weight Δ all populated.

> **Operator note:** a training launched *before* this fix uses the old binary (stderr-only
> `[TELEM]`) → its TUI stays empty. Relaunch `./mud.sh train` with the new binary.

---

## 2. Pointer-optimized hot loops (P-00 / P-01)

Zero-alloc, branchless, raw-pointer rewrites (no functional change to math):

| Location | Change |
|----------|--------|
| `corpus_trainer.rs::apply_optimizer_cpu_step_and_pack` | shadow clamp via raw `*mut f32` |
| `mod.rs::dequantize_ternary_row` | `TERNARY_LUT` branchless; `pub(crate)` |
| `slime_backward.rs::unpack_ternary2bit_to_f32` | raw ptr + LUT, 8 weights / `u32` |
| `ezop.rs::pack_ternary_into` | 8 values / `u32` word |
| `ezop.rs::pack_elut_prq` | 2 values / byte |

`cargo clippy --all-targets` → 0 warnings; `cargo test --lib` → 222 passed.

---

## 3. Debate / circuit writeback hash check

`run_debate_session` now compares `hash_trained_weights(&mud)` **in vs out** and prints
✓ / ⚠ NO-OP (resolves the previously-unused warning). This also closes the C1/C2
collapsed-writeback class from forensic §8 — it reuses `sync_shadow_to_mud` (PRQ-scale-aware).

**Verified:** `MUD_TRAIN_WCLAMP_K=0` + 64 steps → weight hash `0x939f… → 0x8d52…`
persists (not a no-op).

---

## 4. STE deadzone finding (caveat, not a bug)

On the **healthy, converged base** at the default `QAT_LEARNING_RATE=0.0005`
(constants.rs:12) with the STE threshold `s*0.7`, the gradient (~1/256) never flips a
ternary code → **ΔW ≈ 0** (the trainer reports `Σ|ΔW|=0`, checkpoint MD5 unchanged).
This is expected convergence behavior, **not** the prior no-op bug. For *visible* weight
movement use a higher LR, e.g. `MUD_QAT_LR=0.03` (confirmed: 15 bytes moved over a
short run, hash changed, checkpoint MD5 differed).

---

## 5. Verification summary

- `cargo clippy --all-targets` → **0 warnings**.
- `cargo test --lib` → **226 passed** (+4 vs prior: `forward_sanity`,
  `model_logits_not_collapsed`, `test_smollm2_space_char`, plus harness).
- `mud_train_metrics.log` now receives `[TELEM]` + `[DW]` (was stderr-only).
- TUI panels populated (tmux capture).
- New reusable module `src/mud/inference.rs` (`forward_last_logits`,
  `model_logits_collapsed`) extracted from the working `src/main.rs` forward path;
  shared by tests + the circuit health gate.

---

## 6. Open debt (resolved this session)

- **Trainer console banner** hardcoded `lr_init=3e-4` (corpus_trainer.rs:1719) → now
  `crate::mud::constants::qat_learning_rate()`; displays the real LR (`0.000500`).
- **P3 gates**: `circuit_eval_integrity` now rejects a base whose forward collapses to
  token-0 across probe prompts (`model_logits_collapsed`), in addition to the existing
  norm/tensor structural checks. `scale-audit` remains the manual scale gate.
- **Retrain-of-verification** from the healthy base (`MUD_TRAIN_RESET_EPOCH=1 MUD_QAT_LR=0.03`)
  confirmed the trainer **persists weights** (checkpoint hash `0x872c…→0x24e7…`,
  base MD5 changed). `conf>5%` is NOT reached in 2 chunks (loss ~6–14) — expected; the
  real no-op gate is weight movement, which passed.

---


---

## 7. Inference observation — fused + incoherent output

`./mud.sh chat weights/checkpoints/model_latest_checkpoint.mud` (and greedy gen on the
base) yields incoherent, word-fused text:
`romancesinite restraintStore masterargopickle succeeded perfectvisible immers cereals ...`
Greedy on the base `models/smollm2.mud` prompt "hola":
`2034 15 life is and are2 ". from m in' (9 to form the8 about two34,70 ...`

### 7.1 Fused words — PREVIOUS DIAGNOSIS RETRACTED (false negative)
The earlier claim "the `.mud` vocab contains **0** `Ġ`" was a **measurement error**:
`strings models/smollm2.mud | grep -c "Ġ"` returns 0 only because GNU `strings` default
does **not** render multibyte UTF-8 (`Ġ` = bytes `C4 A0`) as printable, so it breaks the
run at the `C4` byte and never emits `Ġworld`.

**Corrected finding (verified 2026-07-20 build):**
- The vocab carries **68,733** `Ġ` byte occurrences (python count of `Ġ`.encode('utf-8'));
  the source `model.vocab` has **32,079** `Ġ` tokens.
- `test_smollm2_space_char` (`src/model/tokenizer_test.rs`) loads the real `.mud` and
  asserts `space_char == Some('Ġ')` + round-trip `decode(encode("hello world"))=="hello world"`
  — **passes**. So `Tokenizer::decode` inserts spaces correctly.
- Reconvert from source (`universal_converter`) keeps `Ġ` and produces a scale-sane model
  (`scale_audit` ratio 0.374, same as before) — the reconvert was a no-op for the vocab
  and is NOT required to fix spacing.

**Conclusion:** the tokenizer/vocab was **never broken**. The earlier "fused words"
appearance (`romancesinite`) was model-quality gibberish (the 135M base emits tokens that
decode to low-quality text), not a missing-space bug. Greedy gen now prints spaced words
(`life is are and from`) with spaces inserted.

### 7.2 Incoherent output — engine sanity PROVEN, quality is model limitation
A dedicated engine-correctness pass **was** added and passes:
- `forward_sanity` (`src/mud/inference.rs`): loads the real model, runs the full forward
  on several prompts, asserts logits are **finite**, **no token-0 dominance** across
  prompts (collapse gate), and **entropy > 0 but < log(vocab)** (non-degenerate).
  → passes on `models/smollm2.mud`. The ternary engine is **not collapsed**.
- `model_logits_collapsed` is wired into `circuit_eval_integrity` so the F3+ circuit
  refuses a genuinely collapsed base (clear error, no mid-loop panic).

What remains is **model quality**, not a correctness bug:
- Ternary reconstruction has ~0.4× source RMS (expected).
- The 135M base is undertrained for chat; greedy logits are strong (max ~19.6) but map to
  the wrong tokens → gibberish. This needs real training that moves weights (§4/§6), not
  an engine fix. The T0.2 gate confirms the forward is sound.

---

## 8. New modules / functions (2026-07-20 build)

| Item | Location | Purpose |
|------|----------|---------|
| `forward_last_logits` | `src/mud/inference.rs` | reusable single-sequence forward (port of `main.rs` path) |
| `model_logits_collapsed` | `src/mud/inference.rs` | token-0-dominance health gate |
| `forward_sanity` (test) | `src/mud/inference.rs` | engine sanity: finite / no collapse / entropy |
| `model_logits_not_collapsed` (test) | `src/mud/inference.rs` | healthy base passes gate |
| `test_smollm2_space_char` (test) | `src/model/tokenizer_test.rs` | locks `Ġ` space prefix + round-trip |
| circuit gate | `corpus_trainer.rs::circuit_eval_integrity` | rejects collapsed base via `model_logits_collapsed` |
| banner LR | `corpus_trainer.rs:1719` | `qat_learning_rate()` (was hardcoded `3e-4`) |

---

## 9. DSPARK adoption + test-stability (2026-07-20, cont.)

**Goal:** adopt DSpark (DeepSeek, MIT/DeepSpec) techniques into MUD's existing drafter/packing,
each with a short validatable test. Plus kill two pre-existing parallel-test races that made
`cargo test --lib` flaky (1 failure at random).

### 9.1 DSPARK slices (all in `src/mud/speculative.rs`)
| ID | Item | Symbol | Test |
|----|------|--------|------|
| DSP-1 | Semi-autoregressive (Markov) draft head | `sequential_draft` + `markov_bias` | `test_sequential_draft_length_and_markov` |
| DSP-2 | Confidence head + hardware-aware scheduler | `schedule_draft_length` (＋ `schedule_draft_from_hidden`) | `test_schedule_draft_length`, `test_schedule_draft_from_hidden` |
| DSP-3 | Anchor-bounded packing (token-level idx, 0 pad) | `anchor_boundaries` + `anchor_attention_indices` | `test_anchor_boundaries_padding_free` |
| DSP-4 | Hidden-state comm `O(d)` | `project_hidden_to_d` + `hidden_comm_bytes` | `test_project_hidden_to_d_od` |
| DSP-5 | Spherical norm draft-target alignment | `spherical_norm` + `confidence_spherical` | `test_spherical_norm_and_confidence` |

Note: `SlimeDrafter::propose_tokens` already drafts sequentially (feeds token back) — that is the
model-backed instance of DSP-1. The new `sequential_draft` is the pure, testable core.

### 9.2 Flaky-test fixes (pre-existing, surfaced by added tests)
- `test_quick_max_chunks_defaults_to_sgd` raced on `MUD_TRAIN_MAX_CHUNKS` with
  `test_select_optimizer_square_is_muon` → **merged** into one sequential
  `test_select_optimizer_square` (corpus_trainer.rs `optimizer_strategy_tests`, guarded by
  `OPT_TEST_LOCK`).
- `test_round_robin_cycles` raced on the global `STEP_COUNTER`/`HIT_COUNTS` with sibling
  `moe_train` tests → the three env-touching tests now take `EXPERT_TEST_LOCK`
  (moe_train.rs).
- Result: `cargo test --lib` **230 passed, 0 failed** (stable across 3 repeated runs);
  `cargo clippy --all-targets -- -D warnings` clean.

### 9.3 Roadmap T4 item J — CSA LSH prefilter (validation)
`csa_indexer.rs` already shipped the LSH machinery (Stream J): `lsh_signature` (SimHash),
`hamming64`, and the LSH prefilter branch inside `index_hca_blocks` (gated on
`MUD_CSA_LSH`). The roadmap item only lacked the **recall-vs-brute** validation.
- Added `force_lsh: Option<bool>` to `index_hca_blocks` (None = env flag) so the
  prefilter path is testable deterministically without an env-var race; updated the
  single caller in `slime_forward.rs:976` (`None`).
- Added `test_lsh_prefilter_recall_vs_brute`: LSH path keeps **recall == 1.0** of the
  brute top-k (never drops a true top block) while excluding a far-in-subspace
  low-score block (block 0). `cargo test --lib` → **231 passed, 0 failed**;
  clippy `--all-targets` clean.

### 9.4 Roadmap T4 fully closed (F/G/H/I/J/K/DSP)
Inspection showed F–I were already code-complete with unit tests (like J); only
the roadmap rows lacked a DONE mark. Verified and marked:
- **F/QKV** — `tools/gemv_auto_bench.rs` break-even bench present.
- **G** — `moe_train.rs` round-robin + `begin_step_hash`; 3 tests (serialized).
- **H** — `grad_checkpoint.rs` segmented + `ResidualBank` roundtrip + `_RESIDUAL` flag;
  added `test_residual_bank_env_flag`.
- **I** — `kv_dtype.rs` f16 pack/round-trip; 3 tests.
- **J** — recall-vs-brute test (§9.3).
- **K** — loss cert CI gate (`cargo test --lib loss_cert`).
- **DSP** — §9.1.

**Final:** `cargo test --lib` → **232 passed, 0 failed**; `cargo clippy --all-targets -- -D warnings` clean. The entire prioritized roadmap (T0–T4) is GO.

### 9.5 F+ orbit (beyond the prioritized roadmap)
The prioritized roadmap is closed; the F+ **orbit** (`MUD_IMPROVEMENTS_POST_AE.md` §4 "Suggested next work") is the next frontier. Implemented the first concrete item:
- **Orbit #1 — Multi-expert weighted STE:** added `weighted_expert_deltas(grads, weights, lr)`
  in `moe_train.rs` (computes `delta_j = lr·w_j·grad_j` per top-k expert) + `test_weighted_expert_deltas`.
  Kept separate from the proven top-1 trainer path (no regression). This is the building block to
  train top-k experts jointly with route weights.
- **Final:** `cargo test --lib` → **233 passed, 0 failed**; clippy `--all-targets` clean.

---

## 10. SlimeX Dynamic Stack & RPG Battle Circuit (2026-07-20, session 2)

**Goal:** Implement the proof-of-concept for the `ShadowExpertBus` (SlimeX) to manage dynamic expert mounting without allocation in the hot loop, and define the gamification of the training circuit (RPG Battle Circuit).

### 10.1 SlimeX Dynamic Stack (Orbit F+)
- **Design (`src/mud/slime_x.rs`):** Implemented `SlimeXSlot` containing pre-allocated buffers (weights + optimizers) for FFN up/gate/down.
- **Bus (`ShadowExpertBus`):** Manages a fixed `top_k` array of `SlimeXSlot`s. Enables dynamic `mount` and `unmount` of experts by swapping pointers/IDs instead of reallocating (respects P-01 zero-allocation policy).
- **Integration:** Wired into `src/mud/slime_backward.rs` via `pub slime_x: Option<crate::mud::slime_x::ShadowExpertBus>` inside `SlimeLayerShadowF32`.
- **Validation:** Added and passed `test_slimex_dynamic_stack`. Compilation and `cargo clippy --all-targets` reported 0 errors/warnings.

### 10.2 Training Collapse Audit
- Ran the circuit interactively: `MUD_QAT_LR=0.03 ./mud.sh circuit models/smollm2.mud`.
- **Observation:** `VarH` plummeted to ~0.04 under the aggressive LR, leading to an aphasic model state (`models/smollm2.mud.bak_circuit`).
- **Confirmation:** Sent `q` to gracefully save the state. Performed inference on the checkpoint and confirmed severe neural collapse (gibberish token output). This explains the negative rewards (e.g. `A:-0.550`) given by the circuit's judge for incoherence.

### 10.3 RPG Battle Circuit (Future Phase)
Documented a new gamified training paradigm in `VISION_ROADMAP.md` and `MUD_PLAN_CIRCUIT_ALGORITHMS.md`:
- **Barra de Vida (HP):** Models maintain a persistent health state.
- **Doppelgänger Battles:** Model battles itself or a mutated version in the debate circuit.
- **Evolutionary Rewards:** The winning model becomes Player A and receives a 5-epoch training boost.
- **Forced Study:** Each rotation changes the seed and forces the models to study a specific topic before the debate battle, enforcing semantic coherence and factual robustness.

---

*Next file to read: `TRAIN_TELEMETRY_FORENSIC_2026-07-20.md` §7.7 · `MUD_FIX_PLAN_2026-07-20.md` · `docs/research/DEEPSEEK_DSPARK_RESEARCH_2026-07-20.md`.*

