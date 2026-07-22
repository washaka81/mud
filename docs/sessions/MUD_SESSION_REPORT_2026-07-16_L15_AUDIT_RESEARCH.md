# Session — L-15 + audit tools + gap research (2026-07-16)

## L-15 Gradient checkpointing
- Module: `src/mud/grad_checkpoint.rs`
- Env: `MUD_GRAD_CKPT=1`, `MUD_GRAD_CKPT_SEG=N` (default 4)
- Forward: fill tapes then **discard** if segmented
- Backward: `recompute_from_embedding` for needed prefix, then `backward_slime_block`, discard tape
- `SlimeLayerTape.valid` flag
- Wired in `corpus_trainer::train_on_sequence`

## Audit tooling
| Tool | Command |
|------|---------|
| Full ledger audit | `./mud.sh audit-full [model]` → `mud_full_audit` |
| Health preflight | `./mud.sh health` |
| CI battery | `./mud.sh ci` (+ GHA runs `mud_full_audit`) |

`mud_full_audit` checks: P-13 arch, tensors, L-13 KV policy, L-15 ckpt estimates, optimizer shapes, feature flags, C-MUD algebra, packing smoke.

## Research
`docs/research/MUD_GAP_ANALYSIS_POST_L15.md` — P0–P3 backlog after ledger close.

## Validation
- 159+ lib tests
- `mud_full_audit models/smollm2.mud` → CERTIFIED
