# Session Report — L-13 HCA / 32k-ready KV (2026-07-16)

## Problem
Full dense KV for long context is `O(layers × kv_heads × max_pos × head_dim)`.
A hard `max_pos.min(8192)` cap avoided OOM but blocked the 32k roadmap.

## Solution

### `src/mud/kv_context.rs`
| Field | Role |
|-------|------|
| `logical_max_pos` | Up to **32 768** (generation bound) |
| `dense_cap` | Ring size ≈ `hca_window + hca_ratio` |
| `hca_slots` | Compressed history (capped at 512) |
| Auto ratio | Raised so `hca_slots ≤ 512` for 32k (ratio ≥ 64) |

Env overrides: `MUD_MAX_POS`, `MUD_HCA_WINDOW`, `MUD_HCA_RATIO`.

### Workspace + forward
- Dense K/V stored in a **ring** (`pos % dense_cap`)
- HCA mean-pool still builds compressed history from the ring
- Attention: all HCA blocks + recent dense tokens (unchanged algorithm, less RAM)

### Footprint (smollm-ish 30L × 3 kv × head 64 @ 32k)
Dense ring + HCA ≪ naive full-32k KV (tested &lt; 50 MB).

## Not full CSA
Lightning top-k sparse indexer over 1M tokens is **not** implemented. L-13 is **HCA + ring** (heavily compressed attention path already in tree), memory-scaled for 32k.

## Validation
- **145** lib tests
- clippy clean
- `test_l13_32k_workspace_fits`, `kv_context` property-style tests

## Ledger
L-13 **DONE**. Remaining deferred: L-14 C-MUD, L-15 grad checkpointing.
