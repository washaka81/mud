# Session Report — L-10 Sequence Packing (2026-07-16)

## Problem

`train_on_sequence` built pairs as:

```text
tokens.windows(2).step_by(8).take(batch_size)
```

With `batch_size=16` and ~12 500-token chunks, only the **head ~1%** of each chunk contributed gradients. Fixed chunking also **dropped the stream tail** (`saturating_sub(capacity)`).

## Solution (`src/mud/sequence_pack.rs`)

| API | Role |
|-----|------|
| `split_documents` | BOS/EOS spans (EOS included as last token) |
| `pack_documents` / `pack_stream` | First-fit bins, **no pad tokens** |
| `chunk_ranges_no_pad` | Prefetch ranges including short tail chunk |
| `pairs_from_stream` / `next_token_pairs` | EOS-safe pairs + **uniform subsample over full chunk** |

## Integration

- `corpus_trainer::train_on_sequence` → `pairs_from_stream(...)`
- AOT prefetch → `chunk_ranges_no_pad` (keeps remainder; variable-length last chunk)
- Banner prints `Packing: L-10 no-pad + EOS-safe pairs`

## Guarantees

- No pad zeros inserted for short docs
- No training pair with EOS as **input** (no cross-doc leak)
- Can still train **→ EOS** as target (end-of-document)
- Pair sample covers head **and** tail of each chunk

## Validation

- `cargo test --lib` → **118 passed**
- `cargo clippy --lib -- -D warnings` → clean

## Ledger

| ID | Status |
|----|--------|
| L-10 | **DONE** |
| Next | L-11 Mini MoE / Phase B |
