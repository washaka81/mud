# MUD AUDIT REPORT V13: TOKENIZER APHASIA & BPE GHOST MERGES

**Date:** 2026-06-12
**Status:** RESOLVED
**Author:** Forge LLM Architecture Team (MUD)

## 1. The Anomaly (The "Aphasic" State)
For two months, the 1.58-bit Ternary BitNet model deployed in MUD suffered from a severe linguistic collapse, commonly referred to as "Aphasia" or "Ternary Shock". Despite mathematical audits confirming that the weights, scales, and variances perfectly conformed to mathematical invariants (0% fractional deviation, SQNR > 10.5 dB, negative HiPPO eigenvalues), the model generated complete gibberish (e.g., `"deprecatedelinesunktiliate Route measure revert Datestoodstood"`).

When given the prompt `"hola"`, the model correctly mapped the input, but its generation quickly collapsed. 

## 2. Investigation and Root Cause
The root cause of this catastrophic failure was **NOT** within the neural network's forward pass, nor was it related to ternary quantization. It was a silent, critical bug deeply embedded in the `encode_bpe` logic of `Tokenizer` (introduced during the `O(N log N)` BinaryHeap optimization in commit `c456438`).

### The Ghost Merge Bug
The `BinaryHeap` tokenizer queue prioritized BPE merge rules by their rank (lower is better). However, when two adjacent parts merged (e.g., `"U"` + `"D"` -> `"UD"`), the `BinaryHeap` logic successfully updated the `part.text` but failed to invalidate older, queued merge rules that involved the previous strings.

When a previously queued pair involving the old parts popped from the heap (e.g., `"M"` + `"U"`), it bypassed the `self.merges` dictionary check entirely and blindly concatenated the updated part, resulting in a **Ghost Token** (`"MUD"`). 

Because `"MUD"` did not exist in the official `vocab`, the token ID lookup failed (`self.vocab.get("MUD") == None`). 

### The Silent Drop
The final step of the tokenizer logic checked if the token list was empty to trigger a character-level fallback. Because `"MUD"` was part of a larger sequence (e.g., `"MUD engine optimized"`), the overall token list was **not** empty (it successfully tokenized `" engine optimized"`). This caused the fallback to skip, silently dropping the `MUD` token entirely without any warning or error.

This bug meant the tokenizer was constantly dropping key tokens from both the user's prompt and the model's autoregressive generation. The neural state became completely desynchronized and corrupt, leading to "Aphasia".

## 3. The Fix
1. **Merge Validation:** We introduced a strict `self.merges.get(&current_pair) == Some(&rank)` validation check immediately after `heap.pop()`. This ensures that if a part has mutated, any stale merge pairs are immediately rejected.
2. **Robust Fallback:** The loop that collects token IDs now includes an `else` branch. If a text chunk somehow survives without being in the `vocab`, it gracefully splits back into its constituent characters instead of being silently dropped.
3. **Re-Integration:** `encode_simple` and `decode_simple` were restored to support the API expected by `mud_corpus_trainer.rs` and `sampling.rs`.

## 4. Conclusion and Next Steps
The "Aphasic" state was a pure I/O corruption issue, not a fundamental flaw in the 1.58-bit ternary architecture. With the tokenizer fully stabilized, the model's mathematical fidelity is expected to translate directly into coherent output. 

We have successfully rebuilt the model (`bitnet-b1.58-2B-4T.mud`) and initiated a deep QAT recovery loop (`restore-iq 50 epochs`) to reseat the latent space and re-align the model.
