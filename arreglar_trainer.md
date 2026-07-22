 The user wants me to review the trainer. Let me start by looking at the training...
  Voy a revisar el entrenador (trainer) del proyecto. Según AGENTS.md, los puntos de entrada son:

  • tools/run_trainer.rs — entrada del trainer
  • tools/warp_aligner.rs — TUI
  • src/mud/corpus_trainer.rs — core del entrenamiento

  Voy a lanzar subagentes en paralelo para cubrir todo el código relevante.

● Agent(research: Core trainer reviewer)(Review the core trainer file at /home/ale/proyectos/forge_llm/src/mud/corpus_tra...)

  Lancé 3 subagentes de investigación en paralelo:

  1. Core trainer reviewer — revisa corpus_trainer.rs a fondo
  2. Trainer binary reviewer — revisa run_trainer.rs y warp_aligner.rs
  3. Optimizer/support reviewer — revisa optimizadores, STE, gradientes, corpus pipeline

  Esperando resultados...

○ Schedule(300s: Check on trainer review subagents progress) (ctrl+o to expand)

  Esperando los reportes de los subagentes...

  [From subagent a7fcd4f0-2b9e-4c42-998d-3b0cdce022ea (Optimizer and training support reviewer)]

  Here is my comprehensive report on the optimizer and training support modules in the Forge LLM project.
  ──────
  ## 1. Module Inventory (src/mud/)

  The src/mud/ directory contains 26 modules:

   Module                                                                               │ Role
  ──────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────
   adam_state.rs                                                                        │ Adam/SparseAdam moment storage
   cmud.rs                                                                              │ C-MUD complex manifold
   constants.rs                                                                         │ SSOT constants (EPSILON_FLOOR, etc.)
   corpus_cache.rs                                                                      │ AOT corpus cache
   corpus_trainer.rs                                                                    │ Core training loop
   csa_indexer.rs                                                                       │ CSA top-k lightning indexer
   expert_bus.rs                                                                        │ MoE ExpertBus mount/unmount
   ezop.rs                                                                              │ EZOP raw-pointer pass (TLS grad scratch)
   gemv_policy.rs                                                                       │ GEMV auto CPU/GPU dispatch
   grad_checkpoint.rs                                                                   │ Gradient checkpointing (L-15)
   moe_load.rs                                                                          │ MoE .mud loader
   muon.rs                                                                              │ Muon optimizer (Newton-Schulz)
   optimizer.rs                                                                         │ select_optimizer + apply_optimizer_cpu_step_and_pack
   p13.rs                                                                               │ P-13 property tests
   sequence_pack.rs                                                                     │ Sequence packing (L-10)
   slime_expert.rs                                                                      │ Dense expert FFN
   slime_jepa.rs                                                                        │ JEPA OU tracker + mHC
   slime_workspace.rs                                                                   │ SlimeWorkspace pre-allocation
   telemetry.rs                                                                         │ Training telemetry
   Others                                                                               │ Forward/model/tokenizer support
  ──────
  ## 2. Optimizer Module (optimizer.rs, ~450 lines)

  ### select_optimizer

  • Shape dispatch (lines 42-85): Correctly routes based on matrix aspect ratio:
      • Square-ish → Muon
      • Tall (rows ≫ cols) → GaLore
      • Wide (cols ≫ rows) → ChunkedAdam
      • Huge embedding → SparseAdam
      • Else → Adam
  • ✅ No hardcoded dims — uses row/col ratio thresholds.

  • Returns an OptimizerStrategy enum.
  ### apply_optimizer_cpu_step_and_pack

  • Central dispatch: matches on OptimizerStrategy, calls the appropriate step function, then packs weights back to ELUT ternary via pack_f32_to_ternary2bit.
  • ✅ STE integration: The unpacked f32 shadow is updated, then re-quantized.
  • ✅ Gradient clamping (P-14): clamp_grad_norm called before optimizer step.
  • ✅ PRQ scale update: update_prq_scale after packing.

  ### Issues Found:

  1. Potential allocation in galore_step (line ~210): Creates a temporary Vec<f32> for the projected gradient. This is inside the training loop. Severity: Medium — could be
  pre-allocated in trainer buffers.
  2. chunked_adam_step (line ~280): Allocates chunk buffers per call. Same concern as above. Severity: Medium.
  3. Both are mitigated by the fact that these are per-parameter-tensor operations (not per-token), so the allocation frequency is lower than GEMV hot paths, but still
  violates the spirit of P-01.
  ──────
  ## 3. Adam State (adam_state.rs, ~320 lines)

  ### Structure

  • AdamState stores first/second moment vectors (m, v) as Vec<f32>, plus t (timestep).
  • SparseAdamState extends this with a touched bitset for zero-row skip.
  • adam_step_avx2: AVX2-accelerated Adam update.

  ### Findings:

  • ✅ Pre-allocated moments: allocated once when AdamState::new() is called, reused across steps.
  • ✅ AVX2 path with scalar fallback for remainder elements.
  • ✅ Finite check (line ~145): if !grad.is_finite() { continue; } — skips NaN/Inf gradients per element.
  • ✅ SparseAdam zero-row skip: Only updates rows that have non-zero gradients.
  • ⚠️ Minor: adam_step_avx2 uses _mm256_set1_ps for constants inside the loop body (line ~168). These could be hoisted outside the loop for marginal speedup. Severity: Low —
  compiler likely hoists them anyway.
  ──────
  ## 4. Muon Optimizer (muon.rs, ~280 lines)

  ### Structure

  • muon_preprocess: Applies Newton-Schulz orthogonalization to the gradient.
  • GPU path via AshContext::dispatch_newton_schulz_sync when MUD_USE_VULKAN=1.
  • CPU fallback with 5 NS iterations.

  ### Findings:

  • ✅ GPU/CPU dispatch works correctly.
  • ✅ NS iteration count = 5 (hardcoded but documented as intentional).
  • ⚠️ CPU NS path (line ~95): Allocates temporary matrices for X, Y, Z in each call. These are matrix-sized (rows×cols), so pre-allocation is harder but could use workspace
  scratch. Severity: Medium for large matrices.
  • ✅ P-27 compliance: No Rayon usage.
  ──────
  ## 5. EZOP (ezop.rs, ~400 lines)

  ### Structure

  • TLS (thread-local) gradient scratch buffers.
  • Zero-alloc backward pass using pre-allocated scratch.
  • Raw pointer arithmetic for hot-path gradient accumulation.

  ### Findings:

  • ✅ P-00 compliance: Extensive raw pointer usage with // SAFETY: comments.
  • ✅ P-01 compliance: TLS scratch is allocated once per thread, reused.
  • ✅ Zero-alloc backward: Verified — no allocations in the backward pass hot loop.
  • ✅ pack_ternary_ste: Straight-Through Estimator correctly passes gradients through the quantization floor.
  ──────
  ## 6. Gradient Checkpointing (grad_checkpoint.rs, ~200 lines)

  ### Structure

  • Implements MUD_GRAD_CKPT=1 recompute-on-reverse strategy.
  • Segments the model into checkpoint segments.
  • On backward pass, recomputes forward activations from the last checkpoint.

  ### Findings:

  • ✅ Correctness: Segment boundaries correctly stored and restored.
  • ✅ Memory savings: Only stores activations at segment boundaries.
  • ✅ Residual bank (MUD_GRAD_CKPT_RESIDUAL=1): Stores/restores residual stream at boundaries.
  • ⚠️ Segment count (line ~45): Default segment count is num_layers / 4, minimum 2. This is reasonable but could be tunable via env var. Currently not exposed. Severity: Low.
  ──────
  ## 7. Sequence Packing (sequence_pack.rs, ~250 lines)

  ### Findings:

  • ✅ No padding: Sequences packed end-to-end with EOS boundaries tracked.
  • ✅ No cross-EOS attention: Attention mask correctly blocks cross-sequence attention.
  • ✅ Full-chunk pairs: Tail sequences kept (not discarded).
  • ✅ P-01: Packing buffers pre-allocated.
  ──────
  ## 8. Corpus Cache (corpus_cache.rs, ~180 lines)

  ### Findings:

  • ✅ AOT binary cache: Tokenized sequences cached to disk as .bin files.
  • ✅ Cache invalidation: Checks file modification time.
  • ✅ P-17 compliance: Errors on corrupt cache files rather than silently using bad data.
  ──────
  ## 9. Sampled Softmax

  Found in corpus_trainer.rs (reviewed by other subagent), but the loss function implementation has support code in a few places:

  • sampled_softmax_loss function handles target + hard/random negatives.
  • Negative sampling: ~512 negatives (configurable via env var).
  • ✅ NCE-8 confirmed abandoned — no NCE references found in active code.
  ──────
  ## 10. Training Data Pipeline Integration

  The data flow is:

  1. Corpus files (training/corpus/) → tokenized via BPE
  2. AOT cache (corpus_cache.rs) → binary token sequences
  3. Sequence packing (sequence_pack.rs) → packed batches
  4. Forward pass → SlimeWorkspace activations
  5. Loss → Sampled Softmax
  6. Backward → EZOP zero-alloc gradients
  7. Optimizer → select_optimizer → strategy-specific step
  8. Re-pack → pack_f32_to_ternary2bit + PRQ update
  ──────
  ## Summary of Issues

   Severity                                              │ File                                                 │ Issue
  ───────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────
   Medium                                                │ optimizer.rs                                         │ GaLore step allocates temporary projected gradient vec
   Medium                                                │ optimizer.rs                                         │ ChunkedAdam allocates chunk buffers per call
   Medium                                                │ muon.rs                                              │ CPU Newton-Schulz allocates temporary matrices per call
   Low                                                   │ adam_state.rs                                        │ AVX2 constants could be hoisted (compiler likely does this)
   Low                                                   │ grad_checkpoint.rs                                   │ Segment count not env-var tunable

  All issues are non-critical — the allocations in optimizer.rs and muon.rs are per-parameter-tensor (not per-token), so they don't violate P-01 at the GEMV/forward level.
  However, for very large models or tight training loops, pre-allocating these buffers would be beneficial.

  No policy violations found. The codebase is well-structured with proper P-00/P-01/P-13/P-14/P-17/P-27 compliance throughout.

  [From subagent 76664215-da67-440d-b60a-03a09eba6561 (Trainer binary reviewer)]

  Here is my detailed report on the trainer binary (run_trainer.rs) and TUI (warp_aligner.rs).
  ──────
  ## 1. Cargo.toml Binary Entries

  Found in Cargo.toml:

    [[bin]]
    name = "run_trainer"
    path = "tools/run_trainer.rs"

    [[bin]]
    name = "warp_aligner"
    path = "tools/warp_aligner.rs"

  Both properly registered. mud.sh has corresponding entries.
  ──────
  ## 2. run_trainer.rs (~680 lines)

  ### Architecture

  The file serves as the CLI entry point for training. Structure:

  1. Argument parsing (lines 1-80): Processes CLI args and env vars
  2. Model loading (lines 82-180): Loads .mud file, validates metadata
  3. Corpus loading (lines 182-260): Loads and tokenizes corpus, AOT cache
  4. Workspace setup (lines 262-340): Creates SlimeWorkspace, pre-allocates buffers
  5. Training loop (lines 342-580): Main epoch/step loop
  6. Checkpoint saving (lines 582-650): Periodic + final checkpoint
  7. Cleanup (lines 652-680): GPU teardown

  ### Environment Variables Handled

   Var                                                                                  │ Purpose
  ──────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────
   MUD_TRAIN_FULL_SEQ                                                                   │ Full-sequence training mode
   MUD_TRAIN_SEQ_LEN                                                                    │ Sequence length for full-seq
   MUD_TRAIN_EXPERT                                                                     │ Dense expert index for MoE
   MUD_MOE_TRAIN                                                                        │ MoE training mode (1|hash)
   MUD_GRAD_CKPT                                                                        │ Gradient checkpointing
   MUD_GRAD_CKPT_RESIDUAL                                                               │ Residual bank for grad ckpt
   MUD_GPU_GEMV                                                                         │ GPU GEMV mode
   MUD_USE_VULKAN                                                                       │ Vulkan backend
   MUD_KV_DTYPE                                                                         │ KV cache dtype
   MUD_PCORE_THREADS                                                                    │ PCorePool thread count
   MUD_CSA_LSH                                                                          │ CSA LSH prefilter
   MUD_CMUD_THINK                                                                       │ C-MUD complex manifold
   MUD_LR                                                                               │ Learning rate override
   MUD_EPOCHS                                                                           │ Epoch count
   MUD_BATCH_SIZE                                                                       │ Batch size
   MUD_CHECKPOINT_INTERVAL                                                              │ Steps between checkpoints

  ### Findings

  #### ✅ Compliant

  • P-13: Model dimensions (hidden_size, num_layers, num_heads, vocab_size, ffn_dim) all extracted from metadata. Panics if missing.
  • P-17: Fail-fast on missing metadata, corrupt model files, invalid env vars.
  • P-01: SlimeWorkspace created once before training loop. Trainer buffers (gradient scratch, logits buffer, target buffer) pre-allocated.
  • P-27: No Rayon imports or usage. Uses PCorePool.
  • P-08: No obvious dead code. All functions are called.
  • Checkpoint handling: Saves to weights/checkpoints/model_latest_checkpoint.mud with atomic rename (write to .tmp, then rename). Also saves numbered checkpoints at
  intervals.

  #### ⚠️ Issues Found

  1. Hardcoded learning rate default (line ~48):
    let lr: f32 = env_or("MUD_LR", 1e-4);
    This is a reasonable default, not a P-13 violation (it's a hyperparameter, not a model dimension). Severity: None — acceptable.
  2. Hardcoded max_gen (line ~55):
    let max_gen: usize = 512;
    Used for validation generation length. Mentioned in AGENTS.md §P-13 soft debt. Not a blocking issue but could be env-var configurable. Severity: Low.
  3. Checkpoint directory creation (line ~590):
    std::fs::create_dir_all("weights/checkpoints")?;
    Uses relative path. If the binary is run from a different working directory, checkpoints go to the wrong place. Severity: Low — mud.sh always sets CWD correctly.
  4. Validation generation (lines ~540-570): After each epoch, generates sample text for quality check. This allocates a String for the output. Severity: None — this is
  outside the training hot loop and is acceptable.
  5. GPU teardown (line ~660):
    if let Some(ash) = ash_ctx.take() {
        ash.destroy();
    }
    Properly destroys GPU context. The take() ensures it's only destroyed once. ✅
  ──────
  ## 3. warp_aligner.rs (~920 lines)

  ### Architecture

  Full TUI (terminal UI) for training visualization. Uses crossterm for terminal rendering.

  Sections:

  1. TUI state (lines 1-120): State struct with training metrics, layout info
  2. Rendering (lines 122-480): Terminal drawing functions
  3. Input handling (lines 482-560): Keyboard shortcuts
  4. Telemetry integration (lines 562-700): Receives telemetry from trainer
  5. Main loop (lines 702-920): Event loop + render cycle

  ### Display Panels

  • Loss curve: ASCII loss graph with rolling average
  • Gradient stats: Norm, max, min, finite percentage
  • JEPA diagnostics: VarH, VarJ, gate mean/std
  • Optimizer info: Strategy per layer, step count
  • MoE stats: Expert utilization (when MoE active)
  • Throughput: Tokens/sec, steps/sec
  • Memory: Workspace size, GPU memory (if Vulkan)
  • Checkpoint status: Last save time, path

  ### Findings

  #### ✅ Compliant

  • No allocations in render loop — all display buffers pre-allocated in state init.
  • P-13: Reads model info from telemetry, doesn't hardcode dims.
  • P-08: All render functions are called. No dead panels.
  • P-27: No Rayon. Event loop is single-threaded with non-blocking input.

  #### ⚠️ Issues Found

  1. unwrap_or(0) for display (multiple lines):
    let varh = telemetry.varh.unwrap_or(0.0);
    This is mentioned in AGENTS.md as acceptable P-13 soft debt for display-only values. Severity: None — documented acceptable behavior.
  2. Terminal size handling (line ~135):
    let (cols, rows) = crossterm::terminal::size().unwrap_or((80, 24));
    Falls back to 80x24 if terminal size can't be determined. Reasonable. Severity: None.
  3. Loss history buffer (line ~85):
    loss_history: Vec<f32>,  // grows unboundedly
    The loss history vector grows with each step. For very long training runs (millions of steps), this could consume significant memory. Should be capped with a ring buffer.
  Severity: Medium — could OOM on very long runs.
  4. Refresh rate (line ~715):
    let refresh_ms = 100; // 10 FPS
    Hardcoded. Not a problem, but could be configurable. Severity: None.
  ──────
  ## 4. Integration Between Components

    run_trainer.rs
        ├── Loads model (.mud)
        ├── Creates SlimeWorkspace
        ├── Creates PCorePool
        ├── Loads corpus (corpus_cache)
        ├── Packs sequences (sequence_pack)
        ├── Initializes telemetry channel
        │
        ├── Training loop:
        │   ├── corpus_trainer::train_step()
        │   │   ├── Forward pass (evaluate_slime_block)
        │   │   ├── Sampled Softmax loss
        │   │   ├── Backward pass (EZOP)
        │   │   ├── select_optimizer → strategy step
        │   │   └── Re-pack weights
        │   ├── Updates telemetry
        │   └── Periodic checkpoint save
        │
        └── warp_aligner (optional, via --tui flag)
            ├── Receives telemetry
            └── Renders TUI panels

  The integration is clean. run_trainer.rs orchestrates, corpus_trainer.rs does the heavy lifting, warp_aligner.rs is purely display.
  ──────
  ## Summary

   Severity                                                │ File                                                    │ Issue
  ─────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────
   Medium                                                  │ warp_aligner.rs                                         │ loss_history Vec grows unboundedly — use ring buffer
   Low                                                     │ run_trainer.rs                                          │ max_gen hardcoded (documented soft debt)
   Low                                                     │ run_trainer.rs                                          │ Checkpoint directory uses relative path

  Overall assessment: Both files are well-structured, policy-compliant, and properly integrated. The only actionable issue is the unbounded loss_history in the TUI.

  [From subagent a4ed8ac9-9cfc-435e-9147-0bf1bb3cd158 (Core trainer reviewer)]

  Here is my comprehensive analysis of corpus_trainer.rs and related training modules.
  ──────
  ## 1. Overall Architecture (corpus_trainer.rs, ~1,450 lines)

  The file is the core training engine. Key functions:

   Function                                              │ Lines                                                │ Purpose
  ───────────────────────────────────────────────────────┼──────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────
   train_step                                            │ ~120-380                                             │ Single training step: forward → loss → backward → optimizer
   forward_train                                         │ ~382-520                                             │ Forward pass with optional grad checkpointing
   sampled_softmax_loss                                  │ ~522-680                                             │ Loss computation with hard/random negatives
   backward_pass                                         │ ~682-900                                             │ STE backward through all blocks
   apply_gradients                                       │ ~902-1050                                            │ Per-tensor optimizer dispatch + re-pack
   prepare_batch                                         │ ~1052-1150                                           │ Batch preparation from packed sequences
   build_causal_mask                                     │ ~1152-1200                                           │ Causal attention mask for full-seq mode
   evaluate_train_block                                  │ ~1200-1350                                           │ Single block forward with JEPA/mHC (training variant)
   collect_negatives                                     │ ~1352-1420                                           │ Hard + random negative sampling
   init_trainer_buffers                                  │ ~1422-1450                                           │ Pre-allocates all training buffers

  ### Data Flow

    prepare_batch → forward_train → sampled_softmax_loss → backward_pass → apply_gradients
         ↓              ↓                    ↓                    ↓              ↓
      packed seqs   block-by-block      logits + neg         STE grads      optimizer
      + positions   JEPA/mHC/MoE       sampling              EZOP          + repack
    ──────
  ## 2. Policy Compliance Analysis

  ### P-00 (Raw Pointer Mastery) ✅

  • backward_pass uses raw pointers for gradient accumulation (lines ~700-750):
    let grad_ptr = grad_buf.as_mut_ptr();
    // SAFETY: grad_buf pre-allocated to hidden_size, index checked at allocation
    unsafe { *grad_ptr.add(i) += delta; }

  • forward_train uses raw pointers for GEMV dispatch.
  • All unsafe blocks have // SAFETY: comments.

  ### P-01 (Zero Allocation) ✅ with minor exceptions

  • init_trainer_buffers (line ~1422) pre-allocates:
      • grad_scratch: Vec<f32> (largest tensor size)
      • logits_buf: Vec<f32> (vocab_size)
      • neg_logits: Vec<f32> (num_negatives)
      • target_buf: Vec<f32> (hidden_size)
      • block_grads: Vec<Vec<f32>> (per-block gradient buffers)
      • ffn_scratch: Vec<f32> (ffn_dim * 3)
  • These are reused every step via fill(0.0) / copy_from_slice.
  • Exception: collect_negatives (line ~1370) creates a HashSet<usize> per call for deduplication of negative indices. This is ~512-entry set, allocated per step. Severity:
  Medium — could use a pre-allocated bitset.

  ### P-02 (f32 matmul_accum) ✅

  • All accumulations are f32. No i16 or f16 accumulation found.
  • SlimeWorkspace registers confirmed f32.

  ### P-05 (JEPA + mHC every block) ✅

  • evaluate_train_block (line ~1220):
    // JEPA OU update
    slime_jepa::update_jepa_z(&mut workspace.jepa_z, block_idx, &y_norm, hidden_size);
    // mHC residual blend
    let gate = slime_jepa::compute_jepa_gate(&workspace.jepa_z, block_idx, hidden_size);
    slime_jepa::apply_mhc_blend(y, residual, gate, hidden_size);

  • Called for every block in the forward pass. ✅

  ### P-06 (clippy clean) — No obvious violations

  • Code style is consistent. No unwrap() on fallible operations in hot paths (uses ? or explicit error handling).

  ### P-08 (Dead code) ✅

  • No commented-out functions. All functions are called from the training pipeline.
  • One minor observation: there's a #[allow(unused)] on a debug helper function dump_gradient_stats (line ~1048) that is conditionally compiled with #[cfg(debug_assertions)].
  This is acceptable.

  ### P-13 (No hardcoded dims) ✅

  • All dimensions sourced from ModelMetadata:
    let hidden = meta.hidden_size;
    let num_layers = meta.num_layers;
    let vocab_size = meta.vocab_size;
    let ffn_dim = meta.ffn_dim;
    let num_heads = meta.num_heads;
    let head_dim = hidden / num_heads;

  • Panics if metadata is missing (line ~130):
    let hidden = meta.hidden_size.expect("P-13: hidden_size must be in metadata");


  ### P-14 (Gradient finite + clamp) ✅

  • apply_gradients (line ~920):
    // Finite check
    sanitize_gradients(grad_slice);
    // Gradient norm clipping
    let grad_norm = l2_norm(grad_slice);
    if grad_norm > max_grad_norm {
        let scale = max_grad_norm / grad_norm;
        scale_slice(grad_slice, scale);
    }

  • sanitize_gradients replaces NaN/Inf with 0.0.

  ### P-17 (Fail-fast) ✅

  • Missing metadata → panic with P-13 message.
  • Corrupt model data → error propagation via Result.
  • NaN loss detection (line ~650):
    if loss.is_nan() {
        eprintln!("[WARN] NaN loss at step {step}, skipping gradient update");
        return Ok(TrainStepResult { loss: f32::NAN, skipped: true, .. });
    }
    This logs and skips rather than crashing, which is appropriate for training (you don't want to lose progress).

  ### P-27 (No Rayon) ✅

  • No rayon imports. Uses PCorePool for parallel GEMV.
  ──────
  ## 3. Training Objective: Sampled Softmax ✅

  sampled_softmax_loss (lines ~522-680):

    // 1. Compute target logit
    let target_logit = dot_product(&hidden_state, &target_embedding);

    // 2. Collect negatives (hard + random)
    let negatives = collect_negatives(target_id, vocab_size, num_negatives, &embedding_norms);

    // 3. Compute negative logits
    for (i, &neg_id) in negatives.iter().enumerate() {
        neg_logits[i] = dot_product(&hidden_state, &get_embedding(neg_id));
    }

    // 4. Softmax over [target, negatives]
    let max_logit = target_logit.max(neg_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    let target_exp = (target_logit - max_logit).exp();
    let neg_sum_exp: f32 = neg_logits.iter().map(|&l| (l - max_logit).exp()).sum();
    let loss = -(target_exp / (target_exp + neg_sum_exp)).ln();

    // 5. Gradients
    let softmax_target = target_exp / (target_exp + neg_sum_exp);
    // d_loss/d_hidden = (softmax_target - 1) * target_emb + sum(softmax_neg_i * neg_emb_i)

  • Numerically stable: Uses max-subtraction before exp. ✅
  • Hard negatives: Picks top-K by embedding norm similarity (cosine-adjacent heuristic). ✅
  • No NCE-8 code present. ✅

  ### Potential Issue:

  • Hard negative selection (line ~1380): Uses embedding L2 norms for hard negative mining:
    // Sort by norm similarity to target
    let target_norm = embedding_norms[target_id];
    candidates.sort_by(|&a, &b| {
        let da = (embedding_norms[a] - target_norm).abs();
        let db = (embedding_norms[b] - target_norm).abs();
        da.partial_cmp(&db).unwrap_or(Ordering::Equal)
    });
    This uses L2 norm similarity as a proxy for semantic similarity. It's a reasonable heuristic but not as effective as cosine similarity of actual embeddings. Severity: Low
  — works well enough for training, and computing full cosine similarity per step would be expensive.
  ──────
  ## 4. Full-Sequence Training ✅

  forward_train handles both modes:

    if full_seq_mode {
        // Full-sequence: causal windows with position embeddings
        let mask = build_causal_mask(seq_len);
        for pos in 0..seq_len {
            // KV cache grows with position
            for block_idx in 0..num_layers {
                evaluate_train_block(workspace, block_idx, pos, &mask, ...);
            }
        }
    } else {
        // Pair mode: single input → single target at position 0
        for block_idx in 0..num_layers {
            evaluate_train_block(workspace, block_idx, 0, &None, ...);
        }
    }

  • Causal mask: Correctly lower-triangular. ✅
  • Position handling: Positions increment correctly in full-seq mode. ✅
  • KV cache: Updated per position in full-seq mode. ✅
  • Sequence packing integration: Packed batches have EOS boundaries tracked; attention mask prevents cross-sequence leakage. ✅
  ──────
  ## 5. Gradient Checkpointing ✅

    if grad_ckpt_enabled {
        let segments = grad_checkpoint::compute_segments(num_layers, segment_size);
        // Forward: save activations only at segment boundaries
        for seg in &segments {
            for block_idx in seg.start..seg.end {
                evaluate_train_block(...);
            }
            if seg.is_boundary {
                grad_checkpoint::save_activation(workspace, seg.end);
            }
        }
        // Backward: recompute from saved checkpoints
        for seg in segments.iter().rev() {
            grad_checkpoint::restore_activation(workspace, seg.start);
            // Recompute forward for this segment
            for block_idx in seg.start..seg.end {
                evaluate_train_block(...);
            }
            // Backward for this segment
            backward_segment(workspace, seg.start, seg.end, ...);
        }
    }

  • Residual bank (MUD_GRAD_CKPT_RESIDUAL=1): Saves/restores residual stream at boundaries. ✅
  • Memory reduction: Verified — only stores num_segments activations instead of num_layers. ✅
  ──────
  ## 6. MoE Support ✅

    if let Some(expert_idx) = train_expert {
        // Dense single-expert training
        evaluate_train_block_moe(workspace, block_idx, expert_idx, ...);
    } else if moe_train_mode == "hash" {
        // Hash routing for multi-expert
        let expert = hash_route(token_id, num_experts);
        evaluate_train_block_moe(workspace, block_idx, expert, ...);
    }

  • Round-robin and hash routing: Both implemented. ✅
  • FFN names: w3=up, w1=gate, w2=down — consistent with AGENTS.md. ✅
  • ExpertBus integration: Loads expert weights via expert_bus::mount. ✅
  ──────
  ## 7. Bugs and Issues Found

  ### Issue 1: HashSet allocation in collect_negatives (Medium)

  Location: Line ~1370

    fn collect_negatives(target_id: usize, vocab_size: usize, num_neg: usize, norms: &[f32]) -> Vec<usize> {
        let mut seen = HashSet::with_capacity(num_neg + 1);
        seen.insert(target_id);
        // ...
    }

  Problem: Allocates a HashSet every training step. With ~512 negatives, this is a small allocation, but it violates P-01 spirit.
  Fix: Use a pre-allocated bitset or a Vec<bool> of size vocab_size that gets cleared each step.

  ### Issue 2: candidates Vec in hard negative mining (Medium)

  Location: Line ~1375

    let mut candidates: Vec<usize> = (0..vocab_size).collect();
    candidates.sort_by(...);

  Problem: Allocates a Vec of vocab_size elements (e.g., 128,256 entries = ~1MB) and sorts it every step to find hard negatives.
  Fix: Pre-allocate the candidates vec in trainer buffers. Or better: use a partial sort / nth_element to find top-K without full sort.

  ### Issue 3: Gradient accumulation buffer sizing (Low)

  Location: Line ~1430

    let max_tensor_size = tensors.iter().map(|t| t.num_elements()).max().unwrap_or(0);
    grad_scratch = vec![0.0f32; max_tensor_size];

  Problem: If a model has a very large embedding tensor (e.g., 128K × 2048 = 262M elements), this allocates ~1GB for grad_scratch. This is a one-time allocation so not a P-01
  hot-loop violation, but worth noting for memory planning.
  Mitigation: The allocation is correct; it's the minimum needed for the largest tensor. Could add a log message about the allocation size.

  ### Issue 4: Loss NaN handling — skip vs accumulate (Low)

  Location: Line ~650

    if loss.is_nan() {
        eprintln!("[WARN] NaN loss at step {step}, skipping gradient update");
        return Ok(TrainStepResult { loss: f32::NAN, skipped: true, .. });
    }

  Observation: When NaN loss occurs, the step is skipped but the learning rate schedule still advances (if using a scheduler). This could cause the effective number of update
  steps to differ from the reported step count. Not a bug per se, but could affect training dynamics if NaN losses are frequent.
  ──────
  ## 8. Performance Analysis

  ### Hot Path Allocation Audit:

   Operation                                               │ Allocates?                                              │ Severity
  ─────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────
   train_step outer                                        │ No (buffers pre-alloc)                                  │ ✅
   forward_train                                           │ No (workspace reused)                                   │ ✅
   evaluate_train_block                                    │ No (registers in workspace)                             │ ✅
   sampled_softmax_loss                                    │ No (logits_buf pre-alloc)                               │ ✅
   collect_negatives                                       │ Yes (HashSet + candidates Vec)                          │ ⚠️
   backward_pass                                           │ No (EZOP TLS scratch)                                   │ ✅
   apply_gradients                                         │ No (grad_scratch pre-alloc)                             │ ✅
   build_causal_mask                                       │ Builds once, cached                                     │ ✅

  The only hot-loop allocations are in collect_negatives, which are Issue 1 and Issue 2 above.

  ### GEMV parallelism:

  • Uses PCorePool for all GEMV operations. ✅
  • GPU dispatch via gemv_policy when profitable. ✅
  ──────
  ## 9. Summary

  ### Overall Assessment: Strong 🟢

  The training engine is well-architected, policy-compliant, and correctly implements all the features claimed in AGENTS.md/GEMINI.md (Sampled Softmax, full-seq training,
  gradient checkpointing, MoE, optimizer dispatch, JEPA/mHC, STE QAT).

  ### Issues by Severity

   Severity                             │ Issue                                                   │ Location                            │ Recommended Fix
  ──────────────────────────────────────┼─────────────────────────────────────────────────────────┼─────────────────────────────────────┼─────────────────────────────────────
   Medium                               │ HashSet alloc per step in collect_negatives             │ ~L1370                              │ Pre-allocated bitset
   Medium                               │ Full-sort of vocab_size Vec per step for hard negatives │ ~L1375                              │ Pre-alloc vec + partial sort
   Low                                  │ Gradient buffer can be very large for big embeddings    │ ~L1430                              │ Add size logging
   Low                                  │ LR schedule advances on NaN-skipped steps               │ ~L650                               │ Consider conditional advance
   Low                                  │ Hard negative heuristic uses L2 norms, not cosine       │ ~L1380                              │ Acceptable tradeoff

  No critical bugs or policy violations found.

  Now let me compile the full report.*
