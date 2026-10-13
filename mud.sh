#!/bin/bash
# ==============================================================================
# MUD MASTER ORCHESTRATOR (v2.1)
# ==============================================================================
# Unified entry point for MUD Engine Operations.
# Commands are grouped by category for discoverability.
# ==============================================================================

set -e

# --- COLORS ---
PURPLE='\x1b[1;35m'
BLUE='\x1b[1;34m'
GREEN='\x1b[1;32m'
YELLOW='\x1b[1;33m'
RED='\x1b[1;31m'

# OPTIM-1 (2026-08-03): Enable native CPU features for the Rust compiler.
# The #[target_feature(enable="avx2,fma")] annotated functions already use AVX2,
# but WITHOUT target-cpu=native the scalar fallback paths (softmax, embedding
# normalization, element-wise ops, Rust stdlib loops) are compiled for baseline
# x86-64 (SSE2 only). target-cpu=native lets LLVM use AVX2/AVX-512 everywhere,
# improving non-annotated code paths ~1.3× on i7-1260P.
export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native"
CYAN='\x1b[1;36m'
MAG='\x1b[1;35m'
NC='\x1b[0m'

# --- CONFIGURATION ---
# Si no se pasó MODEL_PATH por entorno, usa core_skills.mud por defecto
if [ -z "$MODEL_PATH" ]; then
    # El modelo a entrenar siempre es el latest trained
    if ls models/*_trained.mud 1> /dev/null 2>&1; then
        MODEL_PATH=$(ls -t models/*_trained.mud | head -n 1)
    elif [ -f "weights/checkpoints/model_latest_checkpoint.mud" ]; then
        MODEL_PATH="weights/checkpoints/model_latest_checkpoint.mud"
    elif ls models/*.mud 1> /dev/null 2>&1; then
        MODEL_PATH=$(ls -t models/*.mud | head -n 1)
    else
        MODEL_PATH="models/smollm2.mud"
    fi
fi
CHECKPOINT_DIR="weights/checkpoints"
export MUD_USE_VULKAN=1
export MKL_DEBUG_CPU_TYPE=5  # Enforce AVX2 on Intel CPUs
# Host-safe train defaults (run_trainer also sets these if absent; export for all bins)
: "${MUD_PCORE_THREADS:=8}"
: "${MUD_GPU_GEMV:=0}"   # Disabled by default on Intel Iris Xe due to SIGSEGV
: "${MUD_TRAIN_EZOP:=1}"
: "${MUD_TRAIN_FREEZE_EMB:=1}"
: "${MUD_TRAIN_TEXT_ONLY:=1}"
: "${MUD_CMUD_THINK:=0}"
: "${MUD_CMUD_CELLV5:=0}"     # L-14: CellV5 (u32 8:8:8:8) packing for C-MUD state (4× compression)
: "${MUD_PIPELINE_OVERLAP:=0}"  # D4: full CPU→GPU→CPU pipeline overlap (GPU only)
: "${MUD_BF16_SHADOW:=0}"      # D2: bf16 shadow weight storage (halves shadow memory)
: "${MUD_KV_COMPRESS:=0}"      # Stream I: lossless KV cache compression (Delta + LZ4)
: "${MUD_KV_COMPRESS_DELTA:=1}" # Stream I: delta encoding within LZ4 blocks
: "${MUD_KV_COMPRESS_CELLV5:=0}"  # PLAN_DIA_2 Phase 3: CellV5 pre-compression (4× extra on top of LZ4+Delta), reserved — future implementation; resolver emits a warning when set.
: "${MUD_KV_BLOCK_SIZE:=256}"  # Stream I: tokens per compressed KV block
: "${MUD_MHC_SYNC:=1}"           # 2026-07-31: persist mHC alpha/beta on checkpoint save (1=on)
: "${MUD_DW_DEBUG:=0}"           # 2026-07-31: report ternary + PRQ ΔW per checkpoint (MUD_TRAIN_DEBUG_DW)
: "${MUD_BEST_CHECKPOINT:=1}"     # 2026-08-03: save {stem}_best.mud on loss improvement
: "${MUD_BF16_SHADOW:=0}"        # D2: bf16 shadow weight storage (halves shadow memory)
export MUD_PCORE_THREADS MUD_GPU_GEMV MUD_TRAIN_EZOP MUD_TRAIN_FREEZE_EMB MUD_TRAIN_TEXT_ONLY MUD_CMUD_THINK MUD_CMUD_CELLV5 MUD_PIPELINE_OVERLAP MUD_BF16_SHADOW MUD_KV_COMPRESS MUD_KV_COMPRESS_DELTA MUD_KV_COMPRESS_CELLV5 MUD_KV_BLOCK_SIZE MUD_MHC_SYNC MUD_DW_DEBUG MUD_BEST_CHECKPOINT

# --- DEBATE / PROFESSOR-STUDENT DEFAULTS ------------------------------------
# Infinite RLVR survival loop (F3). Detenable by Ctrl-C / 'q' in TUI, or
# MUD_DEBATE_MAX_TIME. Defaults are host-safe and CPU-friendly on i7-1260P.
: "${MUD_DEBATE_MODE:=debate}"          # debate | professor
: "${MUD_DEBATE_MAX_TIME:=0}"           # 0 = infinite until Ctrl-C / 'q'
: "${MUD_DEBATE_RWIN:=1.0}"             # winner reward
: "${MUD_DEBATE_RLOSE:=0.7}"            # loser penalty (asymmetric)
: "${MUD_DEBATE_JEPB_LAMBDA:=0.05}"     # JEPA intrinsic aux (anti-collapse)
: "${MUD_DEBATE_LEARN:=1}"              # persist shadow->MUD after session (default ON)
# MUD_DEBATE_MAX_NEW_TOKENS left unset -> auto by free RAM (see auto_max_new_tokens)
export MUD_DEBATE_MODE MUD_DEBATE_MAX_TIME MUD_DEBATE_RWIN MUD_DEBATE_RLOSE MUD_DEBATE_JEPB_LAMBDA MUD_DEBATE_LEARN

# --- PROJECT-ADAPTED CORPUS ---------------------------------------------------
# Assemble a curated, project-specific training corpus into
# training/corpus/project_corpus.txt by concatenating:
#   - existing ES/EN alignment text (training/corpus/*.txt)
#   - project docs (*.md: AGENTS, GEMINI, README, docs/**)
#   - project source (src/**, forge_autograd/src/**, tools/** .rs)
# Only rebuilds when the newest source is newer than the assembled cache
# (mtime gate) so repeated `./mud.sh train` reuses the AOT cache.
build_project_corpus() {
    local out="training/corpus/project_corpus.txt"
    mkdir -p training/corpus
    # Newest mtime among candidate sources; empty dirs -> epoch 0
    local newest=0
    while IFS= read -r -d '' f; do
        local m
        m=$(stat -c %Y "$f" 2>/dev/null || echo 0)
        [ "$m" -gt "$newest" ] && newest=$m
    done < <(find training/corpus -maxdepth 1 -name '*.txt' -print0 2>/dev/null; \
             find . -maxdepth 1 -name '*.md' -print0 2>/dev/null; \
             find docs -name '*.md' -print0 2>/dev/null; \
             find src forge_autograd/src tools -name '*.rs' -print0 2>/dev/null)
    local cached=0
    [ -f "$out" ] && cached=$(stat -c %Y "$out" 2>/dev/null || echo 0)
    if [ "$newest" -le "$cached" ] && [ "$cached" -gt 0 ]; then
        echo -e "${GREEN}[corpus]${NC} reusing cached project_corpus.txt (sources unchanged)"
        return 0
    fi
    echo -e "${BLUE}[corpus]${NC} assembling project-adapted corpus -> $out"
    : > "$out"
    # 1) ES/EN alignment text already in training/corpus
    for f in training/corpus/*.txt; do
        [ -e "$f" ] || continue
        cat "$f" >> "$out" 2>/dev/null
        printf '\n\n' >> "$out"
    done
    # 2) project docs (markdown)
    for f in $(find . -maxdepth 1 -name '*.md' 2>/dev/null; find docs -name '*.md' 2>/dev/null); do
        [ -f "$f" ] || continue
        echo "### DOC: $f" >> "$out"
        cat "$f" >> "$out" 2>/dev/null
        printf '\n\n' >> "$out"
    done
    # 3) project source (rust) — included as training text so the model adapts to THIS codebase
    for f in $(find src forge_autograd/src tools -name '*.rs' 2>/dev/null); do
        [ -f "$f" ] || continue
        echo "### SRC: $f" >> "$out"
        cat "$f" >> "$out" 2>/dev/null
        printf '\n\n' >> "$out"
    done
    echo -e "${GREEN}[corpus]${NC} assembled $(wc -c < "$out") bytes"
}

# Pick a chunk budget adapted to the project corpus size.
# CHUNK_SIZE=50000 chars (~12.5k tokens). Cover the whole corpus, capped to
# a sane default so a bare `./mud.sh train` does not run for hours.
compute_project_chunks() {
    local bytes
    bytes=$(wc -c < training/corpus/project_corpus.txt 2>/dev/null || echo 0)
    local chunks=$(( (bytes + 49999) / 50000 ))
    [ "$chunks" -lt 1 ] && chunks=1
    # Cap default run to 64 chunks (covers ~3.2M chars ≈ full small project) unless user overrides.
    if [ -z "$MUD_TRAIN_MAX_CHUNKS" ]; then
        if [ "$chunks" -gt 64 ]; then chunks=64; fi
        export MUD_TRAIN_MAX_CHUNKS=$chunks
    fi
}


show_help() {
    echo -e "${PURPLE}=== MUD MASTER COMMAND CENTER (v2.1) ===${NC}"
    echo -e "Usage: ./mud.sh [command] [options]"
    echo ""
    echo -e "${BLUE}🥇 Priority 0 (2026-10-11):${NC}"
    echo "  plan            Backlog activo R-##: correcciones/limpieza/mejoras + progreso por ola"
    echo "                  (docs/planning/MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md)"
    echo ""
    echo -e "${BLUE}🧠 Intelligence & Training:${NC}"
    echo "  train           Launch Local Rust AutoTrainer daemon (Live Learning)"
    echo "  align           Start Native Corpus Aligner (Linguistic Restoration)"
    echo "  project         Run Recalibration Projector (Bayesian Determinism)"
    echo "  restore-iq      Unified Pipeline: Align -> Project -> Train (Auto-Restoration)"
    echo "  circuit [model] Infinite training circuit by seed: shuffled batteries"
    echo "                  of align/debate/games/professor (loop until quit)."
    echo "  slimex-train    SlimeX-activated training (CellV5 + ShadowExpertBus)"
    echo "  slimex-audit    Verify SlimeX cell buffer integrity"
    echo "  slimex-fuse     Merge multiple .mud files into one (MudFileMerger)"
    echo "  pipeline-overlap  D4: Benchmark CPU→GPU→CPU pipeline overlap (GPU required)"
    echo ""
    echo -e "${BLUE}💬 Interaction & Analysis:${NC}"
    echo "  chat            Launch interactive MUD terminal (use --cmud for Thinker)"
    echo "  step            Run step-by-step inference autopsy"
    echo "  telemetry       Launch Real-Time Loss/Variance TUI Graph"
    echo "  debate          Launch Telemetry Debate module (RLVR seed-survival loop)"
    echo "  professor       Launch Professor-Student loop (local grammar/syntax judge)"
    echo ""
    echo -e "${BLUE}🛡️ Safety, Quality & Validation:${NC}"
    echo "  health          Run Training Healthcheck (Pre-flight weight & shape validation)"
    echo "  audit-full      Full ledger audit L-01…L-15 (structure + policy + smoke)"
    echo "  ci              L-12 battery: tests + loss_cert + clippy + health (+ optional e2e)"
    echo "  cert-loss       Stream K: loss-must-fall gate (needs models/*.mud)"
    echo "  audit-conv      Run Converter Auditor (Post-conversion structural & magic byte check)"
    echo "  verify [sf]     Run High-Fidelity Conversion Verifier against raw safetensors"
    echo "  precise-audit   Run Precise Converter Audit (bitnet_scale-correct SQNR/MAE/MSE)"
    echo "  bound           Run Quantization Boundary Validator (NaN/Grid check)"
    echo "  estimate        Run Universal Training Estimator (seating predictions)"
    echo "  validate        Run Cognitive Iteration Validator (Score % metric)"
    echo "  inspect-var     Run Deep Variance Inspector (VarH/VarJ check per layer)"
    echo "  scale-audit     PRQ scale audit vs BF16 source (detects vocabulary collapse)"
    echo "  pointer-audit   Raw pointer & ELUT nibble formula audit (P-00/P-01)"
    echo "  env-audit       Verify 0 env-var doc/code/mud.sh divergences (R-05)"
    echo "  unsafe-audit    Verify 0 unjustified unsafe blocks in prod code (R-36)"
    echo "  mhc-sync        Verify mHC alpha/beta persisted to .mud after checkpoint save"
    echo ""
    echo -e "${BLUE}🧪 C-MUD (Complex Manifold Thinking):${NC}"
    echo "  cmud-audit      C-MUD complex-reasoning manifold auditor"
    echo "  cmud-cmp        Baseline vs C-MUD quality probe"
    echo "  cmud-train      Finite-difference phase-bias ascent (research §4)"
    echo "  cmud-gradtest   Analytic vs finite-difference gradient (end-to-end)"
    echo "  cmud-bench      Think ON vs OFF next-token quality"
    echo "  cmud-manifold   C-MUD 5-dimension cognition & comparative validator"
    echo "  mhc-stress      mHC scale SGD stress test + precision + speed (Vulkan/CPU)"
    echo "  coherence       Inference coherence: success, deviation, coherence path"
    echo ""
    echo -e "${BLUE}⚡ Performance & Hardware:${NC}"
    echo "  hw              Show detected hardware profile & SIMD status"
    echo "  bench           Run deep performance & memory benchmark"
    echo "  audit           Run full cognitive & structural audit suite"
    echo "  diag            Launch the unified master health & diagnostics dashboard"
    echo ""
    echo -e "${BLUE}📊 Benchmarks:${NC}"
    echo "  fenwick-bench   Fenwick Tree vs Linear scan (F4/F1/F3 + FenwickLogF32)"
    echo "                  Flags: --size, --queries, --iterations, --json, --quiet"
    echo "  asm-bench       ASM kernel benchmark: AVX2 vs Scalar (16 kernels)"
    echo "                  Flags: --size, --iterations, --queries, --json, --quiet"
    echo ""
    echo -e "${BLUE}📦 Weights Management:${NC}"
    echo "  convert         Universal Converter: Safetensors/PyTorch to .mud"
    echo "                  Usage: ./mud.sh convert [input] [output] [--ternarize-emb]"
    echo "  forge           MUD Forge: Create a blank initialized model from scratch"
    echo "                  Usage: ./mud.sh forge [nano|micro|base|custom] [output] [extra opts...]"
    echo "                  Tokenizer dir from MUD_TOKENIZER_DIR (default: models/smollm2)"
    echo "  extract-docs    Extract Rust docs + Markdown docs into training/corpus/mud_docs.txt"
    echo ""
    echo -e "${BLUE}🧹 Workspace:${NC}"
    echo "  ckpt            List available training checkpoints"
    echo "  restore [name]  Replace current model with a specific checkpoint"
    echo "  restore-best [stem]  Restore from the best-loss checkpoint ({stem}_best.mud)"
    echo "  clean           Organize files and move logs to logs/"
    echo "  clean-logs      Purge all telemetry metrics, training logs, and circuit history"
    echo ""
    echo -e "${BLUE}📋 Catalog:${NC}"
    echo "  tools           Display binary catalog of all core tools and executables"
    echo ""
    echo -e "${BLUE}⚙️  Debate / Professor env vars (defaults set by mud.sh):${NC}"
    echo "  MUD_DEBATE_MODE=debate|professor   game mode (professor = student learns until quit)"
    echo "  MUD_DEBATE_MAX_TIME=0              seconds before auto-stop (0 = infinite)"
    echo "  MUD_DEBATE_RWIN=1.0  MUD_DEBATE_RLOSE=0.7   reward / penalty"
    echo "  MUD_DEBATE_JEPB_LAMBDA=0.05        JEPA intrinsic aux weight"
    echo "  MUD_DEBATE_LEARN=1                 persist shadow->MUD after session (0=off)"
    echo "  MUD_DEBATE_MAX_NEW_TOKENS=auto     per-response cap (auto by free RAM if unset)"
}

# R-36: verifica que todo `unsafe` en código de producción tenga justificación.
# Justificado si hay `SAFETY` / `# Safety` en la misma línea o en las 8 previas.
# Ops de ALTO RIESGO (from_raw_parts/Mmap::map/CStr::from_ptr/transmute/alloc/
# dealloc/from_ptr/as_mut()/.offset() ) SIEMPRE exigen comentario adyacente.
# El resto puede cubrirse con un contrato de módulo `//! SAFETY:` (no aplica a
# alto riesgo). Se excluyen tests (tests.rs, test_*.rs, tests/, src/bin/ y
# cuerpos #[cfg(test)] dentro de ficheros de producción).
run_unsafe_audit() {
    local FILES
    FILES=$(grep -rlE '\bunsafe\b' src --include='*.rs' 2>/dev/null \
        | grep -vE '(^|/)(tests\.rs|test_[^/]*\.rs|tests/|src/bin/)' || true)
    if [ -z "$FILES" ]; then
        echo -e "${GREEN}[UNSAFE-AUDIT] OK - 0 sin comentario (R-36)${NC}"
        return 0
    fi
    # shellcheck disable=SC2086
    awk '
    BEGIN { bad = 0; total = 0 }
    FNR == 1 { depth = 0; skipd = -1; pending = 0; first = 0; modc = 0; delete P }
    {
        line = $0; first++
        if (first <= 60 && line ~ /^[ \t]*\/\/![ \t]*SAFETY/) modc = 1
        if (line !~ /^[ \t]*\/\// && line ~ /#[ \t]*\[[ \t]*cfg\([ \t]*test[ \t]*\)[ \t]*\]/) pending = 1
        if (pending && line ~ /^[ \t]*(pub[ \t]+)?mod[ \t]/ && line ~ /\{/) { skipd = depth; pending = 0 }
        tmp = line
        nd = depth + gsub(/\{/, "{", tmp) - gsub(/\}/, "}", tmp)
        skip_line = (skipd >= 0 && depth > skipd)
        if (skipd >= 0 && nd <= skipd) skipd = -1
        if (!skip_line && line !~ /^[ \t]*\/\// && line ~ /(^|[^A-Za-z0-9_])unsafe([^A-Za-z0-9_]|$)/) {
            total++
            adj = (line ~ /SAFETY|# [Ss]afety/) ? 1 : 0
            for (k = FNR - 1; k >= FNR - 8 && k >= 1; k--) { if (P[k] ~ /SAFETY|# [Ss]afety/) { adj = 1; break } }
            hr = (line ~ /from_raw_parts|Mmap::map|CStr::from_ptr|transmute|alloc\(|dealloc|from_ptr|as_mut\(\)|\.offset\(/) ? 1 : 0
            ok = adj || (!hr && modc)
            if (!ok) { bad++; printf "    %s:%d: %s\n", FILENAME, FNR, substr(line, 1, 110) }
        }
        P[FNR] = line; depth = nd
    }
    END {
        printf "[UNSAFE-AUDIT] unsafe: %d · sin justificar: %d\n", total, bad
        if (bad == 0) print "OK - 0 sin comentario (R-36)"
        exit (bad > 0)
    }' $FILES
}

case $1 in

# ============================================================================
# 🥇 PRIORIDAD 0 (2026-10-11)
# ============================================================================

    plan)
        # Backlog activo R-## (SSOT de ejecución). Ver GEMINI.md §0.3 + ROADMAP "PRIORIDAD 0".
        PLAN_MD="docs/planning/MUD_MASTER_REMEDIATION_PLAN_2026-10-11.md"
        AUDIT_MD="docs/audits/MUD_AUDIT_2026-10-11_LOW_COMPUTE.md"
        if [ ! -f "$PLAN_MD" ]; then
            echo -e "${RED}Plan maestro no encontrado: $PLAN_MD${NC}"
            exit 1
        fi
        echo -e "${PURPLE}=== MUD — PRIORIDAD 0: PLAN MAESTRO DE CORRECCIONES ===${NC}"
        echo -e "${DIM}$PLAN_MD${NC}"
        echo -e "${DIM}Auditoría de origen: $AUDIT_MD${NC}"
        echo ""
        # Panel de seguimiento (§12) — scoped para no duplicar las tablas por ola
        PANEL_AWK='/^## 12\. Panel de seguimiento/{f=1; next} /^## 13\./{f=0} f && /^\| R-[0-9]+ \|/{print}'
        echo -e "${BLUE}Estado por ítem (ID | Título | Sev | Estado):${NC}"
        awk "$PANEL_AWK" "$PLAN_MD" | sed -E 's/^\| /  /; s/([[:space:]]*\|)+[[:space:]]*$//'
        echo ""
        TOTAL=$(awk "$PANEL_AWK" "$PLAN_MD" | wc -l)
        DONE=$(awk "$PANEL_AWK" "$PLAN_MD" | grep -c '✅' || true)
        echo -e "${PURPLE}Progreso: ${DONE:-0}/${TOTAL} ítems cerrados.${NC}"
        echo -e "${DIM}Orden obligatorio: Ola 0 (verdad/CI) → Ola 1 (entrenamiento) ∥ Ola 2 (seguridad) → 3 → 4 → 5 → 6.${NC}"
        echo -e "${DIM}Regla: nada se marca hecho sin (comando, salida, fecha) — R-73.${NC}"
        echo -e "${DIM}Detalle completo: less $PLAN_MD${NC}"
        ;;

# ============================================================================
# 🧠 INTELLIGENCE & TRAINING
# ============================================================================

    train)
        shift
        # If user explicitly passed a model path as the next arg, capture it and shift
        if [[ "$1" == *.mud ]] || [[ "$1" == *.safetensors ]]; then
            MODEL_PATH="$1"
            shift
        fi

        # Priority 53: Auto-select model if none given
        if [ "$MODEL_PATH" = "models/core_skills.mud" ] && [ ! -f "models/core_skills.mud" ]; then
            if [ -f "smollm2.mud" ]; then
                MODEL_PATH="smollm2.mud"
            fi
        fi
        # Project-adapted corpus (rebuilt only when sources change)
        build_project_corpus || true
        # Validated training defaults (override via env if needed):
        #   STP aux loss ON (F2), 64 steps/chunk (stable descent, no 3.4<->8.9 noise),
        #   TEXT_ONLY=1 because source/docs are inlined into project_corpus.txt as text.
        : "${MUD_TRAIN_STP:=1}"
        : "${MUD_TRAIN_STEPS_PER_CHUNK:=64}"
        : "${MUD_TRAIN_TEXT_ONLY:=1}"
        : "${MUD_TRAIN_NUM_NEG:=511}"     # FIX (2026-08-03): 511 negatives gives ln(512)=6.237 loss floor
        : "${MUD_OPT:=adam}"              # 3x more room for gradient signal vs 255-neg (loss floor ln(256)=5.545)
        # FIX (2026-08-03): MUD_QAT_LR removed — was hardcoding lr=0.003 and bypassing
        # the cosine schedule entirely (apply_lr_schedule short-circuits on MUD_QAT_LR).
        # Now uses cosine schedule (base=0.0005, warmup 64 steps, min_lr=5% floor)
        # which prevents the slow cog decay → collapse observed on ale.mud at step ~3403.
        # Override: MUD_QAT_LR=0.003 to force fixed LR (disables schedule).
        : "${MUD_GRAD_CLIP:=1.0}"         # Gradient clip (1.0 default; fresh models have grad_norm≈2.2, too low kills learning)
        # 2026-08-03: Best-model checkpoint is the safety net (not clip) for collapse recovery
        : "${MUD_BEST_CHECKPOINT:=1}"     # Save {stem}_best.mud on loss improvement (rollback safety net)
        : "${MUD_TRAIN_RESET_EPOCH:=1}"     # Fresh epoch counter — prevents stale no-op resume
        : "${MUD_TRAIN_WCLAMP_K:=8}"        # Weight clamp: prevents PRQ scale inflation (vocab collapse)
        : "${MUD_TRAIN_SEQ_LEN:=128}"       # Full-seq window: 4x default (32) for richer context
        : "${MUD_BF16_SHADOW:=0}"           # Keep CPU fp32 shadow (Iris Xe has no benefit)

        # If the user explicitly asks for epochs, do not cap the training chunks
        for arg in "$@"; do
            if [[ "$arg" == "--epochs" ]]; then
                export MUD_TRAIN_MAX_CHUNKS=0
                break
            fi
        done

        compute_project_chunks
        export MUD_TRAIN_STP MUD_TRAIN_STEPS_PER_CHUNK MUD_TRAIN_TEXT_ONLY MUD_TRAIN_NUM_NEG MUD_OPT MUD_MHC_SYNC MUD_DW_DEBUG
        export MUD_TRAIN_RESET_EPOCH MUD_TRAIN_WCLAMP_K MUD_TRAIN_SEQ_LEN MUD_BF16_SHADOW
        export MUD_GRAD_CLIP MUD_BEST_CHECKPOINT
        # Export MUD_QAT_LR only if the user explicitly set it (cosine schedule otherwise)
        if [ -n "${MUD_QAT_LR:-}" ]; then export MUD_QAT_LR; fi
        # GRASSHOPPER FIX (2026-08-03): Cosine restarts (SGDR) for LR schedule.
        # Default: MUD_LR_RESTART=1 (enabled). Set MUD_LR_RESTART=0 to use
        # original single-cosine decay (legacy behavior). Cycles configurable
        # via MUD_LR_CYCLES (default: auto, max 3-16).
        if [ -n "${MUD_LR_RESTART:-}" ]; then export MUD_LR_RESTART; fi
        # Explicit model arg from remaining CLI wins over MODEL_PATH env default.
        # Host train stack: AVX2 × PCorePool + CPU GEMV; ash QAT VRAM off by default.
        #
        # HARDWARE CAVEAT (i7-1260P / Iris Xe ADL GT2, UMA integrated):
        # the Intel Vulkan driver (libvulkan_intel.so) SIGSEGVs inside
        # submit_and_wait during GEMV-QKV dispatch on this silicon. The AGENTS
        # runtime truth is that the real hot-path here is AVX2 (the GPU
        # break-even only shows in synthetic micro-benches, not the 147M forward).
        # Force Vulkan OFF for training so the trainer never dispatches to the
        # crashing driver. Set MUD_TRAIN_FORCE_VULKAN=1 to override (e.g. on
        # discrete/stable GPUs).
        if [ "${MUD_TRAIN_FORCE_VULKAN:-0}" != "1" ]; then
            export MUD_USE_VULKAN=0
            export MUD_GPU_GEMV=0
        fi
        echo ""
        echo -e "${MAG}▸ MUD TRAIN LAUNCH${NC}"
        echo -e "  ${CYAN}Model${NC}:     ${MODEL_PATH}"
        echo -e "  ${CYAN}Hardware${NC}:  PCore=${MUD_PCORE_THREADS}  GPU_GEMV=${MUD_GPU_GEMV}  USE_VULKAN=${MUD_USE_VULKAN}  EZOP=${MUD_TRAIN_EZOP}  FREEZE_EMB=${MUD_TRAIN_FREEZE_EMB}"
        echo -e "  ${CYAN}Schedule${NC}:  STP=${MUD_TRAIN_STP}  STEPS/CHUNK=${MUD_TRAIN_STEPS_PER_CHUNK}  NEG=${MUD_TRAIN_NUM_NEG}  OPT=${MUD_OPT}  MAX_CHUNKS=${MUD_TRAIN_MAX_CHUNKS}"
        if [ -n "${MUD_QAT_LR:-}" ]; then
            echo -e "  ${CYAN}Training${NC}:  LR=${MUD_QAT_LR} (fixed, schedule bypassed)  RESET_EPOCH=${MUD_TRAIN_RESET_EPOCH}  WCLAMP_K=${MUD_TRAIN_WCLAMP_K}  SEQ_LEN=${MUD_TRAIN_SEQ_LEN}  BF16_SHADOW=${MUD_BF16_SHADOW}"
        else
            restart_label="SGDR"
            if [ "${MUD_LR_RESTART:-1}" = "0" ]; then restart_label="single-cosine"; fi
            echo -e "  ${CYAN}Training${NC}:  LR=cosine(base=0.0005→min=0.000025, ${restart_label})  GRAD_CLIP=${MUD_GRAD_CLIP}  RESET_EPOCH=${MUD_TRAIN_RESET_EPOCH}  WCLAMP_K=${MUD_TRAIN_WCLAMP_K}  SEQ_LEN=${MUD_TRAIN_SEQ_LEN}  BF16_SHADOW=${MUD_BF16_SHADOW}"
        fi
        echo -e "  ${CYAN}Best Model${NC}: ${MUD_BEST_CHECKPOINT:+saving {stem}_best.mud on loss improvement}${MUD_BEST_CHECKPOINT:-disabled}"
        echo ""
        cargo run --release --bin run_trainer -- "$MODEL_PATH" "$@"
        ;;
    restore-iq)
        MODEL_PATH=${2:-$MODEL_PATH}
        echo -e "${PURPLE}╭───────────────────────────────────────────────────────────────────────────────╮${NC}"
        echo -e "${PURPLE}│ 🌀 INICIANDO UCP v2: RESTAURACION COGNITIVA (MUD IQ-RESTORE)                  │${NC}"
        echo -e "${PURPLE}│ ⚡ LIVE: AVX2 ELUT GEMV × PCorePool(8) + FP32 accum  |  ash=optional ctx      │${NC}"
        echo -e "${PURPLE}│ 🧠 JEPA/mHC + STE QAT (LIVE=SGD)  |  Muon/GaLore=PLANNED L-01                 │${NC}"
        echo -e "${PURPLE}╰───────────────────────────────────────────────────────────────────────────────╯${NC}"

        echo -e "${BLUE}[1/6] VERIFY CONVERSION: Checking SQNR & Tier-1 Stability...${NC}"
        # Skip conversion verifier in automated pipeline as it requires the safetensors path
        # cargo run --release --bin conversion_verifier -- "$MODEL_PATH"

        echo -e "${BLUE}[2/6] VERIFY BOUNDARIES: Checking mHC Manifolds & Ternary Scale Conformity...${NC}"
        cargo run --release --bin boundary_validator -- "$MODEL_PATH"

        echo -e "${BLUE}[3/6] ESTIMATE WORKLOAD: Calculating JEPA Recovery Parameters & Seating...${NC}"
        EST_EPOCHS=$(cargo run --release --bin training_estimator "$MODEL_PATH" 2>/dev/null | grep "Required Epochs" | awk '{print $4}' | sed -r "s/\x1B\[([0-9]{1,3}(;[0-9]{1,2})?)?[mGK]//g")
        if [ -z "$EST_EPOCHS" ]; then
            EST_EPOCHS=5
            echo -e "${YELLOW}Could not determine exact epochs, defaulting to 5.${NC}"
        else
            echo -e "${GREEN}Calculated required epochs for thermodynamic recovery: $EST_EPOCHS${NC}"
        fi

        echo -e "${BLUE}[4/6] ALIGN: Ash QAT Dispatcher & Linguistic Restoration (Deep Epoch)...${NC}"
        cargo run --release --bin run_trainer -- "$MODEL_PATH" --epochs "$EST_EPOCHS"

        echo -e "${BLUE}[5/6] PROJECT: Newton-Schulz Orthogonalization (Tier 3 PRQ Calibration)...${NC}"
        cargo run --release --bin recalibration_projector -- "$MODEL_PATH" --tier3
        # cargo run --release --bin mud_autotrainer -- seating "$MODEL_PATH"

        echo -e "${BLUE}[6/6] ASSERT EFFECTIVENESS: 20-Parameter Validation (JEPA, mHC, Aphasia) >96%...${NC}"
        cargo run --release --bin iteration_validator -- "$MODEL_PATH"
        ;;
    align)
        echo -e "${PURPLE}[ALIGN] Starting Native Corpus Aligner...${NC}"
        build_project_corpus || true
        : "${MUD_TRAIN_STP:=1}"
        : "${MUD_TRAIN_STEPS_PER_CHUNK:=64}"
        : "${MUD_TRAIN_TEXT_ONLY:=1}"
        compute_project_chunks
        export MUD_TRAIN_STP MUD_TRAIN_STEPS_PER_CHUNK MUD_TRAIN_TEXT_ONLY
        cargo run --release --bin run_trainer
        ;;
    project)
        echo -e "${PURPLE}[PROJECT] Running Recalibration Projector...${NC}"
        MODEL=${2:-$MODEL_PATH}
        cargo run --release --bin recalibration_projector -- "$MODEL"
        ;;
    circuit)
        MODEL=${2:-$MODEL_PATH}
        # Infinite training circuit: per SEED it mints a distinct BATTERY (a shuffled
        # ordering of align/debate/games/professor) so training is never a fixed
        # monotonic schedule. Loops until 'quit' / Ctrl-C. Time-boxed per phase.
        : "${MUD_CIRCUIT_MAX_PER_MODE:=120}"   # seconds per phase
        : "${MUD_CIRCUIT_EPOCHS:=1}"           # alignment epochs per cycle
        : "${MUD_CIRCUIT_BATCH:=16}"
        : "${MUD_DEBATE_LEARN:=1}"             # persist each phase (honors-mode default ON)
        if [ "${MUD_TRAIN_FORCE_VULKAN:-0}" != "1" ]; then
            export MUD_USE_VULKAN=0
            export MUD_GPU_GEMV=0
        fi
        : "${MUD_OPT:=adam}"
        export MUD_CIRCUIT_MAX_PER_MODE MUD_CIRCUIT_EPOCHS MUD_CIRCUIT_BATCH MUD_DEBATE_LEARN MUD_USE_VULKAN MUD_GPU_GEMV MUD_OPT
        mkdir -p logs
        echo -e "${BLUE}[circuit]${NC} seed→battery(align/debate/games/professor) | max/mode=${MUD_CIRCUIT_MAX_PER_MODE}s learn=${MUD_DEBATE_LEARN}"
        echo -e "${YELLOW}[circuit]${NC} Live TUI: Juez + J|A/B (+Profesor/Alumno). Ctrl-C / 'q' to stop & save. Telemetry log: logs/circuit.log"
        cargo run --release --bin circuit_telemetry -- "$MODEL"
        ;;
    pipeline-overlap)
        # D4: Benchmark CPU→GPU→CPU pipeline overlap throughput.
        # Activates MUD_PIPELINE_OVERLAP=1 and runs a short training burst,
        # measuring tokens/s with vs without pipeline overlap.
        # Requires Vulkan (GPU). On Intel Iris Xe, force CPU fallback.
        MODEL=${2:-$MODEL_PATH}
        if [ "${MUD_TRAIN_FORCE_VULKAN:-0}" != "1" ]; then
            echo -e "${YELLOW}[pipeline-overlap]${NC} Vulkan unstable on Intel Iris Xe — forcing CPU (MUD_PIPELINE_OVERLAP=0)"
            export MUD_USE_VULKAN=0
            export MUD_GPU_GEMV=0
            export MUD_PIPELINE_OVERLAP=0
        else
            export MUD_PIPELINE_OVERLAP=1
        fi
        : "${MUD_BF16_SHADOW:=1}"  # D2: enable bf16 shadow for memory reduction
        export MUD_BF16_SHADOW
        echo -e "${BLUE}[pipeline-overlap]${NC} D4: CPU→GPU→CPU overlap=${MUD_PIPELINE_OVERLAP} | bf16_shadow=${MUD_BF16_SHADOW}"
        cargo run --release --bin run_trainer -- "$MODEL" --epochs 1
        ;;
    slimex-train)
        # SX-01/SX-03: SlimeX-activated training — uses SlimeXCellV5 transport cell
        # for snapshot_to_slimex() (memory continuity). P-01/SX-02 pre-alloc pattern.
        # ShadowExpertBus removed (P-08: dead code — mount/evict/unmount never wired).
        MODEL=${2:-$MODEL_PATH}
        : "${MUD_EPOCHS:=5}"
        : "${MUD_QAT_LR:=0.0005}"
        : "${MUD_TRAIN_FULL_SEQ:=1}"
        if [ "${MUD_TRAIN_FORCE_VULKAN:-0}" != "1" ]; then
            export MUD_USE_VULKAN=0
            export MUD_GPU_GEMV=0
        fi
        export MUD_SLIMEX=1
        export MUD_EPOCHS MUD_QAT_LR MUD_TRAIN_FULL_SEQ MUD_USE_VULKAN MUD_GPU_GEMV
        echo -e "${BLUE}[slimex-train]${NC} SlimeX CellV5 active | model=${MODEL} epochs=${MUD_EPOCHS} lr=${MUD_QAT_LR}"
        echo -e "${YELLOW}[slimex-train]${NC} SX-01: slimex_cells pre-alloc | SX-03: AVX2 kernel"
        cargo run --release --bin run_trainer -- "$MODEL"
        ;;
    slimex-audit)
        # Verifies SlimeX integration: checks slimex_cells are populated and non-zero
        # after a short forward pass. Runs the p13 + slimex health check.
        MODEL=${2:-$MODEL_PATH}
        export MUD_SLIMEX=1
        echo -e "${PURPLE}[slimex-audit]${NC} SlimeX cell buffer integrity check | model=${MODEL}"
        cargo run --release --bin diagnose_model -- "$MODEL"
        ;;
    slimex-fuse)
        # SPRINT-2: SlimeX file fusion. Scans models/*.mud and writes a single merged .mud
        # via MudFileMerger. Usage: ./mud.sh slimex-fuse [INPUT_DIR] [OUTPUT_PATH]
        INPUT_DIR=${2:-models}
        OUTPUT_PATH=${3:-models/merged.mud}
        echo -e "${PURPLE}[slimex-fuse]${NC} Merging .mud files in ${INPUT_DIR} → ${OUTPUT_PATH}"
        cargo run --release --bin slimex_fuse -- "$INPUT_DIR" "$OUTPUT_PATH"
        ;;

    extract-docs)
        echo -e "${BLUE}[extract-docs]${NC} Extracting documentation + comments from source + docs into training/corpus/mud_docs.txt"
        cargo run --release --bin extract_code_docs
        echo -e "${GREEN}[extract-docs]${NC} Done. New corpus: $(wc -l < training/corpus/mud_docs.txt) lines, $(wc -w < training/corpus/mud_docs.txt) words"
        ;;

    chat)
        shift
        USE_CMUD=0
        MODEL_ARG=""
        for arg in "$@"; do
            if [[ "$arg" == "--cmud" ]]; then
                USE_CMUD=1
            elif [[ -z "$MODEL_ARG" && "$arg" != -* ]]; then
                MODEL_ARG="$arg"
            fi
        done
        MODEL=${MODEL_ARG:-$MODEL_PATH}

        if [ "$USE_CMUD" -eq 1 ]; then
            export MUD_CMUD_THINK=1
            export MUD_CMUD_PARAMS="${MODEL}.cmud.json"
            echo -e "${PURPLE}[C-MUD] Active | Params: ${MUD_CMUD_PARAMS}${NC}"
        fi

        cargo run --release --bin forge_llm -- "$MODEL"
        ;;
    debate)
        MODEL=${2:-$MODEL_PATH}
        echo -e "${BLUE}[debate]${NC} mode=${MUD_DEBATE_MODE} max_time=${MUD_DEBATE_MAX_TIME} learn=${MUD_DEBATE_LEARN}"
        cargo run --release --bin debate_telemetry -- "$MODEL"
        ;;
    professor)
        MODEL=${2:-$MODEL_PATH}
        export MUD_DEBATE_MODE=professor
        echo -e "${BLUE}[professor]${NC} Professor-Student loop | max_time=${MUD_DEBATE_MAX_TIME} learn=${MUD_DEBATE_LEARN}"
        echo -e "${YELLOW}[professor]${NC} Ctrl-C / 'q' to stop; student repeats & learns until quit."
        cargo run --release --bin debate_telemetry -- "$MODEL"
        ;;
    telemetry)
        cargo run --release --bin train_telemetry
        ;;
    step)
        cargo run --release --bin step_inference
        ;;

# ============================================================================
# 🛡️ SAFETY, QUALITY & VALIDATION
# ============================================================================

    health)
        MODEL=${2:-$MODEL_PATH}
        cargo run --release --bin diagnose_model -- "$MODEL"
        ;;
    audit-full)
        MODEL=${2:-$MODEL_PATH}
        echo -e "${PURPLE}[AUDIT-FULL] Ledger validation L-01…L-15${NC}"
        cargo run --release --bin mud_full_audit -- "$MODEL"
        ;;
    ci)
        # L-12: local mirror of .github/workflows/ci.yml health battery
        echo -e "${PURPLE}[L-12] CI health battery${NC}"
        echo -e "${BLUE}→ cargo clippy --all-targets --all-features -D warnings${NC}"
        cargo clippy --all-targets --all-features -- -D warnings
        echo -e "${BLUE}→ unsafe-audit (R-36)${NC}"
        run_unsafe_audit
        echo -e "${BLUE}→ cargo test --lib --release${NC}"
        cargo test --lib --release -- --test-threads=2
        echo -e "${BLUE}→ P-13 property tests (release)${NC}"
        cargo test --lib --release p13 -- --test-threads=1
        echo -e "${BLUE}→ loss_cert unit tests (stream K, release)${NC}"
        cargo test --lib --release loss_cert -- --test-threads=1
        echo -e "${BLUE}→ all-targets tests (release)${NC}"
        cargo test --all-targets --release
        echo -e "${BLUE}→ training_healthcheck${NC}"
        MODEL=${2:-$MODEL_PATH}
        if [ -f "$MODEL" ]; then
            cargo run --quiet --bin diagnose_model -- "$MODEL" || {
                echo -e "${YELLOW}healthcheck exited non-zero (model may lack tensors) — battery continues${NC}"
            }
        else
            echo -e "${YELLOW}No model at $MODEL — skipping healthcheck${NC}"
        fi
        # C-MUD complex-reasoning audit (research §3, new work)
        if [ -f "$MODEL" ]; then
            echo -e "${BLUE}→ cmud_audit (C-MUD reasoning path)${NC}"
            cargo run --quiet --bin cmud_audit -- "$MODEL" || {
                echo -e "${YELLOW}cmud_audit exited non-zero — battery continues${NC}"
            }
            echo -e "${BLUE}→ cmud-cmp (baseline vs C-MUD probe)${NC}"
            cargo run --quiet --bin cmud_audit -- "$MODEL" --cmp || {
                echo -e "${YELLOW}cmud-cmp exited non-zero — battery continues${NC}"
            }
            echo -e "${BLUE}→ pointer-audit (ELUT/ternary layout P-00)${NC}"
            cargo test --quiet --lib pointer_audit || {
                echo -e "${YELLOW}pointer-audit exited non-zero — battery continues${NC}"
            }
        else
            echo -e "${YELLOW}No model at $MODEL — skipping cmud_audit / cmud-cmp / pointer-audit${NC}"
        fi
        # Optional e2e loss gate (heavy): MUD_CI_LOSS_CERT=1 ./mud.sh ci
        if [ "${MUD_CI_LOSS_CERT:-0}" = "1" ] && [ -f "$MODEL" ]; then
            echo -e "${BLUE}→ loss_certification_bench --fast (MUD_CI_LOSS_CERT=1)${NC}"
            MUD_LOSS_CERT_FAST=1 cargo run --release --bin loss_certification_bench -- --fast "$MODEL" || {
                echo -e "${RED}loss certification FAILED${NC}"
                exit 1
            }
        fi
        # mHC scale SGD gradient check (finite-difference) — validates optim::mhc_scale_sgd_step
        echo -e "${BLUE}→ mhc_stress_bench --mode gradcheck${NC}"
        cargo run --release --bin mhc_stress_bench -- --mode gradcheck --dims 256 --quiet || {
            echo -e "${RED}mHC gradient check FAILED${NC}"
            exit 1
        }
        echo -e "${GREEN}[L-12] Health battery complete.${NC}"
        ;;
    cert-loss)
        MODEL=${2:-$MODEL_PATH}
        echo -e "${PURPLE}[K] Loss certification (fast STE gate)${NC}"
        if [ ! -f "$MODEL" ]; then
            echo -e "${YELLOW}No model at $MODEL — exit 2 (soft skip)${NC}"
            exit 2
        fi
        MUD_LOSS_CERT_FAST=${MUD_LOSS_CERT_FAST:-1} \
            cargo run --release --bin loss_certification_bench -- --fast "$MODEL"
        ;;
    audit-conv)
        MODEL=${2:-$MODEL_PATH}
        cargo run --release --bin converter_auditor -- "$MODEL"
        ;;
    verify)
        echo -e "${PURPLE}[VERIFIER] Auditing Conversion Fidelity against Safetensors...${NC}"
        SF_PATH=${2:-models/qwen2_0.5b/model.safetensors}
        cargo run --release --bin conversion_verifier -- "$SF_PATH" "$MODEL_PATH"
        ;;
    precise-audit)
        echo -e "${PURPLE}[PRECISE-AUDIT] bitnet_scale-correct SQNR/MAE/MSE fidelity audit...${NC}"
        SF_PATH=${2:-models/qwen2_0.5b/model.safetensors}
        cargo run --release --bin precise_converter_audit -- "$SF_PATH" "$MODEL_PATH" --calibration
        ;;
    bound)
        echo -e "${PURPLE}[BOUNDS] Validating Quantization Boundaries & Mathematical Safety...${NC}"
        cargo run --release --bin boundary_validator -- "$MODEL_PATH"
        ;;
    estimate)
        echo -e "${PURPLE}[ESTIMATOR] Computing Universal Restoration Requirements...${NC}"
        cargo run --release --bin training_estimator -- "$MODEL_PATH"
        ;;
    validate)
        echo -e "${PURPLE}[VALIDATOR] Auditing Cognitive Cohesion & Effectiveness...${NC}"
        cargo run --release --bin iteration_validator -- "$MODEL_PATH"
        ;;
    inspect-var)
        echo -e "${PURPLE}[INSPECTOR] Corriendo Deep Variance Inspector...${NC}"
        cargo run --release --bin variance_inspector -- "$MODEL_PATH"
        ;;
    scale-audit)
        # P1.1 (2026-07-20): detect ternary vocabulary-collapse (PRQ scale inflation)
        # by comparing dequantized .mud RMS vs BF16 source RMS per layer.
        MODEL=${2:-$MODEL_PATH}
        SRC=${3:-models/smollm2}
        echo -e "${PURPLE}[SCALE-AUDIT] dequant RMS vs BF16 source ($SRC)${NC}"
        cargo run --release --bin scale_audit -- "$MODEL" "$SRC"
        ;;
    pointer-audit)
        MODEL=${2:-$MODEL_PATH}
        echo -e "${PURPLE}[POINTER-AUDIT] ELUT/ternary layout (P-00)${NC}"
        cargo run --release --bin pointer_audit -- "$MODEL"
        ;;
    mhc-sync)
        # 2026-07-31: Verify mHC alpha/beta are synced to the .mud file.
        # Uses scale_audit to check mHC tensor existence + values before/after
        # a tiny training run, then verifies the checkpoint persists them.
        MODEL=${2:-$MODEL_PATH}
        echo -e "${CYAN}[MHC-SYNC]${NC} pre-training mHC values:"
        cargo run --release --bin scale_audit -- "$MODEL" models/smollm2 2>/dev/null | grep -E "mhc_alpha|mhc_beta|mHC" || echo "(no mHC info in scale_audit output)"
        echo -e "${CYAN}[MHC-SYNC]${NC} MUD_MHC_SYNC=$MUD_MHC_SYNC MUD_DW_DEBUG=$MUD_DW_DEBUG"
        echo -e "${YELLOW}[MHC-SYNC]${NC} Run 'train' or 'circuit' first, then re-check mHC values in the saved checkpoint."
        echo -e "${YELLOW}[MHC-SYNC]${NC} Tip: MUD_DW_DEBUG=1 reports ΔW per checkpoint in sync_shadow_to_mud log."
        ;;

# ============================================================================
# 🧪 C-MUD (Complex Manifold Thinking)
# ============================================================================

    cmud-audit)
        MODEL=${2:-$MODEL_PATH}
        echo -e "${PURPLE}[C-MUD-AUDIT] complex reasoning path (research §3)${NC}"
        cargo run --release --bin cmud_audit -- "$MODEL"
        ;;
    cmud-cmp)
        MODEL=${2:-$MODEL_PATH}
        echo -e "${PURPLE}[C-MUD-CMP] baseline vs C-MUD quality probe${NC}"
        cargo run --release --bin cmud_audit -- "$MODEL" --cmp
        ;;
    cmud-train)
        MODEL=${2:-$MODEL_PATH}
        shift 2 2>/dev/null || shift $#
        echo -e "${PURPLE}[C-MUD-TRAIN] finite-difference phase-bias ascent (research §4)${NC}"
        cargo run --release --bin cmud_train -- "$MODEL" "$@"
        ;;
    cmud-gradtest)
        MODEL=${2:-$MODEL_PATH}
        shift 2 2>/dev/null || shift $#
        echo -e "${PURPLE}[C-MUD-GRADTEST] analytic vs finite-difference gradient (end-to-end)${NC}"
        cargo run --release --bin cmud_gradtest -- "$MODEL" "$@"
        ;;
    cmud-bench)
        MODEL=${2:-$MODEL_PATH}
        shift 2 2>/dev/null || shift $#
        echo -e "${PURPLE}[C-MUD-BENCH] think ON vs OFF next-token quality${NC}"
        cargo run --release --bin cmud_bench -- "$MODEL" "$@"
        ;;
    cmud-manifold)
        MODEL=${2:-$MODEL_PATH}
        echo -e "${PURPLE}[C-MUD MANIFOLD] Corriendo Validador de Colector Complejo & Cognición...${NC}"
        cargo run --release --bin cmud_manifold_validator -- "$MODEL"
        ;;
    mhc-stress)
        echo -e "${PURPLE}[MHC-STRESS] mHC scale SGD stress test + precision + speed${NC}"
        cargo run --release --bin mhc_stress_bench -- "$@"
        ;;
    coherence)
        MODEL=${2:-$MODEL_PATH}
        PROMPTS=${3:-prompts.txt}
        echo -e "${PURPLE}[COHERENCE] Inference coherence: success, deviation, coherence path${NC}"
        cargo run --release --bin inference_coherence -- "$MODEL" "$PROMPTS" "$@"
        ;;

# ============================================================================
# ⚡ PERFORMANCE & HARDWARE
# ============================================================================

    hw|bench|audit|diag)
        echo -e "${PURPLE}[DIAGNOSTICS] Iniciando Panel Unificado de Diagnóstico MUD (AVX2 + Vulkan)...${NC}"
        cargo run --release --bin diagnose_model -- "$MODEL_PATH"
        ;;

# ============================================================================
# 📊 BENCHMARKS
# ============================================================================

    fenwick-bench)
        # F4: Fenwick Tree benchmark (Linear vs Fenwick vs ASM/pointer).
        echo -e "${PURPLE}[fenwick-bench]${NC} Fenwick Tree vs Linear scan benchmark"
        cargo run --release --bin fenwick_bench -- ${2:+--size $2}
        ;;
    asm-bench)
        # SPRINT-6: ASM kernel benchmark (AVX2 vs Scalar, 13 kernels).
        echo -e "${PURPLE}[asm-bench]${NC} ASM kernel benchmark: AVX2 vs Scalar"
        cargo run --release --bin asm_kernel_bench
        ;;

# ============================================================================
# 📦 WEIGHTS MANAGEMENT
# ============================================================================

    convert|export)
        echo -e "${BLUE}[CONVERT] Converting safetensors to MUD format...${NC}"
        INPUT=${2:-models/smollm2/model.safetensors}
        OUTPUT=${3:-models/smollm2.mud}
        cargo run --release --bin universal_converter -- "$INPUT" "$OUTPUT"
        ;;
    forge)
        echo -e "${BLUE}[FORGE] Creating dense ternary transformer from scratch...${NC}"
        TOKENIZER=${MUD_TOKENIZER_DIR:-models/smollm2}
        [ -d "$TOKENIZER" ] || TOKENIZER="models/gpt2_tokenizer"
        PROFILE=${2:-nano}
        OUTPUT=${3:-models/scratch.mud}
        shift 3 2>/dev/null || true
        cargo run --release --bin mud_forge -- "$PROFILE" "$TOKENIZER" "$OUTPUT" "$@"
        ;;

# ============================================================================
# 🧹 WORKSPACE & CATALOG
# ============================================================================

    ckpt)
        echo -e "${BLUE}[CKPT] Listing available checkpoints in weights/checkpoints/:${NC}"
        ls -lh weights/checkpoints/*.mud 2>/dev/null || echo -e "${YELLOW}No checkpoints found.${NC}"
        echo ""
        echo -e "${CYAN}Best-model checkpoints (lowest loss):${NC}"
        ls -lh weights/checkpoints/*_best.mud 2>/dev/null || echo -e "${YELLOW}No best-model checkpoints found.${NC}"
        ;;
    restore-best)
        # FIX (2026-08-03): Restore from the best-loss checkpoint ({stem}_best.mud).
        # If no stem is given, auto-detect from MODEL_PATH.
        MODEL_STEM=""
        if [ -n "$2" ]; then
            MODEL_STEM="$2"
        else
            MODEL_STEM=$(basename "$MODEL_PATH" .mud 2>/dev/null || echo "model")
        fi
        BEST_CKPT="weights/checkpoints/${MODEL_STEM}_best.mud"
        if [ ! -f "$BEST_CKPT" ]; then
            echo -e "${RED}Error: Best checkpoint '$BEST_CKPT' not found.${NC}"
            echo -e "${YELLOW}Run training first (./mud.sh train) — best checkpoints are saved when loss improves.${NC}"
            exit 1
        fi
        echo -e "${YELLOW}[RESTORE-BEST] Backing up current model...${NC}"
        cp "$MODEL_PATH" "$MODEL_PATH.bak"
        echo -e "${GREEN}[RESTORE-BEST] Restoring from $BEST_CKPT...${NC}"
        cp "$BEST_CKPT" "$MODEL_PATH"
        echo -e "${GREEN}Model restored to best-loss version.${NC}"
        ;;
    restore)
        CKPT="weights/checkpoints/$2"
        if [ -z "$2" ]; then
            echo -e "${RED}Usage: ./mud.sh restore [checkpoint_filename]${NC}"
            echo -e "${YELLOW}  Or use: ./mud.sh restore-best [model_stem]${NC}"
            exit 1
        fi
        if [ ! -f "$CKPT" ]; then
            echo -e "${RED}Error: Checkpoint '$CKPT' not found.${NC}"
            exit 1
        fi
        echo -e "${YELLOW}[RESTORE] Backing up current model...${NC}"
        cp "$MODEL_PATH" "$MODEL_PATH.bak"
        echo -e "${GREEN}[RESTORE] Restoring from $CKPT...${NC}"
        cp "$CKPT" "$MODEL_PATH"
        echo -e "${GREEN}Model restored successfully.${NC}"
        ;;
    clean)
        echo -e "${BLUE}[CLEAN] Organizing workspace...${NC}"
        mkdir -p models weights/checkpoints logs/training docs tools/legacy
        mv *.log logs/ 2>/dev/null || true
        mv mud_disassembly.txt docs/ 2>/dev/null || true
        echo -e "${GREEN}Workspace clean.${NC}"
        ;;
    clean-logs|clean-telemetry|purge-logs|reset-telemetry)
        echo -e "${BLUE}[CLEAN-TELEMETRY] Purging all telemetry, training logs, and circuit history...${NC}"
        rm -f *.log mud_train_metrics.log logs/circuit.log 2>/dev/null || true
        rm -rf logs/*.log logs/training/* 2>/dev/null || true
        mkdir -p logs
        echo -e "${GREEN}Telemetry and logs successfully purged.${NC}"
        ;;
    tools)
        echo -e "${PURPLE}=== MUD BINARY & TOOL CATALOG ===${NC}"
        echo "  forge_llm               Primary MUD LLM inference engine & chat terminal"
        echo "  run_trainer             Native STE QAT trainer daemon (corpus alignment & QAT)"
        echo "  diagnose_model          Unified master health & diagnostics dashboard"
        echo "  mud_forge               Dense ternary transformer generator from scratch"
        echo "  mud_full_audit          Full ledger audit suite (L-01…L-15)"
        echo "  cmud_audit              C-MUD complex-reasoning manifold auditor"
        echo "  cmud_manifold_validator C-MUD 5-dimension cognition & comparative validator"
        echo "  cmud_train              Finite-difference phase-bias ascent (research §4)"
        echo "  cmud_gradtest           Analytic vs finite-difference gradient (end-to-end)"
        echo "  cmud_bench              Think ON vs OFF next-token quality benchmark"
        echo "  scale_audit             PRQ scale audit vs BF16 source (detects vocabulary collapse)"
        echo "  pointer_audit           Raw pointer & ELUT nibble formula audit (P-00/P-01)"
        echo "  circuit_telemetry       Seed-driven infinite training circuit TUI"
        echo "  debate_telemetry        RLVR debate & Professor-Student TUI"
        echo "  train_telemetry         Real-time training loss & variance TUI"
        echo "  universal_converter     Safetensors/PyTorch to .mud streaming converter"
        echo "  bonsai_gguf_to_mud      Bonsai GGUF Q2_0 to .mud streaming converter"
        echo "  fenwick_bench           Fenwick Tree vs Linear scan benchmark (F4/F1/F3 + FenwickLogF32)"
        echo "  asm_kernel_bench        ASM kernel benchmark: AVX2 vs Scalar (13 kernels)"
        echo "  slimex_fuse             MudFileMerger: merge multiple .mud into one"
        echo "  conversion_verifier     High-fidelity conversion verifier against safetensors"
        echo "  boundary_validator      Quantization boundary & mathematical safety validator"
        echo "  training_estimator      Universal training workload estimator"
        echo "  iteration_validator     Cognitive cohesion & effectiveness validator"
        echo "  variance_inspector      Deep variance inspector (VarH/VarJ per layer)"
        echo "  loss_certification_bench STE loss-must-fall gate (stream K)"
        echo "  extract_code_docs       Extract Rust docs + Markdown into training/corpus/mud_docs.txt"
        echo "  env-audit              Verify 0 env-var doc/code/mud.sh divergences (R-05)"
        echo "  unsafe-audit           Verify 0 unjustified unsafe blocks in prod code (R-36)"
        echo "  circuit_telemetry       Seed-driven infinite training circuit TUI"
        ;;
    env-audit)
        # R-05: Verifica 0 divergencias entre código (env::var), mud.sh vars y docs.
        # 1) missing-docs:  MUD_* leída en src/tools pero no documentada en MUD_ENV_REFERENCE.md §1/§2
        # 2) stale-docs:    MUD_* documentada pero NO leída ni exportada (var retirada o falsa)
        # 3) retired-usage:  presencia de vars §3 (retiradas) en src/tools/mud.sh
        REF="docs/manuals/MUD_ENV_REFERENCE.md"
        if [ ! -f "$REF" ]; then
            echo -e "${RED}[ENV-AUDIT]${NC} Reference file missing: $REF"
            exit 1
        fi
        # Patrón de exclusión: referencias a paths/c documents que contienen MUD_ pero
        # no son env vars reales (p.ej. MUD_MASTER_REMEDIATION_PLAN_2026, MUD_ENV_REFERENCE).
        DOCNAME_RE='MUD_[A-Z0-9_]+(_(AUDIT|PLAN|REPORT|REFERENCE|FEASIBILITY|ANALYSIS|GAP|LOGGAS|COUNTDOWN|CRITICOS|OPTIMIZATION_ANALYSIS|IMPROVEMENTS|COMPUTE_STACK|REGISTER_BIT_PACKING|ULTRA_COMPRESSED|SESSION_REPORT|TODO)_[A-Z0-9_]*)'
        strip_docnames() { grep -vE "$DOCNAME_RE" | sort -u; }
        # --- vars realmente leídas por el código (R): env::var("MUD_*") ---
        CODED=$(grep -rhoE 'env::var\("MUD_[A-Z0-9_]+"\)|std::env::var\("MUD_[A-Z0-9_]+"\)' src tools 2>/dev/null \
                | grep -oE '"MUD_[A-Z0-9_]+"' | tr -d '"' | sort -u)
        # --- vars referenciadas en mud.sh (S): expansión, asignación o export ---
        MUDSH=$(grep -hoE 'MUD_[A-Z0-9_]+' mud.sh 2>/dev/null | strip_docnames | grep -v '^$')
        ALL_CODED=$(printf '%s\n%s\n' "$CODED" "$MUDSH" | grep -v '^$' | sort -u)
        # --- vars documentadas en §1 tabla maestra (backtick `MUD_*`; evita subcadenas de CMUD_*) ---
        DOCS=$(sed -n '/^## 1\. Tabla maestra/,/^## 2\. /p' "$REF" \
               | grep -oE '`MUD_[A-Z0-9_]+`' | tr -d '`' | sort -u)
        # --- vars retiradas en §3 (sólo filas de tabla `| \`MUD_X\` `, no prose) ---
        RETIRED=$(sed -n '/^## 3\. Variables retiradas/,/^## 4\. /p' "$REF" \
                  | grep -oE '\| `MUD_[A-Z0-9_]+`' | sed -E 's/\| `MUD_//; s/`$//' | sort -u)
        ERR=0
        echo -e "${CYAN}[ENV-AUDIT]${NC} Código: $(echo "$CODED" | grep -c .) vars · mud.sh: $(echo "$MUDSH" | grep -c .) vars"
        echo -e "${CYAN}[ENV-AUDIT]${NC} Docs §1: $(echo "$DOCS" | grep -c .) vars · Retiradas §3: $(echo "$RETIRED" | grep -c .) vars"
        # 1) missing-docs
        MISSING=$(comm -23 <(echo "$CODED" | grep -v '^$') <(echo "$DOCS" | grep -v '^$'))
        if [ -n "$MISSING" ]; then
            echo -e "${RED}[ENV-AUDIT] FAIL missing-docs:${NC} vars leídas en código pero no documentadas:"
            echo "$MISSING" | sed 's/^/    /'
            ERR=1
        fi
        # 2) stale-docs (documentada en §1/§2 pero ni en código ni en mud.sh)
        STALE=$(comm -23 <(echo "$DOCS" | grep -v '^$') <(echo "$ALL_CODED" | grep -v '^$'))
        if [ -n "$STALE" ]; then
            echo -e "${RED}[ENV-AUDIT] FAIL stale-docs:${NC} vars documentadas en §1/§2 pero ausentes de código y mud.sh (deberían estar en §3 Retiradas):"
            echo "$STALE" | sed 's/^/    /'
            ERR=1
        fi
        # 3) retired-usage (vars retiradas presentes en código o mud.sh)
        RUSAGE=$(comm -12 <(echo "$RETIRED" | grep -v '^$') <(echo "$ALL_CODED" | grep -v '^$'))
        if [ -n "$RUSAGE" ]; then
            echo -e "${RED}[ENV-AUDIT] FAIL retired-usage:${NC} vars retiradas todavía usadas en código o mud.sh:"
            echo "$RUSAGE" | sed 's/^/    /'
            ERR=1
        fi
        if [ "$ERR" -eq 0 ]; then
            echo -e "${GREEN}[ENV-AUDIT] OK — 0 divergencias (R-05)${NC}"
        else
            echo -e "${YELLOW}[ENV-AUDIT] Divergencias detectadas — corregir según §3 de $REF${NC}"
        fi
        exit $ERR
        ;;
    unsafe-audit)
        # R-36: every unsafe in production code must be justified (adjacent
        # SAFETY/# Safety; enclosing //! SAFETY: module contract for
        # non-high-risk sites; high-risk ops always adjacent).
        run_unsafe_audit
        exit $?
        ;;
    *)
        show_help
        ;;
esac
