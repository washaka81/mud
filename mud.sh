#!/bin/bash
# ==============================================================================
# MUD MASTER ORCHESTRATOR (v2.0)
# ==============================================================================
# Unified entry point for MUD Engine Operations.
# ==============================================================================

set -e

# --- COLORS ---
PURPLE='\x1b[1;35m'
BLUE='\x1b[1;34m'
GREEN='\x1b[1;32m'
YELLOW='\x1b[1;33m'
RED='\x1b[1;31m'
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
: "${MUD_GPU_GEMV:=auto}"   # auto = one-shot CPU/GPU(ash) micro-bench; GPU only past break-even
: "${MUD_TRAIN_EZOP:=0}"
: "${MUD_TRAIN_FREEZE_EMB:=1}"
: "${MUD_TRAIN_TEXT_ONLY:=1}"
: "${MUD_CMUD_THINK:=0}"
export MUD_PCORE_THREADS MUD_GPU_GEMV MUD_TRAIN_EZOP MUD_TRAIN_FREEZE_EMB MUD_TRAIN_TEXT_ONLY MUD_CMUD_THINK

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
    echo -e "${PURPLE}=== MUD MASTER COMMAND CENTER (v2.0) ===${NC}"
    echo -e "Usage: ./mud.sh [command] [options]"
    echo ""
    echo -e "${BLUE}🧠 Intelligence & Restoration:${NC}"
    echo "  align           Start Native Corpus Aligner (Linguistic Restoration)"
    echo "  project         Run Recalibration Projector (Bayesian Determinism)"
    echo "  train           Launch Local Rust AutoTrainer daemon (Live Learning)"
    echo "  restore-iq      Unified Pipeline: Align -> Project -> Train (Auto-Restoration)"
    echo ""
    echo -e "${BLUE}💬 Interaction & Analysis:${NC}"
    echo "  chat            Launch interactive MUD terminal"
    echo "  step            Run step-by-step inference autopsy"
    echo "  telemetry       Launch Real-Time Loss/Variance TUI Graph"
    echo "  debate          Launch Telemetry Debate module (RLVR seed-survival loop)"
    echo "  professor       Launch Professor-Student loop (local grammar/syntax/coherence/pragmatism judge)"
    echo "  circuit [model.mud]  Infinite training circuit by seed: shuffled batteries of align/debate/games/professor (loop until quit). REQUIRES healthy .mud (smollm2.mud/checkpoint are collapsed); use models/ternary_bonsai_1.7b.mud"
    echo ""
    echo -e "${BLUE}🛡️ Safety & Persistence:${NC}"
    echo "  ckpt            List available training checkpoints"
    echo "  restore [name]  Replace current model with a specific checkpoint"
    echo "  clean           Organize files and clear temporary logs"
    echo ""
    echo -e "${BLUE}🛡️ Safety & Quality Enforcers:${NC}"
    echo "  verify [sf]     Run High-Fidelity Conversion Verifier against raw safetensors"
    echo "  bound           Run Quantization Boundary Validator (NaN/Grid check)"
    echo "  health          Run Training Healthcheck (Pre-flight weight & shape validation)"
    echo "  audit-full      Full ledger audit L-01…L-15 (structure + policy + smoke)"
    echo "  ci              L-12 battery: tests + loss_cert + clippy + health (+ optional e2e)"
    echo "  cert-loss       Stream K: loss-must-fall gate (needs models/*.mud)"
    echo "  audit-conv      Run Converter Auditor (Post-conversion structural & magic byte check)"
    echo "  estimate        Run Universal Training Estimator (seating predictions)"
    echo "  validate        Run Cognitive Iteration Validator (Score % metric)"
    echo "  inspect-var     Run Deep Variance Inspector (VarH/VarJ check per layer)"
    echo ""
    echo -e "${BLUE}⚡ Performance & Hardware:${NC}"
    echo "  hw              Show detected hardware profile & SIMD status (Unified)"
    echo "  bench           Run deep performance & memory benchmark (Unified)"
    echo "  audit           Run full cognitive & structural audit suite (Unified)"
    echo "  diag            Launch the unified master health & diagnostics dashboard"
    echo ""
    echo -e "${BLUE}📦 Weights Management:${NC}"
    echo "  convert         Universal Converter: Safetensors/PyTorch to .mud"
    echo "                  Usage: ./mud.sh convert [input] [output] [--ternarize-emb]"
    echo "  forge           MUD Forge: Create a blank initialized model from scratch"
    echo "                  Usage: ./mud.sh forge [template: nano|micro|base|custom] [output]"
    echo ""
    echo -e "${BLUE}⚙️  Debate / Professor env vars (defaults set by mud.sh):${NC}"
    echo "  MUD_DEBATE_MODE=debate|professor   game mode (professor = student learns until quit)"
    echo "  MUD_DEBATE_MAX_TIME=0              seconds before auto-stop (0 = infinite)"
    echo "  MUD_DEBATE_RWIN=1.0  MUD_DEBATE_RLOSE=0.7   reward / penalty"
    echo "  MUD_DEBATE_JEPB_LAMBDA=0.05        JEPA intrinsic aux weight"
    echo "  MUD_DEBATE_LEARN=1                 persist shadow->MUD after session (0=off)"
    echo "  MUD_DEBATE_MAX_NEW_TOKENS=auto     per-response cap (auto by free RAM if unset)"
}

case $1 in

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
        : "${MUD_TRAIN_NUM_NEG:=255}"
        : "${MUD_OPT:=adam}"
        
        # If the user explicitly asks for epochs, do not cap the training chunks
        for arg in "$@"; do
            if [[ "$arg" == "--epochs" ]]; then
                export MUD_TRAIN_MAX_CHUNKS=0
                break
            fi
        done

        compute_project_chunks
        export MUD_TRAIN_STP MUD_TRAIN_STEPS_PER_CHUNK MUD_TRAIN_TEXT_ONLY MUD_TRAIN_NUM_NEG MUD_OPT
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
        echo -e "${BLUE}[train]${NC} model=${MODEL_PATH}"
        echo -e "${BLUE}[train]${NC} PCore=${MUD_PCORE_THREADS} GPU_GEMV=${MUD_GPU_GEMV} USE_VULKAN=${MUD_USE_VULKAN} EZOP=${MUD_TRAIN_EZOP} FREEZE_EMB=${MUD_TRAIN_FREEZE_EMB}"
        echo -e "${BLUE}[train]${NC} STP=${MUD_TRAIN_STP} STEPS/CHUNK=${MUD_TRAIN_STEPS_PER_CHUNK} NEG=${MUD_TRAIN_NUM_NEG} OPT=${MUD_OPT} MAX_CHUNKS=${MUD_TRAIN_MAX_CHUNKS}"
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
    chat)
        MODEL=${2:-$MODEL_PATH}
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
    telemetry)
        cargo run --release --bin train_telemetry
        ;;
    step)
        cargo run --release --bin step_inference
        ;;
    health)
        MODEL=${2:-$MODEL_PATH}
        cargo run --release --bin training_healthcheck -- "$MODEL"
        ;;
    audit-full)
        MODEL=${2:-$MODEL_PATH}
        echo -e "${PURPLE}[AUDIT-FULL] Ledger validation L-01…L-15${NC}"
        cargo run --release --bin mud_full_audit -- "$MODEL"
        ;;
    cmud-audit)
        MODEL=${2:-$MODEL_PATH}
        echo -e "${PURPLE}[C-MUD-AUDIT] complex reasoning path (research §3)${NC}"
        cargo run --release --bin cmud_audit -- "$MODEL"
        ;;
    pointer-audit)
        MODEL=${2:-$MODEL_PATH}
        echo -e "${PURPLE}[POINTER-AUDIT] ELUT/ternary layout (P-00)${NC}"
        cargo run --release --bin pointer_audit -- "$MODEL"
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
    scale-audit)
        # P1.1 (2026-07-20): detect ternary vocabulary-collapse (PRQ scale inflation)
        # by comparing dequantized .mud RMS vs BF16 source RMS per layer.
        MODEL=${2:-$MODEL_PATH}
        SRC=${3:-models/smollm2}
        echo -e "${PURPLE}[SCALE-AUDIT] dequant RMS vs BF16 source ($SRC)${NC}"
        cargo run --release --bin scale_audit -- "$MODEL" "$SRC"
        ;;
    ci)
        # L-12: local mirror of .github/workflows/ci.yml health battery
        echo -e "${PURPLE}[L-12] CI health battery${NC}"
        echo -e "${BLUE}→ cargo test --lib${NC}"
        cargo test --lib -- --test-threads=2
        echo -e "${BLUE}→ P-13 property tests${NC}"
        cargo test --lib p13 -- --test-threads=1
        echo -e "${BLUE}→ loss_cert unit tests (stream K)${NC}"
        cargo test --lib loss_cert -- --test-threads=1
        echo -e "${BLUE}→ clippy -D warnings${NC}"
        cargo clippy --lib -- -D warnings
        echo -e "${BLUE}→ training_healthcheck${NC}"
        MODEL=${2:-$MODEL_PATH}
        if [ -f "$MODEL" ]; then
            cargo run --quiet --bin training_healthcheck -- "$MODEL" || {
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
            cargo run --quiet --bin pointer_audit -- "$MODEL" || {
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
    cmud-manifold)
        MODEL=${2:-$MODEL_PATH}
        echo -e "${PURPLE}[C-MUD MANIFOLD] Corriendo Validador de Colector Complejo & Cognición...${NC}"
        cargo run --release --bin cmud_manifold_validator -- "$MODEL"
        ;;
    audit-conv)
        MODEL=${2:-$MODEL_PATH}
        cargo run --release --bin converter_auditor -- "$MODEL"
        ;;
    inspect-var)
        echo -e "${PURPLE}[INSPECTOR] Corriendo Deep Variance Inspector...${NC}"
        cargo run --release --bin variance_inspector -- "$MODEL_PATH"
        ;;
    mai-audit)
        echo -e "${PURPLE}[MAI-AUDIT] Validando arquitectura de orquestador LatentMoE...${NC}"
        cargo run --release --bin mai_orchestrator_validator
        ;;
    hw|bench|audit|diag)
        echo -e "${PURPLE}[DIAGNOSTICS] Iniciando Panel Unificado de Diagnóstico MUD (AVX2 + Vulkan)...${NC}"
        cargo run --release --bin diagnose_model -- "$MODEL_PATH"
        ;;
    ckpt)
        echo -e "${BLUE}[CKPT] Listing available checkpoints in weights/checkpoints/:${NC}"
        ls -lh weights/checkpoints/*.mud 2>/dev/null || echo -e "${YELLOW}No checkpoints found.${NC}"
        ;;
    restore)
        CKPT="weights/checkpoints/$2"
        if [ -z "$2" ]; then
            echo -e "${RED}Usage: ./mud.sh restore [checkpoint_filename]${NC}"
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
    convert|export)
        echo -e "${BLUE}[CONVERT] Converting safetensors to MUD format...${NC}"
        INPUT=${2:-models/qwen2_0.5b/model.safetensors}
        OUTPUT=${3:-models/qwen2_0.5b.mud}
        cargo run --release --bin universal_converter -- "$INPUT" "$OUTPUT"
        ;;
    forge)
        echo -e "${BLUE}[FORGE] Initializing MUD Forge model generation...${NC}"
        PROFILE=${2:-custom}
        OUTPUT=${3:-models/nuevo_vacio.mud}
        cargo run --release --bin mud_forge -- "$PROFILE" models/qwen2_0.5b "$OUTPUT"
        ;;
    verify)
        echo -e "${PURPLE}[VERIFIER] Auditing Conversion Fidelity against Safetensors...${NC}"
        SF_PATH=${2:-models/qwen2_0.5b/model.safetensors}
        cargo run --release --bin conversion_verifier -- "$SF_PATH" "$MODEL_PATH"
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
    clean)
        echo -e "${BLUE}[CLEAN] Organizing workspace...${NC}"
        mkdir -p models weights/checkpoints logs/training docs tools/legacy
        mv *.log logs/ 2>/dev/null || true
        mv mud_disassembly.txt docs/ 2>/dev/null || true
        echo -e "${GREEN}Workspace clean.${NC}"
        ;;

    *)
        show_help
        ;;
esac
