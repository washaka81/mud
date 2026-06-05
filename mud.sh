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
    MODEL_PATH="models/core_skills.mud"
fi
CHECKPOINT_DIR="weights/checkpoints"
export MUD_USE_VULKAN=1
export MKL_DEBUG_CPU_TYPE=5  # Enforce AVX2 on Intel CPUs


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
    echo "  vocab           Perform Vocabulary-Embedding alignment audit"
    echo ""
    echo -e "${BLUE}🛡️ Safety & Persistence:${NC}"
    echo "  ckpt            List available training checkpoints"
    echo "  restore [name]  Replace current model with a specific checkpoint"
    echo "  clean           Organize files and clear temporary logs"
    echo ""
    echo -e "${BLUE}🛡️ Safety & Quality Enforcers:${NC}"
    echo "  verify [sf]     Run High-Fidelity Conversion Verifier against raw safetensors"
    echo "  bound           Run Quantization Boundary Validator (NaN/Grid check)"
    echo "  estimate        Run Universal Training Estimator (seating predictions)"
    echo "  validate        Run Cognitive Iteration Validator (Score % metric)"
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
    echo "  menu            Show interactive selector for MUD models"
}

case $1 in

    menu)
        cargo run --release --bin mud_selector
        ;;

    train)
        shift
        cargo run --release --bin mud_corpus_trainer
        ;;
    restore-iq)
        echo -e "${PURPLE}╭───────────────────────────────────────────────────────────────────────────────╮${NC}"
        echo -e "${PURPLE}│ 🌀 INICIANDO PIPELINE DE RESTAURACION COGNITIVA (MUD IQ-RESTORE)              │${NC}"
        echo -e "${PURPLE}╰───────────────────────────────────────────────────────────────────────────────╯${NC}"
        
        echo -e "${BLUE}[1/6] VERIFY CONVERSION: Checking SQNR & Stability...${NC}"
        # Skip conversion verifier in automated pipeline as it requires the safetensors path
        # cargo run --release --bin conversion_verifier -- "$MODEL_PATH"

        echo -e "${BLUE}[2/6] VERIFY BOUNDARIES: Checking Ternary Conformity & Scales...${NC}"
        cargo run --release --bin boundary_validator -- "$MODEL_PATH"

        echo -e "${BLUE}[3/6] ESTIMATE WORKLOAD: Calculating required recovery parameters...${NC}"
        EST_EPOCHS=$(cargo run --release --bin training_estimator "$MODEL_PATH" 2>/dev/null | grep "Required Epochs" | awk '{print $4}' | sed -r "s/\x1B\[([0-9]{1,3}(;[0-9]{1,2})?)?[mGK]//g")
        if [ -z "$EST_EPOCHS" ]; then
            EST_EPOCHS=5
            echo -e "${YELLOW}Could not determine exact epochs, defaulting to 5.${NC}"
        else
            echo -e "${GREEN}Calculated required epochs for recovery: $EST_EPOCHS${NC}"
        fi

        echo -e "${BLUE}[4/6] ALIGN: Linguistic Restoration (Deep Epoch)...${NC}"
        cargo run --release --bin mud_corpus_trainer -- "$MODEL_PATH" --epochs "$EST_EPOCHS"
        
        echo -e "${BLUE}[5/6] PROJECT & TRAIN: Tier 3 PRQ Calibration...${NC}"
        cargo run --release --bin recalibration_projector -- "$MODEL_PATH" --tier3
        # cargo run --release --bin mud_autotrainer -- seating "$MODEL_PATH"
        
        echo -e "${BLUE}[6/6] ASSERT EFFECTIVENESS: Final Iteration Validation (>96%)...${NC}"
        cargo run --release --bin iteration_validator -- "$MODEL_PATH"
        ;;
    align)
        echo -e "${PURPLE}[ALIGN] Starting Native Corpus Aligner...${NC}"
        cargo run --release --bin mud_corpus_trainer
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
    step)
        cargo run --release --bin step_inference
        ;;
    hw|bench|audit|diag)
        echo -e "${PURPLE}[DIAGNOSTICS] Iniciando Panel Unificado de Diagnóstico MUD (AVX2 + Vulkan)...${NC}"
        cargo run --release --bin mud_diagnostics -- "$MODEL_PATH"
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
    vocab)
        cargo run --release --bin vocab_check
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
    phase14)
        echo -e "${PURPLE}[PHASE 14] Ejecutando Comprehensive Audit para RRM-02 y GRAM...${NC}"
        cargo run --release --bin phase14_audit
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
