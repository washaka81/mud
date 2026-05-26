#!/bin/bash
# ==============================================================================
# MUD MASTER ORCHESTRATOR (v1.1)
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
MODEL_PATH="models/core_skills.mud"
CHECKPOINT_DIR="weights/checkpoints"
export MUD_USE_VULKAN=1
export MKL_DEBUG_CPU_TYPE=5  # Enforce AVX2 on Intel CPUs


show_help() {
    echo -e "${PURPLE}=== MUD MASTER COMMAND CENTER (v1.2) ===${NC}"
    echo -e "Usage: ./mud.sh [command] [options]"
    echo ""
    echo -e "${BLUE}🧠 Intelligence & Restoration:${NC}"
    echo "  align           Start Native Corpus Aligner (Linguistic Restoration)"
    echo "  project         Run Recalibration Projector (Bayesian Determinism)"
    echo "  train           Launch Local Rust AutoTrainer daemon (Live Learning)"
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
    echo -e "${BLUE}⚡ Performance & Hardware:${NC}"
    echo "  hw              Show detected hardware profile & SIMD status"
    echo "  bench           Run deep performance & memory benchmark"
    echo "  audit           Run full cognitive & structural audit suite"
    echo ""
    echo -e "${BLUE}📦 Weights Management:${NC}"
    echo "  convert         Universal Converter: Safetensors/PyTorch to .mud"
    echo "                  Usage: ./mud.sh convert [input] [output] [--ternarize-emb]"
}

case $1 in
    profile|iq|colab)
        echo -e "${RED}[MIGRATION] The Python ecosystem has been completely purged.${NC}"
        ;;
    train)
        shift
        cargo run --release --bin mud_autotrainer
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
        cargo run --release --bin forge_llm
        ;;
    step)
        cargo run --release --bin step_inference
        ;;
    hw)
        cargo run --release --bin hw_detect
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
    bench)
        cargo run --release --bin memory_benchmark
        ;;
    vocab)
        cargo run --release --bin vocab_check
        ;;
    audit)
        echo -e "${YELLOW}[AUDIT] Executing Full MUD Suite...${NC}"
        RUSTFLAGS="-C target-cpu=native" cargo build --release --quiet
        TOOLS=("tokenizer_audit" "weight_audit" "moe_audit" "truth_auditor" "deep_math_audit")
        for tool in "${TOOLS[@]}"; do
            echo -e "${BLUE}> Running $tool...${NC}"
            ./target/release/"$tool" "$MODEL_PATH" || echo "  ⚠️  $tool failed."
        done
        ;;
    convert|export)
        echo -e "${BLUE}[CONVERT] Converting safetensors to MUD format...${NC}"
        INPUT=${2:-models/mud_fast_ckpt.safetensors}
        OUTPUT=${3:-models/core_skills.mud}
        cargo run --release --bin universal_converter -- "$INPUT" "$OUTPUT"
        ;;
    clean)
        echo -e "${BLUE}[CLEAN] Organizing workspace...${NC}"
        mkdir -p models weights/checkpoints logs/training docs tools/legacy
        mv *.log logs/ 2>/dev/null || true
        mv mud_disassembly.txt docs/ 2>/dev/null || true
        echo -e "${GREEN}Workspace clean.${NC}"
        ;;
    pull)
        echo -e "${RED}[MIGRATION] Bash pull script pending Rust network module integration.${NC}"
        ;;
    test-handoff)
        echo -e "${RED}[MIGRATION] Handoff test pending Rust network module integration.${NC}"
        ;;
    *)
        show_help
        ;;
esac
