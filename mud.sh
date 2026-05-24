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
    echo -e "${PURPLE}=== MUD COMMAND CENTER ===${NC}"
    echo -e "Usage: ./mud.sh [command] [options]"
    echo ""
    echo -e "${BLUE}Core Commands:${NC}"
    echo "  train           Launch/Resume V1-MASTER training"
    echo "  train --kaggle  Dispatch training to Kaggle cloud"
    echo "  colab           Launch training on Google Colab (GPU T4/A100)"
    echo "  chat            Launch interactive MUD terminal"
    echo "  step            Run step-by-step inference analysis"
    echo ""
    echo -e "${BLUE}Optimization & Audit:${NC}"
    echo "  bench           Run performance & memory benchmark"
    echo "  iq              Calculate current Digital IQ Score"
    echo "  audit           Run full cognitive & structural audit suite"
    echo "  clean           Organize files and clear temporary logs"
    echo ""
    echo -e "${BLUE}Weights Management:${NC}"
    echo "  export          Convert latest PyTorch checkpoint to .mud"
    echo "  pull            Pull latest weights from Kaggle"
    echo "  test-handoff    Run Kaggle->Local synchronization test"
}

case $1 in
    profile)
        echo -e "${BLUE}[PROFILE] Analizando hardware...${NC}"
        python3 tools/hardware_profiler.py
        ;;
    train)
        shift
        ./scripts/train_master.sh "$@"
        ;;
    chat)
        cargo run --release --bin forge_llm
        ;;
    step)
        cargo run --release --bin step_inference
        ;;
    bench)
        cargo run --release --bin memory_benchmark
        ;;
    iq)
        python3 tools/cognitive_dashboard.py
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
    export)
        echo -e "${BLUE}[EXPORT] Converting latest checkpoint...${NC}"
        python3 tools/export_checkpoint.py
        ;;
    clean)
        echo -e "${BLUE}[CLEAN] Organizing workspace...${NC}"
        mkdir -p models weights/checkpoints logs/training docs tools/legacy
        mv *.log logs/ 2>/dev/null || true
        mv mud_disassembly.txt docs/ 2>/dev/null || true
        echo -e "${GREEN}Workspace clean.${NC}"
        ;;
    pull)
        ./training/pull_from_kaggle.sh
        ;;
    colab)
        shift
        echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║     MUD SLIME ENGINE — GOOGLE COLAB LAUNCHER                        ║${NC}"
        echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
        echo -e "${BLUE}  ℹ️  Ejecutar este comando DENTRO de una celda de Google Colab${NC}"
        echo ""
        python3 training/google_colab_trainer.py "$@"
        ;;
    test-handoff)
        echo -e "${BLUE}[TEST] Running Kaggle->Local Handoff Test...${NC}"
        bash test_handoff.sh
        ;;
    *)
        show_help
        ;;
esac
