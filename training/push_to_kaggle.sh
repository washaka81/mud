#!/bin/bash

echo "=== MUD Kaggle Deployment Tool (v35: High-Speed) ==="

# --- CONFIGURATION ---
CONFIG_FILE="$(dirname "$0")/kaggle_config.sh"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
fi

KAG_USER="${KAG_USER:-alejandrofonda}"
KAG_KERNEL="${KAG_KERNEL:-mud-ternary-moe-training-es-en}"

# 1. Prepare Dataset and Vocab
echo "  [1/3] Preparing training assets..."
cp training/vocab_es_en.txt training/vocab_es_en.txt 2>/dev/null
cp training/synthetic_knowledge.txt training/synthetic_knowledge.txt 2>/dev/null

# 2. Safety Check
if [ ! -f "training/synthetic_knowledge.txt" ]; then
    echo "❌ ERROR: synthetic_knowledge.txt not found in training/ directory!"
    echo "Run 'python3 tools/dreamer.py' first."
    exit 1
fi

# 3. Push to Kaggle
echo "  [2/3] Pushing Logic Training Kernel to Kaggle (GPU enabled)..."
kaggle kernels push -p training/

echo "  [3/3] Deployment complete."
echo ""
echo "🚀 MONITOR PROGRESS AT:"
echo "https://www.kaggle.com/code/$KAG_USER/$KAG_KERNEL"
echo ""
echo "Once finished (COMPLETED), run:"
echo "  ./training/pull_from_kaggle.sh"
