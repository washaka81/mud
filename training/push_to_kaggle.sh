#!/bin/bash

echo "=== MUD Kaggle Deployment Tool ==="
echo "Pushing Logic Training Kernel to Kaggle..."

# --- CONFIGURATION ---
CONFIG_FILE="$(dirname "$0")/kaggle_config.sh"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
fi

KAG_USER="${KAG_USER:-alejandrofonda}"
KAG_KERNEL="${KAG_KERNEL:-mud-ternary-moe-training-es-en}"

# Ensure vocab file exists
if [ ! -f "training/vocab_es_en.txt" ]; then
    if [ -f "vocab_es_en.txt" ]; then
        cp vocab_es_en.txt training/vocab_es_en.txt
    else
        echo "Warning: vocab_es_en.txt not found. Using built-in fallback."
    fi
fi

# Push kernel to Kaggle (GPU enabled, internet for dependencies)
kaggle kernels push -p training/

echo ""
echo "Deployment complete. Monitor at:"
echo "https://www.kaggle.com/code/$KAG_USER/$KAG_KERNEL"
echo ""
echo "After training finishes, retrieve model with:"
echo "  ./training/pull_from_kaggle.sh"
