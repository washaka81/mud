#!/bin/bash

echo "=== MUD Kaggle Model Retrieval Tool ==="

# --- CONFIGURATION ---
# To protect your privacy, set these in a local file named 'kaggle_config.sh'
# or as environment variables. DO NOT commit the config file.
CONFIG_FILE="$(dirname "$0")/kaggle_config.sh"

if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
fi

# Fallback to current values if not set in config
KAG_USER="${KAG_USER:-alejandrofonda}"
KAG_KERNEL="${KAG_KERNEL:-mud-ternary-moe-training-es-en}"
KERNEL_ID="$KAG_USER/$KAG_KERNEL"
DEST_DIR="models/"

# 2. Ensure models directory exists
mkdir -p "$DEST_DIR"

# 3. Download outputs from Kaggle
echo "Attempting to download trained .ai files from $KERNEL_ID..."
# Ensure kaggle command is available
if ! command -v kaggle &> /dev/null; then
    echo "Error: Kaggle CLI not found. Please install it and configure your API key."
    exit 1
fi

kaggle kernels output "$KERNEL_ID" -p "$DEST_DIR"

if [ $? -eq 0 ]; then
    echo "Success! Your model should now be in $DEST_DIR"
    ls -lh "$DEST_DIR"/*.ai 2>/dev/null || echo "Note: No .ai files found yet. The training might still be in progress."
else
    echo "Error: Failed to retrieve output. Make sure the kernel is finished and public/private access is correct."
fi
