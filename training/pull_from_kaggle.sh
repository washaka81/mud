#!/bin/bash

echo "=== MUD Kaggle Model Retrieval Tool (Ultra Trainer) ==="

# --- CONFIGURATION ---
# To protect your privacy, set these in a local file named 'kaggle_config.sh'
# or as environment variables. DO NOT commit the config file.
CONFIG_FILE="$(dirname "$0")/kaggle_config.sh"

if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
fi

# Fallback to current values if not set in config
KAG_USER="${KAG_USER:-YOUR_KAGGLE_USERNAME}"
KAG_KERNEL="${KAG_KERNEL:-YOUR_KAGGLE_KERNEL_NAME}"
KERNEL_ID="$KAG_USER/$KAG_KERNEL"
DEST_DIR="models/"

# 2. Ensure models directory exists
mkdir -p "$DEST_DIR"

# 3. Download outputs from Kaggle
echo "Attempting to download trained files from $KERNEL_ID..."
TEMP_DL="models/tmp_kaggle_dl"
mkdir -p "$TEMP_DL"

kaggle kernels output "$KERNEL_ID" -p "$TEMP_DL"

if [ $? -eq 0 ]; then
    echo "Processing downloaded files..."
    mv "$TEMP_DL"/*.mud models/ 2>/dev/null || true
    mv "$TEMP_DL"/*.pt weights/ 2>/dev/null || true
    rm -rf "$TEMP_DL"
    echo "Success! Files distributed to models/ and weights/"
else
    echo "Error: Failed to retrieve output."
fi
