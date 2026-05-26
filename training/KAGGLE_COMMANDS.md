---
lang: en
---

# MUD Training: Kaggle Setup Guide

Paste the following blocks into separate cells in your Kaggle Notebook (set to GPU P100 or T4/L4).

## Cell 1: Environment Installation
```bash
pip install torch datasets transformers
```

## Cell 2: MUD Training Execution
```python
import os

# The training notebook is at training/notebook*.ipynb
# Upload it to Kaggle and execute sequentially.

# For manual training, run the Rust-native pipeline locally:
#   ./mud.sh train
# For cloud sync:
#   bash training/push_to_kaggle.sh   # upload dataset
#   bash training/pull_from_kaggle.sh # download trained model
```

## Cell 3: Export to Local / Drive
```python
# Save the resulting .mud file for your local Forge LLM engine
!cp models/core_skills.mud /kaggle/working/
print("Training Complete. Download core_skills.mud from 'Output' section.")
```

## Optimization Tips for Kaggle:
1. **GPU Selection:** Use the **T4** or **P100** accelerators.
2. **Persistence:** The `.mud` file will appear in `/kaggle/working/`.
3. **Dataset:** Sync your local `knowledge.db` content via `push_to_kaggle.sh` before starting the notebook.
