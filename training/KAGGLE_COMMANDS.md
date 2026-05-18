# MUD Training: Kaggle & Trainer Setup Guide

Paste the following blocks into separate cells in your Kaggle Notebook (set to GPU P100 or T4/L4).

## Cell 1: Environment Installation
```bash
# Install Trainer and dependencies for fast training
pip install --no-deps "Trainer[colab-new] @ git+https://github.com/Trainerai/Trainer.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
pip install datasets transformers anyhow numpy
```

## Cell 2: MUD Training Execution
```python
import os

# 1. Download/Create the MUD environment in Kaggle
# (Assuming files are uploaded or created via script)
# !git clone <your_repo_if_public> or use the code we generated:

# Execute the trainer we developed
# Note: Trainer works best with MUD/Logic bases, 
# our MudMoE is a custom ternary implementation.
# We will use standard PyTorch on Kaggle but optimized via bitsandbytes.

os.system("python training/mud_language_trainer.py")
```

## Cell 3: Export to Local / Drive
```python
# Save the resulting .mud file for your local Forge LLM engine
from google.colab import files # If in Colab
# In Kaggle:
!cp models/mud_multilingual_v1.mud /kaggle/working/
print("Training Complete. Download mud_multilingual_v1.mud from 'Output' section.")
```

## Optimization Tips for Kaggle:
1. **GPU Selection:** Use the **L4** or **P100** accelerators for best compatibility with `bitsandbytes`.
2. **Persistence:** The `.mud` file will appear in `/kaggle/working/`.
3. **Dataset:** If training Spanish-LATAM, consider adding `load_dataset("cispa/culturax", "es", split="train", streaming=True)` for better quality.
