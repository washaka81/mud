import torch
import os

def check_checkpoint(path):
    try:
        ckpt = torch.load(path, map_location="cpu")
        print(f"\n--- {path} ---")
        if "model" in ckpt:
            sd = ckpt["model"]
            layers = 0
            while f"layers.{layers}.gate.weight" in sd:
                layers += 1
            
            # Experts
            experts = 0
            while f"layers.0.experts.{experts}.w1.weight" in sd:
                experts += 1
                
            hidden = sd["layers.0.gate.weight"].shape[1]
            
            print(f"Arch: Layers={layers}, Experts={experts}, Hidden={hidden}")
            
            if "embed" in ckpt:
                vocab_size = ckpt["embed"]["weight"].shape[0]
                print(f"Vocab Size: {vocab_size}")
            
            print(f"Step: {ckpt.get('step', 'N/A')}")
            return True
        else:
            print("No 'model' key found in checkpoint.")
            return False
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return False

# Check some samples
samples = [
    "models/mud_last_checkpoint.pt",
    "weights/checkpoints/ckpt_step_4800.pt",
    "weights/checkpoints/ckpt_step_1000.pt",
    "weights/checkpoints_old_vocab/ckpt_step_1000.pt"
]

for s in samples:
    if os.path.exists(s):
        check_checkpoint(s)
