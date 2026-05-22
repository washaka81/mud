import torch
import os
import sys

def audit_checkpoint(path):
    print(f"=== Auditing Checkpoint: {path} ===")
    if not os.path.exists(path):
        print("Error: File not found.")
        return

    try:
        ckpt = torch.load(path, map_location='cpu')
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Handle nested state_dict
    sd = ckpt.get('model_state_dict', ckpt)
    
    keys = sorted(list(sd.keys()))
    print(f"Total keys: {len(keys)}")
    
    # Analyze layers
    layers = sorted(list(set([k.split('.')[1] for k in keys if k.startswith('layers.')])))
    print(f"Layers found: {layers}")
    
    # Check critical keys
    critical = ['embedding.weight', 'token_embd.weight', 'norm.weight', 'output.weight']
    for c in critical:
        present = any(c in k for k in keys)
        print(f"Critical key '{c}': {'✅' if present else '❌'}")
    
    # Check MoE components in first layer
    if layers:
        l0 = layers[0]
        components = ['gate', 'experts', 'norm']
        for comp in components:
            present = any(f'layers.{l0}.{comp}' in k for k in keys)
            print(f"Layer {l0} component '{comp}': {'✅' if present else '❌'}")

    print("-" * 40)

if __name__ == "__main__":
    paths = ['models/mud_last_checkpoint.pt', 'checkpoints_vulkan/ckpt_step_235.pt']
    for p in paths:
        audit_checkpoint(p)
