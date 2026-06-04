import json
import os
import sys
from safetensors.torch import save_file, load_file

def merge(model_dir, output_path):
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        print("No index file found. Maybe it is already a single file.")
        return
    with open(index_path, 'r') as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    files_to_load = set(weight_map.values())
    
    merged_tensors = {}
    for file in files_to_load:
        print(f"Loading {file}...")
        tensors = load_file(os.path.join(model_dir, file))
        merged_tensors.update(tensors)
    
    print(f"Saving to {output_path}...")
    save_file(merged_tensors, output_path)
    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python merge_safetensors.py <model_dir> <output_path>")
        sys.exit(1)
    merge(sys.argv[1], sys.argv[2])
