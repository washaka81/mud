
import torch
import os
import sys
import struct
import numpy as np
from typing import Dict

# Ensure we can import from training
sys.path.append(os.getcwd())
from training.exporter import MudExporter

def convert_ckpt(ckpt_path, output_path, vocab_path, hidden=384, ffn_hidden=1536, n_layers=4, n_experts=64, top_k=4):
    print(f"Loading checkpoint from {ckpt_path}...")
    # Using weights_only=False because of the custom classes possibly saved or complex dicts
    # but we only need the state dicts which are standard types.
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    model_sd = ckpt.get("model_state_dict", ckpt.get("model", {}))
    embed_sd = ckpt.get("embed_state_dict", ckpt.get("embed", {}))
    
    if not model_sd:
        print("Error: Could not find model_state_dict in checkpoint")
        return

    # Load vocab
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = [l.strip() for l in f if l.strip()]
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}")

    # Prepare for export
    exporter = MudExporter(output_path)
    exporter.add_metadata("hidden_size", str(hidden))
    exporter.add_metadata("ffn_hidden", str(ffn_hidden))
    exporter.add_metadata("num_experts", str(n_experts))
    exporter.add_metadata("num_layers", str(n_layers))
    exporter.add_metadata("top_k", str(top_k))
    exporter.add_metadata("arch", "mud-ternary-moe-v1-master")
    exporter.add_metadata("tokenizer.tokens", "\n".join(vocab))
    
    # Map keys to MUD engine format
    sd = {}
    
    # Embeddings
    if "weight" in embed_sd:
        sd["token_embd.weight"] = embed_sd["weight"]
    elif "token_embd.weight" in model_sd:
        sd["token_embd.weight"] = model_sd["token_embd.weight"]
    elif "embed.weight" in model_sd:
        sd["token_embd.weight"] = model_sd["embed.weight"]
    else:
        # Try to find any embedding weight
        for k in embed_sd:
            if "weight" in k:
                sd["token_embd.weight"] = embed_sd[k]
                break

    # Final norm
    if "norm.weight" in model_sd:
        sd["output_norm.weight"] = model_sd["norm.weight"]

    # Layers
    for l in range(n_layers):
        # Attention
        sd[f"blk.{l}.attn_q.weight"] = model_sd[f"layers.{l}.attention.wq.weight"]
        sd[f"blk.{l}.attn_k.weight"] = model_sd[f"layers.{l}.attention.wk.weight"]
        sd[f"blk.{l}.attn_v.weight"] = model_sd[f"layers.{l}.attention.wv.weight"]
        sd[f"blk.{l}.attn_output.weight"] = model_sd[f"layers.{l}.attention.wo.weight"]
        sd[f"blk.{l}.attn_norm.weight"] = model_sd[f"layers.{l}.attention.norm.weight"]
        
        # MoE Gate & Norm
        sd[f"blk.{l}.gate.weight"] = model_sd[f"layers.{l}.moe.gate.weight"]
        sd[f"blk.{l}.norm.weight"] = model_sd[f"layers.{l}.moe.norm.weight"]
        
        # Experts
        for i in range(n_experts):
            sd[f"blk.{l}.expert.{i}.w1.weight"] = model_sd[f"layers.{l}.moe.experts.{i}.w1.weight"]
            sd[f"blk.{l}.expert.{i}.w2.weight"] = model_sd[f"layers.{l}.moe.experts.{i}.w2.weight"]
            sd[f"blk.{l}.expert.{i}.w3.weight"] = model_sd[f"layers.{l}.moe.experts.{i}.w3.weight"]

    exporter.export(sd)
    print(f"Successfully converted to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--vocab", type=str, default="training/vocab_es_en.txt")
    args = parser.parse_args()
    
    # Use big config as default
    convert_ckpt(args.ckpt, args.out, args.vocab)
