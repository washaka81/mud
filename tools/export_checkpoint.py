import torch
import torch.nn as nn
import os
import sys
import math

# Add training directory to path to import components
sys.path.append(os.path.abspath("training"))

from auto_config import load_training_config
from mud_ultra_trainer import MudMoE, MudExporter, load_vocab, weight_quant

_cfg = load_training_config()
HIDDEN = _cfg["hidden"]
FFN_HIDDEN = _cfg.get("ffn_hidden", HIDDEN * 4)
EXPERTS = _cfg["num_experts"]
NUM_LAYERS = _cfg["num_layers"]
TOP_K = _cfg["top_k"]

def export_latest():
    device = "cpu"
    vocab = load_vocab()
    word_to_id = {w: i for i, w in enumerate(vocab)}
    vocab_size = len(vocab)
    print(f"Vocabulary: {vocab_size} tokens")

    model = MudMoE(dim=HIDDEN, hidden_dim=FFN_HIDDEN, num_experts=EXPERTS, num_layers=NUM_LAYERS, top_k=TOP_K).to(device)
    embed = nn.Embedding(vocab_size, HIDDEN).to(device)
    
    # ... (rest of search logic)
    ckpt_dir = "weights/checkpoints"
    checkpoints = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    if os.path.exists("models/mud_last_checkpoint.pt"):
        checkpoints.append("models/mud_last_checkpoint.pt")
    if os.path.exists("weights/mud_last_checkpoint.pt"):
        checkpoints.append("weights/mud_last_checkpoint.pt")
        
    if not checkpoints:
        print("❌ No checkpoints found to export.")
        return

    latest_ckpt = max(checkpoints, key=os.path.getmtime)
    print(f"📦 Loading latest checkpoint: {latest_ckpt}")
    
    ckpt = torch.load(latest_ckpt, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt.get("model_state_dict", ckpt)), strict=False)
    if "embed" in ckpt:
        embed.load_state_dict(ckpt["embed"])
    
    step = ckpt.get("step", "unknown")
    model_path = "models/core_skills.mud"
    
    # Calculate IQ
    with torch.no_grad():
        all_weights = torch.cat([p.flatten() for p in model.parameters() if p.dim() > 1])
        sigma = all_weights.std().item()
        skew = ((all_weights - all_weights.mean())**3).mean().item() / (sigma**3 + 1e-7)
        steps_val = step if isinstance(step, int) else 1000
        iq_score = 10.0 + (math.log10(steps_val + 1) * 10.0) + (1.0 - abs(skew)) * 20.0
        iq_score = min(200.0, max(10.0, iq_score))
        print(f"🧠 Calculated Digital IQ: {iq_score:.2f} (Sigma: {sigma:.4f}, Skew: {skew:.4f})")

    print(f"🚀 Exporting to {model_path}...")
    exp = MudExporter(model_path)
    exp.add_metadata("arch", "ternary_moe")
    exp.add_metadata("vocab_size", vocab_size)
    exp.add_metadata("hidden", HIDDEN)
    exp.add_metadata("hidden_size", HIDDEN)
    exp.add_metadata("num_layers", NUM_LAYERS)
    exp.add_metadata("num_experts", EXPERTS)
    exp.add_metadata("top_k", TOP_K)
    exp.add_metadata("ffn_hidden", FFN_HIDDEN)
    exp.add_metadata("tokenizer.tokens", "\n".join(vocab))
    exp.add_metadata("iq.score", f"{iq_score:.2f}")
    exp.add_metadata("iq.skew", f"{skew:.4f}")
    exp.add_metadata("step", str(step))
    
    # --- EXPERT TAXONOMY METADATA ---
    taxonomy = [
        "Planificación/CoT", "Lógica Formal", "Evaluador Interno", "Razonamiento Difuso",
        "Gramática/AST", "Optimización/Bajo Nivel", "Algoritmia Avanzada", "Álgebra Lineal",
        "Cálculo/Dinámica", "Estadística Avanzada", "Física Cuántica", "Mecánica Clásica",
        "Química Molecular", "Bioinformática", "Sistemas Complejos", "Taxonomías Fácticas"
    ]
    exp.add_metadata("expert_taxonomy", ",".join(taxonomy))
    
    sd = {"token_embd.weight": weight_quant(embed.weight), "output_norm.weight": model.norm.weight}
    for l in range(NUM_LAYERS):
        layer = model.layers[l]
        sd[f"blk.{l}.attn_q.weight"] = weight_quant(layer.attention.wq.weight)
        sd[f"blk.{l}.attn_k.weight"] = weight_quant(layer.attention.wk.weight)
        sd[f"blk.{l}.attn_v.weight"] = weight_quant(layer.attention.wv.weight)
        sd[f"blk.{l}.attn_output.weight"] = weight_quant(layer.attention.wo.weight)
        sd[f"blk.{l}.attn_norm.weight"] = layer.attention.norm.weight
        sd[f"blk.{l}.gate.weight"] = layer.gate.weight
        sd[f"blk.{l}.norm.weight"] = layer.norm.weight
        sd[f"blk.{l}.attn_q.scale"] = layer.attention.wq.scale
        sd[f"blk.{l}.attn_k.scale"] = layer.attention.wk.scale
        sd[f"blk.{l}.attn_v.scale"] = layer.attention.wv.scale
        sd[f"blk.{l}.attn_output.scale"] = layer.attention.wo.scale
        for i in range(EXPERTS):
            sd[f"blk.{l}.expert.{i}.w1.weight"] = weight_quant(layer.experts[i].w1.weight)
            sd[f"blk.{l}.expert.{i}.w1.scale"] = layer.experts[i].w1.scale
            sd[f"blk.{l}.expert.{i}.w2.weight"] = weight_quant(layer.experts[i].w2.weight)
            sd[f"blk.{l}.expert.{i}.w2.scale"] = layer.experts[i].w2.scale
            sd[f"blk.{l}.expert.{i}.w3.weight"] = weight_quant(layer.experts[i].w3.weight)
            sd[f"blk.{l}.expert.{i}.w3.scale"] = layer.experts[i].w3.scale
    
    exp.export(sd)
    print("✅ Export complete!")

if __name__ == "__main__":
    export_latest()
