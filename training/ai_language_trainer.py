import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct
import os
import math
import sys
from typing import Dict, List
from tqdm import tqdm

# --- HYPERPARAMETERS ---
HIDDEN = 512
FFN_HIDDEN = 2048
EXPERTS = 8
TOP_K = 2
NUM_LAYERS = 6  # Increased for real learning
LR = 5e-4
STEPS = 50000
SEQ_LEN = 128   # Increased for Chain of Thought
BATCH_SIZE = 32 # Optimized for P100/T4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def weight_quant(w):
    scale = w.abs().mean()
    w_scaled = w / (scale + 1e-7)
    w_q = torch.clamp(torch.round(w_scaled), -1, 1)
    return w + (w_q - w).detach()

class BitLinear(nn.Linear):
    def forward(self, x):
        w_q = weight_quant(self.weight)
        return F.linear(x, w_q, self.bias) * (1.0 / math.sqrt(self.in_features))

class CustomRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, xq_.shape[1], 1, xq_.shape[3])
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.wq = BitLinear(dim, dim, bias=False)
        self.wk = BitLinear(dim, dim, bias=False)
        self.wv = BitLinear(dim, dim, bias=False)
        self.wo = BitLinear(dim, dim, bias=False)
        self.norm = CustomRMSNorm(dim)

    def forward(self, x, freqs_cis):
        bsz, seqlen, _ = x.shape
        x_norm = self.norm(x)
        xq, xk, xv = self.wq(x_norm), self.wk(x_norm), self.wv(x_norm)
        xq = xq.view(bsz, seqlen, self.num_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.num_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.num_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis[:seqlen])
        xq = xq.transpose(1, 2); xk = xk.transpose(1, 2); xv = xv.transpose(1, 2)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(seqlen, seqlen, device=x.device), diagonal=1).bool()
        scores.masked_fill_(mask, float("-inf"))
        probs = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(probs, xv)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

class MoEExpert(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = BitLinear(dim, hidden_dim, bias=False)
        self.w2 = BitLinear(hidden_dim, dim, bias=False)
        self.w3 = BitLinear(dim, hidden_dim, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MudBlock(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts, num_heads=8, top_k=2):
        super().__init__()
        self.attention = CausalSelfAttention(dim, num_heads)
        self.experts = nn.ModuleList([MoEExpert(dim, hidden_dim) for _ in range(num_experts)])
        self.gate = BitLinear(dim, num_experts, bias=False)
        self.norm = CustomRMSNorm(dim)
        self.num_experts = num_experts
        self.top_k = top_k
        
    def forward(self, x, freqs_cis):
        x = x + self.attention(x, freqs_cis)
        residual = x
        x_norm = self.norm(x)
        gate_logits = self.gate(x_norm)
        probs = F.softmax(gate_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        
        # Balance Loss
        importance = probs.view(-1, self.num_experts).mean(dim=0)
        balance_loss = importance.var() * 10.0
        
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Optimized MoE Forward
        bsz, seqlen, d = x.shape
        x_flat = x_norm.view(-1, d)
        indices_flat = top_k_indices.view(-1, self.top_k)
        probs_flat = top_k_probs.view(-1, self.top_k)
        out_flat = torch.zeros_like(x_flat)
        
        for i, expert in enumerate(self.experts):
            mask = (indices_flat == i).any(dim=-1)
            if mask.any():
                expert_out = expert(x_flat[mask])
                for k in range(self.top_k):
                    k_mask = (indices_flat[mask][:, k] == i)
                    if k_mask.any():
                        out_flat[mask] += probs_flat[mask][:, k:k+1] * expert_out
                        
        return residual + out_flat.view(bsz, seqlen, d), balance_loss

class MudMoE(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts, num_layers, num_heads=8, top_k=2):
        super().__init__()
        self.layers = nn.ModuleList([MudBlock(dim, hidden_dim, num_experts, num_heads, top_k) for _ in range(num_layers)])
        self.norm = CustomRMSNorm(dim)
        self.freqs_cis = precompute_freqs_cis(dim // num_heads, 1024)
        self.balance_loss = torch.tensor(0.0)
        
    def forward(self, x):
        self.freqs_cis = self.freqs_cis.to(x.device)
        total_bl = 0.0
        for layer in self.layers:
            x, bl = layer(x, self.freqs_cis)
            total_bl += bl
        self.balance_loss = total_bl
        return self.norm(x)

class MudExporter:
    MAGIC = b"MUD\x01"
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.metadata = {}
    def add_metadata(self, key: str, value: str):
        self.metadata[key] = value
    def export(self, tensors: Dict[str, torch.Tensor]):
        with open(self.output_path, "wb") as f:
            f.write(self.MAGIC)
            f.write(struct.pack("<I", len(self.metadata)))
            for k, v in self.metadata.items():
                kb, vb = k.encode("utf-8"), v.encode("utf-8")
                f.write(struct.pack("<I", len(kb)) + kb + struct.pack("<I", len(vb)) + vb)
            f.write(struct.pack("<I", len(tensors)))
            curr_off = 0; tensor_data = []; header = []
            for name, t in tensors.items():
                nb = name.encode("utf-8")
                header.append(struct.pack("<I", len(nb)) + nb)
                t_type = 1 if "norm" in name else 0
                header.append(struct.pack("<I", t_type) + struct.pack("<I", len(t.shape)))
                for d in t.shape: header.append(struct.pack("<Q", d))
                data = t.detach().cpu().numpy().astype(np.float32).tobytes() if t_type == 1 else self._pack_ternary(t)
                header.append(struct.pack("<Q", curr_off))
                tensor_data.append(data); curr_off += len(data)
            f.write(b"".join(header))
            f.write(b"\x00" * ((32 - (f.tell() % 32)) % 32))
            f.write(b"".join(tensor_data))
    def _pack_ternary(self, t) -> bytes:
        r = t.detach().cpu().numpy().flatten()
        scale = np.abs(r).mean(); w_q = np.clip(np.round(r / (scale + 1e-7)), -1, 1).astype(np.int8)
        packed = []
        for i in range(0, len(w_q), 16):
            chunk = w_q[i:i+16]; val = 0
            for j, b in enumerate(chunk):
                bits = 1 if b == 1 else (2 if b == -1 else 0)
                val |= (bits << (j * 2))
            packed.append(struct.pack("<I", val))
        return b"".join(packed)

def train():
    print(f"🔥 Starting High-Speed MUD Training on {DEVICE}")
    vocab_path = "vocab_es_en.txt"
    if not os.path.exists(vocab_path): vocab_path = "training/vocab_es_en.txt"
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = [l.strip() for l in f if l.strip()]
    word_to_id = {w: i for i, w in enumerate(vocab)}
    
    model = MudMoE(HIDDEN, FFN_HIDDEN, EXPERTS, NUM_LAYERS, top_k=TOP_K).to(DEVICE)
    embed = nn.Embedding(len(vocab), HIDDEN).to(DEVICE)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(embed.parameters()), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS)
    scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None

    # Load high-quality corpus (Priority: Synthetic Dataset -> DB Facts -> Hardcoded)
    corpus = []
    synth_path = "synthetic_knowledge.txt"
    if not os.path.exists(synth_path): synth_path = "training/synthetic_knowledge.txt"
    
    if os.path.exists(synth_path):
        print(f"  [MUD Dreaming] Loading synthetic dataset from {synth_path}...")
        with open(synth_path, "r", encoding="utf-8") as f:
            corpus = [line.strip() for line in f if line.strip()]
    
    if not corpus:
        corpus = [
            "Q: explica 1+1 A: <thinking> 1+1 es una operación aritmética básica. Sumando una unidad a otra resulta en dos. </thinking> <answer> 1+1 es 2 </answer>",
            "Q: what is MUD A: <thinking> MUD stands for Modular Understanding Dynamics. It is a ternary MoE architecture. </thinking> <answer> MUD is an advanced AI engine. </answer>",
            "Q: explain logic A: <thinking> Logic is the study of correct reasoning. It involves premises and conclusions. </thinking> <answer> Logic is the foundation of science. </answer>",
            "Q: por qué es rápido A: <thinking> Usa pesos ternarios de 1.58 bits y aceleración Vulkan en iGPU. </thinking> <answer> Es eficiente por su diseño modular. </answer>"
        ]
        if os.path.exists("models/knowledge.db"):
            import sqlite3
            conn = sqlite3.connect("models/knowledge.db")
            db_facts = [r[0] for r in conn.execute("SELECT content FROM facts LIMIT 5000")]
            for fact in db_facts:
                corpus.append(f"<thinking> Procesando información sobre: {fact[:50]} </thinking> <answer> {fact} </answer>")


    encoded = [torch.tensor([word_to_id.get(w, 0) for w in t.split()], device=DEVICE) for t in corpus if len(t.split()) > 2]
    
    for step in tqdm(range(STEPS)):
        batch_idx = torch.randint(0, len(encoded), (BATCH_SIZE,))
        optimizer.zero_grad()
        total_loss = 0
        
        with torch.amp.autocast("cuda" if DEVICE == "cuda" else "cpu"):
            for idx in batch_idx:
                seq = encoded[idx]; x_ids = seq[:-1].unsqueeze(0); target = seq[1:]
                if x_ids.shape[1] == 0: continue
                emb_ste = weight_quant(embed.weight)
                h = model(F.embedding(x_ids, emb_ste))
                logits = torch.matmul(h.squeeze(0), emb_ste.T)
                total_loss += F.cross_entropy(logits, target) + model.balance_loss
            
            total_loss /= BATCH_SIZE
            
        if scaler:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()
        scheduler.step()

        if step % 2000 == 0:
            print(f" Step {step} | Loss: {total_loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    # Export
    with torch.no_grad(): emb_export = weight_quant(embed.weight)
    exp = MudExporter("models/core_skills.ai")
    exp.add_metadata("hidden_size", str(HIDDEN)); exp.add_metadata("num_layers", str(NUM_LAYERS))
    exp.add_metadata("num_experts", str(EXPERTS)); exp.add_metadata("tokenizer.tokens", "\n".join(vocab))
    sd = {"token_embd.weight": emb_export, "output_norm.weight": model.norm.weight}
    for l in range(NUM_LAYERS):
        layer = model.layers[l]
        sd[f"blk.{l}.attn_q.weight"] = layer.attention.wq.weight
        sd[f"blk.{l}.attn_k.weight"] = layer.attention.wk.weight
        sd[f"blk.{l}.attn_v.weight"] = layer.attention.wv.weight
        sd[f"blk.{l}.attn_output.weight"] = layer.attention.wo.weight
        sd[f"blk.{l}.gate.weight"] = layer.gate.weight
        sd[f"blk.{l}.norm.weight"] = layer.norm.weight
        for i in range(EXPERTS):
            sd[f"blk.{l}.expert.{i}.w1.weight"] = layer.experts[i].w1.weight
            sd[f"blk.{l}.expert.{i}.w2.weight"] = layer.experts[i].w2.weight
            sd[f"blk.{l}.expert.{i}.w3.weight"] = layer.experts[i].w3.weight
    exp.export(sd)
    print("✅ High-Speed Training & Export Complete.")

if __name__ == "__main__": train()
