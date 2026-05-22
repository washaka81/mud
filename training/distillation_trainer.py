import torch
import os

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct
import math
from typing import Dict
from tqdm import tqdm

# --- HYPERPARAMETERS ---
HIDDEN = 512
FFN_HIDDEN = 2048
EXPERTS = 8
TOP_K = 2
NUM_LAYERS = 2 # Increased layers for better capacity
LR = 1e-4 # Lower learning rate for fine-tuning
STEPS = 100000 # Massive steps for massive data
BATCH_SIZE = 4
MAX_SEQ_LEN = 128

KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ
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

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
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
        residual = x
        x = self.norm(x)
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.num_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.num_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.num_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis[:seqlen])
        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(seqlen, seqlen, device=x.device), diagonal=1).bool()
        scores.masked_fill_(mask, float("-inf"))
        probs = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(probs, xv)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return residual + self.wo(output)

class MoEExpert(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = BitLinear(dim, hidden_dim, bias=False)
        self.w2 = BitLinear(hidden_dim, dim, bias=False)
        self.w3 = BitLinear(dim, hidden_dim, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MudBlock(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts, num_heads=8, top_k=2, aux_coeff=0.05):
        super().__init__()
        self.attention = CausalSelfAttention(dim, num_heads)
        self.experts = nn.ModuleList([MoEExpert(dim, hidden_dim) for _ in range(num_experts)])
        self.gate = BitLinear(dim, num_experts, bias=False)
        self.norm = CustomRMSNorm(dim)
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_coeff = aux_coeff
        self.register_buffer("_step_ratio", torch.tensor(0.0))
    def forward(self, x, freqs_cis):
        x = self.attention(x, freqs_cis)
        residual = x
        x_norm = self.norm(x)
        gate_logits = self.gate(x_norm)
        # Noisy top-k gating con annealing
        noise_std = 0.1 * (1.0 - self._step_ratio.item())
        noise = torch.randn_like(gate_logits) * noise_std
        gate_logits = gate_logits + noise
        probs = F.softmax(gate_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        out = torch.zeros_like(x)
        bsz, seqlen, d = x.shape
        # 3-Component Balance Loss
        importance = probs.view(-1, self.num_experts).mean(dim=0)
        loss_imp = importance.var() * self.aux_coeff * self.num_experts
        flat_i = top_k_indices.view(-1, self.top_k)
        load = torch.zeros(self.num_experts, device=x.device)
        for e_idx in range(self.num_experts):
            load[e_idx] = (flat_i == e_idx).any(dim=-1).float().mean()
        loss_load = load.var() * self.aux_coeff * self.num_experts
        z_loss = (gate_logits.logsumexp(dim=-1) ** 2).mean() * 1e-4
        balance_loss = loss_imp + loss_load + z_loss
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
    def __init__(self, vocab_size, dim, hidden_dim, num_experts, num_layers=2, aux_coeff=0.05):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([MudBlock(dim, hidden_dim, num_experts, aux_coeff=aux_coeff) for _ in range(num_layers)])
        self.norm = CustomRMSNorm(dim)
        self.freqs_cis = precompute_freqs_cis(dim // 8, 2048)
    def forward(self, idx):
        x = self.embed(idx)
        self.freqs_cis = self.freqs_cis.to(x.device)
        total_bl = 0
        for layer in self.layers:
            x, bl = layer(x, self.freqs_cis)
            total_bl += bl
        return self.norm(x), total_bl

# --- TOKENIZER HELPER ---
class SimpleTokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.tokens = [line.strip() for line in f]
        self.t2i = {t: i for i, t in enumerate(self.tokens)}
    def encode(self, text):
        # Very simple fallback for distillation: match whole tokens or single chars
        res = []
        for word in text.split():
            if word in self.t2i: res.append(self.t2i[word])
            else:
                for c in word:
                    if c in self.t2i: res.append(self.t2i[c])
        return res

def train():
    print(f"Starting Knowledge Distillation on {DEVICE}...")
    vocab_path = "vocab_es_en.txt" if os.path.exists("vocab_es_en.txt") else "/kaggle/input/mud-vocab/vocab_es_en.txt"
    tokenizer = SimpleTokenizer(vocab_path)
    
    with open("massive_knowledge_corpus.txt", "r", encoding="utf-8") as f:
        corpus = [line.strip() for line in f if len(line.strip()) > 10]
    
    print(f"Corpus size: {len(corpus)} items")
    _coeff = 0.5 if EXPERTS <= 16 else (0.1 if EXPERTS <= 64 else 0.05)
    model = MudMoE(len(tokenizer.tokens), HIDDEN, FFN_HIDDEN, EXPERTS, NUM_LAYERS, aux_coeff=_coeff).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    pbar = tqdm(range(STEPS))
    for step in pbar:
        # Annealing del ruido MoE
        step_ratio = step / max(1, STEPS)
        for module in model.modules():
            if isinstance(module, MudBlock):
                module._step_ratio = torch.tensor(step_ratio)

        # Simple batching
        batch_indices = np.random.randint(0, len(corpus), BATCH_SIZE)
        batch_texts = [corpus[i] for i in batch_indices]
        
        input_ids = []
        for text in batch_texts:
            ids = tokenizer.encode(text)[:MAX_SEQ_LEN]
            if len(ids) < 2: ids = [0, 0]
            input_ids.append(torch.tensor(ids))
        
        # Pad to same length
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True).to(DEVICE)
        
        logits, bl = model(input_ids[:, :-1])
        targets = input_ids[:, 1:]
        
        # Flatten for loss
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) + bl
        
        optimizer.zero_grad()

        try:

            if scaler:

                scaler.scale(loss).backward()

                # Numerical guard

                if not torch.isfinite(loss):

                    print(f"\n⚠️  Salto de emergencia: Loss no finita. Ignorando lote.")

                    optimizer.zero_grad(set_to_none=True)

                    continue

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)

                scaler.update()

            else:

                loss.backward()

                if not torch.isfinite(loss):

                    print(f"\n⚠️  Salto de emergencia: Loss no finita. Ignorando lote.")

                    optimizer.zero_grad(set_to_none=True)

                    continue

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

        except RuntimeError as e:

            if "out of memory" in str(e).lower() or "allocation" in str(e).lower():

                print(f"\n⚠️  OOM en paso. Limpiando cachés y omitiendo lote.")

                optimizer.zero_grad(set_to_none=True)

                if torch.cuda.is_available(): torch.cuda.empty_cache()

                import gc; gc.collect()

                try:

                    if VULKAN_AVAILABLE:

                        from training import vulkan_backend

                        vulkan_backend.clear_caches()

                except: pass

                continue

            else:

                raise e

        
        if step % 100 == 0:
            pbar.set_description(f"Loss: {loss.item():.4f}")

    # Export Logic
    torch.save(model.state_dict(), "mud_distilled_weights.pt")
    print("Distillation complete. Weights saved.")

if __name__ == "__main__":
    train()
