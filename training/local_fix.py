import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct
import os
from tqdm import tqdm

# --- CORE MATH: BITNET 1.58b ---
def activation_quant(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    y = (x * scale).round().clamp(-128, 127) / scale
    return x + (y - x).detach()

def weight_quant(w):
    scale = w.abs().mean()
    w_scaled = w / (scale + 1e-7)
    w_q = torch.clamp(torch.round(w_scaled), -1, 1)
    return w + (w_q - w).detach()

class BitLinear(nn.Linear):
    def forward(self, x):
        return F.linear(activation_quant(x), weight_quant(self.weight), self.bias)

class CustomRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6))

class MoEExpert(nn.Module):
    def __init__(self, dim, h_dim):
        super().__init__()
        self.w1 = BitLinear(dim, h_dim, bias=False)
        self.w2 = BitLinear(h_dim, dim, bias=False)
        self.w3 = BitLinear(dim, h_dim, bias=False)
        self.norm = CustomRMSNorm(dim)
    def forward(self, x):
        x = self.norm(x)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MudMoE(nn.Module):
    def __init__(self, dim, h_dim, n_exp):
        super().__init__()
        self.experts = nn.ModuleList([MoEExpert(dim, h_dim) for _ in range(n_exp)])
        self.gate = nn.Linear(dim, n_exp, bias=False)
        self.norm = CustomRMSNorm(dim)
        self.balance_loss = torch.tensor(0.0)
    def forward(self, x):
        x_norm = self.norm(x)
        logits = self.gate(x_norm)
        probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, 2, dim=-1)
        importance = probs.mean(dim=0)
        self.balance_loss = importance.var() * 10.0
        top_k_probs /= top_k_probs.sum(dim=-1, keepdim=True)
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (top_k_indices == i).any(dim=-1)
            if mask.any():
                expert_out = expert(x[mask])
                for k in range(2):
                    k_mask = (top_k_indices[mask][:, k] == i)
                    if k_mask.any(): out[mask] += top_k_probs[mask][:, k:k+1] * expert_out
        return out

# --- EXPANDED BILINGUAL CORPUS ---
HIDDEN = 512; EXPERTS = 8; STEPS = 1500; VOCAB_SIZE = 151936
vocab = ['!', 'MUD', 'Forge', 'hola', 'hello', 'is', 'fast', 'smart', 'eficiente', 'motor', 'de', 'un', 'the', 'future', 'AI', 'engine', 'modular', 'soy', 'asistente', 'bien', 'gracias', 'qué', 'tal', 'estoy', 'listo', 'para', 'ayudarte', 'entendido', 'claro', 'si', 'no', 'proceso', 'conocimiento', 'grafo', 'red', 'neuronal', 'ternario', 'bits', 'velocidad']
vocab += [f't_{i}' for i in range(VOCAB_SIZE - len(vocab))]
word_to_id = {w: i for i, w in enumerate(vocab)}

corpus = [
    "hola MUD", "hello MUD", "MUD is fast", "MUD es eficiente", "soy un motor modular", 
    "the future is AI", "estoy listo para ayudarte", "bien gracias", "qué tal",
    "un motor de bits", "ternario es velocidad", "el grafo de conocimiento",
    "entendido proceso los datos", "MUD is smart", "Forge engine is fast"
]
encoded = [torch.tensor([word_to_id.get(w, 0) for w in s.split()]) for s in corpus]

# --- TRAINING ---
DEVICE = "cpu"
model = MudMoE(HIDDEN, HIDDEN*4, EXPERTS).to(DEVICE)
embedding = nn.Embedding(VOCAB_SIZE, HIDDEN).to(DEVICE)
optimizer = torch.optim.AdamW(list(model.parameters()) + list(embedding.parameters()), lr=1e-3)

print("🚀 Expanding Local Brain (V16)...")
for step in tqdm(range(STEPS)):
    target_ids = encoded[step % len(encoded)]
    x_ids = target_ids.repeat(4, 2)[:4, :16]
    x = embedding(x_ids)
    optimizer.zero_grad()
    output = model(x)
    loss = F.mse_loss(output, x.detach()) + model.balance_loss
    loss.backward()
    optimizer.step()

# --- EXPORT ---
def pack_ternary(row):
    r = torch.clamp(torch.round(row / (row.abs().mean() + 1e-7)), -1, 1).detach().cpu().numpy().flatten().astype(np.int8)
    packed = []
    for i in range(0, len(r), 16):
        chunk = r[i:i+16]; val = 0
        for j, v in enumerate(chunk):
            bits = 1 if v == 1 else (2 if v == -1 else 0)
            val |= (int(bits) << (j * 2))
        packed.append(val)
    return np.array(packed, dtype=np.uint32).tobytes()

with open('models/core_skills.mud', 'wb') as f:
    f.write(b'MUD\x01')
    meta = {'hidden_size': str(HIDDEN), 'num_experts': str(EXPERTS), 'num_layers': '1', 'tokenizer.tokens': '\n'.join(vocab)}
    f.write(struct.pack('<I', len(meta)))
    for k, v in meta.items():
        for s in [k, v]: b = s.encode('utf-8'); f.write(struct.pack('<I', len(b))); f.write(b)
    
    sd = {'token_embd.weight': embedding.weight, 'blk.0.gate.weight': model.gate.weight, 'blk.0.norm.weight': model.norm.weight}
    for i, e in enumerate(model.experts):
        sd[f'blk.0.expert.{i}.w1.weight'] = e.w1.weight; sd[f'blk.0.expert.{i}.w2.weight'] = e.w2.weight; sd[f'blk.0.expert.{i}.w3.weight'] = e.w3.weight
    
    f.write(struct.pack('<I', len(sd)))
    off = 0; tensor_data = []
    for name, t in sd.items():
        is_w = 'weight' in name and t.dim() > 1
        data = pack_ternary(t) if is_w else t.detach().cpu().numpy().astype(np.float32).tobytes()
        f.write(struct.pack('<I', len(name.encode('utf-8')))); f.write(name.encode('utf-8'))
        f.write(struct.pack('<I', 0 if is_w else 1))
        f.write(struct.pack('<I', len(t.shape)))
        for d in t.shape: f.write(struct.pack('<Q', d))
        f.write(struct.pack('<Q', off)); tensor_data.append(data); off += len(data)
    f.write(b'\x00' * ((32 - (f.tell() % 32)) % 32))
    for d in tensor_data: f.write(d)

print("✅ Local Brain V16 Ready.")
