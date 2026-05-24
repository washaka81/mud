import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import math
import time
import random
import argparse
import struct
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple, Any

# Importar auto_config para usar los parámetros recomendados por hardware
sys.path.insert(0, os.getcwd())
from training.auto_config import load_training_config, print_config_report

# ─────────────────────────────────────────────────────────────────────────────
# VULKAN BACKEND INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────
VULKAN_AVAILABLE = False
try:
    if os.environ.get("MUD_USE_VULKAN") == "1":
        from training import vulkan_backend
        vulkan_backend._load_lib()
        VULKAN_AVAILABLE = vulkan_backend._vulkan_available
        if VULKAN_AVAILABLE:
            print("🚀 Vulkan Backend: ACTIVADO")
        else:
            print("⚠️  Vulkan Backend: No disponible en hardware/driver")
except Exception as e:
    print(f"⚠️  Error al cargar Vulkan: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZACIONES GLOBALES DE SISTEMA
# ─────────────────────────────────────────────────────────────────────────────
torch.set_float32_matmul_precision('high')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUPPORTS_BF16 = DEVICE == "cpu" or (DEVICE == "cuda" and torch.cuda.is_bf16_supported())

# --- CONFIGURACIÓN DINÁMICA BASADA EN HARDWARE ---
_cfg = load_training_config()
HIDDEN       = _cfg["hidden"]
FFN_HIDDEN   = _cfg["ffn_hidden"] if "ffn_hidden" in _cfg else _cfg["hidden"] * 4
NUM_EXPERTS  = _cfg["num_experts"]
NUM_LAYERS   = _cfg["num_layers"]
TOP_K        = _cfg["top_k"]
LR           = _cfg.get("lr", 5e-4)
GRAD_CLIP    = _cfg.get("grad_clip", 1.0)
AUX_COEFF    = _cfg.get("aux_coeff", 0.05)

# MoE Clústeres
CLUSTER_SIZE = min(16, NUM_EXPERTS // 4) if NUM_EXPERTS >= 4 else 1
NUM_CLUSTERS = max(1, NUM_EXPERTS // CLUSTER_SIZE)

# ─── CORE MATH: BITNET 1.58b ────────────────────────────────────────────────

def weight_quant(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cuantización ternaria {-1, 0, 1} con escala dinámica."""
    gamma = w.abs().mean().clamp(min=1e-7)
    w_scaled = w / gamma
    w_q = torch.clamp(torch.round(w_scaled), -1, 1)
    return w + (w_q * gamma - w).detach(), gamma

class TernaryLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (1.0 / math.sqrt(in_features)))
        self.register_buffer("scale", torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma = self.weight.abs().mean().clamp(min=1e-7)
        self.scale.copy_(gamma.detach())
        if VULKAN_AVAILABLE and (not self.training or DEVICE == "cpu"):
            from training.vulkan_backend import TernaryLinearFunction
            return TernaryLinearFunction.apply(x, self.weight, self.scale)
        w_q, _ = weight_quant(self.weight)
        return F.linear(x, w_q)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        v = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(v + self.eps) * self.weight

# ─── ROTARY EMBEDDINGS ──────────────────────────────────────────────────────

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t     = torch.arange(end)
    freqs = torch.outer(t, freqs)
    return torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs: torch.Tensor):
    def rot(x, f):
        B, T, H, D = x.shape
        x = x.view(B, T, H, D // 2, 2)
        cos = f[:T, None, :, 0]; sin = f[:T, None, :, 1]
        x0 = x[..., 0]; x1 = x[..., 1]
        out0 = x0 * cos - x1 * sin
        out1 = x0 * sin + x1 * cos
        return torch.stack([out0, out1], dim=-1).flatten(3).type_as(x)
    return rot(xq, freqs), rot(xk, freqs)

# ─── TRANSFORMER COMPONENTS ──────────────────────────────────────────────────

class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.wq = TernaryLinear(dim, dim)
        self.wk = TernaryLinear(dim, dim)
        self.wv = TernaryLinear(dim, dim)
        self.wo = TernaryLinear(dim, dim)
        self.norm = RMSNorm(dim)
    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        xn = self.norm(x)
        q = self.wq(xn).view(B, T, self.heads, self.head_dim)
        k = self.wk(xn).view(B, T, self.heads, self.head_dim)
        v = self.wv(xn).view(B, T, self.heads, self.head_dim)
        q, k = apply_rotary_emb(q, k, freqs)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, C))

class Expert(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = TernaryLinear(dim, hidden)
        self.w2 = TernaryLinear(hidden, dim)
        self.w3 = TernaryLinear(dim, hidden)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoELayer(nn.Module):
    def __init__(self, dim: int, hidden: int, n_experts: int, top_k: int,
                 n_clusters: int, cluster_size: int, aux_coeff: float = 0.05):
        super().__init__()
        self.experts = nn.ModuleList([Expert(dim, hidden) for _ in range(n_experts)])
        self.gate = nn.Linear(dim, n_experts, bias=False)
        self.norm = RMSNorm(dim)
        self.n_experts = n_experts
        self.top_k = top_k
        self.n_clusters = n_clusters
        self.cluster_size = cluster_size
        self.aux_coeff = aux_coeff
        self.register_buffer("cluster_activations", torch.zeros(n_clusters, dtype=torch.long))
        self.register_buffer("_step_ratio", torch.tensor(0.0))

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        xn = self.norm(x)
        logits = self.gate(xn)
        if self.training:
            noise_std = torch.clamp(0.1 * (1.0 - self._step_ratio), min=0.01)
            logits = logits + torch.randn_like(logits) * noise_std
        probs = F.softmax(logits / ((1.0 + (1.0 - self._step_ratio)) if self.training else 1.0), dim=-1)
        topk_p, topk_i = torch.topk(probs, self.top_k, dim=-1)
        topk_p = topk_p / topk_p.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        
        balance_loss = probs.view(-1, self.n_experts).mean(0).var() * self.aux_coeff * self.n_experts

        if not self.training or torch.rand((), device=x.device) < 0.1:
            with torch.no_grad():
                c_ids = topk_i // self.cluster_size
                self.cluster_activations.index_add_(0, c_ids.view(-1).clamp(0, self.n_clusters-1), 
                                                 torch.ones(c_ids.numel(), dtype=torch.long, device=x.device))

        out = torch.zeros_like(xn)
        xn_flat, topk_i_flat, topk_p_flat = xn.view(-1, C), topk_i.view(-1, self.top_k), topk_p.view(-1, self.top_k)
        
        # Optimización: Procesar expertos activos de forma eficiente
        active_indices = torch.unique(topk_i_flat)
        for i in active_indices.tolist():
            expert = self.experts[i]
            # Encontrar tokens donde este experto es el elegido
            mask = (topk_i_flat == i)
            token_indices = mask.any(dim=-1).nonzero(as_tuple=True)[0]
            if token_indices.numel() > 0:
                expert_input = xn_flat[token_indices]
                expert_output = expert(expert_input)
                # Extraer pesos correspondientes
                weights = (mask[token_indices] * topk_p_flat[token_indices]).sum(dim=-1, keepdim=True)
                out.view(-1, C)[token_indices] += weights * expert_output
                
        return out, balance_loss

class MudBlock(nn.Module):
    def __init__(self, dim, hidden, n_experts, top_k, n_clusters, cluster_size, aux_coeff):
        super().__init__()
        self.attention = Attention(dim, 8)
        self.moe = MoELayer(dim, hidden, n_experts, top_k, n_clusters, cluster_size, aux_coeff)
    def forward(self, x: torch.Tensor, freqs: torch.Tensor):
        x = x + self.attention(x, freqs)
        m_out, bl = self.moe(x)
        return x + m_out, bl

class MudModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            MudBlock(config['hidden'], config['ffn_hidden'], config['num_experts'], 
                     config['top_k'], NUM_CLUSTERS, CLUSTER_SIZE, config['aux_coeff'])
            for _ in range(config['num_layers'])
        ])
        self.norm = RMSNorm(config['hidden'])
        self.freqs = precompute_freqs_cis(config['hidden'] // 8, 1024)
    def forward(self, x: torch.Tensor):
        self.freqs = self.freqs.to(x.device)
        total_bl = 0.0
        for b in self.layers:
            x, bl = b(x, self.freqs)
            total_bl += bl
        return self.norm(x), total_bl

# ─── TOKENIZER & DATA ─────────────────────────────────────────────────────────

class FastTokenizer:
    def __init__(self, vocab_path: str):
        if os.path.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            self.word_to_id = {w: i for i, w in enumerate(lines)}
            self.id_to_word = {i: w for i, w in enumerate(lines)}
        else:
            print("⚠️  Vocabulario no encontrado, usando fallback ASCII.")
            self.word_to_id = {}
        self.vocab_size = 9882

    def encode(self, text: str) -> List[int]:
        # Tokenización por palabras simple
        words = text.split()
        return [self.word_to_id.get(w, ord(w[0]) % self.vocab_size if w else 0) for w in words]

class MudDataset(torch.utils.data.Dataset):
    def __init__(self, file_path: str, tokenizer: FastTokenizer, seq_len: int = 128):
        self.seq_len = seq_len
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        
        print(f"📦 Tokenizando {len(lines)} líneas...")
        all_tokens = []
        for line in lines:
            if len(line.strip()) > 10:
                all_tokens.extend(tokenizer.encode(line))
        
        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        self.num_samples = len(self.tokens) // seq_len

    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.tokens[start:start+self.seq_len+1]
        if len(chunk) < self.seq_len + 1:
            chunk = F.pad(chunk, (0, self.seq_len + 1 - len(chunk)))
        return chunk[:-1], chunk[1:]

# ─── TRAINING ────────────────────────────────────────────────────────────────

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--experts", type=int, default=NUM_EXPERTS)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-balance", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    print_config_report(_cfg)
    
    tokenizer = FastTokenizer("training/vocab_es_en.txt")
    model = MudModel(_cfg).to(DEVICE)
    embed = nn.Embedding(9882, HIDDEN).to(DEVICE)
    head = TernaryLinear(HIDDEN, 9882).to(DEVICE)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(embed.parameters()) + list(head.parameters()), lr=LR)
    
    start_step = 0
    ckpt_path = "models/mud_fast_ckpt.pt"
    if args.resume and os.path.exists(ckpt_path):
        print(f"♻️  Cargando checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        if 'embed' in ckpt: embed.load_state_dict(ckpt['embed'])
        if 'head' in ckpt: head.load_state_dict(ckpt['head'])
        model.load_state_dict(ckpt['model'], strict=False)
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
            print("   ✅ Optimizador cargado")
        except:
            print("   ⚠️ Reiniciando optimizador")
        start_step = ckpt.get('step', 0)
        print(f"   ↳ Continuando desde paso {start_step}")

    # ELIMINADO torch.compile POR DEFECTO para evitar SegFaults en CPU con ctypes
    # Solo se habilita si se pide explícitamente y MUD_NO_COMPILE no está puesto
    if not args.no_compile and os.environ.get("MUD_USE_COMPILE") == "1":
        print("⚙️  Compilando modelo...")
        model = torch.compile(model)

    # Buscar el mejor corpus disponible
    corpus_file = None
    possible_paths = [
        "training/massive_knowledge_corpus.txt",
        "massive_knowledge_corpus.txt",
        "training/synthetic_knowledge.txt",
        "synthetic_knowledge.txt",
        "/kaggle/input/mud-training/massive_knowledge_corpus.txt"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            corpus_file = p
            break
            
    if not corpus_file:
        print("❌ No se encontró ningún archivo de corpus (massive_knowledge_corpus.txt o synthetic_knowledge.txt)")
        return

    dataset = MudDataset(corpus_file, tokenizer, seq_len=128)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=_cfg['batch_size'], shuffle=True, num_workers=0)
    data_iter = iter(dataloader)
    
    model.train(); embed.train(); head.train()
    pbar = tqdm(range(start_step, args.steps), desc="Entrenando")
    step_times = []
    
    for step in pbar:
        ts = time.time()
        optimizer.zero_grad(set_to_none=True)
        if VULKAN_AVAILABLE: vulkan_backend.clear_caches()

        if step > 0 and step % 100 == 0:
            with torch.no_grad():
                for p in list(model.parameters()) + list(embed.parameters()) + list(head.parameters()):
                    if p.requires_grad: p.add_(torch.randn_like(p) * 1e-5)

        try:
            x_ids, target = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x_ids, target = next(data_iter)
        
        x_ids, target = x_ids.to(DEVICE), target.to(DEVICE)

        step_ratio = (step - start_step) / max(1, args.steps - start_step)
        # Manejo robusto de modelo compilado
        raw_m = model._orig_mod if hasattr(model, "_orig_mod") else model
        for m in raw_m.modules():
            if isinstance(m, MoELayer): m._step_ratio = torch.tensor(step_ratio, device=DEVICE)

        try:
            # En CPU con backend custom, autocast puede causar inestabilidad. Usamos FP32 directo.
            h, bl = model(embed(x_ids))
            logits = head(h)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=0) + bl
            
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(embed.parameters()) + list(head.parameters()), GRAD_CLIP)
                optimizer.step()
        except Exception as e:
            print(f"\n⚠️ Error en paso {step}: {e}")
            continue

        if step % 500 == 0:
            torch.save({'model': raw_m.state_dict(), 'embed': embed.state_dict(), 'head': head.state_dict(),
                       'optimizer': optimizer.state_dict(), 'step': step}, ckpt_path)

        step_times.append(time.time() - ts)
        if step % 5 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", it_s=f"{1.0/np.mean(step_times[-50:]):.1f}")

    print("🏁 Entrenamiento finalizado.")

if __name__ == "__main__":
    train()
