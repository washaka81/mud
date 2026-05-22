import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct
import os
import math
import sys
import multiprocessing
from typing import Dict, List
from tqdm import tqdm

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass
torch.multiprocessing.set_sharing_strategy('file_system')

NUM_THREADS = min((os.cpu_count() or 4), 12)
torch.set_num_threads(NUM_THREADS)
torch.set_num_interop_threads(1)
os.environ["OMP_NUM_THREADS"]    = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"]    = str(NUM_THREADS)
os.environ["KMP_AFFINITY"]       = "granularity=fine,compact,1,0"
os.environ["KMP_BLOCKTIME"]      = "1"
os.environ["OMP_WAIT_POLICY"]    = "PASSIVE"
torch.backends.mkldnn.enabled = True
torch.set_float32_matmul_precision("high")

# --- AUTO-CONFIG (opcional, override con MUD_AUTO_CONFIG=0) ---
if os.environ.get("MUD_AUTO_CONFIG", "1") == "1":
    try:
        from auto_config import load_training_config
        _ac = load_training_config("small")
        HIDDEN     = _ac.get("hidden", HIDDEN)
        FFN_HIDDEN = _ac.get("ffn_hidden", FFN_HIDDEN)
        EXPERTS    = _ac.get("num_experts", EXPERTS)
        TOP_K      = _ac.get("top_k", TOP_K)
        NUM_LAYERS = _ac.get("num_layers", NUM_LAYERS)
        LR         = _ac.get("lr", LR)
        print(f"  ⚙️  Auto-config: {EXPERTS} experts, {NUM_LAYERS} layers, hidden={HIDDEN}")
    except Exception:
        pass

# --- HYPERPARAMETERS ---
HIDDEN = 512; FFN_HIDDEN = 2048; EXPERTS = 8
TOP_K = 2; NUM_LAYERS = 6; LR = 5e-4
STEPS = 50000; SEQ_LEN = 128; BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def check_gpu():
    if DEVICE == "cuda":
        prop = torch.cuda.get_device_properties(0)
        print(f"🚀 [MUD GPU Check] Using: {prop.name} (Capability {prop.major}.{prop.minor})")
        if prop.major < 7:
            print("⚠️ [WARNING] Old GPU detected. FP16 may be slow. Consider switching to T4/L4 on Kaggle.")
    else:
        print("❌ [MUD ERROR] No GPU detected! Running on CPU will be 100x slower.")
        # We don't exit, but we warn loudly.

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
        x = x + self.attention(x, freqs_cis)
        residual = x
        x_norm = self.norm(x)
        gate_logits = self.gate(x_norm)
        
        # Noisy top-k gating con annealing
        noise_std = 0.1 * (1.0 - self._step_ratio.item())
        noise = torch.randn_like(gate_logits) * noise_std
        gate_logits = gate_logits + noise
        
        probs = F.softmax(gate_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        
        # 3-Component Balance Loss
        importance = probs.view(-1, self.num_experts).mean(dim=0)
        loss_imp = importance.var() * self.aux_coeff * self.num_experts
        # Load-based
        flat_i = top_k_indices.view(-1, self.top_k)
        load = torch.zeros(self.num_experts, device=x.device)
        for e_idx in range(self.num_experts):
            load[e_idx] = (flat_i == e_idx).any(dim=-1).float().mean()
        loss_load = load.var() * self.aux_coeff * self.num_experts
        # Z-loss
        z_loss = (gate_logits.logsumexp(dim=-1) ** 2).mean() * 1e-4
        balance_loss = loss_imp + loss_load + z_loss
        
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
    def __init__(self, dim, hidden_dim, num_experts, num_layers, num_heads=8, top_k=2, aux_coeff=0.05):
        super().__init__()
        self.layers = nn.ModuleList([MudBlock(dim, hidden_dim, num_experts, num_heads, top_k, aux_coeff) for _ in range(num_layers)])
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
    check_gpu()
    print(f"🔥 Starting High-Speed MUD Training on {DEVICE}")
    vocab_path = "vocab_es_en.txt"
    if not os.path.exists(vocab_path): vocab_path = "training/vocab_es_en.txt"
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = [l.strip() for l in f if l.strip()]
    word_to_id = {w: i for i, w in enumerate(vocab)}
    
    # Calcular aux_coeff según expertos
    if EXPERTS <= 16:
        _coeff = 0.5
    elif EXPERTS <= 64:
        _coeff = 0.1
    else:
        _coeff = 0.05
    model = MudMoE(HIDDEN, FFN_HIDDEN, EXPERTS, NUM_LAYERS, top_k=TOP_K, aux_coeff=_coeff).to(DEVICE)
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
        print("  ⚠️ No synthetic dataset found. Using default patterns.")
        corpus = ["Q: what is MUD A: <thinking> MUD is ternary MoE. </thinking> <answer> MUD is AI. </answer>"]

    # Pre-encode sequences to save time
    encoded = []
    for t in corpus:
        tokens = [word_to_id.get(w, 0) for w in t.split()]
        if len(tokens) > 2:
            encoded.append(torch.tensor(tokens, device=DEVICE))
    
    print(f"Training on {len(encoded)} sequences. Sequence Length: {SEQ_LEN}")

    for step in tqdm(range(STEPS)):
        # Annealing del ruido MoE
        step_ratio = step / max(1, STEPS)
        for module in model.modules():
            if isinstance(module, MudBlock):
                module._step_ratio = torch.tensor(step_ratio)

        # Randomly select a batch of sequences
        batch_idx = torch.randint(0, len(encoded), (BATCH_SIZE,))
        optimizer.zero_grad()
        total_loss = 0
        
        with torch.amp.autocast("cuda" if DEVICE == "cuda" else "cpu"):
            # We process individually since sequences have different lengths
            # A future optimization would be to pack them into a single tensor with padding
            for idx in batch_idx:
                seq = encoded[idx]
                if seq.shape[0] <= 1: continue
                # Truncate if too long for SEQ_LEN
                if seq.shape[0] > SEQ_LEN: seq = seq[:SEQ_LEN]
                
                x_ids = seq[:-1].unsqueeze(0)
                target = seq[1:]
                
                emb_ste = weight_quant(embed.weight)
                h = model(F.embedding(x_ids, emb_ste))
                logits = torch.matmul(h.squeeze(0), emb_ste.T)
                total_loss += F.cross_entropy(logits, target) + model.balance_loss
            
            total_loss /= BATCH_SIZE

            
        try:

            
            if scaler:

            
                scaler.scale(total_loss).backward()

            
                # Numerical guard

            
                if not torch.isfinite(total_loss):

            
                    print(f"\n⚠️  Salto de emergencia: Loss no finita. Ignorando lote.")

            
                    optimizer.zero_grad(set_to_none=True)

            
                    continue

            
                scaler.unscale_(optimizer)

            
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            
                scaler.step(optimizer)

            
                scaler.update()

            
            else:

            
                total_loss.backward()

            
                if not torch.isfinite(total_loss):

            
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

        scheduler.step()

        if step % 2000 == 0:
            print(f" Step {step} | Loss: {total_loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    # Export
    print("📦 Exporting consolidated 6-layer model...")
    with torch.no_grad(): emb_export = weight_quant(embed.weight)
    exp = MudExporter("core_skills.mud")
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
