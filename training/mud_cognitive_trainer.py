import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import time
import math
import struct
import sqlite3
import argparse
import multiprocessing
from typing import Dict, List
from tqdm import tqdm

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass
torch.multiprocessing.set_sharing_strategy('file_system')

NUM_THREADS = min((os.cpu_count() or 4), 8)
torch.set_num_threads(NUM_THREADS)
torch.set_num_interop_threads(1)
os.environ["OMP_NUM_THREADS"]      = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"]      = str(NUM_THREADS)
os.environ["KMP_AFFINITY"]         = "granularity=fine,compact,1,0"
os.environ["KMP_BLOCKTIME"]        = "1"
os.environ["OMP_WAIT_POLICY"]      = "PASSIVE"
torch.backends.mkldnn.enabled = True
torch.set_float32_matmul_precision("high")

# --- AUTO-CONFIG (opcional) ---
_HAS_AC = False
_AC = {}
if os.environ.get("MUD_AUTO_CONFIG", "1") == "1":
    try:
        from auto_config import load_training_config
        _AC = load_training_config("small")
        _HAS_AC = True
    except Exception:
        pass

# Load Vulkan backend for local hardware offloading (con fallback seguro)
VULKAN_AVAILABLE = False
try:
    from training import vulkan_backend
    from training.vulkan_backend import TernaryLinearFunction, _load_lib
    _load_lib()
    VULKAN_AVAILABLE = vulkan_backend._vulkan_available
    if VULKAN_AVAILABLE:
        print(f"[CognitiveTrainer] Vulkan: ACTIVADO")
    else:
        print(f"[CognitiveTrainer] Vulkan: No disponible, usando CPU")
except Exception as e:
    print(f"[CognitiveTrainer] Vulkan no disponible: {e}, usando CPU fallback")

# --- CPU/GPU Threads Optimization ---
# Saturate Intel Core i7-1260p high-performance P-cores and prevent core-migration lag
torch.set_num_threads(8)
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

# --- HYPERPARAMETERS (con override de auto-config) ---
HIDDEN     = _AC.get("hidden", 512)      if _HAS_AC else 512
FFN_HIDDEN = _AC.get("ffn_hidden", 2048) if _HAS_AC else 2048
EXPERTS    = _AC.get("num_experts", 8)   if _HAS_AC else 8
TOP_K      = _AC.get("top_k", 2)         if _HAS_AC else 2
NUM_LAYERS = _AC.get("num_layers", 6)    if _HAS_AC else 6
LR         = _AC.get("lr", 5e-4)         if _HAS_AC else 5e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if _HAS_AC:
    print(f"  ⚙️  Auto-config: {EXPERTS} experts, {NUM_LAYERS} layers, hidden={HIDDEN}")

class TernaryLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.register_buffer("scale", torch.tensor(1.0))

    def forward(self, x):
        with torch.no_grad():
            self.scale.copy_(self.weight.abs().mean().clamp(min=1e-7))
        if VULKAN_AVAILABLE:
            return TernaryLinearFunction.apply(x, self.weight, self.scale)
        # CPU fallback con STE
        w_q = (self.weight / self.scale).clamp(-1, 1)
        w_q = self.weight + (w_q.round() - w_q).detach()
        return F.linear(x, w_q * self.scale)

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
        self.wq = TernaryLinear(dim, dim)
        self.wk = TernaryLinear(dim, dim)
        self.wv = TernaryLinear(dim, dim)
        self.wo = TernaryLinear(dim, dim)
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
        self.w1 = TernaryLinear(dim, hidden_dim)
        self.w2 = TernaryLinear(hidden_dim, dim)
        self.w3 = TernaryLinear(dim, hidden_dim)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MudBlock(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts, num_heads=8, top_k=2, aux_coeff=0.05):
        super().__init__()
        self.attention = CausalSelfAttention(dim, num_heads)
        self.experts = nn.ModuleList([MoEExpert(dim, hidden_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(dim, num_experts, bias=False)
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
        flat_i = top_k_indices.view(-1, self.top_k)
        load = torch.zeros(self.num_experts, device=x.device)
        for e_idx in range(self.num_experts):
            load[e_idx] = (flat_i == e_idx).any(dim=-1).float().mean()
        loss_load = load.var() * self.aux_coeff * self.num_experts
        z_loss = (gate_logits.logsumexp(dim=-1) ** 2).mean() * 1e-4
        balance_loss = loss_imp + loss_load + z_loss
        
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
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
                t_type = 1 if "norm" in name or "gate" in name else 0
                header.append(struct.pack("<I", t_type) + struct.pack("<I", len(t.shape)))
                for d in t.shape: header.append(struct.pack("<Q", d))
                
                # Dynamic scale calculation
                if t_type == 1:
                    data = t.detach().cpu().numpy().astype(np.float32).tobytes()
                else:
                    data = self._pack_ternary(t)
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

# --- Cognitive Classification ---
def classify_text(text: str) -> str:
    text_lower = text.lower()
    if any(k in text_lower for k in ["mud", "forge", "moe", "expert", "ternary", "quantization", "bits", "cognitive", "iq"]):
        return "system"
    if any(k in text_lower for k in ["python", "rust", "code", "fn ", "def ", "import", "class ", "println", "compile", "struct"]):
        return "code"
    if any(k in text_lower for k in ["plus", "minus", "sum", "math", "logic", "true", "false", "equals", "addition", "multiplicar", "verdad"]):
        return "logic"
    if any(k in text_lower for k in ["hello", "hola", "buenos", "gracias", "thank", "speak", "language", "hablo", "traduce", "translate"]):
        return "linguistics"
    return "general"

# --- Curiosity Engine (SQLite Query) ---
def retrieve_curiosity_facts(area: str, limit: int = 3) -> List[str]:
    if not os.path.exists("models/knowledge.db"):
        return []
    try:
        conn = sqlite3.connect("models/knowledge.db")
        cursor = conn.cursor()
        keywords = {
            "linguistics": ["language", "hablar", "grammar", "dialogue"],
            "logic": ["math", "logic", "number", "sum", "reasoning"],
            "code": ["python", "rust", "code", "programming", "software"],
            "general": ["science", "history", "knowledge", "world"],
            "system": ["modular", "inference", "quantization", "model"]
        }
        kw_list = keywords.get(area, ["knowledge"])
        facts = []
        for kw in kw_list:
            cursor.execute(
                "SELECT content FROM facts WHERE (content LIKE ? OR source LIKE ?) AND status = 0 ORDER BY rank DESC LIMIT ?",
                (f"%{kw}%", f"%{kw}%", limit)
            )
            for row in cursor.fetchall():
                facts.append(row[0])
            if len(facts) >= limit:
                break
        if len(facts) < limit:
            cursor.execute("SELECT content FROM facts WHERE status = 0 ORDER BY rank DESC LIMIT ?", (limit - len(facts),))
            for row in cursor.fetchall():
                facts.append(row[0])
        conn.close()
        return facts[:limit]
    except Exception as e:
        print(f"⚠️  Curiosity Engine error reading SQLite: {e}")
        return []

def mark_facts_as_assimilated(contents: List[str]):
    if not os.path.exists("models/knowledge.db") or not contents:
        return
    try:
        conn = sqlite3.connect("models/knowledge.db")
        cursor = conn.cursor()
        for content in contents:
            cursor.execute("UPDATE facts SET status = 1 WHERE content = ?", (content,))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️  Curiosity Engine error updating SQLite: {e}")

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to train")
    args = parser.parse_args()

    print(f"🧠 Starting MUD Cognitive IQ & Curiosity Pipeline on {DEVICE}")
    
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
    model = MudMoE(HIDDEN, FFN_HIDDEN, EXPERTS, NUM_LAYERS, aux_coeff=_coeff).to(DEVICE)
    embed = nn.Embedding(len(vocab), HIDDEN).to(DEVICE)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(embed.parameters()), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    
    # Load primary synthetic corpus
    corpus = []
    synth_path = "synthetic_knowledge.txt"
    if not os.path.exists(synth_path): synth_path = "training/synthetic_knowledge.txt"
    if os.path.exists(synth_path):
        with open(synth_path, "r", encoding="utf-8") as f:
            corpus = [line.strip() for line in f if line.strip()]
    
    if not corpus:
        corpus = [
            "hola MUD es un modelo modular y ternario",
            "MUD tiene ocho expertos de alta capacidad",
            "la suma de dos mas dos es igual a cuatro",
            "python y rust son lenguajes rapidos y eficientes",
            "la inteligencia artificial esta cambiando el mundo",
        ]
        
    # Categorize corpus
    categories = {
        "linguistics": [],
        "logic": [],
        "code": [],
        "general": [],
        "system": []
    }
    
    for text in corpus:
        area = classify_text(text)
        import re
        tokens_raw = re.findall(r"\w+|[^\w\s]|\s+", text, re.UNICODE)
        tokens = [word_to_id.get(w, 0) for w in tokens_raw]
        if len(tokens) > 2:
            categories[area].append(torch.tensor(tokens, device=DEVICE))
            
    for k, v in categories.items():
        print(f"  📂 Area [{k.capitalize()}]: {len(v)} sequences.")

    # Cognitive IQ initial state
    area_losses = {k: 1.5 for k in categories.keys()}
    iq_scores = {k: 100.0 for k in categories.keys()}
    
    print("\n🚀 Commencing Accelerated Training Loops...")
    start_time = time.time()
    
    for step in tqdm(range(args.steps)):
        # Select target knowledge area dynamically (prioritizing the one with lowest IQ)
        target_area = min(iq_scores, key=iq_scores.get)
        
        # Curiosity Engine Activation
        if iq_scores[target_area] < 100.0:
            print(f"\\n🔮 [CURIOSITY ENGINE] '{target_area.upper()}' IQ has dropped to {iq_scores[target_area]:.1f}!")
            print(f"   Querying SQLite models/knowledge.db to reinforce understanding...")
            facts = retrieve_curiosity_facts(target_area, limit=3)
            
            if facts:
                print(f"   Reinforcing with {len(facts)} unassimilated facts:")
                for f_text in facts:
                    print(f"    - {f_text[:60]}...")
                    tokens = [word_to_id.get(w, 0) for w in f_text.split()]
                    if len(tokens) > 2:
                        categories[target_area].append(torch.tensor(tokens, device=DEVICE))
                mark_facts_as_assimilated(facts)
            else:
                print("   No unassimilated facts found in SQLite. Learning from core memory.")
        
        # Select sequence from category
        sequences = categories[target_area]
        if not sequences:
            # Fallback to general
            sequences = categories["general"]
        if not sequences:
            continue
            
        # Fix: Re-tokenize from raw facts to handle spaces if needed
        # (Assuming word_to_id has space tokens)
        
        seq = sequences[torch.randint(0, len(sequences), (1,)).item()]
        if seq.shape[0] <= 1:
            continue
            
        # Annealing del ruido MoE
        step_ratio = step / max(1, args.steps)
        for module in model.modules():
            if isinstance(module, MudBlock):
                module._step_ratio = torch.tensor(step_ratio)

        optimizer.zero_grad()
        
        # BitNet weight quant
        with torch.no_grad():
            scale_emb = embed.weight.abs().mean()
            emb_ste = torch.clamp(torch.round(embed.weight / (scale_emb + 1e-7)), -1, 1)
            emb_ste = embed.weight + (emb_ste - embed.weight).detach()
            
        h = model(F.embedding(x_ids, emb_ste))
        logits = torch.matmul(h.squeeze(0), emb_ste.T)
        loss = F.cross_entropy(logits, target) + model.balance_loss

        
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

        scheduler.step()
        
        # Update metrics
        step_loss = loss.item()
        area_losses[target_area] = area_losses[target_area] * 0.9 + step_loss * 0.1
        iq_scores[target_area] = max(50.0, min(200.0, 100.0 + (1.5 - area_losses[target_area]) * 20.0))
        
        if (step + 1) % 20 == 0:
            print(f"\\n📈 Step {step+1}/{args.steps} | Current Low: {target_area.capitalize()} (Loss: {step_loss:.4f})")
            print("🧠 IQ Dashboard:")
            for area, score in iq_scores.items():
                bar = "■" * int(score / 10) + "░" * (20 - int(score / 10))
                print(f"  - {area.capitalize():12} [{bar}] {score:.1f}")

    total_time = time.time() - start_time
    print(f"\\n🏁 Cognitive Training Loop finished in {total_time:.2f}s!")
    
    # Export MUD file with visual metadata
    mud_out = "models/core_skills.mud"
    print(f"📦 Consolidating weights and exporting to {mud_out}...")
    
    exp = MudExporter(mud_out)
    exp.add_metadata("hidden_size", str(HIDDEN))
    exp.add_metadata("num_layers", str(NUM_LAYERS))
    exp.add_metadata("num_experts", str(EXPERTS))
    exp.add_metadata("ffn_hidden", str(FFN_HIDDEN))
    exp.add_metadata("tokenizer.tokens", "\n".join(vocab))
    
    # Append final IQ scores to mud global metadata for MUD CLI Dashboard readouts!
    for area, score in iq_scores.items():
        exp.add_metadata(f"iq.{area}", f"{score:.1f}")
        
    sd = {"token_embd.weight": embed.weight, "output_norm.weight": model.norm.weight}
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
    print(f"✅ Successful Consolidation. Annex complete!")

if __name__ == "__main__":
    train()
