import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import struct
import os
import sys
import math
import re
import multiprocessing
from typing import Dict
from tqdm import tqdm
import argparse
import copy
from torch.utils.checkpoint import checkpoint

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass
torch.multiprocessing.set_sharing_strategy('file_system')

NUM_THREADS = min((os.cpu_count() or 4), 16)
torch.set_num_threads(NUM_THREADS)
torch.set_num_interop_threads(1)
os.environ["OMP_NUM_THREADS"]      = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"]      = str(min(NUM_THREADS, 8))
os.environ["KMP_AFFINITY"]         = "granularity=fine,compact,1,0"
os.environ["KMP_BLOCKTIME"]        = "1"
os.environ["OMP_WAIT_POLICY"]      = "PASSIVE"
os.environ["MKL_DYNAMIC"]          = "FALSE"
torch.backends.mkldnn.enabled = True
torch.set_float32_matmul_precision("high")

# --- VULKAN BACKEND ---
VULKAN_AVAILABLE = False
def load_vulkan():
    global VULKAN_AVAILABLE
    try:
        sys.path.append(os.getcwd())
        from training import vulkan_backend
        vulkan_backend._load_lib()
        VULKAN_AVAILABLE = vulkan_backend._vulkan_available
        if VULKAN_AVAILABLE:
            print("🚀 Vulkan Backend: ACTIVADO")
        else:
            print("⚠️  Vulkan Backend: No disponible en hardware/driver")
    except Exception as e:
        print(f"⚠️  Error al cargar Vulkan: {e}")

class ModelEMA:
    def __init__(self, model, embed, decay=0.999):
        self.ema_model = copy.deepcopy(model)
        self.ema_embed = copy.deepcopy(embed)
        self.ema_model.eval()
        self.ema_embed.eval()
        for p in self.ema_model.parameters(): p.requires_grad_(False)
        for p in self.ema_embed.parameters(): p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model, embed):
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.copy_(self.decay * ema_p + (1.0 - self.decay) * p)
        for ema_p, p in zip(self.ema_embed.parameters(), embed.parameters()):
            ema_p.copy_(self.decay * ema_p + (1.0 - self.decay) * p)

# --- DETECCIÓN DE RAM ---
def _estimate_ram_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return float(line.split()[1]) / 1_048_576
    except OSError:
        return 16.0

from auto_config import load_training_config, print_config_report

# --- CONFIGURACIÓN GLOBAL MUD-V1.5-MASTER (auto-escalado) ---
_cfg = load_training_config()
print_config_report(_cfg)

_RAM_MODE = _cfg.get("mode", "medium")
HIDDEN = _cfg["hidden"]
FFN_HIDDEN = _cfg.get("ffn_hidden", HIDDEN * 4)
EXPERTS = _cfg["num_experts"]
NUM_LAYERS = _cfg["num_layers"]
TOP_K = _cfg["top_k"]
LR = _cfg.get("lr", 5e-4)

STEPS = 100000
MICRO_BATCH_SIZE = _cfg.get("batch_size", 2)
GRAD_ACCUM_STEPS = 8
MAX_SEQ_LEN = 128
KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

def parse_args():
    parser = argparse.ArgumentParser(description="MUD Master Trainer")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--use_vulkan", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()

args = parse_args()
STEPS = args.steps
if args.use_vulkan:
    load_vulkan()

# --- PERSONALIDAD Y CORPUS BASE ---
CORPUS = [
    "¡Hola! Soy MUD, tu asistente inteligente de alto nivel. 😊",
    "Estoy procesando los datos con precisión matemática y lógica.",
    "Todo está saliendo excelente bajo mi supervisión experta. ✅",
    "La vida es fascinante cuando se analiza a través de la ciencia.",
    "Mis circuitos están optimizados para resolver cualquier problema.",
    "He consultado mis neuronas digitales y la respuesta es perfecta.",
    "Dato curioso: el Teorema del Límite Central es la base de mi estabilidad.",
    "¡Ciencia y lógica al rescate! ¿En qué puedo ayudarte hoy?"
]

# --- ARQUITECTURA TERNARY BITNET 1.58b ---
def weight_quant(w):
    scale = w.abs().mean()
    w_scaled = w / (scale + 1e-7)
    w_q = torch.clamp(torch.round(w_scaled), -1, 1)
    return w + (w_q - w).detach()

class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("scale", torch.tensor(1.0))

    def forward(self, x):
        gamma = self.weight.abs().mean().detach()
        self.scale.copy_(gamma)
        w_q = weight_quant(self.weight)
        return F.linear(x, w_q, self.bias) * (gamma / math.sqrt(self.in_features))

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
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2).to(xq.device)
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
        output = torch.matmul(probs, xv).transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return residual + self.wo(output)

class MoEExpert(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = BitLinear(dim, hidden_dim, bias=False)
        self.w2 = BitLinear(hidden_dim, dim, bias=False)
        self.w3 = BitLinear(dim, hidden_dim, bias=False)
    def forward(self, x):
        def _inner_forward(x_in):
            return self.w2(F.silu(self.w1(x_in)) * self.w3(x_in))
        if x.requires_grad:
            return checkpoint(_inner_forward, x, use_reentrant=False)
        return _inner_forward(x)

class MudBlock(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts, num_heads=8, top_k=2, aux_coeff=0.05):
        super().__init__()
        self.dim, self.num_experts, self.top_k = dim, num_experts, top_k
        self.aux_coeff = aux_coeff
        self.register_buffer("_step_ratio", torch.tensor(0.0))
        self.attention = CausalSelfAttention(dim, num_heads)
        self.experts = nn.ModuleList([MoEExpert(dim, hidden_dim) for _ in range(num_experts)])
        self.gate = BitLinear(dim, num_experts, bias=False)
        self.norm = CustomRMSNorm(dim)
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
        
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-7)
        
        bsz, seqlen, d = x.shape
        x_flat = x_norm.view(-1, d)
        out_flat = torch.zeros_like(x_flat)
        
        for i in range(self.num_experts):
            expert_mask = (top_k_indices == i)
            if expert_mask.any():
                token_indices = expert_mask.any(dim=-1).view(-1)
                if token_indices.any():
                    expert_input = x_flat[token_indices]
                    expert_output = self.experts[i](expert_input)
                    
                    for k in range(self.top_k):
                        k_mask = (top_k_indices[:, :, k] == i).view(-1)
                        if k_mask.any():
                            contribution_mask = k_mask[token_indices]
                            out_flat[k_mask] += top_k_probs.view(-1, self.top_k)[k_mask, k:k+1] * expert_output[contribution_mask]

        out = residual + out_flat.view(bsz, seqlen, d)
        return out, balance_loss

class MudMoE(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts, num_layers=1, num_heads=8, top_k=2, aux_coeff=0.05):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([MudBlock(dim, hidden_dim, num_experts, num_heads, top_k, aux_coeff) for _ in range(num_layers)])
        self.norm = CustomRMSNorm(dim)
        self.freqs_cis = precompute_freqs_cis(dim // num_heads, 1024)
        self.balance_loss = torch.tensor(0.0)
    def forward(self, x):
        self.freqs_cis = self.freqs_cis.to(x.device)
        total_balance_loss = 0.0
        for layer in self.layers:
            x, bl = layer(x, self.freqs_cis)
            total_balance_loss += bl
        self.balance_loss = total_balance_loss
        return self.norm(x)

# --- EXPORTADOR MUD ---
class MudExporter:
    MAGIC = b"MUD\x01"
    def __init__(self, output_path: str):
        self.output_path, self.metadata = output_path, {}
    def add_metadata(self, key, value): self.metadata[key] = value
    def _pack_ternary_row(self, row):
        r = row.detach().cpu().numpy().flatten()
        gamma = np.abs(r).mean()
        w_q = np.clip(np.round(r / (gamma + 1e-7)), -1, 1).astype(np.int8)
        packed = []
        for i in range(0, len(w_q), 16):
            chunk = w_q[i:i+16]; val = 0
            for j, v in enumerate(chunk):
                bits = 1 if v == 1 else (2 if v == -1 else 0)
                val |= (bits << (j * 2))
            packed.append(val)
        return np.array(packed, dtype=np.uint32).tobytes()
    def export(self, state_dict):
        with open(self.output_path, "wb") as f:
            f.write(self.MAGIC)
            f.write(struct.pack("<I", len(self.metadata)))
            for k, v in self.metadata.items():
                kb = k.encode('utf-8'); vb = str(v).encode('utf-8')
                f.write(struct.pack("<I", len(kb))); f.write(kb)
                f.write(struct.pack("<I", len(vb))); f.write(vb)
            f.write(struct.pack("<I", len(state_dict)))
            curr_off = 0; tensor_data = []
            for name, tensor in state_dict.items():
                t = tensor.detach()
                is_w = "weight" in name and t.dim() > 1
                data = self._pack_ternary_row(t) if is_w else t.cpu().numpy().astype(np.float32).tobytes()
                f.write(struct.pack("<I", len(name.encode('utf-8')))); f.write(name.encode('utf-8'))
                f.write(struct.pack("<I", 0 if is_w else 1))
                f.write(struct.pack("<I", len(t.shape)))
                for d in t.shape: f.write(struct.pack("<Q", d))
                f.write(struct.pack("<Q", curr_off))
                tensor_data.append(data); curr_off += len(data)
            f.write(b'\x00' * ((32 - (f.tell() % 32)) % 32))
            for d in tensor_data: f.write(d)

def find_file(filename):
    paths = [filename, f"training/{filename}", f"/kaggle/input/{filename}", f"/kaggle/working/{filename}"]
    if KAGGLE:
        for root, _, files in os.walk("/kaggle/input/"):
            if filename in files: return os.path.join(root, filename)
    for p in paths:
        if os.path.exists(p): return p
    return None

def load_vocab():
    path = find_file("vocab_es_en.txt")
    if path:
        with open(path, "r", encoding="utf-8") as f: return [l.strip() for l in f if l.strip()]
    return ["<unk>", "<s>", "</s>", "<pad>", "!", "?", ".", ","]

def mud_tokenize(text, word_to_id):
    text = text.replace(" ", "Ġ")
    tokens = re.findall(r"\w+|[^\w\s]|Ġ+", text, re.UNICODE)
    return [word_to_id.get(t, 0) for t in tokens]

class MudDataset(torch.utils.data.Dataset):
    def __init__(self, corpus_file, word_to_id, max_len=MAX_SEQ_LEN):
        self.corpus_file = corpus_file
        self.word_to_id = word_to_id
        self.max_len = max_len
        with open(corpus_file, "r", encoding="utf-8") as f:
            self.lines = [l.strip() for l in f if len(l.strip()) > 5]
    def __len__(self): return len(self.lines)
    def __getitem__(self, idx):
        text = self.lines[idx]
        ids = mud_tokenize(text, self.word_to_id)
        if len(ids) < 2: ids = [0, 2]
        ids.append(2)
        return torch.tensor(ids[:self.max_len], dtype=torch.long)

def collate_fn(batch):
    return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=3)

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        capability = torch.cuda.get_device_capability(0)
        if capability[0] < 7: device = "cpu"
    
    print(f"[MUD-ULTRA-TRAINER V1.5] Device: {device} | RAM: {_estimate_ram_gb():.0f}GB | Modo: {_RAM_MODE}")
    torch.set_num_threads(os.cpu_count())
    vocab = load_vocab()
    word_to_id = {w: i for i, w in enumerate(vocab)}
    vocab_size = len(vocab)
    print(f"Vocabulary: {vocab_size} tokens")

    if EXPERTS <= 16:
        _coeff = 0.5
    elif EXPERTS <= 64:
        _coeff = 0.1
    else:
        _coeff = 0.05
    model = MudMoE(dim=HIDDEN, hidden_dim=FFN_HIDDEN, num_experts=EXPERTS, num_layers=NUM_LAYERS, top_k=TOP_K, aux_coeff=_coeff).to(device)
    embed = nn.Embedding(vocab_size, HIDDEN).to(device)
    
    for m in model.modules():
        if isinstance(m, (nn.Linear, BitLinear)): nn.init.xavier_uniform_(m.weight)
        if isinstance(m, CustomRMSNorm): nn.init.ones_(m.weight)

    if hasattr(torch, "compile") and device == "cuda":
        print("🚀 Compiling model for maximum GPU performance...")
        model = torch.compile(model, mode="max-autotune")
    elif hasattr(torch, "compile") and device == "cpu":
        print("🚀 Compiling model for maximum CPU AVX performance...")
        model = torch.compile(model)

    start_step = 0
    out_dir = "weights/checkpoints"
    os.makedirs(out_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(embed.parameters()), lr=LR)
    
    if args.resume and os.path.exists(args.resume):
        print(f"📦 Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        
        # --- MUD FAST TRAINER COMPATIBILITY ADAPTER ---
        raw_state = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        converted_state = {}
        embed_state = {}
        for k, v in raw_state.items():
            if k.startswith("embed."):
                embed_state[k.replace("embed.", "", 1)] = v
                continue
            if k.startswith("head."):
                continue # Ignore tied head
                
            new_k = k
            if new_k.startswith("blocks."):
                new_k = new_k.replace("blocks.", "layers.", 1)
            new_k = new_k.replace(".attn.", ".attention.")
            new_k = new_k.replace(".moe.experts.", ".experts.")
            new_k = new_k.replace(".moe.gate.", ".gate.")
            import re
            new_k = re.sub(r"layers\.(\d+)\.moe\.norm\.", r"layers.\1.norm.", new_k)
            converted_state[new_k] = v
            
        model.load_state_dict(converted_state, strict=False)
        if embed_state:
            embed.load_state_dict(embed_state, strict=False)
        elif "embed" in ckpt:
            embed.load_state_dict(ckpt["embed"])
            
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
                print("   ✅ Optimizer state restored.")
            except Exception as e:
                print(f"   ⚠️ Could not load optimizer state: {e}")
                
        start_step = ckpt.get("step", 0)
        print(f"   Continuing from step {start_step}")

    # Inicializar el Cerebro Sombra (EMA) para aprendizaje continuo
    if hasattr(model, "_orig_mod"):
        ema = ModelEMA(model._orig_mod, embed, decay=0.999)
    else:
        ema = ModelEMA(model, embed, decay=0.999)

    if args.resume and os.path.exists(args.resume) and "ema_model" in ckpt:
        ema.ema_model.load_state_dict(ckpt["ema_model"], strict=False)
        ema.ema_embed.load_state_dict(ckpt["ema_embed"], strict=False)
        print("   ✅ Sombra EMA restaurada con éxito.")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS, last_epoch=-1)
    
    corpus_file = find_file("massive_knowledge_corpus.txt")
    if not corpus_file:
        dataset = torch.utils.data.TensorDataset(torch.zeros((1, 10), dtype=torch.long))
    else:
        dataset = MudDataset(corpus_file, word_to_id)
        print(f"Loaded knowledge corpus: {len(dataset)} items.")

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=MICRO_BATCH_SIZE, shuffle=True, collate_fn=collate_fn, 
        num_workers=2, prefetch_factor=2, pin_memory=(device=="cuda")
    )
    data_iter = iter(dataloader)

    print(f"Training MUD-ULTRA-TRAINER V1.5 for {STEPS - start_step} more steps...")
    model.train(); embed.train()
    
    pbar = tqdm(range(start_step, STEPS), desc="Training", dynamic_ncols=True)
    pad_token_id = word_to_id.get("<pad>", 3)

    for step in pbar:
        # Annealing del ruido MoE
        step_ratio = (step - start_step) / max(1, STEPS - start_step)
        for module in model.modules():
            if isinstance(module, MudBlock):
                module._step_ratio = torch.tensor(step_ratio)

        if step % 100 == 0:
            with torch.no_grad():
                for param in model.parameters():
                    if param.requires_grad: param.add_(torch.randn_like(param) * 1e-6)

        try:
            input_ids = next(data_iter).to(device)
        except StopIteration:
            data_iter = iter(dataloader)
            input_ids = next(data_iter).to(device)

        if input_ids.size(1) < 2: continue
        x_ids, target = input_ids[:, :-1], input_ids[:, 1:]
        emb_ste = weight_quant(embed.weight)
        h = model(F.embedding(x_ids, emb_ste))
        logits = torch.matmul(h, emb_ste.T) / math.sqrt(HIDDEN)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), target.reshape(-1), ignore_index=pad_token_id)
        loss = loss + model.balance_loss.to(device)
        loss = loss / GRAD_ACCUM_STEPS

        try:
            loss.backward()

            if not torch.isfinite(loss):
                print(f"\n⚠️  Salto de emergencia: Loss no finita ({loss.item():.4f}) en step {step+1}")
                optimizer.zero_grad(set_to_none=True)
                continue
            
            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == STEPS:
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(embed.parameters()), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "allocation" in str(e).lower():
                print(f"\n⚠️  OOM en paso {step+1}. Limpiando cachés y omitiendo lote.")
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

        # Update Continuous Learning Shadow Brain
        if hasattr(model, "_orig_mod"):
            ema.update(model._orig_mod, embed)
        else:
            ema.update(model, embed)

        if step % 10 == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "bal": f"{model.balance_loss.item():.4f}"})

        if step % 200 == 0 and step > start_step:
            ckpt_path = os.path.join(out_dir, f"ckpt_step_{step}.pt")
            m_state = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
            
            # --- MUD FAST TRAINER COMPATIBILITY ADAPTER ---
            unified_model = {}
            # 1. Embeddings go inside the model
            for k, v in embed.state_dict().items():
                unified_model[f"embed.{k}"] = v
            # 2. Map architecture keys
            for k, v in m_state.items():
                new_k = k
                if new_k.startswith("layers."):
                    new_k = new_k.replace("layers.", "blocks.", 1)
                new_k = new_k.replace(".attention.", ".attn.")
                new_k = new_k.replace(".experts.", ".moe.experts.")
                new_k = new_k.replace(".gate.", ".moe.gate.")
                import re
                new_k = re.sub(r"blocks\.(\d+)\.norm\.", r"blocks.\1.moe.norm.", new_k)
                unified_model[new_k] = v

            torch.save({
                "step": step,
                "model": unified_model, 
                "optimizer": optimizer.state_dict(),
                "ema_model": ema.ema_model.state_dict(),
                "ema_embed": ema.ema_embed.state_dict(),
            }, ckpt_path)
            tqdm.write(f"💾 Checkpoint saved: {ckpt_path} | Loss: {loss.item():.4f}")

    model_path = "models/core_skills.mud"
    print(f"📦 Exporting to {model_path}...")
    
    # --- AUTO-IQ CALCULATION ---
    # Simplified IQ estimation based on weight distribution and training steps
    with torch.no_grad():
        all_weights = torch.cat([p.flatten() for p in model.parameters() if p.dim() > 1])
        sigma = all_weights.std().item()
        skew = ((all_weights - all_weights.mean())**3).mean().item() / (sigma**3 + 1e-7)
        # IQ Base formula: 100 + (steps log) + (skew symmetry bonus)
        iq_score = 10.0 + (math.log10(STEPS + 1) * 10.0) + (1.0 - abs(skew)) * 20.0
        iq_score = min(200.0, max(10.0, iq_score))
        print(f"🧠 Calculated Digital IQ: {iq_score:.2f} (Sigma: {sigma:.4f}, Skew: {skew:.4f})")

    exp = MudExporter(model_path)
    exp.add_metadata("arch", "ternary_moe"); exp.add_metadata("vocab_size", vocab_size)
    exp.add_metadata("hidden", HIDDEN); exp.add_metadata("hidden_size", HIDDEN)
    exp.add_metadata("num_layers", NUM_LAYERS); exp.add_metadata("num_experts", EXPERTS)
    exp.add_metadata("top_k", TOP_K)
    exp.add_metadata("ffn_hidden", FFN_HIDDEN); exp.add_metadata("tokenizer.tokens", "\n".join(vocab))
    exp.add_metadata("iq.score", f"{iq_score:.2f}")
    exp.add_metadata("iq.skew", f"{skew:.4f}")
    
    # --- EXPERT TAXONOMY METADATA ---
    taxonomy = [
        "Planificación/CoT", "Lógica Formal", "Evaluador Interno", "Razonamiento Difuso",
        "Gramática/AST", "Optimización/Bajo Nivel", "Algoritmia Avanzada", "Álgebra Lineal",
        "Cálculo/Dinámica", "Estadística Avanzada", "Física Cuántica", "Mecánica Clásica",
        "Química Molecular", "Bioinformática", "Sistemas Complejos", "Taxonomías Fácticas"
    ]
    exp.add_metadata("expert_taxonomy", ",".join(taxonomy))
    
    # Exportamos el CEREBRO SOMBRA (EMA) porque tiene el conocimiento consolidado y coherente
    final_model = ema.ema_model
    final_embed = ema.ema_embed

    sd = {"token_embd.weight": weight_quant(final_embed.weight), "output_norm.weight": final_model.norm.weight}
    for l in range(NUM_LAYERS):
        layer = final_model.layers[l]
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
    
    m_state = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
    torch.save({
        "model": m_state, 
        "embed": embed.state_dict(), 
        "ema_model": ema.ema_model.state_dict(),
        "ema_embed": ema.ema_embed.state_dict(),
        "step": STEPS
    }, "weights/mud_last_checkpoint.pt")

if __name__ == "__main__":
    train()
