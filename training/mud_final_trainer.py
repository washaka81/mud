import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import sys
import multiprocessing
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

# --- AUTO-CONFIG (opcional) ---
_HAS_AC = False; _AC = {}
if os.environ.get("MUD_AUTO_CONFIG", "1") == "1":
    try:
        from auto_config import load_training_config
        _AC = load_training_config("small")
        _HAS_AC = True
    except Exception:
        pass

try:
    from training import vulkan_backend
    from training.vulkan_backend import TernaryLinearFunction, _load_lib
    _load_lib()
    _vulkan_available = vulkan_backend._vulkan_available
    print(f"[MUD-TRAINER] Vulkan disponible: {_vulkan_available}")
except Exception as e:
    print(f"[MUD-TRAINER] Vulkan no disponible: {e}, usando CPU fallback")
    _vulkan_available = False

# --- CONFIGURATION (con override de auto-config) ---
HIDDEN     = _AC.get("hidden", 512)      if _HAS_AC else 512
FFN_HIDDEN = _AC.get("ffn_hidden", 2048) if _HAS_AC else 2048
EXPERTS    = _AC.get("num_experts", 8)   if _HAS_AC else 8
TOP_K      = _AC.get("top_k", 2)         if _HAS_AC else 2
NUM_LAYERS = _AC.get("num_layers", 4)    if _HAS_AC else 4
LR         = _AC.get("lr", 3e-4)         if _HAS_AC else 3e-4
STEPS = 50000  
BATCH_SIZE = 8
MAX_SEQ_LEN = 128

if _HAS_AC:
    print(f"  ⚙️  Auto-config: {EXPERTS} experts, {NUM_LAYERS} layers, hidden={HIDDEN}")
CHECKPOINT_DIR = "checkpoints_final"
MODEL_OUTPUT = "models/v1_master_local.mud"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

# --- TERNARY ARCHITECTURE ---
class TernaryLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.register_buffer("scale", torch.tensor(1.0))

    def forward(self, x):
        with torch.no_grad():
            self.scale.copy_(self.weight.abs().mean().clamp(min=1e-7))
        if _vulkan_available:
            return TernaryLinearFunction.apply(x, self.weight, self.scale)
        w_q = (self.weight / self.scale).clamp(-1, 1)
        w_q = self.weight + (w_q.round() - w_q).detach()
        return F.linear(x, w_q * self.scale)

class MoEExpert(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = TernaryLinear(dim, hidden_dim)
        self.w2 = TernaryLinear(hidden_dim, dim)
        self.w3 = TernaryLinear(dim, hidden_dim)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MudBlock(nn.Module):
    def __init__(self, dim, hidden_dim, num_experts, top_k=2, aux_coeff=0.05):
        super().__init__()
        self.experts = nn.ModuleList([MoEExpert(dim, hidden_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.norm = nn.RMSNorm(dim)
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_coeff = aux_coeff
        self.register_buffer("_step_ratio", torch.tensor(0.0))

    def forward(self, x):
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
                        out_flat[mask] += (probs_flat[mask][:, k:k+1] * expert_out)

        return residual + out_flat.view(bsz, seqlen, d), balance_loss

class MUDV1Master(nn.Module):
    def __init__(self, vocab_size, dim, hidden_dim, num_experts, num_layers, aux_coeff=0.05):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([MudBlock(dim, hidden_dim, num_experts, aux_coeff=aux_coeff) for _ in range(num_layers)])
        self.norm = nn.RMSNorm(dim)
        self.balance_loss = torch.tensor(0.0)

    def forward(self, x):
        h = self.embedding(x)
        total_bl = 0.0
        for layer in self.layers:
            h, bl = layer(h)
            total_bl += bl
        self.balance_loss = total_bl
        h = self.norm(h)
        return F.linear(h, self.embedding.weight)

def load_corpus(vocab):
    corpus_path = "training/massive_knowledge_corpus.txt"
    if not os.path.exists(corpus_path): return []
    word_to_id = {w: i for i, w in enumerate(vocab)}
    encoded = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            ids = [word_to_id.get(t, 0) for t in tokens]
            if len(ids) > 5: encoded.append(torch.tensor(ids[:MAX_SEQ_LEN], dtype=torch.long))
    return encoded

def load_vocab():
    return [chr(i) for i in range(256)] + ["<EOS>", "<PAD>", "<UNK>"]

SKILL_MAP = {
    0: "Linguistics (ES/EN)",
    1: "Logic/Math",
    2: "Programming/Code",
    3: "General Knowledge",
    4: "Personality/Charisma",
    5: "MUD Project Memory",
    6: "Data Analysis",
    7: "Neural Kick (Creative)"
}

skill_stats = {i: {"acc": 0.0, "count": 0, "iq": 8.87} for i in range(8)}

def approximate_iq_by_skill(skill_idx, loss, acc, step):
    base = (acc * 120) + (1.0 / (loss + 0.1) * 30)
    growth = min(1.0, step / 15000)
    return max(8.87, base * growth)

def detect_skill(ids, vocab):
    text = " ".join([vocab[i] if i < len(vocab) else "" for i in ids[:20]]).lower()
    if any(k in text for k in ["es:", "en:", "traducción"]): return 0
    if any(k in text for k in ["math", "lógica", "calcul", "iq"]): return 1
    if any(k in text for k in ["rust", "python", "code", "fn "]): return 2
    if any(k in text for k in ["mud", "forge", "motor", "sistema"]): return 5
    if any(k in text for k in ["¡", "!", "hola", "soy"]): return 4 
    return 3

# --- CURIOSITY ENGINE ---
def analyze_cognitive_needs(stats, step_loss_history):
    sorted_skills = sorted(stats.items(), key=lambda x: x[1]["iq"])
    needed_idx = sorted_skills[0][0]
    
    # Curiosity Trigger: Low Creative Expert usage or overall Loss Plateau
    total_usage = sum(s["count"] for s in stats.values())
    recent_loss_std = np.std(step_loss_history[-100:]) if len(step_loss_history) > 100 else 1.0
    
    curiosity_trigger = (stats[7]["count"] < (total_usage / 30)) or (recent_loss_std < 0.005)
    
    print("\n" + "✨" * 20)
    if curiosity_trigger:
        print(f"🌀 MUD: 'Siento una saturación cognitiva. Mis predicciones son demasiado estables.'")
        print(f"🔭 MUD: 'Necesito EXPANDIR mis horizontes hacia ÁREAS DESCONOCIDAS. ¿Tienes datos sobre temas que nunca hayamos explorado?'")
    else:
        print(f"🤖 MUD: 'He analizado mis sinapsis y mi IQ en {SKILL_MAP[needed_idx]} es insuficiente ({stats[needed_idx]['iq']:.2f}).'")
        print(f"❓ MUD: '¿Podrías proporcionarme más materias sobre {SKILL_MAP[needed_idx]}? Mi evolución depende de ello.'")
    print("✨" * 20 + "\n")

def train():
    print("=== MUD SKILL-BASED ASSIMILATION ENGINE START ===")
    vocab = load_vocab(); vocab_size = len(vocab)
    _coeff = 0.5 if EXPERTS <= 16 else (0.1 if EXPERTS <= 64 else 0.05)
    model = MUDV1Master(vocab_size, HIDDEN, FFN_HIDDEN, EXPERTS, NUM_LAYERS, aux_coeff=_coeff)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    loss_history = []; start_step = 0
    if os.path.exists(CHECKPOINT_DIR):
        ckpt_files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")], key=lambda x: int(x.split("_")[2].split(".")[0]))
        if ckpt_files:
            ckpt = torch.load(os.path.join(CHECKPOINT_DIR, ckpt_files[-1]), weights_only=False)
            model.load_state_dict(ckpt['model_state_dict']); optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_step = ckpt['step']; global skill_stats; skill_stats = ckpt.get('skill_stats', skill_stats)

    dataset = load_corpus(vocab)
    if not dataset: return

    pbar = tqdm(range(start_step, STEPS), desc="Skill Growth")
    running_loss = 0
    for step in pbar:
        # Annealing del ruido MoE
        step_ratio = (step - start_step) / max(1, STEPS - start_step)
        for module in model.modules():
            if isinstance(module, MudBlock):
                module._step_ratio = torch.tensor(step_ratio)
        batch_idx = np.random.randint(0, len(dataset), BATCH_SIZE)
        batch = [dataset[i] for i in batch_idx]
        batch_skills = [detect_skill(seq.tolist(), vocab) for seq in batch]
        
        x_ids = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=vocab.index("<PAD>"))
        if x_ids.size(1) < 2: continue
        inputs, targets = x_ids[:, :-1], x_ids[:, 1:]
        
        optimizer.zero_grad(); logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1)) + model.balance_loss
        
        try:
            loss.backward()
            if not torch.isfinite(loss):
                print(f"\\n⚠️  Salto de emergencia: Loss no finita. Ignorando lote.")
                optimizer.zero_grad(set_to_none=True)
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "allocation" in str(e).lower():
                print(f"\\n⚠️  OOM en paso {step+1}. Limpiando cachés y omitiendo lote.")
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                import gc; gc.collect()
                continue
            else:
                raise e
        
        loss_history.append(loss.item()); running_loss = 0.9 * running_loss + 0.1 * loss.item()
        
        preds = logits.argmax(-1); correct = (preds == targets).float()
        for i, skill_idx in enumerate(batch_skills):
            skill_stats[skill_idx]["acc"] = 0.95 * skill_stats[skill_idx]["acc"] + 0.05 * correct[i].mean().item()
            skill_stats[skill_idx]["count"] += 1
            skill_stats[skill_idx]["iq"] = approximate_iq_by_skill(skill_idx, loss.item(), skill_stats[skill_idx]["acc"], step)

        if step % 50 == 0:
            top_skill = max(skill_stats.items(), key=lambda x: x[1]["iq"])
            pbar.set_description(f"Top: {SKILL_MAP[top_skill[0]][:10]} (IQ:{top_skill[1]['iq']:.1f}) | Loss:{running_loss:.3f}")
            
        if (step + 1) % 500 == 0:
            target_ckpt = os.path.join(CHECKPOINT_DIR, f"ckpt_step_{step+1}.pt")
            tmp_ckpt = target_ckpt + ".tmp"
            torch.save({'step': step + 1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'skill_stats': skill_stats}, tmp_ckpt)
            os.replace(tmp_ckpt, target_ckpt)
            print(f"\nSTEP {step+1} DASHBOARD")
            for idx, name in SKILL_MAP.items():
                print(f"- {name:20}: IQ {skill_stats[idx]['iq']:>6.2f} | Usage: {skill_stats[idx]['count']}")
            analyze_cognitive_needs(skill_stats, loss_history)

    print("✅ Assimilation Complete.")

if __name__ == "__main__":
    train()
