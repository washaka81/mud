import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from tqdm import tqdm
from training import vulkan_backend
from training.vulkan_backend import TernaryLinearFunction, _load_lib

# Load the Vulkan backend
_load_lib()
_vulkan_available = vulkan_backend._vulkan_available
print(f"Vulkan available: {_vulkan_available}")

# --- CONFIG ---
HIDDEN = 256  # Smaller for speed
FFN_HIDDEN = 1024
EXPERTS = 4
TOP_K = 2
NUM_LAYERS = 1
LR = 1e-3
STEPS = 235 # Increased for verification test
BATCH_SIZE = 4
MAX_SEQ_LEN = 32
CHECKPOINT_DIR = "checkpoints_vulkan"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class TernaryLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.register_buffer("scale", torch.tensor(1.0))

    def forward(self, x):
        # Dynamic scale for BitNet 1.58b
        with torch.no_grad():
            self.scale.copy_(self.weight.abs().mean().clamp(min=1e-7))
        return TernaryLinearFunction.apply(x, self.weight, self.scale)

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
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_coeff = aux_coeff
        self.register_buffer("_step_ratio", torch.tensor(0.0))
        self.experts = nn.ModuleList([MoEExpert(dim, hidden_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.norm = nn.RMSNorm(dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        gate_logits = self.gate(x)

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
        
        # Normalize top-k probs
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        bsz, seqlen, d = x.shape
        x_flat = x.view(-1, d)
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

class VulkanModel(nn.Module):
    def __init__(self, vocab_size, dim, hidden_dim, num_experts, num_layers, aux_coeff=0.05):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([MudBlock(dim, hidden_dim, num_experts, aux_coeff=aux_coeff) for _ in range(num_layers)])
        self.norm = nn.RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.balance_loss = torch.tensor(0.0)

    def forward(self, x):
        h = self.embedding(x)
        total_bl = 0.0
        for layer in self.layers:
            h, bl = layer(h)
            total_bl += bl
        self.balance_loss = total_bl
        h = self.norm(h)
        return self.output(h)

def get_latest_checkpoint():
    if not os.path.exists(CHECKPOINT_DIR):
        return None
    files = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")]
    if not files:
        return None
    # Extract step number from ckpt_step_N.pt
    steps = [int(f.split("_")[2].split(".")[0]) for f in files]
    latest_step = max(steps)
    return os.path.join(CHECKPOINT_DIR, f"ckpt_step_{latest_step}.pt")

def benchmark(resume=True):
    print("--- Starting Vulkan Training Session ---")
    vocab_size = 1000
    _coeff = 0.5 if EXPERTS <= 16 else (0.1 if EXPERTS <= 64 else 0.05)
    model = VulkanModel(vocab_size, HIDDEN, FFN_HIDDEN, EXPERTS, NUM_LAYERS, aux_coeff=_coeff)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    start_step = 0
    if resume:
        ckpt_path = get_latest_checkpoint()
        if ckpt_path:
            print(f"Resuming from checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_step = checkpoint['step']
            print(f"Resumed at step {start_step}")

    # Mock data
    data = torch.randint(0, vocab_size, (BATCH_SIZE, MAX_SEQ_LEN))
    
    start_time = time.time()
    for step in range(start_step, STEPS):
        step_start = time.time()

        # Annealing del ruido MoE
        step_ratio = (step - start_step) / max(1, STEPS - start_step)
        for module in model.modules():
            if isinstance(module, MudBlock):
                module._step_ratio = torch.tensor(step_ratio)

        optimizer.zero_grad()
        logits = model(data[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), data[:, 1:].reshape(-1)) + model.balance_loss
        loss.backward()
        optimizer.step()

        # Limpiar caches Vulkan entre pasos para evitar acumulación de buffers stale
        if _vulkan_available:
            from training.vulkan_backend import clear_caches
            clear_caches()
        
        step_time = time.time() - step_start
        if (step + 1) % 5 == 0:
            print(f"Step {step+1}/{STEPS} | Loss: {loss.item():.4f} | Time: {step_time:.4f}s")
            
            # Checkpoint
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"ckpt_step_{step+1}.pt")
            torch.save({
                'step': step + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            # Remove old checkpoints to save space (keep only latest 3)
            manage_checkpoints()

    total_time = time.time() - start_time
    print(f"--- Session Finished ---")
    if STEPS > start_step:
        print(f"Time for {STEPS - start_step} steps: {total_time:.2f}s")
        print(f"Average time per step: {total_time/(STEPS-start_step):.4f}s")

def manage_checkpoints():
    files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")], 
                   key=lambda x: int(x.split("_")[2].split(".")[0]))
    if len(files) > 3:
        for f in files[:-3]:
            os.remove(os.path.join(CHECKPOINT_DIR, f))

if __name__ == "__main__":
    benchmark(resume=True)
