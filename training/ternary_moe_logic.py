import torch
import torch.nn as nn
import torch.nn.functional as F

def ternary_quantize(w):
    """
    Quantizes weights to {-1, 0, 1} using BitNet 1.58b logic.
    W_q = round(clip(W / gamma, -1, 1))
    """
    gamma = w.abs().mean()
    w_scaled = w / (gamma + 1e-7)
    w_quant = torch.clamp(torch.round(w_scaled), -1, 1)
    return w_quant

class TernaryLinear(nn.Module):
    """
    A linear layer that uses ternary weights during the forward pass.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # Use straight-through estimator (STE) for training
        w_q = ternary_quantize(self.weight)
        w_q = self.weight + (w_q - self.weight).detach()
        return F.linear(x, w_q, self.bias)

class MoEExpert(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = TernaryLinear(dim, hidden_dim, bias=False)
        self.w2 = TernaryLinear(hidden_dim, dim, bias=False)
        self.w3 = TernaryLinear(dim, hidden_dim, bias=False)

    def forward(self, x):
        # SwiGLU with ternary weights
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MudMoE(nn.Module):
    """
    Modular Understanding Dynamics MoE Layer.
    """
    def __init__(self, dim, hidden_dim, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([MoEExpert(dim, hidden_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x):
        # x shape: [batch, seq, dim]
        orig_shape = x.shape
        x = x.view(-1, x.shape[-1])
        
        # Simple Top-K routing
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        top_k_probs /= top_k_probs.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (top_k_indices == i).any(dim=-1)
            if mask.any():
                # Weighted contribution from active expert
                expert_out = expert(x[mask])
                # Find which position in top_k matches this expert to get the prob
                for k in range(self.top_k):
                    k_mask = (top_k_indices[mask][:, k] == i)
                    if k_mask.any():
                        out[mask] += top_k_probs[mask][:, k:k+1] * expert_out
        
        return out.view(*orig_shape)

if __name__ == "__main__":
    # Quick test
    model = MudMoE(dim=512, hidden_dim=1024, num_experts=8)
    test_input = torch.randn(1, 10, 512)
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Ternary MoE Logic Prototype Ready.")
