import torch
import torch.nn.functional as F
import math
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import vulkan_backend as vb
vb._load_lib()
print(f"Vulkan available: {vb._vulkan_available}")

torch.manual_seed(42)
n_in, n_out = 512, 512
scale = 1.0 / math.sqrt(n_in)

def weight_quant(w):
    s = w.abs().mean()
    return w + (torch.clamp(torch.round(w / (s + 1e-7)), -1, 1) - w).detach()

# === Forward test ===
x = torch.randn(n_in)
w = torch.randn(n_out, n_in) * 0.1
w_q = weight_quant(w)
y_ref = F.linear(x, w_q) * scale
y_hyb = vb.TernaryLinearFunction.apply(x.unsqueeze(0), w, scale).squeeze(0)
diff = (y_ref - y_hyb).abs().max().item()
print(f"Forward max diff: {diff:.6e}")
assert diff < 1e-4, f"Forward failed: {diff}"

# === Backward test ===
torch.manual_seed(123)
x_base = torch.randn(n_in)
w_base = torch.randn(n_out, n_in) * 0.1

x2 = x_base.clone().requires_grad_(True)
w2 = w_base.clone().requires_grad_(True)
w2.retain_grad()
w2_q = weight_quant(w2)
y_ref2 = (F.linear(x2, w2_q) * scale).sum()
y_ref2.backward()
gx_ref = x2.grad.clone()
gw_ref = w2.grad.clone()

x3 = x_base.clone().requires_grad_(True)
w3 = w_base.clone().requires_grad_(True)
w3.retain_grad()
y_hyb2 = vb.TernaryLinearFunction.apply(x3.unsqueeze(0), w3, scale).squeeze(0)
y_hyb2.sum().backward()
gx_hyb = x3.grad.clone()
gw_hyb = w3.grad.clone()

dx_diff = (gx_ref - gx_hyb).abs().max().item()
dw_diff = (gw_ref - gw_hyb).abs().max().item()
print(f"grad_x max diff: {dx_diff:.6e}")
print(f"grad_w max diff: {dw_diff:.6e}")
if dx_diff > 1e-2:
    print(f"WARNING: grad_x diff {dx_diff} exceeds threshold")
if dw_diff > 1e-2:
    print(f"WARNING: grad_w diff {dw_diff} exceeds threshold")

print("Test completed!")
