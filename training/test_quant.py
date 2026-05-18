import torch
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import vulkan_backend as vb
import ctypes

vb._load_lib()
torch.manual_seed(42)
w = torch.randn(512, 512) * 0.1

# Python quantize
def weight_quant(w):
    s = w.abs().mean()
    w_scaled = w / (s + 1e-7)
    w_q = torch.clamp(torch.round(w_scaled), -1, 1)
    return w_q  # pure quantized, no STE

py_q = weight_quant(w)

# Rust quantize
w_flat = w.contiguous().view(-1)
n = w_flat.numel()
packed = torch.empty((n + 15) // 16, dtype=torch.int32)
vb._lib.vb_quantize(
    ctypes.cast(w_flat.data_ptr(), ctypes.POINTER(ctypes.c_float)),
    ctypes.c_uint32(n),
    ctypes.cast(packed.data_ptr(), ctypes.POINTER(ctypes.c_uint32)),
)

# Dequantize Rust
rs_q = torch.zeros(512, 512)
for i in range(512):
    for j in range(512):
        block = packed[i * 32 + j // 16].item()
        bits = (block >> ((j % 16) * 2)) & 3
        rs_q[i, j] = {1: 1.0, 2: -1.0}.get(bits, 0.0)

diff = (py_q - rs_q).abs().max().item()
print(f"Quantization max diff: {diff:.10f}")
print(f"Py unique values: {py_q.unique().tolist()}")
print(f"RS unique values: {rs_q.unique().tolist()}")
print(f"Py non-zero: {(py_q != 0).sum().item()}")
print(f"RS non-zero: {(rs_q != 0).sum().item()}")

if diff > 0:
    idx = (py_q - rs_q).abs().argmax().item()
    i, j = idx // 512, idx % 512
    print(f"First diff at ({i},{j}): py={py_q[i,j].item()}, rs={rs_q[i,j].item()}")
