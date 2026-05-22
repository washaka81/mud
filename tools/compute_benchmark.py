import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from training import vulkan_backend
from training.vulkan_backend import TernaryLinearFunction, _load_lib

# --- CONFIG ---
HIDDEN = 512
FFN_HIDDEN = 2048
BATCH_SIZE = 16
SEQ_LEN = 128
ITERATIONS = 50

# Load Lib for Vulkan
try:
    _load_lib()
    VULKAN_AVAILABLE = vulkan_backend._vulkan_available
except:
    VULKAN_AVAILABLE = False

# --- KERNELS ---

# 1. Native PyTorch CPU (AVX2)
def pytorch_cpu_ternary(x, w, scale):
    # Simulation of BitNet 1.58b
    w_q = torch.clamp(torch.round(w / (scale + 1e-7)), -1, 1)
    return F.linear(x, w_q) * scale

# 2. Forge Vulkan (iGPU)
def forge_vulkan_ternary(x, w, scale):
    return TernaryLinearFunction.apply(x, w, scale)

def run_benchmark():
    print(f"=== MUD COMPUTE BENCHMARK ===")
    print(f"Hardware: i7-1260P | Iris Xe")
    print(f"Vulkan Available: {VULKAN_AVAILABLE}")
    print(f"Config: Hidden={HIDDEN}, FFN={FFN_HIDDEN}, Batch={BATCH_SIZE}, Seq={SEQ_LEN}")
    print("-" * 40)

    x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN)
    w = torch.randn(FFN_HIDDEN, HIDDEN)
    scale = torch.tensor(0.5)

    # --- TEST 1: CPU ONLY (PyTorch Native) ---
    print("Testing CPU ONLY (Native PyTorch AVX2)...")
    start = time.time()
    for _ in range(ITERATIONS):
        _ = pytorch_cpu_ternary(x, w, scale)
    cpu_time = (time.time() - start) / ITERATIONS
    print(f"  > Avg Time: {cpu_time*1000:.2f} ms")

    # --- TEST 2: GPU ONLY (Forge Vulkan) ---
    if VULKAN_AVAILABLE:
        print("Testing GPU ONLY (Forge Vulkan Iris Xe)...")
        # Warmup
        for _ in range(5): _ = forge_vulkan_ternary(x, w, scale)
        start = time.time()
        for _ in range(ITERATIONS):
            _ = forge_vulkan_ternary(x, w, scale)
        gpu_time = (time.time() - start) / ITERATIONS
        print(f"  > Avg Time: {gpu_time*1000:.2f} ms")
    else:
        gpu_time = float('inf')
        print("  > GPU NOT AVAILABLE")

    # --- TEST 3: HYBRID (Strategy Proposal) ---
    print("Testing HYBRID STRATEGY (Simulated)...")
    # In hybrid, we might run Attention on CPU and MoE on GPU
    # Here we simulate the cost of 1 CPU op + 1 GPU op
    if VULKAN_AVAILABLE:
        hybrid_time = (cpu_time * 0.3) + (gpu_time * 0.7) # Weights based on MUD architecture
        print(f"  > Estimated Avg Time: {hybrid_time*1000:.2f} ms")
    else:
        print("  > HYBRID NOT POSSIBLE")

    print("-" * 40)
    if VULKAN_AVAILABLE:
        speedup = cpu_time / gpu_time
        print(f"FINAL VERDICT: Vulkan is {speedup:.2x} faster than CPU.")
    else:
        print("FINAL VERDICT: Stick to CPU AVX2.")

if __name__ == "__main__":
    run_benchmark()
