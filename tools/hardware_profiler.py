"""
hardware_profiler.py — MUD Hardware Auto-Detection & Benchmarking
================================================================
Detecta CPU, RAM, GPU, Vulkan, AVX features.
Ejecuta micro-benchmarks para calibrar tamaños óptimos.
Persiste todo en knowledge.db para auto-config escalable.
"""

import os, sys, json, time, math, struct, platform, subprocess
import sqlite3
from typing import Dict, Optional
from dataclasses import dataclass, asdict

DB_PATH = "models/knowledge.db"

# ─── Tipos de CPU conocidos con sus benchmarks relativos ──────────────────
CPU_TIERS = {
    "12th Gen Intel Core i7-1260P":   {"cores": 16, "avx2": True, "avx512": False, "tier": "mobile_high"},
    "13th Gen Intel Core i7-1360P":   {"cores": 16, "avx2": True, "avx512": False, "tier": "mobile_high"},
    "Intel(R) Core(TM) i7-1260P":     {"cores": 16, "avx2": True, "avx512": False, "tier": "mobile_high"},
    "Intel(R) Core(TM) i9-13900K":    {"cores": 24, "avx2": True, "avx512": False, "tier": "desktop_ultra"},
    "Intel(R) Core(TM) i7-13700K":    {"cores": 16, "avx2": True, "avx512": False, "tier": "desktop_high"},
    "AMD Ryzen 9 7950X":              {"cores": 16, "avx2": True, "avx512": True,  "tier": "desktop_ultra"},
    "AMD Ryzen 7 7800X3D":            {"cores": 8,  "avx2": True, "avx512": True,  "tier": "desktop_high"},
}

@dataclass
class HardwareProfile:
    cpu_name: str = ""
    cpu_cores: int = 0
    cpu_avx2: bool = False
    cpu_avx512: bool = False
    ram_gb: float = 0.0
    ram_swap_gb: float = 0.0
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    vulkan_available: bool = False
    cpu_tier: str = "unknown"
    benchmark_it_s: float = 0.0
    benchmark_tok_s: float = 0.0
    benchmark_loss: float = 0.0
    num_experts_optimal: int = 8
    hidden_optimal: int = 256
    layers_optimal: int = 2
    timestamp: str = ""


def detect_cpu() -> Dict:
    info = {"name": platform.processor() or platform.machine(), "cores": os.cpu_count() or 1}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["name"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    flags = set()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    flags = set(line.strip().split())
                    break
    except OSError:
        pass
    info["avx2"] = "avx2" in flags
    info["avx512"] = "avx512f" in flags
    return info


def detect_ram() -> Dict:
    info = {"gb": 0.0, "swap_gb": 0.0}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    info["gb"] = float(line.split()[1]) / 1_048_576
                if line.startswith("SwapTotal:"):
                    info["swap_gb"] = float(line.split()[1]) / 1_048_576
    except OSError:
        pass
    return info


def detect_gpu_vulkan() -> Dict:
    info = {"name": "", "vram_gb": 0.0, "available": False}
    try:
        result = subprocess.run(
            ["vulkaninfo", "--summary"],
            capture_output=True, text=True, timeout=15
        )
        for line in result.stdout.split("\n"):
            if "deviceName" in line:
                info["name"] = line.split("=", 1)[1].strip()
                info["available"] = True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    if not info["available"]:
        try:
            import torch
            if torch.cuda.is_available():
                info["name"] = torch.cuda.get_device_name(0)
                info["vram_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
                info["available"] = True
        except Exception:
            pass
    return info


def estimate_optimal_config(ram_gb: float, cpu_cores: int,
                             avx2: bool, avx512: bool, gpu_vram: float) -> Dict:
    # Reduce memory footprint by 15% to avoid swapping during training
    effective_ram = ram_gb * 0.85
    if effective_ram >= 32:
        return {"num_experts": 256, "hidden": 512, "ffn_hidden": 2048,
                "layers": 4, "top_k": 4, "batch_size": 8, "mode": "big"}
    elif effective_ram >= 14:
        return {"num_experts": 64,  "hidden": 384, "ffn_hidden": 1536,
                "layers": 4, "top_k": 4, "batch_size": 4, "mode": "medium"}
    elif effective_ram >= 8:
        return {"num_experts": 16,  "hidden": 256, "ffn_hidden": 1024,
                "layers": 3, "top_k": 2, "batch_size": 4, "mode": "small"}
    else:
        return {"num_experts": 8,   "hidden": 192, "ffn_hidden": 768,
                "layers": 2, "top_k": 2, "batch_size": 2, "mode": "tiny"}


def run_micro_benchmark(cfg: Dict) -> Dict:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        E, H, L, K = cfg["num_experts"], cfg["hidden"], cfg["layers"], cfg["top_k"]
        B, T = min(cfg["batch_size"], 4), 64
        device = "cuda" if torch.cuda.is_available() else "cpu"

        class DummyExpert(nn.Module):
            def __init__(self): super().__init__()
            w1 = nn.Linear(H, H * 2)
            w2 = nn.Linear(H * 2, H)
            def forward(self, x): return self.w2(F.relu(self.w1(x)))

        class DummyMoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = nn.ModuleList([DummyExpert() for _ in range(E)])
                self.gate = nn.Linear(H, E)
            def forward(self, x):
                logits = self.gate(x)
                topk_p, topk_i = torch.topk(F.softmax(logits, dim=-1), K, dim=-1)
                flat = x.view(-1, H)
                out = torch.zeros_like(flat)
                for i in range(E):
                    mask = (topk_i.view(-1, K) == i).any(dim=-1)
                    if mask.any(): out[mask] += self.experts[i](flat[mask])
                return out.view(x.shape)

        model = DummyMoE().to(device)
        inp = torch.randn(B, T, H, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        t0 = time.time()
        steps = 10
        for _ in range(steps):
            optimizer.zero_grad()
            out = model(inp)
            loss = out.mean()
            loss.backward()
            optimizer.step()
        elapsed = time.time() - t0
        it_s = steps / elapsed
        tok_s = steps * B * T / elapsed
        return {"it_s": round(it_s, 2), "tok_s": round(tok_s, 0),
                "loss": round(loss.item(), 4)}
    except Exception as e:
        return {"it_s": 0, "tok_s": 0, "loss": 0, "error": str(e)}


def save_profile_to_db(profile: HardwareProfile):
    os.makedirs(os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else ".", exist_ok=True)
    try:
        conn = sqlite3.connect(DB_PATH, timeout=15.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS hardware_profile (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cpu_name TEXT,
                cpu_cores INTEGER,
                cpu_avx2 INTEGER,
                cpu_avx512 INTEGER,
                ram_gb REAL,
                ram_swap_gb REAL,
                gpu_name TEXT,
                gpu_vram_gb REAL,
                vulkan_available INTEGER,
                cpu_tier TEXT,
                benchmark_it_s REAL,
                benchmark_tok_s REAL,
                benchmark_loss REAL,
                num_experts_optimal INTEGER,
                hidden_optimal INTEGER,
                layers_optimal INTEGER,
                timestamp TEXT DEFAULT (datetime('now')),
                UNIQUE(timestamp)
            );
            CREATE TABLE IF NOT EXISTS training_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mode TEXT UNIQUE,
                num_experts INTEGER,
                hidden INTEGER,
                ffn_hidden INTEGER,
                num_layers INTEGER,
                top_k INTEGER,
                batch_size INTEGER,
                lr REAL,
                aux_coeff REAL,
                grad_clip REAL,
                updated_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS training_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                mode TEXT,
                num_experts INTEGER,
                num_layers INTEGER,
                hidden INTEGER,
                steps INTEGER,
                loss_final REAL,
                avg_it_s REAL,
                avg_tok_s REAL,
                total_time_s REAL,
                ram_gb REAL,
                cpu_cores INTEGER,
                timestamp TEXT DEFAULT (datetime('now'))
            );
        """)
        conn.execute("""
            INSERT OR REPLACE INTO hardware_profile
            (cpu_name, cpu_cores, cpu_avx2, cpu_avx512, ram_gb, ram_swap_gb,
             gpu_name, gpu_vram_gb, vulkan_available, cpu_tier,
             benchmark_it_s, benchmark_tok_s, benchmark_loss,
             num_experts_optimal, hidden_optimal, layers_optimal)
            VALUES (?,?,?,?,?,?, ?,?,?,?, ?,?,?, ?,?,?)
        """, (
            profile.cpu_name, profile.cpu_cores,
            int(profile.cpu_avx2), int(profile.cpu_avx512),
            round(profile.ram_gb, 1), round(profile.ram_swap_gb, 1),
            profile.gpu_name, round(profile.gpu_vram_gb, 1),
            int(profile.vulkan_available), profile.cpu_tier,
            profile.benchmark_it_s, profile.benchmark_tok_s,
            profile.benchmark_loss,
            profile.num_experts_optimal, profile.hidden_optimal,
            profile.layers_optimal
        ))
        config = estimate_optimal_config(
            profile.ram_gb, profile.cpu_cores,
            profile.cpu_avx2, profile.cpu_avx512, profile.gpu_vram_gb
        )
        conn.execute("""
            INSERT OR REPLACE INTO training_config
            (mode, num_experts, hidden, ffn_hidden, num_layers, top_k,
             batch_size, lr, aux_coeff, grad_clip)
            VALUES (?,?,?,?,?,?, ?,?,?,?)
        """, (
            config["mode"], config["num_experts"], config["hidden"],
            config["ffn_hidden"], config["layers"], config["top_k"],
            config["batch_size"], 5e-4,
            0.01 if config["num_experts"] >= 256 else
            0.05 if config["num_experts"] >= 64 else
            0.2 if config["num_experts"] >= 16 else 0.5,
            1.0
        ))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"  ⚠️  DB error saving profile: {e}")
        return False


def load_latest_profile() -> Optional[Dict]:
    if not os.path.exists(DB_PATH):
        return None
    try:
        conn = sqlite3.connect(DB_PATH, timeout=5.0)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT * FROM hardware_profile ORDER BY id DESC LIMIT 1"
        )
        row = cur.fetchone()
        conn.close()
        if row: return dict(row)
    except Exception:
        pass
    return None


def profile() -> HardwareProfile:
    print("  🔍 Perfilando hardware...")
    cpu = detect_cpu()
    ram = detect_ram()
    gpu = detect_gpu_vulkan()

    cpu_tier = "unknown"
    for known_name, info in CPU_TIERS.items():
        if known_name.lower() in cpu["name"].lower():
            cpu_tier = info["tier"]
            break

    print(f"     CPU: {cpu['name']} ({cpu['cores']} cores)")
    print(f"     AVX2: {'✅' if cpu['avx2'] else '❌'}  AVX512: {'✅' if cpu['avx512'] else '❌'}")
    print(f"     RAM: {ram['gb']:.1f}GB  Swap: {ram['swap_gb']:.1f}GB")
    print(f"     GPU: {gpu['name'] or 'No detectada'}  VRAM: {gpu['vram_gb']:.1f}GB" if gpu['name'] else "     GPU: No detectada")

    config = estimate_optimal_config(ram["gb"], cpu["cores"], cpu["avx2"], cpu["avx512"], gpu["vram_gb"])
    print(f"     Config óptima estimada: {config['mode']} ({config['num_experts']} expertos)")

    print("     Ejecutando micro-benchmark (10 steps)...")
    bench = run_micro_benchmark(config)

    profile = HardwareProfile(
        cpu_name=cpu["name"],
        cpu_cores=cpu["cores"],
        cpu_avx2=cpu["avx2"],
        cpu_avx512=cpu["avx512"],
        ram_gb=ram["gb"],
        ram_swap_gb=ram["swap_gb"],
        gpu_name=gpu["name"],
        gpu_vram_gb=gpu["vram_gb"],
        vulkan_available=gpu["available"],
        cpu_tier=cpu_tier,
        benchmark_it_s=bench.get("it_s", 0),
        benchmark_tok_s=bench.get("tok_s", 0),
        benchmark_loss=bench.get("loss", 0),
        num_experts_optimal=config["num_experts"],
        hidden_optimal=config["hidden"],
        layers_optimal=config["layers"],
    )
    save_profile_to_db(profile)
    print(f"     Benchmark: {bench.get('it_s', 0):.1f} it/s, {bench.get('tok_s', 0):.0f} tok/s")
    print(f"  ✅ Perfil guardado en {DB_PATH}")
    return profile


if __name__ == "__main__":
    profile()
