"""
bench_trainer.py — Benchmark de velocidad del entrenamiento local MUD
Mide it/s y tokens/s con y sin BF16 / torch.compile para comparación.
"""
import torch, time, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.mud_fast_trainer import (
    MudModel, HIDDEN, FFN_HIDDEN, EXPERTS, NUM_LAYERS, NUM_HEADS, TOP_K,
    NUM_THREADS
)

VOCAB_SIZE = 8000   # aproximado
SEQ_LEN    = 64
WARMUP     = 5
BENCH      = 30

def run_bench(label: str, model, use_bf16: bool):
    model.train()
    dummy_ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
    dummy_tgt = torch.randint(0, VOCAB_SIZE, (SEQ_LEN,))
    import torch.nn.functional as F

    opt = torch.optim.AdamW(
        (model._orig_mod if hasattr(model, "_orig_mod") else model).parameters(),
        lr=1e-4
    )

    # Warmup
    for _ in range(WARMUP):
        ctx = torch.autocast("cpu", dtype=torch.bfloat16, enabled=use_bf16)
        with ctx:
            logits, bl = model(dummy_ids)
            loss = F.cross_entropy(logits.squeeze(0).float(), dummy_tgt) + bl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(BENCH):
        ctx = torch.autocast("cpu", dtype=torch.bfloat16, enabled=use_bf16)
        with ctx:
            logits, bl = model(dummy_ids)
            loss = F.cross_entropy(logits.squeeze(0).float(), dummy_tgt) + bl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (model._orig_mod if hasattr(model, "_orig_mod") else model).parameters(), 1.0
        )
        opt.step()

    elapsed = time.perf_counter() - t0
    it_s    = BENCH / elapsed
    tok_s   = BENCH * SEQ_LEN / elapsed
    print(f"  {label:<35} {it_s:>6.2f} it/s   {tok_s:>8.0f} tok/s")
    return it_s

if __name__ == "__main__":
    torch.set_num_threads(NUM_THREADS)
    print(f"\n{'='*60}")
    print(f"  MUD Trainer — Benchmark de velocidad")
    print(f"  Threads: {NUM_THREADS} | seq_len: {SEQ_LEN} | pasos: {BENCH}")
    print(f"{'='*60}")

    # Baseline: FP32, sin compile
    m_base = MudModel(VOCAB_SIZE, HIDDEN, FFN_HIDDEN, EXPERTS, NUM_LAYERS, NUM_HEADS, TOP_K)
    r1 = run_bench("FP32  / sin compile", m_base, use_bf16=False)

    # BF16, sin compile
    m_bf16 = MudModel(VOCAB_SIZE, HIDDEN, FFN_HIDDEN, EXPERTS, NUM_LAYERS, NUM_HEADS, TOP_K)
    r2 = run_bench("BF16  / sin compile", m_bf16, use_bf16=True)

    # FP32, con compile
    m_comp = MudModel(VOCAB_SIZE, HIDDEN, FFN_HIDDEN, EXPERTS, NUM_LAYERS, NUM_HEADS, TOP_K)
    print("  (compilando — primer paso lento)...")
    m_comp = torch.compile(m_comp, mode="reduce-overhead")
    r3 = run_bench("FP32  / torch.compile", m_comp, use_bf16=False)

    # BF16, con compile — configuración óptima
    m_best = MudModel(VOCAB_SIZE, HIDDEN, FFN_HIDDEN, EXPERTS, NUM_LAYERS, NUM_HEADS, TOP_K)
    print("  (compilando — primer paso lento)...")
    m_best = torch.compile(m_best, mode="reduce-overhead")
    r4 = run_bench("BF16  / torch.compile  ⭐ ÓPTIMO", m_best, use_bf16=True)

    print(f"\n  Mejora total vs baseline: {r4/r1:.1f}x más rápido")
    print(f"{'='*60}\n")
