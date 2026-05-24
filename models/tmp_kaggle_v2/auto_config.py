"""
auto_config.py — MUD Auto-Configurador Escalable
================================================
Lee el perfil de hardware desde knowledge.db y genera configuraciones
óptimas para todos los trainers. Escalable: aprende de ejecuciones
anteriores y ajusta parámetros progresivamente.
"""

import os, json, math, sqlite3, time
from typing import Dict, Optional

DB_PATH = "models/knowledge.db"


def _get_db() -> Optional[sqlite3.Connection]:
    if not os.path.exists(DB_PATH):
        return None
    try:
        conn = sqlite3.connect(DB_PATH, timeout=15.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn
    except Exception:
        return None


def _estimate_ram_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return float(line.split()[1]) / 1_048_576
    except OSError:
        pass
    return 16.0


def _fallback_config() -> Dict:
    ram = _estimate_ram_gb()
    effective_ram = ram * 0.85
    if effective_ram >= 32:
        return dict(mode="big",    num_experts=256, hidden=512,  ffn_hidden=2048,
                    num_layers=4,  top_k=4, batch_size=8,  lr=5e-4, aux_coeff=0.01,  grad_clip=1.0)
    elif effective_ram >= 14:
        return dict(mode="medium", num_experts=64,  hidden=384,  ffn_hidden=1536,
                    num_layers=4,  top_k=4, batch_size=4,  lr=5e-4, aux_coeff=0.05,  grad_clip=1.0)
    elif effective_ram >= 8:
        return dict(mode="small",  num_experts=16,  hidden=256,  ffn_hidden=1024,
                    num_layers=3,  top_k=3, batch_size=4,  lr=5e-4, aux_coeff=0.5,   grad_clip=1.0)
    else:
        return dict(mode="tiny",   num_experts=8,   hidden=192,  ffn_hidden=768,
                    num_layers=2,  top_k=2, batch_size=2,  lr=5e-4, aux_coeff=0.5,   grad_clip=1.0)


def load_training_config(mode_override: Optional[str] = None) -> Dict:
    cfg = _fallback_config()
    conn = _get_db()
    if conn is None:
        return cfg
    try:
        if mode_override:
            cur = conn.execute(
                "SELECT * FROM training_config WHERE mode = ?", (mode_override,)
            )
        else:
            cur = conn.execute(
                "SELECT * FROM training_config ORDER BY id DESC LIMIT 1"
            )
        row = cur.fetchone()
        if row:
            cfg["mode"]        = row["mode"]
            cfg["num_experts"] = row["num_experts"]
            cfg["hidden"]      = row["hidden"]
            cfg["ffn_hidden"]  = row.get("ffn_hidden") or row["hidden"] * 4
            cfg["num_layers"]  = row["num_layers"]
            cfg["top_k"]       = row["top_k"]
            cfg["batch_size"]  = row["batch_size"]
            cfg["lr"]          = row["lr"]
            cfg["aux_coeff"]   = row["aux_coeff"]
            cfg["grad_clip"]   = row["grad_clip"]
        else:
            cur2 = conn.execute(
                "SELECT * FROM hardware_profile ORDER BY id DESC LIMIT 1"
            )
            hw = cur2.fetchone()
            if hw:
                cfg["num_experts"] = hw["num_experts_optimal"]
                cfg["hidden"]      = hw["hidden_optimal"]
                cfg["layers"]      = hw["layers_optimal"]
    except Exception:
        pass
    finally:
        conn.close()
    return cfg


def save_training_result(session_id: str, mode: str, num_experts: int,
                          num_layers: int, hidden: int, steps: int,
                          loss_final: float, avg_it_s: float,
                          avg_tok_s: float, total_time_s: float):
    conn = _get_db()
    if conn is None:
        return
    try:
        ram = _estimate_ram_gb()
        conn.execute("""
            INSERT INTO training_history
            (session_id, mode, num_experts, num_layers, hidden,
             steps, loss_final, avg_it_s, avg_tok_s, total_time_s,
             ram_gb, cpu_cores)
            VALUES (?,?,?,?,?, ?,?,?,?,?, ?,?)
        """, (
            session_id, mode, num_experts, num_layers, hidden,
            steps, loss_final, avg_it_s, avg_tok_s, total_time_s,
            round(ram, 1), os.cpu_count() or 0
        ))
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()


def get_best_config_for_ram(ram_gb: float) -> Dict:
    conn = _get_db()
    if conn is None:
        return _fallback_config()
    try:
        cur = conn.execute("""
            SELECT mode, num_experts, num_layers, hidden, lr, avg_it_s
            FROM training_history
            WHERE ram_gb BETWEEN ? AND ?
            ORDER BY avg_it_s DESC
            LIMIT 1
        """, (ram_gb * 0.85, ram_gb * 1.15))
        best = cur.fetchone()
        if best:
            cfg = load_training_config(best["mode"])
            cfg["historical_best_it_s"] = best["avg_it_s"]
            return cfg
    except Exception:
        pass
    finally:
        conn.close()
    return load_training_config()


def print_config_report(cfg: Dict):
    print(f"\n  📋 Auto-Config [{cfg.get('mode', 'auto').upper()}]")
    print(f"     {cfg['num_experts']} expertos × {cfg['num_layers']} capas")
    print(f"     Hidden={cfg['hidden']}  FFN={cfg.get('ffn_hidden', cfg['hidden']*4)}")
    print(f"     Top-K={cfg['top_k']}  Batch={cfg['batch_size']}  LR={cfg.get('lr', 5e-4):.0e}")
    print(f"     Aux-coeff={cfg.get('aux_coeff', 0.1)}  Grad-clip={cfg.get('grad_clip', 1.0)}")
    if "historical_best_it_s" in cfg:
        print(f"     📈 Baseline histórico: {cfg['historical_best_it_s']:.1f} it/s")
    print()


if __name__ == "__main__":
    cfg = load_training_config()
    print_config_report(cfg)
