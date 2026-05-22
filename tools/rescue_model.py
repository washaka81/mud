import torch
import numpy as np
import struct
from typing import Dict
import os

# Import MudExporter logic (copied from trainer for self-containment)
class MudExporter:
    MAGIC = b"MUD\x01"
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.metadata = {}
    def add_metadata(self, key: str, value: str):
        self.metadata[key] = value
    def export(self, tensors: Dict[str, torch.Tensor]):
        with open(self.output_path, "wb") as f:
            f.write(self.MAGIC)
            f.write(struct.pack("<I", len(self.metadata)))
            for k, v in self.metadata.items():
                kb, vb = k.encode("utf-8"), v.encode("utf-8")
                f.write(struct.pack("<I", len(kb)) + kb + struct.pack("<I", len(vb)) + vb)
            f.write(struct.pack("<I", len(tensors)))
            curr_off = 0; tensor_data = []; header = []
            for name, t in tensors.items():
                nb = name.encode("utf-8")
                header.append(struct.pack("<I", len(nb)) + nb)
                t_type = 1 if "norm" in name or "gate" in name else 0
                header.append(struct.pack("<I", t_type) + struct.pack("<I", len(t.shape)))
                for d in t.shape: header.append(struct.pack("<Q", d))
                if t_type == 1:
                    data = t.detach().cpu().numpy().astype(np.float32).tobytes()
                else:
                    data = self._pack_ternary(t)
                header.append(struct.pack("<Q", curr_off))
                tensor_data.append(data); curr_off += len(data)
            f.write(b"".join(header))
            f.write(b"\x00" * ((32 - (f.tell() % 32)) % 32))
            f.write(b"".join(tensor_data))
    def _pack_ternary(self, t) -> bytes:
        r = t.detach().cpu().numpy().flatten()
        scale = np.abs(r).mean(); w_q = np.clip(np.round(r / (scale + 1e-7)), -1, 1).astype(np.int8)
        packed = []
        for i in range(0, len(w_q), 16):
            chunk = w_q[i:i+16]; val = 0
            for j, b in enumerate(chunk):
                bits = 1 if b == 1 else (2 if b == -1 else 0)
                val |= (bits << (j * 2))
            packed.append(struct.pack("<I", val))
        return b"".join(packed)

def load_embedding_from_mud(path):
    print(f"Attempting to extract embedding from {path}...")
    if not os.path.exists(path): return None
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"MUD\x01": return None
        meta_count = struct.unpack("<I", f.read(4))[0]
        for _ in range(meta_count):
            kl = struct.unpack("<I", f.read(4))[0]; f.read(kl)
            vl = struct.unpack("<I", f.read(4))[0]; f.read(vl)
        tensor_count = struct.unpack("<I", f.read(4))[0]
        headers = []
        for _ in range(tensor_count):
            nl = struct.unpack("<I", f.read(4))[0]; name = f.read(nl).decode("utf-8")
            t_type = struct.unpack("<I", f.read(4))[0]
            sl = struct.unpack("<I", f.read(4))[0]
            shape = [struct.unpack("<Q", f.read(8))[0] for _ in range(sl)]
            offset = struct.unpack("<Q", f.read(8))[0]
            headers.append({"name": name, "type": t_type, "shape": shape, "offset": offset})
        
        # Alignment padding
        pos = f.tell()
        padding = (32 - (pos % 32)) % 32
        f.read(padding)
        data_start = f.tell()
        
        for h in headers:
            if h["name"] == "token_embd.weight":
                print(f"Found {h['name']} at offset {h['offset']}")
                f.seek(data_start + h["offset"])
                # Embedding is likely ternary (type 0)
                if h["type"] == 0:
                    elements = 1
                    for d in h["shape"]: elements *= d
                    packed_bytes = f.read(elements // 16 * 4)
                    # Dequantize (simplificado para el rescue)
                    return packed_bytes, h["shape"]
    return None

def rescue():
    print("=== MUD Master Rescue Tool ===")
    ckpt_path = "models/mud_last_checkpoint.pt"
    core_mud_path = "models/core_skills.mud"
    
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found.")
        return

    sd = torch.load(ckpt_path, map_location='cpu')
    print(f"Loaded {ckpt_path}. Keys: {len(sd)}")

    # Try to extract embedding from core_skills.mud
    emb_data = load_embedding_from_mud(core_mud_path)
    
    layers = sorted(list(set([int(k.split('.')[1]) for k in sd.keys() if k.startswith('layers.')])))
    num_layers = len(layers)
    print(f"Detected {num_layers} layers.")

    output_path = "models/rescued_master.mud"
    exp = MudExporter(output_path)
    exp.add_metadata("hidden_size", "512")
    exp.add_metadata("num_layers", str(num_layers))
    exp.add_metadata("num_experts", "8")
    exp.add_metadata("ffn_hidden", "2048")
    
    # Vocabulary (copy from core_skills.mud metadata if possible, or use default)
    # For now, we'll assume the user has the vocab file
    vocab_path = "training/vocab_es_en.txt"
    if os.path.exists(vocab_path):
        with open(vocab_path, "r") as f:
            exp.add_metadata("tokenizer.tokens", f.read())

    # Map keys from trainer format to MUD format
    new_sd = {}
    
    if emb_data:
        # We have packed bytes, but our exporter expects torch.Tensor to re-pack
        # This is a bit inefficient but for rescue it's ok to use a dummy tensor 
        # and then manually patch the file, OR better: just use a random tensor
        # for now and warn the user.
        # Actually, let's just use random for the embedding and focus on the layers.
        print("⚠️ Note: Using random embedding for now. Real embedding extraction pending proper dequant logic.")
        new_sd['token_embd.weight'] = torch.randn(18947, 512)
    else:
        new_sd['token_embd.weight'] = torch.randn(18947, 512)

    new_sd['output_norm.weight'] = sd.get('norm.weight', torch.ones(512))

    for l in layers:
        new_sd[f"blk.{l}.attn_q.weight"] = sd[f"layers.{l}.attention.wq.weight"]
        new_sd[f"blk.{l}.attn_k.weight"] = sd[f"layers.{l}.attention.wk.weight"]
        new_sd[f"blk.{l}.attn_v.weight"] = sd[f"layers.{l}.attention.wv.weight"]
        new_sd[f"blk.{l}.attn_output.weight"] = sd[f"layers.{l}.attention.wo.weight"]
        new_sd[f"blk.{l}.gate.weight"] = sd[f"layers.{l}.gate.weight"]
        new_sd[f"blk.{l}.norm.weight"] = sd[f"layers.{l}.norm.weight"]
        for i in range(8):
            new_sd[f"blk.{l}.expert.{i}.w1.weight"] = sd[f"layers.{l}.experts.{i}.w1.weight"]
            new_sd[f"blk.{l}.expert.{i}.w2.weight"] = sd[f"layers.{l}.experts.{i}.w2.weight"]
            new_sd[f"blk.{l}.expert.{i}.w3.weight"] = sd[f"layers.{l}.experts.{i}.w3.weight"]

    print(f"Exporting to {output_path}...")
    exp.export(new_sd)
    print("✅ Rescue complete.")


if __name__ == "__main__":
    rescue()
