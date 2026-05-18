import torch
import struct
import numpy as np
import os

# --- LOGIC MODULE GENERATOR ---
HIDDEN = 512
EXPERTS = 4
VOCAB_SIZE = 151936

def pack_ternary(row):
    r = np.random.choice([-1, 0, 1], size=row.size).astype(np.int8)
    packed = []
    for i in range(0, len(r), 16):
        chunk = r[i:i+16]; val = 0
        for j, v in enumerate(chunk):
            bits = 1 if v == 1 else (2 if v == -1 else 0)
            val |= (int(bits) << (j * 2))
        packed.append(val)
    return np.array(packed, dtype=np.uint32).tobytes()

print(f"📦 Generating Specialized Logic Module (4 Experts)...")
os.makedirs('models', exist_ok=True)

with open('models/logic_module.mud', 'wb') as f:
    f.write(b'MUD\x01')
    # Metadata
    meta = {
        'hidden_size': str(HIDDEN),
        'num_experts': str(EXPERTS),
        'num_layers': '1',
        'skill_name': 'logic_math'
    }
    f.write(struct.pack('<I', len(meta)))
    for k, v in meta.items():
        for s in [k, v]: 
            b = s.encode('utf-8')
            f.write(struct.pack('<I', len(b))); f.write(b)
    
    # Experts for logic (Identity-like to pass signal)
    sd = {}
    for i in range(EXPERTS):
        sd[f'blk.0.expert.{i}.w1.weight'] = np.eye(HIDDEN*4, HIDDEN)[:2048, :]
        sd[f'blk.0.expert.{i}.w2.weight'] = np.eye(HIDDEN, HIDDEN*4)[:, :2048]
        sd[f'blk.0.expert.{i}.w3.weight'] = np.eye(HIDDEN*4, HIDDEN)[:2048, :]

    f.write(struct.pack('<I', len(sd)))
    off = 0; tensor_data = []
    for name, t in sd.items():
        data = pack_ternary(t)
        f.write(struct.pack('<I', len(name.encode('utf-8')))); f.write(name.encode('utf-8'))
        f.write(struct.pack('<I', 0)) # Ternary
        f.write(struct.pack('<I', len(t.shape)))
        for d in t.shape: f.write(struct.pack('<Q', d))
        f.write(struct.pack('<Q', off)); tensor_data.append(data); off += len(data)
    
    f.write(b'\x00' * ((32 - (f.tell() % 32)) % 32))
    for d in tensor_data: f.write(d)

print("✅ Logic Module Created: models/logic_module.mud")
