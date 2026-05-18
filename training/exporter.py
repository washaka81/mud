import torch
import numpy as np
import struct
from typing import Dict, List

class MudExporter:
    """
    Exporter for MUD (Modular Understanding Dynamics) models.
    Packs ternary weights {-1, 0, 1} into 2-bit representations.
    """
    MAGIC = b"MUD\x01"
    
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.tensors = []
        self.metadata = {}

    def add_metadata(self, key: str, value: str):
        self.metadata[key] = value

    def _pack_ternary_row(self, row: torch.Tensor) -> bytes:
        """
        Packs a row of ternary weights into 2-bit packed u32s.
        Mapping: 0 -> 00, 1 -> 01, -1 -> 10
        """
        vals = row.cpu().numpy().flatten().astype(np.int8)
        n = len(vals)
        assert n % 16 == 0, "Row size must be a multiple of 16 for packing"
        
        packed = []
        for i in range(0, n, 16):
            chunk = vals[i:i+16]
            u32_val = 0
            for j, v in enumerate(chunk):
                # 2 bits per value
                bits = 0
                if v == 1: bits = 1
                elif v == -1: bits = 2
                u32_val |= (bits << (j * 2))
            packed.append(u32_val)
        
        return np.array(packed, dtype=np.uint32).tobytes()

    def export(self, state_dict: Dict[str, torch.Tensor]):
        """
        Converts PyTorch state_dict to .ai format.
        """
        print(f"Exporting {len(state_dict)} tensors to {self.output_path}...")
        
        with open(self.output_path, "wb") as f:
            # 1. Write Header
            f.write(self.MAGIC)
            
            # 2. Write Metadata Count & Data
            f.write(struct.pack("<I", len(self.metadata)))
            for k, v in self.metadata.items():
                k_bytes = k.encode('utf-8')
                v_bytes = v.encode('utf-8')
                f.write(struct.pack("<I", len(k_bytes)))
                f.write(k_bytes)
                f.write(struct.pack("<I", len(v_bytes)))
                f.write(v_bytes)

            # 3. Write Tensor Descriptors
            # We first collect data to calculate offsets
            tensor_data = []
            f.write(struct.pack("<I", len(state_dict)))
            
            current_offset = 0
            for name, tensor in state_dict.items():
                is_weight = "weight" in name and tensor.dim() > 1
                t_type = 0 if is_weight else 1 # 0: TernaryPacked, 1: Float32
                
                if is_weight:
                    # Quantize to ternary if not already
                    gamma = tensor.abs().mean()
                    w_q = torch.clamp(torch.round(tensor / (gamma + 1e-7)), -1, 1)
                    data = self._pack_ternary_row(w_q)
                else:
                    data = tensor.cpu().numpy().astype(np.float32).tobytes()

                shape = list(tensor.shape)
                
                # Descriptor: NameLen, Name, Type, DimsCount, Dims, Offset
                name_b = name.encode('utf-8')
                f.write(struct.pack("<I", len(name_b)))
                f.write(name_b)
                f.write(struct.pack("<I", t_type))
                f.write(struct.pack("<I", len(shape)))
                for d in shape:
                    f.write(struct.pack("<Q", d))
                
                f.write(struct.pack("<Q", current_offset))
                tensor_data.append(data)
                current_offset += len(data)

            # 4. Write Tensor Data (aligned to 32 bytes)
            # Alignment padding
            pos = f.tell()
            padding = (32 - (pos % 32)) % 32
            f.write(b'\x00' * padding)
            
            for data in tensor_data:
                f.write(data)

        print("Export completed successfully.")

if __name__ == "__main__":
    # Example usage for MUD Inference testing:
    from ternary_moe_logic import MudMoE
    import torch
    
    hidden_size = 512
    ffn_hidden = 1024
    num_layers = 2
    num_experts = 4
    
    # Define a simple state dict with correct naming for MudInference
    sd = {}
    sd["token_embd.weight"] = torch.randn(5000, hidden_size) # Dummy vocab
    
    for l in range(num_layers):
        sd[f"blk.{l}.gate.weight"] = torch.randn(num_experts, hidden_size)
        sd[f"blk.{l}.norm.weight"] = torch.randn(hidden_size)
        for e in range(num_experts):
            sd[f"blk.{l}.expert.{e}.w1.weight"] = torch.randn(ffn_hidden, hidden_size)
            sd[f"blk.{l}.expert.{e}.w2.weight"] = torch.randn(hidden_size, ffn_hidden)
            sd[f"blk.{l}.expert.{e}.w3.weight"] = torch.randn(ffn_hidden, hidden_size)

    exporter = MudExporter("test_model.ai")
    exporter.add_metadata("hidden_size", str(hidden_size))
    exporter.add_metadata("num_layers", str(num_layers))
    exporter.add_metadata("num_experts", str(num_experts))
    exporter.add_metadata("ffn_hidden", str(ffn_hidden))
    exporter.add_metadata("arch", "mud-ternary-moe-v1")
    
    # --- STANDALONE STARTER VOCABULARY ---
    # Load from vocab file if available
    import os
    vocab_path = "training/vocab_es_en.txt"
    if not os.path.exists(vocab_path):
        vocab_path = "vocab_es_en.txt"
        
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = [line.strip() for line in f if line.strip()]
    else:
        # Common words for Spanish and English interaction (fallback)
        vocab = [
            "!", "MUD", "Forge", "hola", "hello", "engine", "is", "rápido", "fast",
            "inteligente", "smart", "modular", "el", "the", "la", "a", "que", "what",
            "cómo", "how", "estoy", "ready", "listo", "processing", "pensando", "learning",
            "futuro", "future", "efficient", "eficiente", "de", "of", "en", "in", "si", "yes",
            "no", "con", "with", "un", "una", "por", "for", "para", "lo", "it", "yo", "I",
            "soy", "am", "your", "tu", "mi", "my", "assistant", "asistente", "bilingüe", "bilingual"
        ]
    
    exporter.add_metadata("tokenizer.tokens", ",".join(vocab))
    exporter.add_metadata("tokenizer.merges", "")
    
    exporter.export(sd)
