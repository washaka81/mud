import struct

def inspect_mud(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"MUD\x01":
            print("Invalid magic")
            return
        
        meta_count = struct.unpack("<I", f.read(4))[0]
        metadata = {}
        for _ in range(meta_count):
            k_len = struct.unpack("<I", f.read(4))[0]
            k = f.read(k_len).decode('utf-8')
            v_len = struct.unpack("<I", f.read(4))[0]
            v = f.read(v_len).decode('utf-8')
            metadata[k] = v
            
        print(f"Metadata keys: {list(metadata.keys())}")
        if "tokenizer.tokens" in metadata:
            tokens = metadata["tokenizer.tokens"]
            print(f"Tokens string length: {len(tokens)}")
            # Detect separator
            if "\n" in tokens:
                sep = "\n"
                print("Separator: \\n")
            elif "," in tokens:
                sep = ","
                print("Separator: ,")
            else:
                sep = " "
                print("Separator: space")
            
            token_list = tokens.split(sep)
            print(f"Token count: {len(token_list)}")
            print(f"First 10 tokens: {token_list[:10]}")
            print(f"Last 10 tokens: {token_list[-10:]}")
            
            # Search for specific strings
            for t in token_list:
                if "Hola" in t or "multiple" in t:
                    print(f"Found suspicious token: {t}")

inspect_mud("models/core_skills.mud")
