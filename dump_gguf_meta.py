import gguf
import sys
reader = gguf.GGUFReader(sys.argv[1])
for key, field in reader.fields.items():
    if not key.startswith('tokenizer') and not key.startswith('general'):
        print(f"{key}: {field.parts}")
