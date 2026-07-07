import os, json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, decoders

VOCAB_SIZE = int(os.environ.get("BPE_VOCAB_SIZE", "32000"))
WORDLIST_PATH = os.environ.get("WORDLIST", "vocabulario_es.txt")
OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "build/bpe")

os.makedirs(os.path.dirname(OUTPUT_PREFIX) or ".", exist_ok=True)

print(f"Cargando lista de palabras desde {WORDLIST_PATH}...")
with open(WORDLIST_PATH, "r") as f:
    words = [line.strip() for line in f if line.strip()]

# Build a corpus where each word appears once
corpus_path = OUTPUT_PREFIX + "_corpus.txt"
with open(corpus_path, "w") as f:
    for w in words:
        f.write(w + "\n")

print(f"Entrenando BPE tokenizer (vocab_size={VOCAB_SIZE}) sobre {len(words)} palabras...")

# Whitespace-based pre-tokenizer: splits on whitespace and punctuation,
# keeps Unicode chars directly (no byte-level encoding)
tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
tokenizer.decoder = decoders.BPEDecoder()

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<MASK>"],
    min_frequency=1,
    show_progress=True,
)

tokenizer.train([corpus_path], trainer)

# Test encoding
test_text = "hola mundo como estas"
encoded = tokenizer.encode(test_text)
print(f"\nTest encode('{test_text}'):")
print(f"  IDs: {encoded.ids}")
print(f"  Tokens: {tokenizer.decode(encoded.ids)}")

# Save HuggingFace format
tokenizer.save(OUTPUT_PREFIX + "_hf.json")

# Export for C++:
# vocab: "token id" per line
# merges: "left right" per line (in priority order)
with open(OUTPUT_PREFIX + "_hf.json", "r") as f:
    hf_data = json.load(f)

vocab = hf_data["model"]["vocab"]
sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

with open(OUTPUT_PREFIX + "_vocab.txt", "w") as f:
    for token, tid in sorted_vocab:
        f.write(f"{tid} {token}\n")

merges = hf_data["model"].get("merges", [])
with open(OUTPUT_PREFIX + "_merges.txt", "w") as f:
    for left, right in merges:
        f.write(f"{left} {right}\n")

print(f"Vocabulario exportado: {len(sorted_vocab)} tokens")
print(f"Total merges: {len(merges)}")
print(f"Archivos generados:")
print(f"  {OUTPUT_PREFIX}_hf.json")
print(f"  {OUTPUT_PREFIX}_vocab.txt")
print(f"  {OUTPUT_PREFIX}_merges.txt")
