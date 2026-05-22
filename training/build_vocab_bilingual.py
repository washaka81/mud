import re
from collections import Counter

def build_separated_vocab():
    # 1. Base Bytes & Special Tokens
    # We keep 256 byte-tokens to ensure 100% coverage
    special_tokens = ["<unk>", "<s>", "</s>", "<pad>", "<thinking>", "</thinking>", "<answer>", "</answer>"]
    
    # 2. English Core (Common words)
    english_words = ["the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me"]
    
    # 3. Spanish Core (Common words)
    spanish_words = ["de", "la", "que", "el", "en", "y", "a", "los", "se", "del", "las", "un", "por", "con", "no", "una", "su", "para", "es", "al", "lo", "como", "ms", "pero", "sus", "le", "este", "bien", "si", "si", "ya", "hay", "esta", "todo", "esta", "cuando", "nos", "muy", "desde", "porque", "todos", "ser", "son", "hacer", "tienen", "sobre", "un", "donde", "est", "era"]

    # 4. Read massive corpus to find high-frequency bilingual tokens
    try:
        with open("training/massive_knowledge_corpus.txt", "r", encoding="utf-8") as f:
            text = f.read(10_000_000) # Sample 10MB
            # Basic tokenization: split by spaces and preserve some punctuation
            raw_tokens = re.findall(r"[\wáéíóúñÁÉÍÓÚÑ]+|[^\w\s]", text)
            counts = Counter(raw_tokens)
            
            # Filter top 10000 frequent tokens
            frequent = [word for word, count in counts.most_common(10000) if len(word) > 1]
    except:
        frequent = []

    # Assemble Final Vocab
    # Block 1: Special
    # Block 2: ASCII Bytes (to avoid UNKNOWNs)
    # Block 3: Spanish (Primary)
    # Block 4: English (Secondary)
    
    vocab = []
    vocab.extend(special_tokens)
    
    # Add ASCII printable bytes
    for i in range(32, 127):
        vocab.append(chr(i))
    
    # Unique high-freq tokens
    seen = set(vocab)
    
    print("Adding Spanish Core...")
    for w in spanish_words:
        if w not in seen:
            vocab.append(w)
            seen.add(w)
            
    print("Adding English Core...")
    for w in english_words:
        if w not in seen:
            vocab.append(w)
            seen.add(w)
            
    print("Adding Mixed High-Freq...")
    for w in frequent:
        if w not in seen:
            vocab.append(w)
            seen.add(w)

    with open("training/vocab_es_en.txt", "w", encoding="utf-8") as f:
        for v in vocab:
            f.write(v + "\n")
            
    print(f"✅ New Bilingual Vocab generated with {len(vocab)} tokens.")

if __name__ == "__main__":
    build_separated_vocab()
