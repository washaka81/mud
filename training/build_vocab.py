import urllib.request
import re

def download_and_parse_en(url, max_words=10000):
    print("Downloading English vocabulary...")
    response = urllib.request.urlopen(url)
    content = response.read().decode('utf-8')
    words = []
    for line in content.splitlines():
        w = line.strip()
        if w and w.isalpha():
            words.append(w)
        if len(words) >= max_words:
            break
    return words

def download_and_parse_es(url, max_words=10000):
    print("Downloading Spanish vocabulary...")
    response = urllib.request.urlopen(url)
    content = response.read().decode('utf-8')
    words = []
    for line in content.splitlines():
        # Format is typically "word frequency"
        parts = line.strip().split()
        if parts:
            w = parts[0]
            if w.isalpha():  # Basic check
                words.append(w)
        if len(words) >= max_words:
            break
    return words

def main():
    en_url = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-no-swears.txt"
    es_url = "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2016/es/es_50k.txt"
    
    en_words = download_and_parse_en(en_url, 15000)
    es_words = download_and_parse_es(es_url, 15000)
    
    # Base tokens
    special_tokens = [
        "<unk>", "<s>", "</s>", "<pad>", "!", "?", ".", ",", "MUD", "Forge", "AI",
        "<thinking>", "</thinking>", "<answer>", "</answer>"
    ]
    
    # Combine and deduplicate while maintaining order
    seen = set(special_tokens)
    combined = list(special_tokens)
    
    # Alternate adding words to balance the vocabulary early on
    for e, s in zip(en_words, es_words):
        if e not in seen:
            seen.add(e)
            combined.append(e)
        if s not in seen:
            seen.add(s)
            combined.append(s)
            
    print(f"Total unique vocabulary size: {len(combined)}")
    
    with open("training/vocab_es_en.txt", "w", encoding="utf-8") as f:
        for w in combined:
            f.write(w + "\n")
            
    print("Saved vocabulary to training/vocab_es_en.txt")

if __name__ == "__main__":
    main()
