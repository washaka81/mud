import os

print("=== MUD Knowledge Merger for Kaggle ===")

SOURCES = [
    "training/extra_knowledge.txt",
    "models/knowledge_package.txt",
    "training/synthetic_knowledge.txt",
    "training/massive_knowledge_corpus.txt"
]

OUTPUT = "training/massive_knowledge_corpus_updated.txt"

seen_content = set()
total_lines = 0

with open(OUTPUT, "w", encoding="utf-8") as out_f:
    for src in SOURCES:
        if not os.path.exists(src):
            print(f"⚠️  Warning: Source {src} not found, skipping.")
            continue
        
        print(f"📦 Processing {src}...")
        with open(src, "r", encoding="utf-8", errors="ignore") as in_f:
            for line in in_f:
                if "CONTENT:" in line:
                    content = line.split("CONTENT:", 1)[1].strip()
                    if content and content not in seen_content:
                        out_f.write(f"CONTENT: {content}\n")
                        seen_content.add(content)
                        total_lines += 1

print(f"✅ Merging complete. Total unique facts: {total_lines}")
print(f"💾 Saved to {OUTPUT}")

# Replace the original massive corpus with the updated one
if os.path.exists(OUTPUT):
    os.replace(OUTPUT, "training/massive_knowledge_corpus.txt")
    print("🚀 training/massive_knowledge_corpus.txt is now updated and ready for Kaggle!")
