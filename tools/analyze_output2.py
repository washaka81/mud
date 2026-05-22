import sys
from collections import Counter
import re

with open('/tmp/mud_output2.txt', 'r') as f:
    text = f.read()

# Filter out the engine initialization lines
text = text.split("❯  hola\n")[-1] if "❯  hola\n" in text else text

words = re.findall(r'\b\w+\b', text.lower())
counts = Counter(words)

print("Top 15 most frequent words:")
for word, count in counts.most_common(15):
    print(f"  {word}: {count}")
