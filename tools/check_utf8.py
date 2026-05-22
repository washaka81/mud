import sys

# Simulation of what bytes_to_unicode in Rust does
def bytes_to_unicode():
    bs = list(range(ord('!'), ord('~') + 1))
    bs.extend(range(0xA1, 0xAC + 1))
    bs.extend(range(0xAE, 0xFF + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

byte_encoder = bytes_to_unicode()

def encode_bpe_simulation(text):
    # This is how the Rust code prepares the string for BPE
    bytes_data = text.encode('utf-8')
    words = [byte_encoder[b] for b in bytes_data]
    return words

test_str = "áéíóú ñ 😊"
encoded_words = encode_bpe_simulation(test_str)
print(f"Original: {test_str}")
print(f"Encoded (prepared for BPE): {''.join(encoded_words)}")
