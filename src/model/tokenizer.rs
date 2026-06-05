use crate::gguf::{GGUFModel, MetadataValue};
use std::collections::HashMap;

/// Implementation of a BPE (Byte Pair Encoding) tokenizer.
/// Compatible with Transformer-Base, Code, and Core style tokenization.
pub struct Tokenizer {
    /// Mapping from string token to its ID.
    pub vocab: HashMap<String, u32>,
    /// Mapping from token ID to its string representation.
    pub id_to_token: Vec<String>,
    /// BPE merge ranks for subword construction.
    pub merges: HashMap<(String, String), u32>,
    /// Map of special tokens (like <|im_start|>) to their IDs.
    pub special_tokens: HashMap<String, u32>,
    /// Mapping of byte values to their Unicode escape representations.
    pub byte_encoder: HashMap<u8, char>,
    /// Character used as a space prefix in BPE tokenization (e.g. 'Ġ' or ' ')
    pub space_char: Option<char>,
}

impl Tokenizer {
    /// Loads the tokenizer from raw strings (used by MUD format).
    pub fn from_mud_metadata(tokens_str: &str, merges_str: &str) -> Self {
        let mut id_to_token = Vec::new();
        let mut vocab = HashMap::new();

        // Autodetect separator: prefer \n, fallback to ,
        let sep = if tokens_str.contains('\n') { '\n' } else { ',' };

        for (i, t) in tokens_str.split(sep).enumerate() {
            let clean_t = t.trim();
            if clean_t.is_empty() && i > 0 {
                continue;
            }
            id_to_token.push(clean_t.to_string());
            vocab.insert(clean_t.to_string(), i as u32);
        }

        let mut merges = HashMap::new();
        for (rank, m) in merges_str.split('\n').enumerate() {
            if m.is_empty() {
                continue;
            }
            let parts: Vec<&str> = m.split(' ').collect();
            if parts.len() == 2 {
                merges.insert((parts[0].to_string(), parts[1].to_string()), rank as u32);
            }
        }

        let mut special_tokens = HashMap::new();
        let mut count_gpt_space = 0;
        let mut count_sp_space = 0;

        for (i, t) in id_to_token.iter().enumerate() {
            // 1. Detect standard special control marks
            if (t.starts_with('<') && t.ends_with('>')) || (t.starts_with('[') && t.ends_with(']'))
            {
                special_tokens.insert(t.clone(), i as u32);
            }

            // 2. Count space prefix representations to determine concordance
            if t.contains('Ġ') {
                count_gpt_space += 1;
            }
            if t.contains('\u{2581}') {
                // SentencePiece space prefix (U+2581)
                count_sp_space += 1;
            }
        }

        let space_char = if count_sp_space > count_gpt_space {
            Some('\u{2581}') // SentencePiece space prefix
        } else if count_gpt_space > 0 {
            Some('Ġ') // GPT space prefix
        } else {
            None
        };

        // Space prefix auto-detected — silenced
        // Special control marks auto-detected — silenced

        Self {
            vocab,
            id_to_token,
            merges,
            special_tokens,
            byte_encoder: bytes_to_unicode(),
            space_char,
        }
    }

    /// Loads the tokenizer vocabulary and merges from a GGUF model.
    pub fn from_gguf(model: &GGUFModel) -> anyhow::Result<Self> {
        let tokens_val = model
            .get_metadata_array("tokenizer.ggml.tokens")
            .ok_or_else(|| anyhow::anyhow!("No tokens found in GGUF"))?;

        let mut vocab = HashMap::with_capacity(tokens_val.len());
        let mut id_to_token = Vec::with_capacity(tokens_val.len());

        for (i, val) in tokens_val.iter().enumerate() {
            if let MetadataValue::String(s) = val {
                vocab.insert(s.clone(), i as u32);
                id_to_token.push(s.clone());
            }
        }

        let mut special_tokens = HashMap::new();
        // Identify special tokens based on standard Code naming patterns
        for (i, token) in id_to_token.iter().enumerate() {
            if token.starts_with("<|") && token.ends_with("|>") {
                special_tokens.insert(token.clone(), i as u32);
            }
        }

        let merges_val = model
            .get_metadata_array("tokenizer.ggml.merges")
            .ok_or_else(|| anyhow::anyhow!("No merges found in GGUF"))?;

        let mut merges = HashMap::with_capacity(merges_val.len());
        for (rank, val) in merges_val.iter().enumerate() {
            if let MetadataValue::String(s) = val {
                let parts: Vec<&str> = s.split(' ').collect();
                if parts.len() == 2 {
                    merges.insert((parts[0].to_string(), parts[1].to_string()), rank as u32);
                }
            }
        }

        let mut count_gpt_space = 0;
        let mut count_sp_space = 0;
        for token in &id_to_token {
            if token.contains('Ġ') {
                count_gpt_space += 1;
            }
            if token.contains('\u{2581}') {
                count_sp_space += 1;
            }
        }
        let space_char = if count_sp_space > count_gpt_space {
            Some('\u{2581}')
        } else if count_gpt_space > 0 {
            Some('Ġ')
        } else {
            None
        };

        Ok(Self {
            vocab,
            id_to_token,
            merges,
            special_tokens,
            byte_encoder: bytes_to_unicode(),
            space_char,
        })
    }

    /// Encodes a string into a list of token IDs.
    /// Handles special tokens first, then applies BPE. Falls back to character-level IDs if needed.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![];
        }

        let mut final_tokens = Vec::new();

        // 1. Handle special tokens
        let mut parts = vec![text.to_string()];
        for special in self.special_tokens.keys() {
            let mut new_parts = Vec::new();
            for part in parts {
                if self.special_tokens.contains_key(&part) {
                    new_parts.push(part);
                } else {
                    let split: Vec<_> = part.split(special).collect();
                    for (i, s) in split.iter().enumerate() {
                        if !s.is_empty() {
                            new_parts.push(s.to_string());
                        }
                        if i < split.len() - 1 {
                            new_parts.push(special.clone());
                        }
                    }
                }
            }
            parts = new_parts;
        }

        for part in parts {
            if let Some(&id) = self.special_tokens.get(&part) {
                final_tokens.push(id);
            } else {
                // 2. Standard BPE process
                let mut tokens = self.encode_bpe(&part);

                // 3. ROBUST FALLBACK: If BPE failed to produce tokens for this part, use byte/char IDs
                if tokens.is_empty() && !part.trim().is_empty() {
                    for b in part.as_bytes() {
                        final_tokens.push(*b as u32);
                    }
                } else {
                    final_tokens.append(&mut tokens);
                }
            }
        }
        final_tokens
    }

    /// Internal BPE encoder for a single text fragment.
    fn encode_bpe(&self, text: &str) -> Vec<u32> {
        let bytes = text.as_bytes();
        // Pre-tokenization: map bytes to special unicode characters
        let words: Vec<String> = bytes
            .iter()
            .map(|&b| self.byte_encoder.get(&b).unwrap().to_string())
            .collect();

        if words.is_empty() {
            return vec![];
        }

        #[derive(Clone)]
        struct Part {
            text: String,
            prev: isize,
            next: isize,
        }

        let mut parts = Vec::with_capacity(words.len());
        for (i, w) in words.iter().enumerate() {
            parts.push(Part {
                text: w.clone(),
                prev: i as isize - 1,
                next: if i == words.len() - 1 { -1 } else { i as isize + 1 },
            });
        }

        use std::collections::BinaryHeap;
        use std::cmp::Ordering;

        #[derive(Eq, PartialEq)]
        struct MergePair {
            rank: u32,
            left_idx: usize,
            right_idx: usize,
        }

        impl Ord for MergePair {
            fn cmp(&self, other: &Self) -> Ordering {
                // Min-heap on rank
                other.rank.cmp(&self.rank)
            }
        }

        impl PartialOrd for MergePair {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut heap = BinaryHeap::new();
        for i in 0..parts.len() - 1 {
            let pair = (parts[i].text.clone(), parts[i + 1].text.clone());
            if let Some(&rank) = self.merges.get(&pair) {
                heap.push(MergePair { rank, left_idx: i, right_idx: i + 1 });
            }
        }

        while let Some(MergePair { rank: _, left_idx, right_idx }) = heap.pop() {
            // Check if the pair is still valid and adjacent
            if parts[left_idx].next != right_idx as isize || parts[right_idx].prev != left_idx as isize {
                continue;
            }

            // Merge right_idx into left_idx
            let right_text = parts[right_idx].text.clone();
            parts[left_idx].text.push_str(&right_text);
            
            let next_idx = parts[right_idx].next;
            parts[left_idx].next = next_idx;
            if next_idx != -1 {
                parts[next_idx as usize].prev = left_idx as isize;
            }

            // Push new adjacent pairs to heap
            let next_idx = parts[left_idx].next;
            if next_idx != -1 {
                let pair = (parts[left_idx].text.clone(), parts[next_idx as usize].text.clone());
                if let Some(&rank) = self.merges.get(&pair) {
                    heap.push(MergePair { rank, left_idx, right_idx: next_idx as usize });
                }
            }

            let prev_idx = parts[left_idx].prev;
            if prev_idx != -1 {
                let pair = (parts[prev_idx as usize].text.clone(), parts[left_idx].text.clone());
                if let Some(&rank) = self.merges.get(&pair) {
                    heap.push(MergePair { rank, left_idx: prev_idx as usize, right_idx: left_idx });
                }
            }
        }

        let mut tokens = Vec::new();
        let mut curr = 0isize;
        while curr != -1 {
            let part = &parts[curr as usize];
            if let Some(&id) = self.vocab.get(&part.text) {
                tokens.push(id);
            }
            curr = part.next;
        }
        tokens
    }

    /// Decodes a list of token IDs back into a human-readable string.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut raw_text = String::new();
        for &id in ids {
            if let Some(token) = self.id_to_token.get(id as usize) {
                raw_text.push_str(token);
            }
        }

        if raw_text.is_empty() {
            return String::new();
        }

        // Reverse the mapping of bytes to unicode escape characters
        let byte_decoder: HashMap<char, u8> =
            self.byte_encoder.iter().map(|(&b, &c)| (c, b)).collect();

        let mut decoded_bytes = Vec::new();
        for c in raw_text.chars() {
            if let Some(sc) = self.space_char {
                if c == sc {
                    decoded_bytes.push(b' ');
                    continue;
                }
            }

            if let Some(&b) = byte_decoder.get(&c) {
                decoded_bytes.push(b);
            } else {
                // If character is not in byte_encoder, it's likely a direct UTF-8 char
                let mut buf = [0; 4];
                for b in c.encode_utf8(&mut buf).as_bytes() {
                    decoded_bytes.push(*b);
                }
            }
        }

        String::from_utf8_lossy(&decoded_bytes).into_owned()
    }
}

/// Creates a mapping of all byte values to unique Unicode characters.
/// This prevents loss of information during BPE and ensures all strings are valid UTF-8.
fn bytes_to_unicode() -> HashMap<u8, char> {
    let mut bs: Vec<u8> = (b'!'..=b'~').collect();
    bs.extend(0xA1..=0xAC_u8);
    bs.extend(0xAE..=0xFF_u8);

    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    let mut n = 0;
    for b in 0..=255 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }

    let mut map = HashMap::new();
    for (b, c) in bs.into_iter().zip(cs.into_iter()) {
        map.insert(b, std::char::from_u32(c).unwrap());
    }
    map
}
