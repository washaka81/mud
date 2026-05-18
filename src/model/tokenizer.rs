use std::collections::HashMap;
use crate::gguf::{GGUFModel, MetadataValue};

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
            if clean_t.is_empty() && i > 0 { continue; }
            id_to_token.push(clean_t.to_string());
            vocab.insert(clean_t.to_string(), i as u32);
        }

        let mut merges = HashMap::new();
        for (rank, m) in merges_str.split('\n').enumerate() {
            if m.is_empty() { continue; }
            let parts: Vec<&str> = m.split(' ').collect();
            if parts.len() == 2 {
                merges.insert((parts[0].to_string(), parts[1].to_string()), rank as u32);
            }
        }

        Self {
            vocab,
            id_to_token,
            merges,
            special_tokens: HashMap::new(),
            byte_encoder: bytes_to_unicode(),
        }
    }

    /// Loads the tokenizer vocabulary and merges from a GGUF model.
    pub fn from_gguf(model: &GGUFModel) -> anyhow::Result<Self> {
        let tokens_val = model.get_metadata_array("tokenizer.ggml.tokens")
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
        
        let merges_val = model.get_metadata_array("tokenizer.ggml.merges")
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
        
        Ok(Self { 
            vocab, 
            id_to_token, 
            merges,
            special_tokens,
            byte_encoder: bytes_to_unicode(),
        })
    }

    /// Encodes a string into a list of token IDs.
    /// Handles special tokens first, then applies BPE. Falls back to character-level IDs if needed.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() { return vec![]; }
        
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
                        if !s.is_empty() { new_parts.push(s.to_string()); }
                        if i < split.len() - 1 { new_parts.push(special.clone()); }
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
        let mut tokens = Vec::new();
        let bytes = text.as_bytes();
        // Pre-tokenization: map bytes to special unicode characters
        let mut words: Vec<String> = bytes.iter()
            .map(|&b| self.byte_encoder.get(&b).unwrap().to_string())
            .collect();
        
        if words.is_empty() { return vec![]; }
        
        // Iteratively merge the best pairs according to the merge ranks
        loop {
            let mut best_pair: Option<(String, String)> = None;
            let mut best_rank = u32::MAX;
            
            for i in 0..words.len().saturating_sub(1) {
                let pair = (words[i].clone(), words[i+1].clone());
                if let Some(&rank) = self.merges.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_pair = Some(pair);
                    }
                }
            }
            
            if let Some(pair) = best_pair {
                let mut new_words = Vec::new();
                let mut i = 0;
                while i < words.len() {
                    if i < words.len() - 1 && words[i] == pair.0 && words[i+1] == pair.1 {
                        new_words.push(format!("{}{}", pair.0, pair.1));
                        i += 2;
                    } else {
                        new_words.push(words[i].clone());
                        i += 1;
                    }
                }
                words = new_words;
            } else {
                break;
            }
        }
        
        for w in words {
            if let Some(&id) = self.vocab.get(&w) {
                tokens.push(id);
            }
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
        
        if raw_text.is_empty() { return String::new(); }

        // Reverse the mapping of bytes to unicode escape characters
        let byte_decoder: HashMap<char, u8> = self.byte_encoder.iter()
            .map(|(&b, &c)| (c, b)).collect();
            
        let mut decoded_bytes = Vec::new();
        for c in raw_text.chars() {
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
