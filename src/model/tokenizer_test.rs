use crate::gguf::GGUFModel;
use crate::model::tokenizer::Tokenizer;
use crate::mud::MudFile;
use std::path::Path;

#[test]
fn test_tokenizer_prompt() {
    let model_path = "models/MUD2.5-coder-1.5b-instruct-q4_0.gguf";
    if !Path::new(model_path).exists() {
        return;
    }

    let model = GGUFModel::load(model_path).unwrap();
    let tokenizer = Tokenizer::from_gguf(&model).unwrap();

    let text = "def fast_fibonacci(n):";
    let ids = tokenizer.encode_simple(text);
    println!("Prompt: {}", text);
    for id in ids {
        println!("  ID {}: {:?}", id, tokenizer.id_to_token[id as usize]);
    }
}

#[test]
fn test_auto_concordance_gpt_spaces() {
    // GPT-style BPE uses 'Ġ' for spaces.
    let tokens = "hello\nĠworld\n!\n<|im_start|>\n<|im_end|>";
    let merges = "";

    let tokenizer = Tokenizer::from_mud_metadata(tokens, merges);

    // Check space_char detection.
    assert_eq!(tokenizer.space_char, Some('Ġ'));

    // Check special control token detection.
    assert!(tokenizer.special_tokens.contains_key("<|im_start|>"));
    assert!(tokenizer.special_tokens.contains_key("<|im_end|>"));

    // Let's test decoding tokens with space prefix
    // 0: "hello"
    // 1: "Ġworld"
    // 2: "!"
    let decoded = tokenizer.decode_simple(&[0, 1, 2]);
    assert_eq!(decoded, "hello world!");
}

#[test]
fn test_auto_concordance_sp_spaces() {
    // SentencePiece-style BPE uses '\u{2581}' for spaces.
    let tokens = "hello\n\u{2581}world\n!\n[PAD]\n[CLS]";
    let merges = "";

    let tokenizer = Tokenizer::from_mud_metadata(tokens, merges);

    // Check space_char detection (space prefix character is U+2581)
    assert_eq!(tokenizer.space_char, Some('\u{2581}'));

    // Check special control token detection
    assert!(tokenizer.special_tokens.contains_key("[PAD]"));
    assert!(tokenizer.special_tokens.contains_key("[CLS]"));

    // Let's test decoding tokens with SP space prefix
    // 0: "hello"
    // 1: "\u{2581}world"
    // 2: "!"
    let decoded = tokenizer.decode_simple(&[0, 1, 2]);
    assert_eq!(decoded, "hello world!");
}

#[test]
fn test_smollm2_space_char() {
    let path = "models/smollm2.mud";
    if !Path::new(path).exists() {
        return;
    }
    let mud = MudFile::load(path).expect("load smollm2.mud");
    let tokens = mud
        .global_metadata
        .get("tokenizer.tokens")
        .expect("tokenizer.tokens present");
    let merges = mud
        .global_metadata
        .get("tokenizer.merges")
        .cloned()
        .unwrap_or_default();
    let tk = Tokenizer::from_mud_metadata(tokens, &merges);
    // The smollm2 vocab uses 'Ġ' (U+0120) as the space prefix; decode must
    // re-insert spaces so words are not fused (e.g. "romancesinite").
    assert_eq!(
        tk.space_char,
        Some('Ġ'),
        "smollm2 vocab must retain the 'Ġ' space prefix"
    );
    let ids = tk.encode_simple("hello world");
    let decoded = tk.decode_simple(&ids);
    assert_eq!(decoded, "hello world", "roundtrip must preserve spaces");
}

#[test]
fn test_has_space_prefix_subwords() {
    let tokens = "hello\nĠworld\ncan\ncion\nĠroja";
    let merges = "";
    let tk = Tokenizer::from_mud_metadata(tokens, merges);

    // ID 0: "hello" (no space prefix, start of text)
    // ID 1: "Ġworld" (has space prefix)
    // ID 2: "can" (subword)
    // ID 3: "cion" (subword continuation, no space prefix)
    // ID 4: "Ġroja" (start of new word, has space prefix)

    assert!(!tk.has_space_prefix(0));
    assert!(tk.has_space_prefix(1));
    assert!(!tk.has_space_prefix(2));
    assert!(!tk.has_space_prefix(3));
    assert!(tk.has_space_prefix(4));

    // "can" + "cion" -> "cancion" (same word)
    assert_eq!(tk.decode_simple(&[2, 3]), "cancion");

    // "cancion" + "Ġroja" -> "cancion roja" (two separate words)
    assert_eq!(tk.decode_simple(&[2, 3, 4]), "cancion roja");
}
