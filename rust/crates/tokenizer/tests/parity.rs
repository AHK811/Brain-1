use brainlab_tokenizer::{
    char_class, pre_tokenize, split_digit_run, vocab::Vocab, word_chunks, Class,
};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct Fixture {
    version: u32,
    vocab_size: usize,
    eos_id: usize,
    pad_id: usize,
    tokens_hex: Vec<String>,
    merges: Vec<[u32; 2]>,
    docs: Vec<String>,
    pre_tokenize: Vec<Vec<String>>,
    word_chunks_hex: Vec<Vec<String>>,
    ids: Vec<Vec<u32>>,
}

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("crate must sit at rust/crates/<name>")
        .join("tests")
        .join("golden")
        .join("tokenizer_parity.json")
}

fn load() -> Fixture {
    let path = fixture_path();
    let raw = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) => panic!(
            "{} missing or unreadable ({error}); run: python -m brainlab.tok.parity_fixture",
            path.display()
        ),
    };
    serde_json::from_slice(&raw).expect("tokenizer_parity.json is malformed")
}

fn hex_bytes(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(&text[index..index + 2], 16).expect("bad hex"))
        .collect()
}

fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|value| format!("{value:02x}")).collect()
}

fn vocab_from(fixture: &Fixture) -> Vocab {
    let tokens: Vec<Vec<u8>> = fixture
        .tokens_hex
        .iter()
        .map(|text| hex_bytes(text))
        .collect();
    let merges: Vec<(u32, u32)> = fixture
        .merges
        .iter()
        .map(|pair| (pair[0], pair[1]))
        .collect();
    Vocab::new(tokens, merges).expect("fixture vocab failed validation")
}

#[test]
fn fixture_matches_python_pretokenizer() {
    let fixture = load();
    assert_eq!(fixture.version, 1);
    for (document, expected) in fixture.docs.iter().zip(fixture.pre_tokenize.iter()) {
        let got = pre_tokenize(document);
        assert_eq!(&got, expected, "pre_tokenize differs for {document:?}");
    }
}

#[test]
fn fixture_matches_python_word_chunking() {
    let fixture = load();
    let mut index = 0usize;
    for document in &fixture.docs {
        for word in pre_tokenize(document) {
            let got: Vec<String> = word_chunks(&word, 32)
                .iter()
                .map(|chunk| to_hex(chunk))
                .collect();
            assert_eq!(
                got, fixture.word_chunks_hex[index],
                "chunks differ for word {word:?}"
            );
            index += 1;
        }
    }
    assert_eq!(index, fixture.word_chunks_hex.len(), "word count mismatch");
}

#[test]
fn fixture_matches_python_encoder() {
    let fixture = load();
    let vocab = vocab_from(&fixture);
    assert_eq!(vocab.vocab_size(), fixture.vocab_size);
    assert_eq!(vocab.eos_id as usize, fixture.eos_id);
    assert_eq!(vocab.pad_id as usize, fixture.pad_id);
    for (document, expected) in fixture.docs.iter().zip(fixture.ids.iter()) {
        let got = vocab.encode(document);
        assert_eq!(
            got,
            *expected,
            "encoder differs for {document:?}\n  rust={:?}\n  py  ={:?}",
            got.iter().take(24).collect::<Vec<_>>(),
            expected.iter().take(24).collect::<Vec<_>>()
        );
    }
}

#[test]
fn decoder_roundtrips_every_fixture_document() {
    let fixture = load();
    let vocab = vocab_from(&fixture);
    for (document, ids) in fixture.docs.iter().zip(fixture.ids.iter()) {
        let decoded = vocab.decode(ids).expect("decode failed");
        let valid_utf8 = std::str::from_utf8(document.as_bytes()).is_ok();
        if valid_utf8 {
            assert_eq!(&decoded, document, "decode(encode(text)) changed the text");
        }
    }
}

#[test]
fn binary_and_json_forms_agree() {
    let fixture = load();
    let vocab = vocab_from(&fixture);
    let bytes = vocab.to_bin();
    let rebuilt = Vocab::from_bin(&bytes).expect("from_bin failed");
    assert_eq!(rebuilt.tokens, vocab.tokens);
    assert_eq!(rebuilt.merges, vocab.merges);
    assert_eq!(rebuilt.eos_id, vocab.eos_id);
    assert_eq!(rebuilt.pad_id, vocab.pad_id);
    for document in &fixture.docs {
        assert_eq!(rebuilt.encode(document), vocab.encode(document));
    }
    let json_vocab = Vocab::from_json_and_merges(&vocab.vocab_json(), &vocab.merges_txt())
        .expect("json roundtrip failed");
    assert_eq!(json_vocab.tokens, vocab.tokens);
    assert_eq!(json_vocab.merges, vocab.merges);
}

#[test]
fn digit_runs_split_from_the_right_in_threes() {
    assert_eq!(split_digit_run("1234567"), ["1", "234", "567"]);
    assert_eq!(split_digit_run("1234"), ["1", "234"]);
    assert_eq!(split_digit_run("123"), ["123"]);
    assert_eq!(split_digit_run("12a34"), ["12a34"]);
    assert_eq!(split_digit_run("٠١٢٣٤"), ["٠١٢٣٤"]);
}

#[test]
fn class_rules_follow_the_spec() {
    let text: Vec<char> = "state-of-the-art x; y".chars().collect();
    assert_eq!(
        pre_tokenize("state-of-the-art x; y"),
        ["state-of-the-art", " ", "x", ";", " ", "y"]
    );
    assert_eq!(char_class(&text, 5), Class::Alnum);
    assert_eq!(char_class(&text, 16), Class::Whitespace);
    assert_eq!(char_class(&text, 18), Class::Other);
}
