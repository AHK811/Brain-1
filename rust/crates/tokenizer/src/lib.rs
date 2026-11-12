pub mod pre_tokenize;
pub mod train;
pub mod vocab;

pub use pre_tokenize::{
    char_class, pre_tokenize, split_digit_run, word_chunks, Class, MAX_WORD_BYTES,
};
pub use train::{train_bpe, TrainStats};
pub use vocab::{Vocab, BYTE_TOKENS, FORMAT_VERSION, MAGIC};

pub struct Tokenizer {
    vocab: Vocab,
}

pub fn read_corpus_files(
    root: impl AsRef<std::path::Path>,
) -> anyhow::Result<Vec<(String, String)>> {
    let root = root.as_ref();
    let mut paths: Vec<std::path::PathBuf> = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(directory) = stack.pop() {
        for entry in std::fs::read_dir(&directory)? {
            let path = entry?.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|value| value.to_str()) == Some("txt") {
                paths.push(path);
            }
        }
    }
    paths.sort();
    let mut out = Vec::with_capacity(paths.len());
    for path in paths {
        let name = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        let raw = std::fs::read(&path)?;
        out.push((name, String::from_utf8_lossy(&raw).into_owned()));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

pub fn read_corpus_dir(root: impl AsRef<std::path::Path>) -> anyhow::Result<Vec<String>> {
    Ok(read_corpus_files(root)?
        .into_iter()
        .map(|(_name, text)| text)
        .collect())
}

pub fn read_lossy(path: impl AsRef<std::path::Path>) -> anyhow::Result<String> {
    let raw = std::fs::read(path.as_ref())?;
    Ok(String::from_utf8_lossy(&raw).into_owned())
}

impl Tokenizer {
    pub fn load(dir: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        Ok(Self {
            vocab: Vocab::load(dir)?,
        })
    }

    pub fn from_vocab(vocab: Vocab) -> Self {
        Self { vocab }
    }

    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.tokens.len()
    }

    pub fn eos_id(&self) -> usize {
        self.vocab.eos_id as usize
    }

    pub fn pad_id(&self) -> usize {
        self.vocab.pad_id as usize
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        self.vocab
            .encode(text)
            .into_iter()
            .map(|id| id as usize)
            .collect()
    }

    pub fn encode_ids(&self, text: &str) -> Vec<u32> {
        self.vocab.encode(text)
    }

    pub fn decode(&self, ids: &[u32]) -> anyhow::Result<String> {
        self.vocab.decode(ids)
    }
}
