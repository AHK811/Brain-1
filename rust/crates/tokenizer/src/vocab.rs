use crate::pre_tokenize::{pre_tokenize, word_chunks, MAX_WORD_BYTES};
use anyhow::{bail, ensure, Context, Result};
use std::collections::{HashMap, HashSet};
use std::path::Path;

pub const BYTE_TOKENS: usize = 256;
pub const MAGIC: [u8; 4] = *b"BLTK";
pub const FORMAT_VERSION: u16 = 2;
pub const CONTROL_TOKENS: [&str; 12] = [
    "<|image_start|>",
    "<|image_end|>",
    "<|thought_start|>",
    "<|thought_end|>",
    "<|call_start|>",
    "<|call_end|>",
    "<|response_start|>",
    "<|response_end|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|turn_end|>",
];
pub const PAD_LABEL: &[u8] = b"<pad>";
pub const EOS_LABEL: &[u8] = b"<eos>";
pub const SPECIAL_TOKEN_COUNT: usize = CONTROL_TOKENS.len() + 2;

pub(crate) fn special_labels() -> impl Iterator<Item = &'static str> {
    CONTROL_TOKENS.into_iter().chain(["<pad>", "<eos>"])
}

pub(crate) fn protected_spans(text: &str) -> impl Iterator<Item = (usize, &'static str)> + '_ {
    text.match_indices('<').filter_map(|(start, _)| {
        special_labels()
            .find(|label| text[start..].starts_with(label))
            .map(|label| (start, label))
    })
}

#[derive(Debug, Clone)]
pub struct Vocab {
    pub tokens: Vec<Vec<u8>>,
    pub merges: Vec<(u32, u32)>,
    pub eos_id: u32,
    pub pad_id: u32,
    ranks: HashMap<(u32, u32), (u32, u32)>,
}

fn build_ranks(merges: &[(u32, u32)]) -> HashMap<(u32, u32), (u32, u32)> {
    merges
        .iter()
        .enumerate()
        .map(|(index, pair)| (*pair, (index as u32, (BYTE_TOKENS + index) as u32)))
        .collect()
}

impl Vocab {
    pub fn new(tokens: Vec<Vec<u8>>, merges: Vec<(u32, u32)>) -> Result<Self> {
        ensure!(
            tokens.len() >= BYTE_TOKENS + 2 && tokens.len() <= u32::MAX as usize,
            "vocab needs byte tokens and specials within the u32 id space"
        );
        let special_start = BYTE_TOKENS + merges.len();
        let special_count = tokens.len().checked_sub(special_start);
        ensure!(
            special_count == Some(2) || special_count == Some(SPECIAL_TOKEN_COUNT),
            "vocab must contain bytes, ranked merges, then the complete special suffix"
        );
        let skip = if special_count == Some(2) { CONTROL_TOKENS.len() } else { 0 };
        for (token, label) in tokens[special_start..].iter().zip(special_labels().skip(skip)) {
            ensure!(token == label.as_bytes(), "special tokens are missing or out of order");
        }
        for (index, token) in tokens[..BYTE_TOKENS].iter().enumerate() {
            if token.as_slice() != [index as u8] {
                bail!("id {index} must be the single byte {index:02X}, spec/tokenizer_spec.md");
            }
        }
        let mut seen: HashSet<&[u8]> = tokens[..BYTE_TOKENS].iter().map(Vec::as_slice).collect();
        for (rank, &(left, right)) in merges.iter().enumerate() {
            let token_id = BYTE_TOKENS + rank;
            ensure!(
                (left as usize) < token_id && (right as usize) < token_id,
                "merge {rank} must reference earlier ordinary ids"
            );
            let token = &tokens[token_id];
            let mut combined = tokens[left as usize].clone();
            combined.extend_from_slice(&tokens[right as usize]);
            ensure!(*token == combined, "merge {rank} bytes disagree with its children");
            ensure!(
                token.len() <= MAX_WORD_BYTES
                    && !special_labels().any(|label| token.windows(label.len()).any(|part| part == label.as_bytes())),
                "merge {rank} is too long or contains protected bytes"
            );
            ensure!(seen.insert(token), "duplicate merge {rank}");
        }
        let eos_id = tokens.len() as u32 - 1;
        let pad_id = tokens.len() as u32 - 2;
        let ranks = build_ranks(&merges);
        Ok(Self {
            tokens,
            merges,
            eos_id,
            pad_id,
            ranks,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn format_version(&self) -> u16 {
        if self.tokens.len() == BYTE_TOKENS + self.merges.len() + SPECIAL_TOKEN_COUNT {
            FORMAT_VERSION
        } else {
            1
        }
    }

    pub fn special_id(&self, text: &str) -> Option<u32> {
        let skip = if self.format_version() == 1 { CONTROL_TOKENS.len() } else { 0 };
        special_labels().skip(skip).position(|label| label == text)
            .map(|index| (BYTE_TOKENS + self.merges.len() + index) as u32)
    }

    pub fn special_tokens(&self) -> HashMap<&'static str, u32> {
        special_labels().filter_map(|label| self.special_id(label).map(|id| (label, id))).collect()
    }

    pub fn encode_chunk(&self, chunk: &[u8]) -> Vec<u32> {
        let mut syms: Vec<u32> = chunk.iter().map(|byte| *byte as u32).collect();
        loop {
            let mut best_rank = u32::MAX;
            let mut best_pos: Option<usize> = None;
            let mut best_id = 0u32;
            for position in 0..syms.len().saturating_sub(1) {
                if let Some((rank, id)) = self.ranks.get(&(syms[position], syms[position + 1])) {
                    if *rank < best_rank {
                        best_rank = *rank;
                        best_pos = Some(position);
                        best_id = *id;
                    }
                }
            }
            let Some(position) = best_pos else { break };
            syms[position] = best_id;
            syms.remove(position + 1);
        }
        syms
    }

    pub fn encode_word(&self, word: &str) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::new();
        for chunk in word_chunks(word, MAX_WORD_BYTES) {
            out.extend(self.encode_chunk(&chunk));
        }
        out
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::new();
        for word in pre_tokenize(text) {
            out.extend(self.encode_word(&word));
        }
        out
    }

    pub fn encode_with_special(&self, text: &str, allow_special: bool) -> Vec<u32> {
        if !allow_special {
            return self.encode(text);
        }
        let mut out = Vec::new();
        let mut start = 0;
        for (position, label) in protected_spans(text) {
            if let Some(id) = self.special_id(label) {
                out.extend(self.encode(&text[start..position]));
                out.push(id);
                start = position + label.len();
            }
        }
        out.extend(self.encode(&text[start..]));
        out
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut bytes: Vec<u8> = Vec::new();
        for id in ids {
            let index = *id as usize;
            ensure!(
                index < self.tokens.len(),
                "token id {index} outside vocab {}",
                self.tokens.len()
            );
            bytes.extend_from_slice(&self.tokens[index]);
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    pub fn to_bin(&self) -> Vec<u8> {
        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&self.format_version().to_le_bytes());
        out.extend_from_slice(&(self.tokens.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.merges.len() as u32).to_le_bytes());
        for token in &self.tokens {
            out.push(token.len() as u8);
            out.extend_from_slice(token);
        }
        for (left, right) in &self.merges {
            out.extend_from_slice(&left.to_le_bytes());
            out.extend_from_slice(&right.to_le_bytes());
        }
        out.extend_from_slice(&self.eos_id.to_le_bytes());
        out.extend_from_slice(&self.pad_id.to_le_bytes());
        out
    }

    pub fn from_bin(raw: &[u8]) -> Result<Self> {
        ensure!(raw.len() >= 22, "tokenizer.bin is truncated");
        ensure!(raw[0..4] == MAGIC, "bad magic {:?}", &raw[0..4]);
        let version = u16::from_le_bytes([raw[4], raw[5]]);
        ensure!(
            version == 1 || version == FORMAT_VERSION,
            "unsupported format version {version}"
        );
        let vocab_size = u32::from_le_bytes(raw[6..10].try_into()?) as usize;
        let n_merges = u32::from_le_bytes(raw[10..14].try_into()?) as usize;
        let special_count = if version == 1 { 2 } else { SPECIAL_TOKEN_COUNT };
        ensure!(
            vocab_size as u64 == BYTE_TOKENS as u64 + n_merges as u64 + special_count as u64,
            "vocab size does not match version and merge count"
        );
        ensure!(
            vocab_size as u64 + n_merges as u64 * 8 + 22 <= raw.len() as u64,
            "tokenizer.bin is truncated"
        );
        let mut offset = 14usize;
        let mut tokens: Vec<Vec<u8>> = Vec::with_capacity(vocab_size);
        for _ in 0..vocab_size {
            ensure!(
                offset < raw.len(),
                "tokenizer.bin truncated in the token table"
            );
            let length = raw[offset] as usize;
            offset += 1;
            ensure!(
                offset + length <= raw.len(),
                "tokenizer.bin truncated in a token body"
            );
            tokens.push(raw[offset..offset + length].to_vec());
            offset += length;
        }
        let mut merges: Vec<(u32, u32)> = Vec::with_capacity(n_merges);
        for _ in 0..n_merges {
            ensure!(
                offset + 8 <= raw.len(),
                "tokenizer.bin truncated in the merge table"
            );
            merges.push((
                u32::from_le_bytes(raw[offset..offset + 4].try_into()?),
                u32::from_le_bytes(raw[offset + 4..offset + 8].try_into()?),
            ));
            offset += 8;
        }
        ensure!(offset + 8 <= raw.len(), "tokenizer.bin truncated in special ids");
        ensure!(offset + 8 == raw.len(), "trailing bytes after special ids");
        let eos_id = u32::from_le_bytes(raw[offset..offset + 4].try_into()?);
        let pad_id = u32::from_le_bytes(raw[offset + 4..offset + 8].try_into()?);
        let vocab = Self::new(tokens, merges)?;
        ensure!(eos_id == vocab.eos_id, "eos id must be the last id");
        ensure!(pad_id == vocab.pad_id, "pad id must be second to last");
        Ok(vocab)
    }

    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let binary = dir.join("tokenizer.bin");
        if binary.is_file() {
            let raw =
                std::fs::read(&binary).with_context(|| format!("reading {}", binary.display()))?;
            return Self::from_bin(&raw).with_context(|| format!("parsing {}", binary.display()));
        }
        let vocab = std::fs::read_to_string(dir.join("vocab.json"))
            .or_else(|_| std::fs::read_to_string(dir.join("tokenizer.json")))
            .context("no tokenizer.bin or vocab.json in the directory")?;
        let merges =
            std::fs::read_to_string(dir.join("merges.txt")).context("missing merges.txt")?;
        Self::from_json_and_merges(&vocab, &merges)
            .with_context(|| format!("loading {}", dir.display()))
    }

    pub fn from_json_and_merges(vocab_json: &str, merges_txt: &str) -> Result<Self> {
        let parsed: HashMap<String, u32> = serde_json::from_str(vocab_json)?;
        let size = parsed.values().max().copied().context("empty vocab.json")? as usize + 1;
        ensure!(
            parsed.len() == size,
            "vocab.json ids are not contiguous 0..{size}"
        );
        ensure!(parsed.values().collect::<HashSet<_>>().len() == size, "duplicate token ids");
        let mut tokens: Vec<Vec<u8>> = vec![Vec::new(); size];
        for (key, id) in &parsed {
            tokens[*id as usize] = latin1_bytes(key)?;
        }
        let mut lines = merges_txt.lines();
        let version = match lines.next() {
            Some("#version: 1.0") => 1,
            Some("#version: 2.0") => FORMAT_VERSION,
            _ => bail!("unsupported or missing merges.txt version comment"),
        };
        let mut merges: Vec<(u32, u32)> = Vec::new();
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            let mut parts = line.split_whitespace();
            let left = parts.next().context("bad merges.txt line")?;
            let right = parts.next().context("bad merges.txt line")?;
            ensure!(
                parts.next().is_none() && left.bytes().chain(right.bytes()).all(|byte| byte.is_ascii_digit()),
                "merge ids must be two unsigned decimal integers"
            );
            merges.push((left.parse()?, right.parse()?));
        }
        let vocab = Self::new(tokens, merges)?;
        ensure!(vocab.format_version() == version, "version does not match special token layout");
        Ok(vocab)
    }

    pub fn save_dir(&self, dir: impl AsRef<Path>) -> Result<()> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("tokenizer.bin"), self.to_bin())?;
        std::fs::write(dir.join("vocab.json"), self.vocab_json())?;
        std::fs::write(dir.join("merges.txt"), self.merges_txt())?;
        Ok(())
    }

    pub fn vocab_json(&self) -> String {
        let mut out = String::from("{\n");
        for (index, token) in self.tokens.iter().enumerate() {
            if index > 0 {
                out.push_str(",\n");
            }
            out.push_str(&format!(
                "  \"{}\": {}",
                escape(&latin1_string(token)),
                index
            ));
        }
        out.push_str("\n}");
        out
    }

    pub fn merges_txt(&self) -> String {
        let mut out = String::from("#version: 1.0\n");
        for (left, right) in &self.merges {
            out.push_str(&format!("{left} {right}\n"));
        }
        out
    }

    pub fn byte_token(&self, id: u32) -> Option<&[u8]> {
        self.tokens.get(id as usize).map(|token| token.as_slice())
    }
}

fn latin1_string(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| char::from(*byte)).collect()
}

fn latin1_bytes(text: &str) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len());
    for ch in text.chars() {
        let code = ch as u32;
        ensure!(
            code < 256,
            "vocab key has U+{code:04X}, which is not a latin-1 byte"
        );
        out.push(code as u8);
    }
    Ok(out)
}

fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            value if (value as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", value as u32)),
            value if (value as u32) < 0x80 => out.push(value),
            value => out.push_str(&format!("\\u{:04x}", value as u32)),
        }
    }
    out
}
