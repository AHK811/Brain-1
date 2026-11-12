use crate::pre_tokenize::{pre_tokenize, word_chunks, MAX_WORD_BYTES};
use anyhow::{bail, ensure, Context, Result};
use std::collections::HashMap;
use std::path::Path;

pub const BYTE_TOKENS: usize = 256;
pub const MAGIC: [u8; 4] = *b"BLTK";
pub const FORMAT_VERSION: u16 = 1;

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
            tokens.len() >= BYTE_TOKENS + 2,
            "vocab needs the 256 byte tokens plus two specials"
        );
        for (index, token) in tokens[..BYTE_TOKENS].iter().enumerate() {
            if token.as_slice() != [index as u8] {
                bail!("id {index} must be the single byte {index:02X}, spec/tokenizer_spec.md");
            }
        }
        for token in &tokens {
            if token.len() > 255 {
                bail!(
                    "token of {} bytes cannot be stored in tokenizer.bin",
                    token.len()
                );
            }
        }
        if merges.len() + BYTE_TOKENS + 2 != tokens.len() {
            bail!(
                "{} merges produce {} ids but the vocab has {}",
                merges.len(),
                merges.len() + BYTE_TOKENS + 2,
                tokens.len()
            );
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
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
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
        ensure!(raw.len() >= 14, "tokenizer.bin is truncated");
        ensure!(raw[0..4] == MAGIC, "bad magic {:?}", &raw[0..4]);
        let version = u16::from_le_bytes([raw[4], raw[5]]);
        ensure!(
            version == FORMAT_VERSION,
            "unsupported format version {version}"
        );
        let vocab_size = u32::from_le_bytes(raw[6..10].try_into()?) as usize;
        let n_merges = u32::from_le_bytes(raw[10..14].try_into()?) as usize;
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
        ensure!(
            offset + 8 == raw.len(),
            "{} trailing bytes after specials",
            raw.len() - offset - 8
        );
        let eos_id = u32::from_le_bytes(raw[offset..offset + 4].try_into()?);
        let pad_id = u32::from_le_bytes(raw[offset + 4..offset + 8].try_into()?);
        ensure!(
            eos_id as usize + 1 == vocab_size,
            "eos id {eos_id} must be the last of {vocab_size}"
        );
        ensure!(
            pad_id as usize + 2 == vocab_size,
            "pad id {pad_id} must be second to last"
        );
        let ranks = build_ranks(&merges);
        Ok(Self {
            tokens,
            merges,
            eos_id,
            pad_id,
            ranks,
        })
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
        let mut tokens: Vec<Vec<u8>> = vec![Vec::new(); size];
        for (key, id) in &parsed {
            tokens[*id as usize] = latin1_bytes(key)?;
        }
        let mut merges: Vec<(u32, u32)> = Vec::new();
        for (index, line) in merges_txt.lines().enumerate() {
            if index == 0 {
                ensure!(
                    line.starts_with("#version:"),
                    "merges.txt must start with a version comment"
                );
                continue;
            }
            if line.trim().is_empty() {
                continue;
            }
            let mut parts = line.split_whitespace();
            let left = parts
                .next()
                .and_then(|value| value.parse::<u32>().ok())
                .context("bad merges.txt line")?;
            let right = parts
                .next()
                .and_then(|value| value.parse::<u32>().ok())
                .context("bad merges.txt line")?;
            merges.push((left, right));
        }
        let eos_id = size as u32 - 1;
        let pad_id = size as u32 - 2;
        let ranks = build_ranks(&merges);
        Ok(Self {
            tokens,
            merges,
            eos_id,
            pad_id,
            ranks,
        })
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
