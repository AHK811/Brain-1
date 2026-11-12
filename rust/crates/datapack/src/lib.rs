use anyhow::{bail, ensure, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct PackConfig {
    pub seq_len: usize,
    pub shard_tokens: usize,
    pub eos_id: u32,
}

impl PackConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.seq_len > 0, "seq_len must be positive");
        ensure!(
            self.shard_tokens.is_multiple_of(self.seq_len),
            "shard_tokens {} must be a multiple of seq_len {}",
            self.shard_tokens,
            self.seq_len
        );
        if self.eos_id > u16::MAX as u32 {
            bail!("eos id {} does not fit uint16", self.eos_id);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Manifest {
    pub n_tokens: usize,
    pub seq_len: usize,
    pub dtype: String,
    pub split: String,
    pub byte_order: String,
    pub row_major: bool,
    pub source_files: Vec<String>,
    pub sha256: String,
}

pub fn is_ascii_ws(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

pub fn split_documents(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let mut documents: Vec<String> = Vec::new();
    let mut start = 0usize;
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index] != b'\n' {
            index += 1;
            continue;
        }
        let mut run = index + 1;
        while run < bytes.len() && is_ascii_ws(bytes[run]) {
            run += 1;
        }
        let last_newline = (index + 1..run).rev().find(|probe| bytes[*probe] == b'\n');
        match last_newline {
            Some(end) => {
                documents.push(text[start..index].trim().to_string());
                start = end + 1;
                index = end + 1;
            }
            None => index += 1,
        }
    }
    documents.push(text[start..].trim().to_string());
    documents.retain(|document| !document.is_empty());
    documents
}

pub fn val_stride(val_ratio: f64) -> usize {
    if val_ratio <= 0.0 {
        usize::MAX
    } else {
        (1.0 / val_ratio).round().max(1.0) as usize
    }
}

pub fn split_by_file(names: &[String], val_ratio: f64) -> (Vec<String>, Vec<String>) {
    let stride = val_stride(val_ratio);
    let mut train = Vec::new();
    let mut val = Vec::new();
    for (index, name) in names.iter().enumerate() {
        if stride != usize::MAX && index % stride == 0 {
            val.push(name.clone());
        } else {
            train.push(name.clone());
        }
    }
    (train, val)
}

pub fn pack_stream(documents: &[Vec<u32>], cfg: &PackConfig) -> Result<Vec<u16>> {
    let mut stream: Vec<u16> = Vec::new();
    for doc in documents {
        for id in doc {
            if *id > u16::MAX as u32 {
                bail!("token id {id} does not fit uint16 shards");
            }
            stream.push(*id as u16);
        }
        stream.push(cfg.eos_id as u16);
    }
    let usable = stream.len() - (stream.len() % cfg.seq_len);
    stream.truncate(usable);
    Ok(stream)
}

pub fn write_shards(
    stream: &[u16],
    out_dir: &Path,
    split: &str,
    cfg: &PackConfig,
    source_files: &[String],
) -> Result<Vec<PathBuf>> {
    std::fs::create_dir_all(out_dir)?;
    let mut written = Vec::new();
    for (index, chunk) in stream.chunks(cfg.shard_tokens).enumerate() {
        if chunk.len() < cfg.seq_len {
            break;
        }
        let base = out_dir.join(format!("{index:06}"));
        let bin_path = {
            let path = base.with_extension("bin");
            let bytes: Vec<u8> = chunk.iter().flat_map(|value| value.to_le_bytes()).collect();
            std::fs::write(&path, &bytes)?;
            path
        };
        let digest = Sha256::digest(&std::fs::read(&bin_path)?);
        let manifest = Manifest {
            n_tokens: chunk.len(),
            seq_len: cfg.seq_len,
            dtype: "uint16".to_string(),
            split: split.to_string(),
            byte_order: "little".to_string(),
            row_major: true,
            source_files: source_files.to_vec(),
            sha256: digest.iter().map(|byte| format!("{byte:02x}")).collect(),
        };
        std::fs::write(
            base.with_extension("json"),
            serde_json::to_string_pretty(&manifest)?,
        )?;
        written.push(bin_path);
    }
    Ok(written)
}

pub fn pack_split(
    documents: &[Vec<u32>],
    out_dir: &Path,
    split: &str,
    cfg: &PackConfig,
    source_files: &[String],
) -> Result<(usize, usize)> {
    let stream = pack_stream(documents, cfg)?;
    let shards = write_shards(&stream, out_dir, split, cfg, source_files)?;
    Ok((stream.len(), shards.len()))
}
