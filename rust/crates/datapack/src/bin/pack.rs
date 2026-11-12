use anyhow::{bail, ensure, Result};
use brainlab_datapack::{pack_split, split_by_file, split_documents, val_stride, PackConfig};
use brainlab_tokenizer::{read_corpus_files, Tokenizer};
use clap::Parser;
use rayon::prelude::*;
use serde_json::json;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "pack",
    version,
    about = "Tokenize a text corpus into memory-mappable uint16 shards"
)]
struct Args {
    #[arg(long)]
    corpus: PathBuf,
    #[arg(long)]
    tokenizer: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 1024)]
    seq_len: usize,
    #[arg(long, default_value_t = 4194304)]
    shard_tokens: usize,
    #[arg(long, default_value_t = 0.005)]
    val_ratio: f64,
    #[arg(long)]
    report: Option<PathBuf>,
}

fn documents_for(
    wanted: &[String],
    texts: &HashMap<String, String>,
    tokenizer: &Tokenizer,
) -> Result<Vec<Vec<u32>>> {
    let vocab = tokenizer.vocab();
    let missing: Vec<&String> = wanted
        .iter()
        .filter(|name| !texts.contains_key(*name))
        .collect();
    if !missing.is_empty() {
        bail!("{} files vanished from the corpus listing", missing.len());
    }
    let per_file: Vec<Result<Vec<Vec<u32>>>> = wanted
        .par_iter()
        .map(|name| {
            let text = texts
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("missing {}", name))?;
            Ok(split_documents(text)
                .into_iter()
                .map(|document| vocab.encode(&document))
                .collect())
        })
        .collect();
    let mut documents: Vec<Vec<u32>> = Vec::new();
    for entry in per_file {
        documents.extend(entry?);
    }
    Ok(documents)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let tokenizer = Tokenizer::load(&args.tokenizer)?;
    let files = read_corpus_files(&args.corpus)?;
    ensure!(
        !files.is_empty(),
        "no .txt files under {}",
        args.corpus.display()
    );
    let texts: HashMap<String, String> = files.into_iter().collect();
    let mut names: Vec<String> = texts.keys().cloned().collect();
    names.sort();
    let (train_names, val_names) = split_by_file(&names, args.val_ratio);
    let cfg = PackConfig {
        seq_len: args.seq_len,
        shard_tokens: args.shard_tokens,
        eos_id: tokenizer.eos_id() as u32,
    };
    cfg.validate()?;

    let train_docs = documents_for(&train_names, &texts, &tokenizer)?;
    let val_docs = documents_for(&val_names, &texts, &tokenizer)?;
    let (train_tokens, train_shards) = pack_split(
        &train_docs,
        &args.out.join("train"),
        "train",
        &cfg,
        &train_names,
    )?;
    let (val_tokens, val_shards) =
        pack_split(&val_docs, &args.out.join("val"), "val", &cfg, &val_names)?;
    let report = json!({
        "files": names.len(),
        "train_files": train_names.len(),
        "val_files": val_names.len(),
        "val_stride": val_stride(args.val_ratio),
        "train_documents": train_docs.len(),
        "val_documents": val_docs.len(),
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_sequences": train_tokens / args.seq_len,
        "val_sequences": val_tokens / args.seq_len,
        "train_shards": train_shards,
        "val_shards": val_shards,
        "seq_len": args.seq_len,
        "shard_tokens": cfg.shard_tokens,
        "vocab_size": tokenizer.vocab_size(),
        "eos_id": tokenizer.eos_id(),
        "out": args.out.display().to_string(),
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    if let Some(path) = args.report {
        std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
    }
    Ok(())
}
