use anyhow::Result;
use brainlab_tokenizer::{read_corpus_dir, train_bpe};
use clap::Parser;
use serde_json::json;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "train-bpe",
    version,
    about = "Train the byte-level BPE described in spec/tokenizer_spec.md"
)]
struct Args {
    #[arg(long)]
    corpus: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 16384)]
    vocab_size: usize,
    #[arg(long)]
    report: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let texts = read_corpus_dir(&args.corpus)?;
    if texts.is_empty() {
        anyhow::bail!("no .txt files under {}", args.corpus.display());
    }
    let started = std::time::Instant::now();
    let (vocab, stats) = train_bpe(&texts, args.vocab_size)?;
    let seconds = started.elapsed().as_secs_f64();
    vocab.save_dir(&args.out)?;
    let roundtrip_ok = texts
        .iter()
        .take(64)
        .all(|text| matches!(vocab.decode(&vocab.encode(text)), Ok(decoded) if decoded == *text));
    let report = json!({
        "corpus_files": texts.len(),
        "word_types": stats.word_types,
        "word_instances": stats.corpus_word_instances,
        "merges": stats.merges,
        "vocab_size": stats.vocab_size,
        "target_vocab": args.vocab_size,
        "stopped_early": stats.stopped_early,
        "largest_pair_count": stats.largest_pair_count,
        "decode_roundtrip_ok": roundtrip_ok,
        "seconds": seconds,
        "out": args.out.display().to_string(),
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    if let Some(path) = args.report {
        std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
    }
    Ok(())
}
