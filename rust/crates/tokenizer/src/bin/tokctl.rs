use anyhow::{ensure, Result};
use brainlab_tokenizer::{read_lossy, Tokenizer};
use clap::{Parser, Subcommand};
use serde::Deserialize;
use serde_json::json;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "tokctl",
    version,
    about = "Encode, decode and inspect a trained tokenizer"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Info {
        #[arg(long)]
        tokenizer: PathBuf,
    },
    Encode {
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        ids_out: Option<PathBuf>,
    },
    EncodeDocs {
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        json_in: PathBuf,
        #[arg(long)]
        json_out: Option<PathBuf>,
    },
    Decode {
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        ids: PathBuf,
    },
}

#[derive(Deserialize)]
struct DocsPayload {
    docs: Vec<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Info { tokenizer } => {
            let model = Tokenizer::load(&tokenizer)?;
            let vocab = model.vocab();
            println!(
                "{}",
                json!({
                    "vocab_size": model.vocab_size(),
                    "merges": vocab.merges.len(),
                    "eos_id": model.eos_id(),
                    "pad_id": model.pad_id(),
                    "longest_token": vocab.tokens.iter().map(|token| token.len()).max().unwrap_or(0),
                })
            );
        }
        Command::Encode {
            tokenizer,
            input,
            ids_out,
        } => {
            let model = Tokenizer::load(&tokenizer)?;
            let text = read_lossy(&input)?;
            let ids = model.encode_ids(&text);
            if let Some(path) = &ids_out {
                let mut bytes = Vec::with_capacity(ids.len() * 2);
                for id in &ids {
                    ensure!(*id <= u16::MAX as u32, "token id {id} does not fit uint16");
                    bytes.extend_from_slice(&(*id as u16).to_le_bytes());
                }
                std::fs::write(path, bytes)?;
            }
            println!("{}", json!({"count": ids.len(), "ids": ids}));
        }
        Command::EncodeDocs {
            tokenizer,
            json_in,
            json_out,
        } => {
            let model = Tokenizer::load(&tokenizer)?;
            let payload: DocsPayload = serde_json::from_str(&read_lossy(json_in)?)?;
            let encoded: Vec<Vec<u32>> = payload
                .docs
                .iter()
                .map(|text| model.encode_ids(text))
                .collect();
            let doc = json!({"docs": encoded});
            match json_out {
                Some(path) => std::fs::write(path, serde_json::to_string(&doc)?)?,
                None => println!("{}", serde_json::to_string(&doc)?),
            }
        }
        Command::Decode { tokenizer, ids } => {
            let model = Tokenizer::load(&tokenizer)?;
            let raw = std::fs::read(&ids)?;
            ensure!(
                raw.len().is_multiple_of(2),
                "ids file must be uint16 little endian"
            );
            let values: Vec<u32> = raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| u16::from_le_bytes(*pair) as u32)
                .collect();
            print!("{}", model.decode(&values)?);
        }
    }
    Ok(())
}
