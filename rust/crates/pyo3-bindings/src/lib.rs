#![allow(clippy::useless_conversion)]

use anyhow::{Context, Result};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::Deserialize;
use serde_json::json;
use std::sync::{Arc, Mutex, OnceLock};

static VOCAB: OnceLock<Mutex<Option<Arc<brainlab_tokenizer::Vocab>>>> = OnceLock::new();

fn vocab() -> Result<Arc<brainlab_tokenizer::Vocab>> {
    let guard = VOCAB
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    guard
        .clone()
        .ok_or_else(|| anyhow::anyhow!("call brainlab_rs.load_tokenizer before encoding"))
}

#[derive(Deserialize)]
struct DocsPayload {
    docs: Vec<String>,
}

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
fn cpu_features() -> (bool, bool, usize) {
    #[cfg(target_arch = "x86_64")]
    let detected = (
        std::is_x86_feature_detected!("avx"),
        std::is_x86_feature_detected!("avx2"),
    );
    #[cfg(not(target_arch = "x86_64"))]
    let detected = (false, false);
    (
        detected.0,
        detected.1,
        std::thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(1),
    )
}

#[pyfunction]
fn load_tokenizer(dir: &str) -> PyResult<usize> {
    let loaded = brainlab_tokenizer::Vocab::load(dir).context("load_tokenizer")?;
    let size = loaded.vocab_size();
    *VOCAB
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(Arc::new(loaded));
    Ok(size)
}

#[pyfunction]
fn train_bpe(corpus_dir: &str, out_dir: &str, vocab_size: usize) -> PyResult<String> {
    let texts = brainlab_tokenizer::read_corpus_dir(corpus_dir).context("read corpus")?;
    if texts.is_empty() {
        return Err(PyRuntimeError::new_err(format!(
            "no .txt files under {corpus_dir}"
        )));
    }
    let (vocab, stats) = brainlab_tokenizer::train_bpe(&texts, vocab_size).context("train_bpe")?;
    vocab.save_dir(out_dir).context("save tokenizer")?;
    Ok(json!({
        "word_types": stats.word_types,
        "merges": stats.merges,
        "vocab_size": stats.vocab_size,
        "stopped_early": stats.stopped_early,
    })
    .to_string())
}

#[pyfunction]
fn encode_docs<'py>(py: Python<'py>, blob: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
    let payload: DocsPayload = serde_json::from_slice(blob).context("decode payload")?;
    let loaded = vocab()?;
    let encoded = py.allow_threads(|| {
        payload
            .docs
            .iter()
            .map(|text| loaded.encode(text))
            .collect::<Vec<Vec<u32>>>()
    });
    let bytes = serde_json::to_vec(&json!({ "docs": encoded })).context("encode json")?;
    Ok(PyBytes::new_bound(py, &bytes))
}

#[pyfunction]
fn decode_ids(ids: Vec<u32>) -> PyResult<String> {
    let loaded = vocab()?;
    Ok(loaded.decode(&ids).context("decode_ids")?)
}

#[pyfunction]
fn encode_text(text: &str) -> PyResult<Vec<u32>> {
    let loaded = vocab()?;
    Ok(loaded.encode(text))
}

#[pyfunction]
fn pack_shards(
    corpus_dir: &str,
    tokenizer_dir: &str,
    out_dir: &str,
    seq_len: usize,
    shard_tokens: usize,
    val_ratio: f64,
) -> PyResult<String> {
    let tokenizer = brainlab_tokenizer::Tokenizer::load(tokenizer_dir).context("load tokenizer")?;
    let files = brainlab_tokenizer::read_corpus_files(corpus_dir).context("read corpus")?;
    let mut names: Vec<String> = files.iter().map(|(name, _)| name.clone()).collect();
    names.sort();
    let texts: std::collections::HashMap<String, String> = files.into_iter().collect();
    let (train_names, val_names) = brainlab_datapack::split_by_file(&names, val_ratio);
    let cfg = brainlab_datapack::PackConfig {
        seq_len,
        shard_tokens,
        eos_id: tokenizer.eos_id() as u32,
    };
    cfg.validate().context("pack config")?;
    let encode_split = |wanted: &[String]| -> Result<Vec<Vec<u32>>> {
        let vocab = tokenizer.vocab();
        Ok(wanted
            .iter()
            .flat_map(|name| {
                let text = texts.get(name).cloned().unwrap_or_default();
                brainlab_datapack::split_documents(&text)
                    .into_iter()
                    .map(|document| vocab.encode(&document))
                    .collect::<Vec<Vec<u32>>>()
            })
            .collect())
    };
    let train = encode_split(&train_names).context("encode train")?;
    let val = encode_split(&val_names).context("encode val")?;
    let out = std::path::Path::new(out_dir);
    let (train_tokens, train_shards) =
        brainlab_datapack::pack_split(&train, &out.join("train"), "train", &cfg, &train_names)
            .context("write train")?;
    let (val_tokens, val_shards) =
        brainlab_datapack::pack_split(&val, &out.join("val"), "val", &cfg, &val_names)
            .context("write val")?;
    Ok(json!({
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_shards": train_shards,
        "val_shards": val_shards,
    })
    .to_string())
}

#[pyfunction]
fn rmsnorm_f32<'py>(
    py: Python<'py>,
    x: &[u8],
    w: &[u8],
    d: usize,
    eps: f32,
    threads: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    let rows = x.len() / 4;
    let values: Vec<f32> = x
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect();
    let weights: Vec<f32> = w
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect();
    if !values.len().is_multiple_of(d) {
        return Err(PyRuntimeError::new_err(format!(
            "{} floats are not a whole number of rows of {d}",
            values.len()
        )));
    }
    let mut out = vec![0.0f32; rows];
    let result = if threads > 1 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .context("thread pool")?;
        pool.install(|| brainlab_infer::kernels::rmsnorm_par(&values, &weights, &mut out, d, eps))
    } else {
        brainlab_infer::kernels::rmsnorm(&values, &weights, &mut out, d, eps)
    };
    result.context("rmsnorm")?;
    let mut bytes = Vec::with_capacity(out.len() * 4);
    for value in out {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(PyBytes::new_bound(py, &bytes))
}

#[pymodule]
fn brainlab_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(cpu_features, m)?)?;
    m.add_function(wrap_pyfunction!(load_tokenizer, m)?)?;
    m.add_function(wrap_pyfunction!(train_bpe, m)?)?;
    m.add_function(wrap_pyfunction!(encode_docs, m)?)?;
    m.add_function(wrap_pyfunction!(encode_text, m)?)?;
    m.add_function(wrap_pyfunction!(decode_ids, m)?)?;
    m.add_function(wrap_pyfunction!(pack_shards, m)?)?;
    m.add_function(wrap_pyfunction!(rmsnorm_f32, m)?)?;
    m.add("__rust_version__", rustc_version())?;
    Ok(())
}

fn rustc_version() -> &'static str {
    option_env!("RUSTC_VERSION").unwrap_or("unknown")
}
