use anyhow::{bail, ensure, Context, Result};
use brainlab_infer::attention::attention_causal;
use brainlab_infer::golden::{max_abs_error, rel_l2_error, GoldenSet, WorkloadRecord};
use brainlab_infer::kernels::{rmsnorm, rmsnorm_par, softmax_rows};
use brainlab_infer::matmul::{linear, linear_tiled, transpose};
use brainlab_infer::model::{Engine, Weights};
use brainlab_infer::sample::{argmax, pick, Rng, Sampler};
use clap::{Parser, Subcommand};
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "brainlab",
    version,
    about = "Rust CPU inference engine, kernel benchmarker and sampler"
)]
struct Cli {
    #[arg(
        long,
        global = true,
        default_value_t = 0,
        help = "rayon threads, 0 leaves the pool at its default"
    )]
    threads: usize,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    List {
        #[arg(long, default_value = "tests/golden")]
        golden_dir: PathBuf,
    },
    Info {
        #[arg(long)]
        model: PathBuf,
    },
    Bench {
        #[arg(long, default_value = "tests/golden")]
        golden_dir: PathBuf,
        #[arg(long)]
        key: String,
        #[arg(long, default_value_t = 200)]
        reps: usize,
        #[arg(long, default_value_t = 20)]
        warmup: usize,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    Forward {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value = "tests/golden")]
        golden_dir: PathBuf,
        #[arg(long)]
        key: String,
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long, default_value_t = 20)]
        reps: usize,
        #[arg(long, default_value_t = 3)]
        warmup: usize,
    },
    Generate {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long, default_value = "")]
        prompt: String,
        #[arg(long, default_value_t = 64)]
        max_new: usize,
        #[arg(long, default_value_t = 0.8)]
        temperature: f32,
        #[arg(long, default_value_t = 40)]
        top_k: usize,
        #[arg(long, default_value_t = 0.9)]
        top_p: f32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    SelfTest {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value_t = 16)]
        prefill: usize,
        #[arg(long, default_value_t = 8)]
        steps: usize,
    },
    Profile {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value = "tests/golden")]
        golden_dir: PathBuf,
        #[arg(long, default_value = "forward_fp32_1x1024")]
        key: String,
        #[arg(long, default_value_t = 3)]
        reps: usize,
        #[arg(long, default_value_t = 0)]
        decode: usize,
    },
}

fn median(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[middle]
    } else {
        0.5 * (sorted[middle - 1] + sorted[middle])
    }
}

fn summary(samples: &[f64]) -> Map<String, Value> {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let index = ((0.9 * (sorted.len() - 1) as f64).round() as usize).min(sorted.len() - 1);
    Map::from_iter(vec![
        ("ms_median".to_string(), json!(median(samples))),
        ("ms_min".to_string(), json!(sorted[0])),
        ("ms_p90".to_string(), json!(sorted[index])),
        ("ms_max".to_string(), json!(sorted[sorted.len() - 1])),
    ])
}

fn hardware(threads: usize) -> Value {
    #[cfg(target_arch = "x86_64")]
    let (avx, avx2) = (
        std::is_x86_feature_detected!("avx"),
        std::is_x86_feature_detected!("avx2"),
    );
    #[cfg(not(target_arch = "x86_64"))]
    let (avx, avx2) = (false, false);
    json!({
        "backend": "rust_cpu",
        "engine_version": brainlab_infer::version(),
        "rayon_threads": rayon::current_num_threads(),
        "threads_requested": threads,
        "available_parallelism": std::thread::available_parallelism().map(|value| value.get()).unwrap_or(1),
        "pointer_width": std::mem::size_of::<usize>() * 8,
        "avx": avx,
        "avx2": avx2,
    })
}

fn parity_block(actual: &[f32], expected: &[f32], record: &WorkloadRecord) -> Result<Value> {
    let atol = record.atol()?;
    let rtol = record.rtol()?;
    let rel_limit = record.rel_l2_limit()?;
    let max_abs = max_abs_error(actual, expected)?;
    let rel = rel_l2_error(actual, expected)?;
    let elementwise_ok = actual
        .iter()
        .zip(expected)
        .all(|(a, b)| (a - b).abs() as f64 <= atol + rtol * b.abs() as f64);
    Ok(json!({
        "max_abs_err": max_abs,
        "rel_l2_err": rel,
        "atol": atol,
        "rtol": rtol,
        "rel_l2_limit": rel_limit,
        "elementwise_pass": elementwise_ok,
        "rel_pass": rel <= rel_limit,
        "parity_pass": elementwise_ok && rel <= rel_limit,
    }))
}

fn timed(reps: usize, warmup: usize, mut run: impl FnMut()) -> Vec<f64> {
    for _ in 0..warmup {
        run();
    }
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let started = Instant::now();
        run();
        samples.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    samples
}

fn write_out(path: Option<&Path>, values: &[f32]) -> Result<()> {
    let Some(path) = path else {
        return Ok(());
    };
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    std::fs::write(path, bytes)?;
    Ok(())
}

fn two_dims(shape: &[usize]) -> Result<(usize, usize)> {
    match shape {
        [rows, cols] => Ok((*rows, *cols)),
        other => bail!("expected a 2-D shape, got {other:?}"),
    }
}

fn four_dims(shape: &[usize]) -> Result<(usize, usize, usize, usize)> {
    match shape {
        [batch, heads, seq, head_dim] => Ok((*batch, *heads, *seq, *head_dim)),
        other => bail!("expected a 4-D shape, got {other:?}"),
    }
}

fn bench_rmsnorm(
    record: &WorkloadRecord,
    tensors: &std::collections::BTreeMap<String, brainlab_infer::golden::Array>,
    reps: usize,
    warmup: usize,
    threads: usize,
    out: Option<&Path>,
) -> Result<Value> {
    let x = tensors.get("x").expect("x");
    let w = tensors.get("w").expect("w");
    let expected = &tensors.get(&record.output).expect("reference output").data;
    let (_, cols) = two_dims(&x.shape)?;
    ensure!(
        w.data.len() == cols,
        "weight width {} != {cols}",
        w.data.len()
    );
    let eps = record.eps()?;
    let mut out_buffer = vec![0.0f32; x.data.len()];
    let run = |target: &mut [f32]| {
        if threads > 1 {
            rmsnorm_par(&x.data, &w.data, target, cols, eps)
        } else {
            rmsnorm(&x.data, &w.data, target, cols, eps)
        }
    };
    run(&mut out_buffer)?;
    let parity = parity_block(&out_buffer, expected, record)?;
    let samples = timed(reps, warmup, || {
        run(&mut out_buffer).expect("rmsnorm");
    });
    write_out(out, &out_buffer)?;
    let ms = median(&samples);
    let traffic = record.traffic()? as f64;
    Ok(json!({
        "kernel": if threads > 1 { "rmsnorm_par" } else { "rmsnorm" },
        "threads": threads,
        "shape": format!("{}x{}", x.shape[0], cols),
        "timing_mode": "per-rep wall clock",
        "gb_s": traffic / (ms * 1e-3) / 1e9,
        "flops": record.flop_count()?,
        "summary": summary(&samples),
        "parity": parity,
    }))
}

fn bench_softmax(
    record: &WorkloadRecord,
    tensors: &std::collections::BTreeMap<String, brainlab_infer::golden::Array>,
    reps: usize,
    warmup: usize,
    out: Option<&Path>,
) -> Result<Value> {
    let x = tensors.get("x").expect("x");
    let expected = &tensors.get(&record.output).expect("reference output").data;
    let (rows, cols) = two_dims(&x.shape)?;
    let mut produced = vec![0.0f32; x.data.len()];
    softmax_rows(&x.data, &mut produced, cols)?;
    let parity = parity_block(&produced, expected, record)?;
    let samples = timed(reps, warmup, || {
        softmax_rows(&x.data, &mut produced, cols).expect("softmax");
    });
    write_out(out, &produced)?;
    let ms = median(&samples);
    Ok(json!({
        "kernel": "softmax_rows",
        "shape": format!("{rows}x{cols}"),
        "gb_s": (record.traffic()? as f64) / (ms * 1e-3) / 1e9,
        "summary": summary(&samples),
        "parity": parity,
    }))
}

fn bench_attention(
    record: &WorkloadRecord,
    tensors: &std::collections::BTreeMap<String, brainlab_infer::golden::Array>,
    reps: usize,
    warmup: usize,
    out: Option<&Path>,
) -> Result<Value> {
    let q = tensors.get("q").expect("q");
    let k = tensors.get("k").expect("k");
    let v = tensors.get("v").expect("v");
    let expected = &tensors.get(&record.output).expect("reference output").data;
    let (batch, heads, seq, head_dim) = four_dims(&q.shape)?;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut produced = vec![0.0f32; q.data.len()];
    attention_causal(
        &q.data,
        &k.data,
        &v.data,
        &mut produced,
        batch,
        heads,
        seq,
        head_dim,
        scale,
    )?;
    let parity = parity_block(&produced, expected, record)?;
    let samples = timed(reps, warmup, || {
        attention_causal(
            &q.data,
            &k.data,
            &v.data,
            &mut produced,
            batch,
            heads,
            seq,
            head_dim,
            scale,
        )
        .expect("attention");
    });
    write_out(out, &produced)?;
    let ms = median(&samples);
    Ok(json!({
        "kernel": "attention_causal_online_softmax",
        "shape": format!("{batch}x{heads}x{seq}x{head_dim}"),
        "gflops": record.flop_count()? / (ms * 1e-3) / 1e9,
        "gb_s": (record.traffic()? as f64) / (ms * 1e-3) / 1e9,
        "summary": summary(&samples),
        "parity": parity,
    }))
}

fn bench_matmul(
    record: &WorkloadRecord,
    tensors: &std::collections::BTreeMap<String, brainlab_infer::golden::Array>,
    reps: usize,
    warmup: usize,
    out: Option<&Path>,
) -> Result<Value> {
    let a = tensors.get("a").expect("a");
    let b = tensors.get("b").expect("b");
    let expected = &tensors.get(&record.output).expect("reference output").data;
    let (m, k) = two_dims(&a.shape)?;
    let (k2, n) = two_dims(&b.shape)?;
    ensure!(k == k2, "inner dimensions differ {k} vs {k2}");
    let mut produced = vec![0.0f32; m * n];
    let mut tiled = vec![0.0f32; m * n];
    let mut bt = vec![0.0f32; k * n];
    transpose(&b.data, k, n, &mut bt)?;
    linear(&a.data, &bt, &mut produced, m, k, n, None)?;
    let parity = parity_block(&produced, expected, record)?;
    let baseline_samples = timed(reps, warmup, || {
        linear(&a.data, &bt, &mut produced, m, k, n, None).expect("linear");
    });
    linear_tiled(&a.data, &bt, &mut tiled, m, k, n, None)?;
    let tiled_parity = parity_block(&tiled, expected, record)?;
    let tiled_samples = timed(reps, warmup, || {
        linear_tiled(&a.data, &bt, &mut tiled, m, k, n, None).expect("linear_tiled");
    });
    write_out(out, &tiled)?;
    let ms = median(&baseline_samples);
    let tiled_ms = median(&tiled_samples);
    let agreement = max_abs_error(&produced, &tiled)?;
    Ok(json!({
        "kernel": "block_4x2_tiled",
        "shape": format!("{m}x{k}x{n}"),
        "gflops": record.flop_count()? / (tiled_ms * 1e-3) / 1e9,
        "ms_median": tiled_ms,
        "transpose_once": true,
        "summary": summary(&tiled_samples),
        "baseline": {
            "kernel": "dot_fma",
            "ms_median": ms,
            "gflops": record.flop_count()? / (ms * 1e-3) / 1e9,
            "summary": summary(&baseline_samples),
            "parity": parity.clone(),
        },
        "speedup_vs_dot": if tiled_ms > 0.0 { ms / tiled_ms } else { 0.0 },
        "baseline_vs_tiled_max_abs": agreement,
        "parity": tiled_parity,
    }))
}

fn bench(
    key: &str,
    golden_dir: &Path,
    reps: usize,
    warmup: usize,
    threads: usize,
    out: Option<&Path>,
) -> Result<Value> {
    let set = GoldenSet::open(golden_dir)?;
    let record = set.record(key)?;
    let tensors = set.load(record)?;
    let body = match record.op.as_str() {
        "rmsnorm" => bench_rmsnorm(record, &tensors, reps, warmup, threads, out)?,
        "softmax" => bench_softmax(record, &tensors, reps, warmup, out)?,
        "attention" => bench_attention(record, &tensors, reps, warmup, out)?,
        "matmul" => bench_matmul(record, &tensors, reps, warmup, out)?,
        other => bail!("no rust bench for op {other}"),
    };
    Ok(json!({
        "backend": "rust_cpu",
        "key": record.key,
        "op": record.op,
        "dtype": "fp32",
        "golden_dtype": record.dtype,
        "reps": reps,
        "warmup": warmup,
        "hardware": hardware(threads),
        "result": body,
    }))
}

fn forward_batch(
    engine: &mut Engine,
    ids: &[u32],
    batch: usize,
    seq: usize,
    logits: &mut [f32],
) -> Result<()> {
    let vocab = logits.len() / (batch * seq);
    for row in 0..batch {
        let start = row * seq * vocab;
        engine.forward(
            &ids[row * seq..(row + 1) * seq],
            &mut logits[start..start + seq * vocab],
        )?;
    }
    Ok(())
}

fn forward_bench(
    model: &Path,
    golden_dir: &Path,
    key: &str,
    reps: usize,
    warmup: usize,
    out_file: Option<PathBuf>,
) -> Result<Value> {
    let set = GoldenSet::open(golden_dir)?;
    let record = set.record(key)?;
    ensure!(
        record.op == "forward",
        "key {key} is not a forward workload"
    );
    let tensors = set.load(record)?;
    let ids = &tensors.get("ids").expect("ids").data;
    let expected = &tensors.get(&record.output).expect("logits").data;
    let dims = record.shape_map()?;
    let shape = dims
        .get("ids")
        .cloned()
        .unwrap_or_else(|| vec![1, ids.len()]);
    let (batch, seq) = (shape[0], shape[1]);
    let id_list: Vec<u32> = ids.iter().map(|value| *value as u32).collect();
    let weights = Weights::load(model)?;
    let mut engine = Engine::new(&weights)?;
    let vocab = weights.spec.vocab_size;
    let mut logits = vec![0.0f32; batch * seq * vocab];
    forward_batch(&mut engine, &id_list, batch, seq, &mut logits)?;
    let parity = parity_block(&logits, expected, record)?;
    let samples = timed(reps, warmup, || {
        forward_batch(&mut engine, &id_list, batch, seq, &mut logits).expect("forward");
    });
    let ms = median(&samples);
    if let Some(path) = out_file {
        let bytes: Vec<u8> = logits
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(path, bytes)?;
    }
    let tokens = batch * seq;
    Ok(json!({
        "backend": "rust_cpu",
        "key": record.key,
        "op": "forward",
        "dtype": "fp32",
        "reps": reps,
        "warmup": warmup,
        "param_count": weights.spec.param_count(),
        "model_tensors": weights.n_tensors(),
        "model_bytes": weights.file_bytes,
        "batch": batch,
        "sequence": seq,
        "ms_median": ms,
        "summary": summary(&samples),
        "tokens_per_sec": (tokens as f64) / (ms * 1e-3),
        "ms_per_token": ms / tokens as f64,
        "hardware": hardware(rayon::current_num_threads()),
        "parity": parity,
    }))
}

#[cfg(feature = "tokenizer")]
fn generate(args: &GenerateArgs) -> Result<Value> {
    use brainlab_tokenizer::Tokenizer;
    let weights = Weights::load(&args.model)?;
    let tokenizer = Tokenizer::load(&args.tokenizer)?;
    let ids = tokenizer.encode_ids(&args.prompt);
    let mut engine = Engine::new(&weights)?;
    let vocab = weights.spec.vocab_size;
    let mut logits = vec![0.0f32; vocab];
    let started = Instant::now();
    if ids.is_empty() {
        bail!("the prompt tokenized to nothing");
    }
    engine.prefill_last(&ids, &mut logits)?;
    let ttft_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut rng = Rng::new(args.seed);
    let sampler = Sampler {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
    };
    let mut produced: Vec<u32> = Vec::new();
    let decode_started = Instant::now();
    let mut next = pick(&logits, &sampler, &mut rng);
    while produced.len() < args.max_new {
        if next as usize == tokenizer.eos_id() {
            break;
        }
        produced.push(next as u32);
        engine.decode(next as u32, &mut logits)?;
        next = pick(&logits, &sampler, &mut rng);
    }
    let decode_ms = decode_started.elapsed().as_secs_f64() * 1000.0;
    let text = tokenizer.decode(
        &ids.iter()
            .copied()
            .chain(produced.iter().copied())
            .collect::<Vec<u32>>(),
    )?;
    Ok(json!({
        "backend": "rust_cpu",
        "text": text,
        "prompt_tokens": ids.len(),
        "generated": produced.len(),
        "ttft_ms": ttft_ms,
        "decode_tokens_per_sec": (produced.len() as f64) / (decode_ms / 1000.0).max(1e-9),
        "argmax_first": argmax(&logits),
        "kv_len": engine.kv_len(),
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "seed": args.seed,
    }))
}

struct GenerateArgs {
    model: PathBuf,
    tokenizer: PathBuf,
    prompt: String,
    max_new: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: u64,
}

fn self_test(model: &Path, prefill: usize, steps: usize) -> Result<Value> {
    let weights = Weights::load(model)?;
    let spec = weights.spec.clone();
    let vocab = spec.vocab_size;
    let mut rng = Rng::new(7);
    let ids: Vec<u32> = (0..prefill)
        .map(|_| (rng.next_u64() % vocab as u64) as u32)
        .collect();
    let mut full_engine = Engine::new(&weights)?;
    let mut full_logits = vec![0.0f32; prefill * vocab];
    let full_started = Instant::now();
    full_engine.forward(&ids, &mut full_logits)?;
    let full_forward_ms = full_started.elapsed().as_secs_f64() * 1e3;
    let tail = &full_logits[(prefill - 1) * vocab..prefill * vocab];

    let mut wide = Engine::new(&weights)?;
    let mut wide_row = vec![0.0f32; vocab];
    let wide_started = Instant::now();
    wide.prefill(&ids, &mut wide_row)?;
    let prefill_all_rows_ms = wide_started.elapsed().as_secs_f64() * 1e3;

    let mut last = Engine::new(&weights)?;
    let mut last_row = vec![0.0f32; vocab];
    let last_started = Instant::now();
    last.prefill_last(&ids, &mut last_row)?;
    let prefill_last_row_ms = last_started.elapsed().as_secs_f64() * 1e3;

    let cache_identical = {
        let (wide_k, wide_v) = wide.kv_cache();
        let (last_k, last_v) = last.kv_cache();
        wide_k == last_k && wide_v == last_v
    };
    let worst_against_forward = max_abs_error(&last_row, tail)?;
    let worst_against_wide = max_abs_error(&last_row, &wide_row)?;

    let mut worst_decode = 0.0f32;
    let mut produced = ids.clone();
    for _ in 0..steps {
        let next = argmax(&last_row) as u32;
        produced.push(next);
        last.decode(next, &mut last_row)?;
        let mut reference = Engine::new(&weights)?;
        let mut reference_logits = vec![0.0f32; produced.len() * vocab];
        reference.forward(&produced, &mut reference_logits)?;
        let begin = (produced.len() - 1) * vocab;
        let delta = max_abs_error(&last_row, &reference_logits[begin..begin + vocab])?;
        worst_decode = worst_decode.max(delta);
    }
    let scratch_floats_avoided = (prefill - 1) * vocab;
    Ok(json!({
        "backend": "rust_cpu",
        "prefill_tokens": prefill,
        "vocab_size": vocab,
        "prefill_all_rows_ms": prefill_all_rows_ms,
        "prefill_last_row_ms": prefill_last_row_ms,
        "prefill_speedup": if prefill_last_row_ms > 0.0 { prefill_all_rows_ms / prefill_last_row_ms } else { 0.0 },
        "logits_scratch_floats_avoided": scratch_floats_avoided,
        "logits_scratch_bytes_avoided": scratch_floats_avoided * 4,
        "cache_identical_to_full_path": cache_identical,
        "worst_last_row_against_full_forward": worst_against_forward,
        "worst_last_row_against_all_row_prefill": worst_against_wide,
        "full_forward_ms": full_forward_ms,
        "prefill_matches_full_forward": worst_against_forward <= 1e-2,
        "decode_steps": steps,
        "decode_budget": 1e-2,
        "decode_matches_full_forward": worst_decode <= 1e-2,
        "worst_decode_abs_error": worst_decode,
        "context_len": spec.context_len,
        "param_count": spec.param_count(),
    }))
}

fn share_table(snapshot: Vec<(&'static str, f64, u64)>, divisor: f64) -> (Vec<Value>, f64) {
    let total: f64 = snapshot.iter().map(|(_, ms, _)| ms).sum();
    let rows = snapshot
        .into_iter()
        .map(|(name, ms, calls)| {
            json!({
                "op": name,
                "ms_per_run": ms / divisor,
                "share_pct": if total > 0.0 { 100.0 * ms / total } else { 0.0 },
                "calls_per_run": calls as f64 / divisor,
            })
        })
        .collect();
    (rows, total / divisor.max(1e-9))
}

fn profile_run(
    model: &Path,
    golden_dir: &Path,
    key: &str,
    reps: usize,
    decode_steps: usize,
) -> Result<Value> {
    if !brainlab_infer::profile::enabled() {
        bail!(
            "this binary was built without the profile feature; use: cargo run --features profile"
        );
    }
    let weights = Weights::load(model)?;
    let set = GoldenSet::open(golden_dir)?;
    let record = set.record(key)?;
    ensure!(
        record.op == "forward",
        "profile needs a forward workload, {key} is {}",
        record.op
    );
    let tensors = set.load(record)?;
    let ids = tensors.get("ids").context("ids tensor")?;
    let dims = record.shape_map()?;
    let shape = dims
        .get("ids")
        .cloned()
        .unwrap_or_else(|| vec![1, ids.data.len()]);
    let (batch, seq) = (shape[0], shape[1]);
    let tokens: Vec<u32> = ids.data.iter().map(|value| *value as u32).collect();
    let vocab = weights.spec.vocab_size;
    let mut engine = Engine::new(&weights)?;
    let mut logits = vec![0.0f32; tokens.len() * vocab];
    forward_batch(&mut engine, &tokens, batch, seq, &mut logits)?;
    brainlab_infer::profile::reset();
    let started = Instant::now();
    for _ in 0..reps {
        forward_batch(&mut engine, &tokens, batch, seq, &mut logits)?;
    }
    let forward_seconds = started.elapsed().as_secs_f64().max(1e-9);
    let (forward, forward_accounted) =
        share_table(brainlab_infer::profile::snapshot(), reps as f64);
    let mut decode = Value::Null;
    if decode_steps > 0 {
        if decode_steps >= weights.spec.context_len {
            bail!(
                "decode_steps {decode_steps} leaves no room inside context_len {}",
                weights.spec.context_len
            );
        }
        let prefix = seq.min(weights.spec.context_len - decode_steps).max(1);
        let mut cached = Engine::new(&weights)?;
        let mut row = vec![0.0f32; vocab];
        cached.prefill(&tokens[..prefix], &mut row)?;
        brainlab_infer::profile::reset();
        let decode_started = Instant::now();
        for _ in 0..decode_steps {
            let next = argmax(&row) as u32;
            cached.decode(next, &mut row)?;
        }
        let decode_seconds = decode_started.elapsed().as_secs_f64().max(1e-9);
        let (decode_rows, decode_accounted) =
            share_table(brainlab_infer::profile::snapshot(), decode_steps as f64);
        decode = json!({
            "steps": decode_steps,
            "prefill_prefix": prefix,
            "ms_per_step": decode_seconds * 1000.0 / decode_steps as f64,
            "accounted_ms_per_step": decode_accounted,
            "tokens_per_sec": decode_steps as f64 / decode_seconds,
            "time_share": decode_rows,
        });
    }
    Ok(json!({
        "backend": "rust_cpu",
        "op": "profile",
        "key": record.key,
        "batch": batch,
        "sequence": seq,
        "reps": reps,
        "forward_ms_per_run": forward_seconds * 1000.0 / reps as f64,
        "forward_accounted_ms_per_run": forward_accounted,
        "threads": rayon::current_num_threads(),
        "forward_time_share": forward,
        "decode": decode,
        "note": "share_pct is of measured op time; compare forward_accounted_ms_per_run against forward_ms_per_run to see what the table does not cover",
    }))
}

fn dispatch(cli: Cli) -> Result<Value> {
    match cli.command {
        Command::List { golden_dir } => {
            let set = GoldenSet::open(&golden_dir)?;
            let keys: Vec<Value> = set
                .index
                .workloads
                .iter()
                .map(|record| json!({"key": record.key, "op": record.op, "dtype": record.dtype, "file": record.file}))
                .collect();
            Ok(json!({ "golden_dir": golden_dir, "workloads": keys }))
        }
        Command::Info { model } => {
            let weights = Weights::load(&model)?;
            Ok(json!({
                "backend": "rust_cpu",
                "spec": weights.spec,
                "param_count": weights.spec.param_count(),
                "tensors": weights.n_tensors(),
                "file_bytes": weights.file_bytes,
                "metadata": weights.metadata,
            }))
        }
        Command::Bench {
            golden_dir,
            key,
            reps,
            warmup,
            out,
        } => bench(&key, &golden_dir, reps, warmup, cli.threads, out.as_deref()),
        Command::Forward {
            model,
            golden_dir,
            key,
            out,
            reps,
            warmup,
        } => forward_bench(&model, &golden_dir, &key, reps, warmup, out),
        Command::SelfTest {
            model,
            prefill,
            steps,
        } => self_test(&model, prefill, steps),
        Command::Profile {
            model,
            golden_dir,
            key,
            reps,
            decode,
        } => profile_run(&model, &golden_dir, &key, reps, decode),
        #[cfg(feature = "tokenizer")]
        Command::Generate {
            model,
            tokenizer,
            prompt,
            max_new,
            temperature,
            top_k,
            top_p,
            seed,
        } => generate(&GenerateArgs {
            model,
            tokenizer,
            prompt,
            max_new,
            temperature,
            top_k,
            top_p,
            seed,
        }),
        #[cfg(not(feature = "tokenizer"))]
        Command::Generate { .. } => bail!("build with the tokenizer feature to generate text"),
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let json = if cli.threads > 0 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(cli.threads)
            .build()?;
        pool.install(|| dispatch(cli))?
    } else {
        dispatch(cli)?
    };
    println!("{}", serde_json::to_string_pretty(&json)?);
    Ok(())
}
