use brainlab_infer::attention::attention_causal;
use brainlab_infer::golden::{max_abs_error, rel_l2_error, Array, GoldenSet, WorkloadRecord};
use brainlab_infer::kernels::{rmsnorm, rmsnorm_par, softmax_rows, sumsq};
use brainlab_infer::matmul::{linear, linear_tiled, matmul, transpose};
use brainlab_infer::model::{Engine, Weights};
use brainlab_infer::rope::RopeTable;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("crate must sit at rust/crates/<name>")
        .to_path_buf()
}

fn golden_set() -> Option<GoldenSet> {
    let dir = repo_root().join("tests").join("golden");
    match GoldenSet::open(&dir) {
        Ok(set) => {
            let missing: Vec<String> = set
                .index
                .workloads
                .iter()
                .filter(|record| !set.path(record).is_file())
                .map(|record| record.key.clone())
                .collect();
            if missing.is_empty() {
                Some(set)
            } else {
                eprintln!(
                    "SKIP: {} golden files absent under {} (first: {}); run `just golden`",
                    missing.len(),
                    dir.display(),
                    missing[0]
                );
                None
            }
        }
        Err(error) => {
            eprintln!("SKIP: golden tensors unavailable ({error})");
            None
        }
    }
}

fn array<'a>(tensors: &'a BTreeMap<String, Array>, name: &str) -> &'a Array {
    tensors
        .get(name)
        .unwrap_or_else(|| panic!("golden file is missing tensor {name}"))
}

fn assert_within(record: &WorkloadRecord, actual: &[f32], expected: &[f32]) {
    let atol = record.atol().unwrap();
    let rtol = record.rtol().unwrap();
    let limit = record.rel_l2_limit().unwrap();
    let max_abs = max_abs_error(actual, expected).unwrap();
    let rel = rel_l2_error(actual, expected).unwrap();
    let worst_element = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs() as f64 - (atol + rtol * b.abs() as f64))
        .fold(0.0f64, f64::max);
    assert!(
        worst_element <= 0.0 && rel <= limit,
        "{}: max_abs={max_abs:.3e} worst_element={worst_element:.3e} rel_l2={rel:.3e} (limits atol={atol:.1e} rtol={rtol:.1e} rel_l2={limit:.1e})",
        record.key
    );
}

fn iter<'a>(set: &'a GoldenSet, op: &str) -> Vec<(&'a WorkloadRecord, BTreeMap<String, Array>)> {
    set.index
        .workloads
        .iter()
        .filter(|record| record.op == op)
        .map(|record| {
            let tensors = set
                .load(record)
                .unwrap_or_else(|error| panic!("{}: {error}", record.key));
            (record, tensors)
        })
        .collect()
}

#[test]
fn rmsnorm_matches_golden() {
    let Some(set) = golden_set() else { return };
    let cases = iter(&set, "rmsnorm");
    assert!(
        !cases.is_empty(),
        "no rmsnorm workloads in the golden index"
    );
    for (record, tensors) in cases {
        let x = array(&tensors, "x");
        let w = array(&tensors, "w");
        let expected = array(&tensors, &record.output);
        let width = w.shape[0];
        let mut out = vec![0.0f32; x.data.len()];
        rmsnorm(&x.data, &w.data, &mut out, width, record.eps().unwrap()).unwrap();
        assert_within(record, &out, &expected.data);
        let mut parallel = vec![0.0f32; x.data.len()];
        rmsnorm_par(
            &x.data,
            &w.data,
            &mut parallel,
            width,
            record.eps().unwrap(),
        )
        .unwrap();
        assert_within(record, &parallel, &expected.data);
        for (a, b) in out.iter().zip(&parallel) {
            assert!(
                (a - b).abs() <= 1e-6 + 1e-6 * b.abs(),
                "{}: sequential and parallel differ",
                record.key
            );
        }
    }
}

#[test]
fn softmax_matches_golden() {
    let Some(set) = golden_set() else { return };
    for (record, tensors) in iter(&set, "softmax") {
        let x = array(&tensors, "x");
        let expected = array(&tensors, &record.output);
        let (rows, cols) = (x.shape[0], x.shape[1]);
        let mut out = vec![0.0f32; x.data.len()];
        softmax_rows(&x.data, &mut out, cols).unwrap();
        assert_within(record, &out, &expected.data);
        assert_eq!(out.len(), rows * cols);
    }
}

#[test]
fn attention_matches_golden() {
    let Some(set) = golden_set() else { return };
    for (record, tensors) in iter(&set, "attention") {
        let q = array(&tensors, "q");
        let k = array(&tensors, "k");
        let v = array(&tensors, "v");
        let expected = array(&tensors, &record.output);
        let (batch, heads, seq, head_dim) = (q.shape[0], q.shape[1], q.shape[2], q.shape[3]);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![0.0f32; q.data.len()];
        attention_causal(
            &q.data, &k.data, &v.data, &mut out, batch, heads, seq, head_dim, scale,
        )
        .unwrap();
        assert_within(record, &out, &expected.data);
    }
}

#[test]
fn matmul_matches_golden() {
    let Some(set) = golden_set() else { return };
    for (record, tensors) in iter(&set, "matmul") {
        let a = array(&tensors, "a");
        let b = array(&tensors, "b");
        let expected = array(&tensors, &record.output);
        let (m, k) = (a.shape[0], a.shape[1]);
        let (k2, n) = (b.shape[0], b.shape[1]);
        assert_eq!(k, k2);
        let mut c = vec![0.0f32; m * n];
        matmul(&a.data, &b.data, &mut c, m, k, n).unwrap();
        assert_within(record, &c, &expected.data);
    }
}

#[test]
fn forward_matches_golden_when_model_present() {
    let Some(set) = golden_set() else { return };
    let model_path = match set.model_path() {
        Ok(path) => path,
        Err(error) => {
            eprintln!("SKIP forward: {error}");
            return;
        }
    };
    let weights = match Weights::load(&model_path) {
        Ok(weights) => weights,
        Err(error) => {
            eprintln!("SKIP forward: {error}");
            return;
        }
    };
    let cases = iter(&set, "forward");
    if cases.is_empty() {
        eprintln!("SKIP forward: no forward workloads in the golden index");
        return;
    }
    for (record, tensors) in cases {
        let ids = array(&tensors, "ids");
        let expected = array(&tensors, &record.output);
        let tokens: Vec<u32> = ids.data.iter().map(|value| *value as u32).collect();
        let (batch, seq) = (ids.shape[0], ids.shape[1]);
        let vocab = weights.spec.vocab_size;
        let mut got = Vec::with_capacity(expected.data.len());
        for row in 0..batch {
            let context = &tokens[row * seq..(row + 1) * seq];
            let mut engine = Engine::new(&weights).unwrap();
            let mut logits = vec![0.0f32; seq * vocab];
            engine.forward(context, &mut logits).unwrap();
            got.extend_from_slice(&logits);
        }
        assert_within(record, &got, &expected.data);
    }
}

#[test]
fn cached_decode_matches_full_forward() {
    let Some(set) = golden_set() else { return };
    let model_path = match set.model_path() {
        Ok(path) => path,
        Err(error) => {
            eprintln!("SKIP cached decode: {error}");
            return;
        }
    };
    let weights = match Weights::load(&model_path) {
        Ok(weights) => weights,
        Err(error) => {
            eprintln!("SKIP cached decode: {error}");
            return;
        }
    };
    let spec = weights.spec.clone();
    let mut rng = 7u64;
    let next = |state: &mut u64| {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    };
    let prompt: Vec<u32> = (0..37)
        .map(|_| (next(&mut rng) % spec.vocab_size as u64) as u32)
        .collect();
    let mut cached = Engine::new(&weights).unwrap();
    let mut row = vec![0.0f32; spec.vocab_size];
    cached.prefill(&prompt, &mut row).unwrap();
    let mut full = Engine::new(&weights).unwrap();
    let mut reference = vec![0.0f32; prompt.len() * spec.vocab_size];
    full.forward(&prompt, &mut reference).unwrap();
    let begin = (prompt.len() - 1) * spec.vocab_size;
    let prefill_worst = max_abs_error(&row, &reference[begin..begin + spec.vocab_size]).unwrap();
    assert!(
        prefill_worst <= 1e-3,
        "prefill last-row logits differ from a full forward by {prefill_worst:.3e}"
    );
    let mut produced = prompt.clone();
    for step in 0..5 {
        let token = (next(&mut rng) % spec.vocab_size as u64) as u32;
        produced.push(token);
        cached.decode(token, &mut row).unwrap();
        let mut fresh = Engine::new(&weights).unwrap();
        let mut all = vec![0.0f32; produced.len() * spec.vocab_size];
        fresh.forward(&produced, &mut all).unwrap();
        let start = (produced.len() - 1) * spec.vocab_size;
        let worst = max_abs_error(&row, &all[start..start + spec.vocab_size]).unwrap();
        assert!(
            worst <= 1e-2,
            "step {step}: cached decode differs from a full forward by {worst:.3e}; the kv cache layout is wrong"
        );
        assert_eq!(
            cached.kv_len(),
            produced.len(),
            "cache length must advance one row per token"
        );
    }
}

#[test]
fn transpose_roundtrips() {
    let source: Vec<f32> = (0..24).map(|value| value as f32).collect();
    let mut transposed = vec![0.0f32; 24];
    transpose(&source, 4, 6, &mut transposed).unwrap();
    let mut back = vec![0.0f32; 24];
    transpose(&transposed, 6, 4, &mut back).unwrap();
    assert_eq!(source, back);
    assert_eq!(transposed[0], 0.0);
    assert_eq!(transposed[1], 6.0);
    assert_eq!(transposed[3 * 4 + 3], 21.0);
}

#[test]
fn sumsq_matches_naive() {
    for width in [1usize, 7, 8, 31, 64, 384, 512, 1023] {
        let values: Vec<f32> = (0..width)
            .map(|index| (index as f32 + 1.0) * 0.25 - 3.0)
            .collect();
        let naive: f64 = values
            .iter()
            .map(|value| (*value as f64) * (*value as f64))
            .sum();
        let fast = sumsq(&values) as f64;
        assert!(
            (fast - naive).abs() <= 1e-4 * naive.abs().max(1.0),
            "width {width}: {fast} vs {naive}"
        );
    }
}

#[test]
fn linear_equals_matmul() {
    let rows = 5;
    let inner = 48;
    let out = 33;
    let x: Vec<f32> = (0..rows * inner)
        .map(|value| ((value % 17) as f32) * 0.1 - 0.8)
        .collect();
    let w: Vec<f32> = (0..out * inner)
        .map(|value| ((value % 23) as f32) * 0.05 - 0.5)
        .collect();
    let mut y = vec![0.0f32; rows * out];
    linear(&x, &w, &mut y, rows, inner, out, None).unwrap();
    let mut b_transposed = vec![0.0f32; inner * out];
    transpose(&w, out, inner, &mut b_transposed).unwrap();
    let mut reference = vec![0.0f32; rows * out];
    matmul(&x, &b_transposed, &mut reference, rows, inner, out).unwrap();
    for (a, b) in y.iter().zip(&reference) {
        assert!((a - b).abs() <= 1e-5 + 1e-5 * b.abs());
    }
}

#[test]
fn rope_matches_closed_form() {
    let head_dim = 8;
    let half = head_dim / 2;
    let theta = 10000.0f32;
    let table = RopeTable::new(4, head_dim, theta).unwrap();
    let row: Vec<f32> = (0..head_dim).map(|index| index as f32 + 1.0).collect();
    let mut applied = row.clone();
    table.apply_rows(&mut applied, 1).unwrap();
    for lane in 0..half {
        let inv_freq = theta.powf(-2.0 * lane as f32 / head_dim as f32);
        let angle = inv_freq;
        let first = row[lane];
        let second = row[lane + half];
        let expected_first = first * angle.cos() - second * angle.sin();
        let expected_second = second * angle.cos() + first * angle.sin();
        assert!(
            (applied[lane] - expected_first).abs() < 1e-6,
            "lane {lane}: {} vs {expected_first}",
            applied[lane]
        );
        assert!(
            (applied[lane + half] - expected_second).abs() < 1e-6,
            "lane {lane}+half: {} vs {expected_second}",
            applied[lane + half]
        );
    }
    let mut rows = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    ];
    table.apply_rows(&mut rows, 0).unwrap();
    assert!(
        (rows[0] - row[0]).abs() < 1e-6,
        "rotation at position 0 is the identity"
    );
    assert!(
        (rows[half] - row[half]).abs() < 1e-6,
        "rotation at position 0 is the identity"
    );
    let angle_lane0_position1 = 1.0f32;
    let expected = row[0] * angle_lane0_position1.cos() - row[half] * angle_lane0_position1.sin();
    assert!(
        (rows[head_dim] - expected).abs() < 1e-6,
        "row at position 1: got {} expected {expected}",
        rows[head_dim]
    );
}

fn naive_linear(x: &[f32], w: &[f32], y: &mut [f32], m: usize, k: usize, n: usize) {
    for row in 0..m {
        for column in 0..n {
            let mut sum = 0.0f64;
            for inner in 0..k {
                sum += f64::from(x[row * k + inner]) * f64::from(w[column * k + inner]);
            }
            y[row * n + column] = sum as f32;
        }
    }
}

#[test]
fn tiled_linear_matches_dot_linear_on_odd_shapes() {
    for &(m, k, n) in &[
        (1usize, 512, 16384),
        (4, 512, 512),
        (5, 512, 1536),
        (1024, 512, 1536),
        (7, 13, 9),
        (33, 8, 3),
        (16, 17, 16),
        (128, 1344, 512),
    ] {
        let x: Vec<f32> = (0..m * k)
            .map(|value| ((value % 31) as f32) * 0.05 - 0.7)
            .collect();
        let w: Vec<f32> = (0..n * k)
            .map(|value| ((value % 23) as f32) * 0.04 - 0.45)
            .collect();
        let bias: Vec<f32> = (0..n).map(|value| (value as f32) * 0.01 - 0.2).collect();
        let mut expected = vec![0.0f32; m * n];
        naive_linear(&x, &w, &mut expected, m, k, n);
        for (offset, value) in expected.iter_mut().enumerate() {
            *value += bias[offset % n];
        }
        let mut by_dot = vec![0.0f32; m * n];
        let mut by_tile = vec![0.0f32; m * n];
        linear(&x, &w, &mut by_dot, m, k, n, Some(&bias)).unwrap();
        linear_tiled(&x, &w, &mut by_tile, m, k, n, Some(&bias)).unwrap();
        for (row, (value, want)) in by_tile.chunks(n).zip(expected.chunks(n)).enumerate() {
            for (column, (got, want)) in value.iter().zip(want).enumerate() {
                let budget = 1e-4 + 1e-4 * want.abs();
                assert!(
                    (got - want).abs() <= budget,
                    "{m}x{k}x{n} at ({row},{column}): tiled {got} vs float64 {want}"
                );
                let tight = 1e-5 + 1e-5 * want.abs();
                assert!(
                    (by_dot[row * n + column] - got).abs() <= budget.max(tight) * 4.0,
                    "{m}x{k}x{n} at ({row},{column}): tiled {got} disagrees with dot kernel {}",
                    by_dot[row * n + column]
                );
            }
        }
    }
}

#[test]
fn prefill_last_row_matches_the_all_row_path() {
    let Some(set) = golden_set() else { return };
    let model_path = match set.model_path() {
        Ok(path) => path,
        Err(error) => {
            eprintln!("SKIP prefill_last: {error}");
            return;
        }
    };
    let weights = match Weights::load(&model_path) {
        Ok(weights) => weights,
        Err(error) => {
            eprintln!("SKIP prefill_last: {error}");
            return;
        }
    };
    let spec = weights.spec.clone();
    let vocab = spec.vocab_size;
    let mut state = 11u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % vocab as u64) as u32
    };
    for seq in [1usize, 7, 64, 257] {
        let ids: Vec<u32> = (0..seq).map(|_| next()).collect();
        let mut wide = Engine::new(&weights).unwrap();
        let mut wide_row = vec![0.0f32; vocab];
        wide.prefill(&ids, &mut wide_row).unwrap();
        let mut last = Engine::new(&weights).unwrap();
        let mut last_row = vec![0.0f32; vocab];
        last.prefill_last(&ids, &mut last_row).unwrap();
        let (wide_k, wide_v) = wide.kv_cache();
        let (last_k, last_v) = last.kv_cache();
        assert_eq!(wide_k, last_k, "seq {seq}: key cache differs");
        assert_eq!(wide_v, last_v, "seq {seq}: value cache differs");
        assert_eq!(
            wide.kv_len(),
            last.kv_len(),
            "seq {seq}: cache length differs"
        );
        assert_eq!(
            last.kv_len(),
            seq,
            "seq {seq}: cache must hold every prompt row"
        );
        let worst = max_abs_error(&last_row, &wide_row).unwrap();
        assert!(
            worst <= 1e-4,
            "seq {seq}: last-row logits differ from the all-row path by {worst:.3e}"
        );
        let mut full = Engine::new(&weights).unwrap();
        let mut all = vec![0.0f32; seq * vocab];
        full.forward(&ids, &mut all).unwrap();
        let tail = &all[(seq - 1) * vocab..seq * vocab];
        let against_forward = max_abs_error(&last_row, tail).unwrap();
        assert!(
            against_forward <= 1e-4,
            "seq {seq}: last-row logits differ from a full forward by {against_forward:.3e}"
        );
        let argmax_row = |values: &[f32]| {
            values.iter().enumerate().fold(
                0usize,
                |best, (index, value)| {
                    if *value > values[best] {
                        index
                    } else {
                        best
                    }
                },
            )
        };
        assert_eq!(
            argmax_row(&last_row),
            argmax_row(tail),
            "seq {seq}: the picked token changed between paths"
        );
    }
}
