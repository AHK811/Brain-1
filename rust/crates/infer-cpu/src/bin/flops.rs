use anyhow::{bail, Result};
use clap::Parser;
use serde_json::json;
use std::collections::BTreeMap;
use std::hint::black_box;
use std::thread;
use std::time::{Duration, Instant};

const LANES: usize = 8;

#[derive(Parser)]
#[command(
    name = "flops",
    about = "Measure this host's AVX2+FMA throughput ceiling for the matmul micro-kernel"
)]
struct Cli {
    #[arg(long, default_value_t = 600)]
    ms: u64,
    #[arg(long, default_value_t = 3)]
    rounds: usize,
    #[arg(long, default_value_t = 5)]
    repeats: usize,
    #[arg(long, default_value_t = 16)]
    accumulators: usize,
    #[arg(long, default_value_t = 0)]
    threads: usize,
    #[arg(long, default_value_t = false)]
    sweep: bool,
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn fma_span(accumulators: usize, iters: u64) -> f32 {
    use std::arch::x86_64::*;
    let multiplier = _mm256_set1_ps(1.0000001f32);
    let addend = _mm256_set1_ps(1e-7f32);
    let mut acc: Vec<__m256> = (0..accumulators)
        .map(|index| _mm256_set1_ps(1.0 + index as f32 * 0.001))
        .collect();
    for _ in 0..iters {
        for lane in acc.iter_mut() {
            *lane = _mm256_fmadd_ps(*lane, multiplier, addend);
        }
    }
    let mut total = _mm256_setzero_ps();
    for lane in acc.iter() {
        total = _mm256_add_ps(total, *lane);
    }
    let folded = _mm_add_ps(
        _mm256_castps256_ps128(total),
        _mm256_extractf128_ps(total, 1),
    );
    let swapped = _mm_add_ps(folded, _mm_movehl_ps(folded, folded));
    let scalar = _mm_add_ss(swapped, _mm_shuffle_ps(swapped, swapped, 0x55));
    black_box(_mm_cvtss_f32(scalar))
}

#[cfg(target_arch = "x86_64")]
fn seconds_for(accumulators: usize, iters: u64) -> f64 {
    let started = Instant::now();
    let sink = unsafe { fma_span(accumulators, iters) };
    black_box(sink);
    started.elapsed().as_secs_f64().max(1e-9)
}

#[cfg(target_arch = "x86_64")]
fn flops_of(iters: u64, accumulators: usize) -> f64 {
    iters as f64 * accumulators as f64 * (LANES * 2) as f64
}

#[cfg(target_arch = "x86_64")]
fn worker(accumulators: usize, budget: Duration, rounds: usize) -> f64 {
    let probe_iters = 20_000u64;
    let probe_seconds = seconds_for(accumulators, probe_iters);
    let wanted = budget.as_secs_f64();
    let iters = ((probe_iters as f64 * wanted / probe_seconds) as u64).max(1);
    let mut best = 0.0f64;
    for _ in 0..rounds {
        let seconds = seconds_for(accumulators, iters);
        best = best.max(flops_of(iters, accumulators) / seconds);
    }
    best
}

#[cfg(target_arch = "x86_64")]
fn measure(threads: usize, accumulators: usize, budget: Duration, rounds: usize) -> f64 {
    let handles: Vec<_> = (0..threads)
        .map(|_| thread::spawn(move || worker(accumulators, budget, rounds)))
        .collect();
    handles
        .into_iter()
        .map(|handle| handle.join().unwrap_or(0.0))
        .sum::<f64>()
        / 1e9
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[middle]
    } else {
        0.5 * (sorted[middle - 1] + sorted[middle])
    }
}

fn min_of(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::INFINITY, f64::min)
}

fn max_of(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::MIN, f64::max)
}

fn round1(value: f64) -> f64 {
    (value * 10.0).round() / 10.0
}

fn cpu_mhz() -> Option<f64> {
    let text = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    text.lines().find_map(|line| {
        let mut parts = line.split(':');
        if parts.next()?.trim() != "cpu MHz" {
            return None;
        }
        parts.next()?.trim().parse::<f64>().ok()
    })
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    #[cfg(not(target_arch = "x86_64"))]
    bail!("the FMA ceiling probe is x86-64 only");
    #[cfg(target_arch = "x86_64")]
    if !std::is_x86_feature_detected!("fma") || !std::is_x86_feature_detected!("avx2") {
        bail!("this host lacks AVX2+FMA, so the micro-kernel ceiling cannot be measured here");
    }
    let cores = std::thread::available_parallelism().map(|value| value.get())?;
    let requested = if cli.threads == 0 {
        cores
    } else {
        cli.threads.min(cores)
    };
    let budget = Duration::from_millis(cli.ms);
    let thread_counts: Vec<usize> = if cli.sweep {
        (1..=cores).collect()
    } else {
        vec![requested]
    };
    let mut samples: BTreeMap<usize, Vec<f64>> = thread_counts
        .iter()
        .copied()
        .map(|threads| (threads, Vec::new()))
        .collect();
    for _round in 0..cli.repeats {
        for threads in &thread_counts {
            #[cfg(target_arch = "x86_64")]
            let gflops = measure(*threads, cli.accumulators, budget, cli.rounds);
            #[cfg(not(target_arch = "x86_64"))]
            let gflops = 0.0f64;
            samples.get_mut(threads).expect("seeded").push(gflops);
        }
    }
    let mut median_map = serde_json::Map::new();
    let mut min_map = serde_json::Map::new();
    let mut max_map = serde_json::Map::new();
    let mut worst_spread = 0.0f64;
    for (threads, values) in &samples {
        median_map.insert(format!("threads_{threads}"), json!(round1(median(values))));
        min_map.insert(format!("threads_{threads}"), json!(round1(min_of(values))));
        max_map.insert(format!("threads_{threads}"), json!(round1(max_of(values))));
        if min_of(values) > 0.0 {
            worst_spread = worst_spread.max(max_of(values) / min_of(values));
        }
    }
    let peak = samples
        .get(&cores)
        .map(|values| median(values))
        .unwrap_or(0.0);
    let mhz = cpu_mhz();
    let one_unit = mhz.map(|value| (cores as f64 * value * 1e6 * (LANES * 2) as f64) / 1e9);
    println!(
        "{}",
        json!({
            "backend": "rust_cpu",
            "kernel": "avx2_fma_chain",
            "accumulators": cli.accumulators,
            "lanes_per_vector": LANES,
            "flops_per_fma_instruction": LANES * 2,
            "cores": cores,
            "cpu_mhz": mhz,
            "rounds": cli.rounds,
            "repeats": cli.repeats,
            "gflops_median_by_threads": median_map,
            "gflops_min_by_threads": min_map,
            "gflops_max_by_threads": max_map,
            "worst_repeat_spread_ratio": round1(worst_spread),
            "ceiling_measured_at_threads": cores,
            "gflops_peak": round1(peak),
            "theoretical_gflops_one_fma_per_cycle": one_unit,
            "theoretical_gflops_two_fma_per_cycle": one_unit.map(|value| value * 2.0),
            "fraction_of_one_fma_unit_theoretical": match one_unit {
                Some(limit) if peak > 0.0 => Some(peak / limit),
                _ => None,
            },
            "ceiling_to_compare_against": round1(peak),
            "interpretation": "compare matmul, attention and forward GFLOP/s against gflops_peak, not against a paper spec",
        })
    );
    Ok(())
}
