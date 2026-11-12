use anyhow::{bail, ensure, Result};
use rayon::prelude::*;

#[inline]
pub fn sumsq(x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx") {
            return unsafe { sumsq_avx(x) };
        }
    }
    sumsq_scalar(x)
}

#[allow(clippy::chunks_exact_to_as_chunks)]
fn sumsq_scalar(x: &[f32]) -> f32 {
    let mut chunks = x.chunks_exact(4);
    let (mut a, mut b, mut c, mut d) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for chunk in &mut chunks {
        a += chunk[0] * chunk[0];
        b += chunk[1] * chunk[1];
        c += chunk[2] * chunk[2];
        d += chunk[3] * chunk[3];
    }
    let mut tail = 0.0f32;
    for value in chunks.remainder() {
        tail += value * value;
    }
    (a + b) + (c + d) + tail
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn sumsq_avx(x: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = x.len();
    let base = x.as_ptr();
    let (mut a, mut b, mut c, mut d) = (
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
    );
    let mut i = 0usize;
    while i + 32 <= n {
        for lane in 0..4 {
            let v = _mm256_loadu_ps(base.add(i + lane * 8));
            let squared = _mm256_mul_ps(v, v);
            match lane {
                0 => a = _mm256_add_ps(a, squared),
                1 => b = _mm256_add_ps(b, squared),
                2 => c = _mm256_add_ps(c, squared),
                _ => d = _mm256_add_ps(d, squared),
            }
        }
        i += 32;
    }
    while i + 8 <= n {
        let v = _mm256_loadu_ps(base.add(i));
        a = _mm256_add_ps(a, _mm256_mul_ps(v, v));
        i += 8;
    }
    let folded = _mm256_add_ps(_mm256_add_ps(a, b), _mm256_add_ps(c, d));
    let mut pair = _mm_add_ps(
        _mm256_castps256_ps128(folded),
        _mm256_extractf128_ps(folded, 1),
    );
    pair = _mm_add_ps(pair, _mm_movehl_ps(pair, pair));
    pair = _mm_add_ss(pair, _mm_shuffle_ps(pair, pair, 0x55));
    let mut total = _mm_cvtss_f32(pair);
    while i < n {
        let value = *x.get_unchecked(i);
        total += value * value;
        i += 1;
    }
    total
}

#[inline]
pub fn mean_sq_inverse(x: &[f32], eps: f32) -> f32 {
    1.0 / (sumsq(x) / x.len() as f32 + eps).sqrt()
}

fn scale_row_scalar(x: &[f32], w: &[f32], out: &mut [f32], inv: f32) {
    for index in 0..x.len() {
        out[index] = x[index] * inv * w[index];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn scale_row_avx(x: &[f32], w: &[f32], out: &mut [f32], inv: f32) {
    use std::arch::x86_64::*;
    let scale = _mm256_set1_ps(inv);
    let n = x.len();
    let (xp, wp, op) = (x.as_ptr(), w.as_ptr(), out.as_mut_ptr());
    let mut i = 0usize;
    while i + 16 <= n {
        for lane in 0..2 {
            let offset = i + lane * 8;
            let xv = _mm256_mul_ps(_mm256_loadu_ps(xp.add(offset)), scale);
            let result = _mm256_mul_ps(xv, _mm256_loadu_ps(wp.add(offset)));
            _mm256_storeu_ps(op.add(offset), result);
        }
        i += 16;
    }
    while i + 8 <= n {
        let xv = _mm256_mul_ps(_mm256_loadu_ps(xp.add(i)), scale);
        _mm256_storeu_ps(op.add(i), _mm256_mul_ps(xv, _mm256_loadu_ps(wp.add(i))));
        i += 8;
    }
    while i < n {
        *out.get_unchecked_mut(i) = *x.get_unchecked(i) * inv * *w.get_unchecked(i);
        i += 1;
    }
}

#[inline]
fn scale_row(x: &[f32], w: &[f32], out: &mut [f32], inv: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx") {
            unsafe { scale_row_avx(x, w, out, inv) };
            return;
        }
    }
    scale_row_scalar(x, w, out, inv);
}

pub fn rmsnorm_row(x: &[f32], w: &[f32], out: &mut [f32], eps: f32) {
    let inv = mean_sq_inverse(x, eps);
    scale_row(x, w, out, inv);
}

fn check_rows(x: &[f32], w: &[f32], out: &[f32], d: usize) -> Result<usize> {
    ensure!(d > 0, "width must be positive");
    ensure!(w.len() == d, "weight length {} != width {}", w.len(), d);
    ensure!(
        x.len().is_multiple_of(d),
        "input length {} not divisible by width {}",
        x.len(),
        d
    );
    ensure!(
        out.len() == x.len(),
        "output length {} != input length {}",
        out.len(),
        x.len()
    );
    Ok(x.len() / d)
}

pub fn rmsnorm(x: &[f32], w: &[f32], out: &mut [f32], d: usize, eps: f32) -> Result<usize> {
    let rows = check_rows(x, w, out, d)?;
    for row in 0..rows {
        let start = row * d;
        let end = start + d;
        rmsnorm_row(&x[start..end], w, &mut out[start..end], eps);
    }
    Ok(rows)
}

pub fn rmsnorm_par(x: &[f32], w: &[f32], out: &mut [f32], d: usize, eps: f32) -> Result<usize> {
    let rows = check_rows(x, w, out, d)?;
    if rows < 8 {
        return rmsnorm(x, w, out, d, eps);
    }
    out.par_chunks_mut(d)
        .zip(x.par_chunks(d))
        .for_each(|(out_row, in_row)| rmsnorm_row(in_row, w, out_row, eps));
    Ok(rows)
}

pub fn rmsnorm_with_threads(
    x: &[f32],
    w: &[f32],
    out: &mut [f32],
    d: usize,
    eps: f32,
    threads: usize,
) -> Result<usize> {
    if threads <= 1 {
        return rmsnorm(x, w, out, d, eps);
    }
    if threads == rayon::current_num_threads() {
        return rmsnorm_par(x, w, out, d, eps);
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|err| anyhow::anyhow!("thread pool: {err}"))?;
    pool.install(|| rmsnorm_par(x, w, out, d, eps))
}

pub fn softmax_rows(source: &[f32], target: &mut [f32], cols: usize) -> Result<()> {
    if cols == 0 {
        bail!("softmax needs a positive row width");
    }
    ensure!(
        source.len() == target.len(),
        "softmax lengths differ {} vs {}",
        source.len(),
        target.len()
    );
    ensure!(
        source.len().is_multiple_of(cols),
        "length {} not divisible by {}",
        source.len(),
        cols
    );
    for (row, out) in source.chunks(cols).zip(target.chunks_mut(cols)) {
        out.copy_from_slice(row);
        softmax_row(out);
    }
    Ok(())
}

pub fn softmax_row(row: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for value in row.iter() {
        if *value > max {
            max = *value;
        }
    }
    let mut sum = 0.0f32;
    for value in row.iter_mut() {
        let exp = (*value - max).exp();
        *value = exp;
        sum += exp;
    }
    let inverse = 1.0 / sum;
    for value in row.iter_mut() {
        *value *= inverse;
    }
}
