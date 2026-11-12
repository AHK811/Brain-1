use anyhow::{ensure, Result};
use rayon::prelude::*;

const ROW_BLOCK: usize = 4;

#[inline]
#[allow(clippy::chunks_exact_to_as_chunks)]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut chunks_a = a.chunks_exact(4);
    let mut chunks_b = b.chunks_exact(4);
    let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    while let (Some(x), Some(y)) = (chunks_a.next(), chunks_b.next()) {
        s0 += x[0] * y[0];
        s1 += x[1] * y[1];
        s2 += x[2] * y[2];
        s3 += x[3] * y[3];
    }
    let mut tail = 0.0f32;
    for (x, y) in chunks_a.remainder().iter().zip(chunks_b.remainder()) {
        tail += x * y;
    }
    (s0 + s1) + (s2 + s3) + tail
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if a.len() >= 8 {
            if std::is_x86_feature_detected!("fma") {
                return unsafe { dot_fma(a, b) };
            }
            if std::is_x86_feature_detected!("avx") {
                return unsafe { dot_avx(a, b) };
            }
        }
    }
    dot_scalar(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn dot_fma(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let (ap, bp) = (a.as_ptr(), b.as_ptr());
    let (mut acc, mut second) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let mut i = 0usize;
    while i + 16 <= n {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(i)), _mm256_loadu_ps(bp.add(i)), acc);
        second = _mm256_fmadd_ps(
            _mm256_loadu_ps(ap.add(i + 8)),
            _mm256_loadu_ps(bp.add(i + 8)),
            second,
        );
        i += 16;
    }
    while i + 8 <= n {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(ap.add(i)), _mm256_loadu_ps(bp.add(i)), acc);
        i += 8;
    }
    let mut total = reduce256(_mm256_add_ps(acc, second));
    while i < n {
        total += *a.get_unchecked(i) * *b.get_unchecked(i);
        i += 1;
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn reduce256(value: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let mut pair = _mm_add_ps(
        _mm256_castps256_ps128(value),
        _mm256_extractf128_ps(value, 1),
    );
    pair = _mm_add_ps(pair, _mm_movehl_ps(pair, pair));
    pair = _mm_add_ss(pair, _mm_shuffle_ps(pair, pair, 0x55));
    _mm_cvtss_f32(pair)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn dot_avx(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let (ap, bp) = (a.as_ptr(), b.as_ptr());
    let (mut acc, mut second) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let mut i = 0usize;
    while i + 16 <= n {
        acc = _mm256_add_ps(
            acc,
            _mm256_mul_ps(_mm256_loadu_ps(ap.add(i)), _mm256_loadu_ps(bp.add(i))),
        );
        second = _mm256_add_ps(
            second,
            _mm256_mul_ps(
                _mm256_loadu_ps(ap.add(i + 8)),
                _mm256_loadu_ps(bp.add(i + 8)),
            ),
        );
        i += 16;
    }
    while i + 8 <= n {
        acc = _mm256_add_ps(
            acc,
            _mm256_mul_ps(_mm256_loadu_ps(ap.add(i)), _mm256_loadu_ps(bp.add(i))),
        );
        i += 8;
    }
    let folded = _mm256_add_ps(acc, second);
    let mut pair = _mm_add_ps(
        _mm256_castps256_ps128(folded),
        _mm256_extractf128_ps(folded, 1),
    );
    pair = _mm_add_ps(pair, _mm_movehl_ps(pair, pair));
    pair = _mm_add_ss(pair, _mm_shuffle_ps(pair, pair, 0x55));
    let mut total = _mm_cvtss_f32(pair);
    while i < n {
        total += *a.get_unchecked(i) * *b.get_unchecked(i);
        i += 1;
    }
    total
}

pub fn transpose(source: &[f32], rows: usize, cols: usize, target: &mut [f32]) -> Result<()> {
    ensure!(
        source.len() == rows * cols,
        "transpose source {} != {rows}x{cols}",
        source.len()
    );
    ensure!(
        target.len() == rows * cols,
        "transpose target {} != {rows}x{cols}",
        target.len()
    );
    let tile = 32;
    for tile_row in (0..rows).step_by(tile) {
        for tile_col in (0..cols).step_by(tile) {
            let row_end = (tile_row + tile).min(rows);
            let col_end = (tile_col + tile).min(cols);
            for r in tile_row..row_end {
                for c in tile_col..col_end {
                    target[c * rows + r] = source[r * cols + c];
                }
            }
        }
    }
    Ok(())
}

pub fn linear(
    x: &[f32],
    w: &[f32],
    y: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
) -> Result<()> {
    ensure!(x.len() == m * k, "input {} != {m}x{k}", x.len());
    ensure!(w.len() == n * k, "weight {} != {n}x{k}", w.len());
    ensure!(y.len() == m * n, "output {} != {m}x{n}", y.len());
    if let Some(values) = bias {
        ensure!(values.len() == n, "bias {} != {n}", values.len());
    }
    let row_stride_x = ROW_BLOCK * k;
    let row_stride_y = ROW_BLOCK * n;
    x.par_chunks(row_stride_x)
        .zip(y.par_chunks_mut(row_stride_y))
        .for_each(|(block_x, block_y)| {
            let rows = block_x.len() / k;
            for column in 0..n {
                let weight_row = &w[column * k..column * k + k];
                let add = bias.map(|values| values[column]).unwrap_or(0.0);
                for slot in 0..rows {
                    let activation = &block_x[slot * k..(slot + 1) * k];
                    block_y[slot * n + column] = dot(activation, weight_row) + add;
                }
            }
        });
    Ok(())
}

pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) -> Result<()> {
    ensure!(b.len() == k * n, "b {} != {k}x{n}", b.len());
    let mut bt = vec![0.0f32; k * n];
    transpose(b, k, n, &mut bt)?;
    linear(a, &bt, c, m, k, n, None)
}

#[cfg(target_arch = "x86_64")]
fn fast_path(k: usize) -> bool {
    k >= 8 && std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
}

#[cfg(not(target_arch = "x86_64"))]
fn fast_path(_k: usize) -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum256(value: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let folded = _mm_add_ps(
        _mm256_castps256_ps128(value),
        _mm256_extractf128_ps(value, 1),
    );
    let swapped = _mm_add_ps(folded, _mm_movehl_ps(folded, folded));
    let scalar = _mm_add_ss(swapped, _mm_shuffle_ps(swapped, swapped, 0x55));
    _mm_cvtss_f32(scalar)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn block_4x2(
    x: &[f32],
    w: &[f32],
    y: &mut [f32],
    n: usize,
    k: usize,
    column: usize,
    add: [f32; 2],
) {
    use std::arch::x86_64::*;
    let (mut a00, mut a01) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let (mut a10, mut a11) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let (mut a20, mut a21) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let (mut a30, mut a31) = (_mm256_setzero_ps(), _mm256_setzero_ps());
    let w0 = w.as_ptr().add(column * k);
    let w1 = w0.add(k);
    let mut tail = k - (k % 8);
    let mut offset = 0usize;
    while offset < tail {
        let x0 = _mm256_loadu_ps(x.as_ptr().add(offset));
        let x1 = _mm256_loadu_ps(x.as_ptr().add(k + offset));
        let x2 = _mm256_loadu_ps(x.as_ptr().add(2 * k + offset));
        let x3 = _mm256_loadu_ps(x.as_ptr().add(3 * k + offset));
        let v0 = _mm256_loadu_ps(w0.add(offset));
        let v1 = _mm256_loadu_ps(w1.add(offset));
        a00 = _mm256_fmadd_ps(x0, v0, a00);
        a01 = _mm256_fmadd_ps(x0, v1, a01);
        a10 = _mm256_fmadd_ps(x1, v0, a10);
        a11 = _mm256_fmadd_ps(x1, v1, a11);
        a20 = _mm256_fmadd_ps(x2, v0, a20);
        a21 = _mm256_fmadd_ps(x2, v1, a21);
        a30 = _mm256_fmadd_ps(x3, v0, a30);
        a31 = _mm256_fmadd_ps(x3, v1, a31);
        offset += 8;
    }
    tail = k;
    let rows = [
        (hsum256(a00), hsum256(a01)),
        (hsum256(a10), hsum256(a11)),
        (hsum256(a20), hsum256(a21)),
        (hsum256(a30), hsum256(a31)),
    ];
    for (index, (left, right)) in rows.into_iter().enumerate() {
        let mut low = left;
        let mut high = right;
        for position in offset..tail {
            let first = *x.get_unchecked(index * k + position);
            low += first * *w0.add(position);
            high += first * *w1.add(position);
        }
        y[index * n + column] = low + add[0];
        y[index * n + column + 1] = high + add[1];
    }
}

pub fn linear_tiled(
    x: &[f32],
    w: &[f32],
    y: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
) -> Result<()> {
    ensure!(x.len() == m * k, "input {} != {m}x{k}", x.len());
    ensure!(w.len() == n * k, "weight {} != {n}x{k}", w.len());
    ensure!(y.len() == m * n, "output {} != {m}x{n}", y.len());
    if let Some(values) = bias {
        ensure!(values.len() == n, "bias {} != {n}", values.len());
    }
    if m < 4 || !fast_path(k) {
        return linear(x, w, y, m, k, n, bias);
    }
    let blocks = m / 4;
    y.par_chunks_mut(4 * n)
        .zip(0..blocks)
        .for_each(|(out, block)| {
            let row0 = block * 4;
            let rows = &x[row0 * k..row0 * k + 4 * k];
            let mut column = 0usize;
            while column + 2 <= n {
                let add = [
                    bias.map(|values| values[column]).unwrap_or(0.0),
                    bias.map(|values| values[column + 1]).unwrap_or(0.0),
                ];
                unsafe { block_4x2(rows, w, out, n, k, column, add) };
                column += 2;
            }
            if column < n {
                let weight = &w[column * k..column * k + k];
                let add = bias.map(|values| values[column]).unwrap_or(0.0);
                for index in 0..4 {
                    out[index * n + column] = dot(&rows[index * k..index * k + k], weight) + add;
                }
            }
        });
    for row in (m / 4) * 4..m {
        for column in 0..n {
            let weight = &w[column * k..column * k + k];
            let add = bias.map(|values| values[column]).unwrap_or(0.0);
            y[row * n + column] = dot(&x[row * k..row * k + k], weight) + add;
        }
    }
    Ok(())
}
