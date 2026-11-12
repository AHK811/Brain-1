use crate::matmul::dot;
use anyhow::{ensure, Result};
use rayon::prelude::*;

#[allow(clippy::too_many_arguments)]
pub fn attention_causal(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    batch: usize,
    heads: usize,
    seq: usize,
    head_dim: usize,
    scale: f32,
) -> Result<()> {
    let total = batch * heads * seq * head_dim;
    ensure!(
        q.len() == total && k.len() == total && v.len() == total && out.len() == total,
        "attention buffers must all be {total} floats"
    );
    ensure!(
        seq > 0 && head_dim > 0,
        "attention needs a positive sequence and head width"
    );
    let plane_stride = head_dim * seq;
    out.par_chunks_mut(plane_stride)
        .zip((0..batch * heads).into_par_iter())
        .for_each(|(out_plane, plane)| {
            let offset = plane * plane_stride;
            let q_plane = &q[offset..offset + plane_stride];
            let k_plane = &k[offset..offset + plane_stride];
            let v_plane = &v[offset..offset + plane_stride];
            let mut acc = vec![0.0f32; head_dim];
            for target in 0..seq {
                let query = &q_plane[target * head_dim..(target + 1) * head_dim];
                for slot in acc.iter_mut() {
                    *slot = 0.0;
                }
                let mut running_max = f32::NEG_INFINITY;
                let mut running_sum = 0.0f32;
                for source in 0..=target {
                    let score =
                        dot(query, &k_plane[source * head_dim..(source + 1) * head_dim]) * scale;
                    let values = &v_plane[source * head_dim..(source + 1) * head_dim];
                    if score > running_max {
                        let correction = (running_max - score).exp();
                        for (accumulator, value) in acc.iter_mut().zip(values) {
                            *accumulator = *accumulator * correction + *value;
                        }
                        running_sum = running_sum * correction + 1.0;
                        running_max = score;
                    } else {
                        let weight = (score - running_max).exp();
                        running_sum += weight;
                        for (accumulator, value) in acc.iter_mut().zip(values) {
                            *accumulator += weight * *value;
                        }
                    }
                }
                let inverse = 1.0 / running_sum;
                let destination = &mut out_plane[target * head_dim..(target + 1) * head_dim];
                for (slot, value) in destination.iter_mut().zip(acc.iter()) {
                    *slot = *value * inverse;
                }
            }
        });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn attention_query_row(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    out: &mut [f32],
    len: usize,
    head_dim: usize,
    scale: f32,
) -> Result<()> {
    ensure!(
        query.len() == head_dim && out.len() == head_dim,
        "one query row of {head_dim}"
    );
    ensure!(
        keys.len() >= len * head_dim && values.len() >= len * head_dim,
        "cache shorter than {len} rows"
    );
    let mut running_max = f32::NEG_INFINITY;
    let mut running_sum = 0.0f32;
    for slot in out.iter_mut() {
        *slot = 0.0;
    }
    for source in 0..len {
        let key = &keys[source * head_dim..(source + 1) * head_dim];
        let score = dot(query, key) * scale;
        let value_row = &values[source * head_dim..(source + 1) * head_dim];
        if score > running_max {
            let correction = (running_max - score).exp();
            for (accumulator, value) in out.iter_mut().zip(value_row) {
                *accumulator = *accumulator * correction + *value;
            }
            running_sum = running_sum * correction + 1.0;
            running_max = score;
        } else {
            let weight = (score - running_max).exp();
            running_sum += weight;
            for (accumulator, value) in out.iter_mut().zip(value_row) {
                *accumulator += weight * *value;
            }
        }
    }
    let inverse = 1.0 / running_sum;
    for slot in out.iter_mut() {
        *slot *= inverse;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn attention_cache(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    out: &mut [f32],
    heads: usize,
    len: usize,
    capacity: usize,
    head_dim: usize,
    scale: f32,
) -> Result<()> {
    let plane_stride = capacity * head_dim;
    ensure!(
        query.len() == heads * head_dim,
        "query must be {heads} rows of {head_dim}"
    );
    ensure!(
        keys.len() == heads * plane_stride && values.len() == heads * plane_stride,
        "cache must be {heads} planes of {capacity} rows"
    );
    ensure!(
        out.len() == heads * head_dim,
        "out must be {heads} rows of {head_dim}"
    );
    ensure!(
        len <= capacity && len > 0,
        "cache length {len} outside 1..{capacity}"
    );
    for head in 0..heads {
        let cache_offset = head * plane_stride;
        attention_query_row(
            &query[head * head_dim..(head + 1) * head_dim],
            &keys[cache_offset..cache_offset + len * head_dim],
            &values[cache_offset..cache_offset + len * head_dim],
            &mut out[head * head_dim..(head + 1) * head_dim],
            len,
            head_dim,
            scale,
        )?;
    }
    Ok(())
}
