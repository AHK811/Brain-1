use crate::kernels::softmax_row;
use anyhow::Result;

#[derive(Debug, Clone, Copy)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E3779B97F4A7C15,
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Sampler {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl Default for Sampler {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.9,
        }
    }
}

impl Sampler {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
        }
    }
}

pub fn pick(logits: &[f32], sampler: &Sampler, rng: &mut Rng) -> usize {
    if sampler.temperature <= 0.0 {
        let mut best = 0usize;
        let mut best_value = f32::NEG_INFINITY;
        for (index, value) in logits.iter().enumerate() {
            if *value > best_value {
                best_value = *value;
                best = index;
            }
        }
        return best;
    }
    let limit = sampler.top_k.min(logits.len()).max(1);
    let mut candidates: Vec<(f32, usize)> = logits
        .iter()
        .cloned()
        .map(|value| value / sampler.temperature)
        .enumerate()
        .map(|(index, value)| (value, index))
        .collect();
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(limit);
    let mut scores: Vec<f32> = candidates.iter().map(|entry| entry.0).collect();
    softmax_row(&mut scores);
    let mut cumulative = 0.0f32;
    let mut usable = scores.len();
    for (index, probability) in scores.iter().enumerate() {
        cumulative += probability;
        if cumulative >= sampler.top_p {
            usable = index + 1;
            break;
        }
    }
    let draw = rng.unit() * cumulative.min(1.0);
    let mut running = 0.0f32;
    for index in 0..usable {
        running += scores[index];
        if draw <= running {
            return candidates[index].1;
        }
    }
    candidates[usable - 1].1
}

pub fn argmax(values: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, value) in values.iter().enumerate() {
        if *value > best_value {
            best_value = *value;
            best = index;
        }
    }
    best
}

pub fn cross_entropy(logits: &[f32], targets: &[u32], vocab: usize) -> Result<f32> {
    let mut total = 0.0f64;
    for (row, target) in logits.chunks(vocab).zip(targets) {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f64 = row.iter().map(|value| ((value - max) as f64).exp()).sum();
        total += (max as f64) + sum.ln() - (row[*target as usize] - max) as f64;
    }
    Ok((total / targets.len() as f64) as f32)
}
