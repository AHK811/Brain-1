use anyhow::{bail, ensure, Result};

#[derive(Debug, Clone)]
pub struct RopeTable {
    cos: Vec<f32>,
    sin: Vec<f32>,
    seq: usize,
    head_dim: usize,
}

impl RopeTable {
    pub fn new(seq: usize, head_dim: usize, theta: f32) -> Result<Self> {
        ensure!(
            head_dim.is_multiple_of(2),
            "head_dim {head_dim} must be even for split-half rope"
        );
        ensure!(
            seq > 0 && head_dim > 0,
            "rope table needs positive dimensions"
        );
        let half = head_dim / 2;
        let mut cos = vec![0.0f32; seq * head_dim];
        let mut sin = vec![0.0f32; seq * head_dim];
        for index in 0..half {
            let inv_freq = 1.0 / theta.powf((2.0 * index as f32) / head_dim as f32);
            for position in 0..seq {
                let angle = position as f32 * inv_freq;
                let (c, s) = (angle.cos(), angle.sin());
                let offset = position * head_dim;
                cos[offset + index] = c;
                cos[offset + half + index] = c;
                sin[offset + index] = s;
                sin[offset + half + index] = s;
            }
        }
        Ok(Self {
            cos,
            sin,
            seq,
            head_dim,
        })
    }

    pub fn capacity(&self) -> usize {
        self.seq
    }

    pub fn apply_rows(&self, x: &mut [f32], base_position: usize) -> Result<()> {
        let d = self.head_dim;
        ensure!(
            x.len().is_multiple_of(d),
            "{} elements are not a whole number of rows of {d}",
            x.len()
        );
        let rows = x.len() / d;
        if base_position + rows > self.seq {
            bail!(
                "rope position {}+{} exceeds table capacity {}",
                base_position,
                rows,
                self.seq
            );
        }
        let half = d / 2;
        for index in 0..rows {
            let position = base_position + index;
            let offset = position * d;
            let row_offset = index * d;
            for lane in 0..half {
                let first = x[row_offset + lane];
                let second = x[row_offset + half + lane];
                let c0 = self.cos[offset + lane];
                let s0 = self.sin[offset + lane];
                let c1 = self.cos[offset + half + lane];
                let s1 = self.sin[offset + half + lane];
                x[row_offset + lane] = first * c0 - second * s0;
                x[row_offset + half + lane] = second * c1 + first * s1;
            }
        }
        Ok(())
    }
}
