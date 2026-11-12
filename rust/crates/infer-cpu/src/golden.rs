use crate::model::ModelSpec;
use anyhow::{bail, ensure, Context, Result};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
pub struct WorkloadRecord {
    pub file: String,
    pub output: String,
    pub bytes: u64,
    #[serde(rename = "brainlab.op")]
    pub op: String,
    #[serde(rename = "brainlab.dtype")]
    pub dtype: String,
    #[serde(rename = "brainlab.key")]
    pub key: String,
    #[serde(rename = "brainlab.eps")]
    pub eps: String,
    #[serde(rename = "brainlab.seed")]
    pub seed: String,
    #[serde(rename = "brainlab.shapes")]
    pub shapes: String,
    #[serde(rename = "brainlab.atol")]
    pub atol: String,
    #[serde(rename = "brainlab.rtol")]
    pub rtol: String,
    #[serde(rename = "brainlab.rel_l2")]
    pub rel_l2: String,
    #[serde(rename = "brainlab.traffic_bytes")]
    pub traffic_bytes: String,
    #[serde(rename = "brainlab.flops")]
    pub flops: String,
}

impl WorkloadRecord {
    pub fn eps(&self) -> Result<f32> {
        Ok(self.eps.parse()?)
    }
    pub fn atol(&self) -> Result<f64> {
        Ok(self.atol.parse()?)
    }
    pub fn rtol(&self) -> Result<f64> {
        Ok(self.rtol.parse()?)
    }
    pub fn rel_l2_limit(&self) -> Result<f64> {
        Ok(self.rel_l2.parse()?)
    }
    pub fn traffic(&self) -> Result<u64> {
        Ok(self.traffic_bytes.parse()?)
    }
    pub fn flop_count(&self) -> Result<f64> {
        Ok(self.flops.parse()?)
    }
    pub fn shape_map(&self) -> Result<BTreeMap<String, Vec<usize>>> {
        let raw: BTreeMap<String, String> = serde_json::from_str(&self.shapes)
            .with_context(|| format!("bad shapes json for {}", self.key))?;
        let mut out = BTreeMap::new();
        for (name, encoded) in raw {
            let dims = encoded
                .split('x')
                .filter(|part| !part.is_empty())
                .map(|part| part.parse::<usize>())
                .collect::<Result<Vec<usize>, _>>()?;
            out.insert(name, dims);
        }
        Ok(out)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Producer {
    pub torch: String,
    pub numpy: String,
    pub cuda_available: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WorkloadIndex {
    pub spec: ModelSpec,
    pub param_count: usize,
    pub model_file: Option<String>,
    pub producer: Producer,
    pub workloads: Vec<WorkloadRecord>,
}

#[derive(Debug, Clone)]
pub struct Array {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Array {
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn rows_cols(&self) -> Result<(usize, usize)> {
        ensure!(
            self.shape.len() == 2,
            "{} has {} dimensions, expected 2",
            self.name,
            self.shape.len()
        );
        Ok((self.shape[0], self.shape[1]))
    }
}

pub struct GoldenSet {
    pub dir: PathBuf,
    pub index: WorkloadIndex,
}

impl GoldenSet {
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let path = dir.join("index.json");
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("run `just golden` to create {}", path.display()))?;
        let index: WorkloadIndex =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        index
            .spec
            .validate()
            .with_context(|| "spec from golden index".to_string())?;
        if index.param_count != index.spec.param_count() {
            bail!(
                "index param_count {} != spec {}",
                index.param_count,
                index.spec.param_count()
            );
        }
        Ok(Self { dir, index })
    }

    pub fn record(&self, key: &str) -> Result<&WorkloadRecord> {
        self.index
            .workloads
            .iter()
            .find(|record| record.key == key)
            .with_context(|| format!("no workload with key {key} in {}", self.dir.display()))
    }

    pub fn path(&self, record: &WorkloadRecord) -> PathBuf {
        self.dir.join(&record.file)
    }

    pub fn load(&self, record: &WorkloadRecord) -> Result<BTreeMap<String, Array>> {
        let tensors = load_arrays(self.path(record))?;
        let expected = record.shape_map()?;
        for (name, shape) in &expected {
            let array = tensors
                .get(name)
                .with_context(|| format!("{}: missing tensor {name}", record.key))?;
            if &array.shape != shape {
                bail!(
                    "{}: tensor {name} has shape {:?}, index says {shape:?}",
                    record.key,
                    array.shape
                );
            }
        }
        if let Some(array) = tensors.get(&record.output) {
            let limit = array.numel();
            if limit == 0 {
                bail!("{}: reference output is empty", record.key);
            }
        }
        Ok(tensors)
    }

    pub fn model_path(&self) -> Result<PathBuf> {
        let name = self.index.model_file.as_ref().with_context(|| {
            "no model file; regenerate with `just golden -- --model`".to_string()
        })?;
        Ok(self.dir.join(name))
    }
}

pub fn load_arrays(path: impl AsRef<Path>) -> Result<BTreeMap<String, Array>> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let handle =
        SafeTensors::deserialize(&bytes).with_context(|| format!("parsing {}", path.display()))?;
    let mut out = BTreeMap::new();
    for name in handle.names() {
        let tensor = handle.tensor(name)?;
        let shape: Vec<usize> = tensor.shape().to_vec();
        let bytes = tensor.data();
        let numel: usize = shape.iter().product::<usize>().max(1);
        let data = match tensor.dtype() {
            Dtype::F32 => {
                ensure_bytes(path, name, bytes.len(), numel, 4)?;
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|chunk| f32::from_le_bytes(*chunk))
                    .collect()
            }
            Dtype::F16 => {
                ensure_bytes(path, name, bytes.len(), numel, 2)?;
                bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|chunk| half::f16::from_le_bytes(*chunk).to_f32())
                    .collect()
            }
            Dtype::BF16 => {
                ensure_bytes(path, name, bytes.len(), numel, 2)?;
                bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|chunk| {
                        let low = chunk[0] as u32;
                        let high = chunk[1] as u32;
                        f32::from_bits((high << 24) | (low << 16))
                    })
                    .collect()
            }
            Dtype::U16 => {
                ensure_bytes(path, name, bytes.len(), numel, 2)?;
                bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|chunk| u16::from_le_bytes(*chunk) as f32)
                    .collect()
            }
            Dtype::U32 => {
                ensure_bytes(path, name, bytes.len(), numel, 4)?;
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|chunk| u32::from_le_bytes(*chunk) as f32)
                    .collect()
            }
            other => bail!(
                "{}: tensor {name} has unsupported dtype {other:?}",
                path.display()
            ),
        };
        out.insert(
            name.to_string(),
            Array {
                name: name.to_string(),
                shape,
                data,
            },
        );
    }
    Ok(out)
}

fn ensure_bytes(path: &Path, name: &str, actual: usize, numel: usize, width: usize) -> Result<()> {
    let expected = numel * width;
    if actual != expected {
        bail!(
            "{}: tensor {name} has {actual} bytes, expected {expected} for {numel} elements",
            path.display()
        );
    }
    Ok(())
}

pub fn max_abs_error(actual: &[f32], expected: &[f32]) -> Result<f32> {
    ensure!(
        actual.len() == expected.len(),
        "length mismatch {} vs {}",
        actual.len(),
        expected.len()
    );
    let mut worst = 0.0f32;
    for (a, b) in actual.iter().zip(expected) {
        let delta = (a - b).abs();
        if delta > worst || !delta.is_finite() {
            worst = delta;
        }
    }
    Ok(worst)
}

pub fn rel_l2_error(actual: &[f32], expected: &[f32]) -> Result<f64> {
    ensure!(
        actual.len() == expected.len(),
        "length mismatch {} vs {}",
        actual.len(),
        expected.len()
    );
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (a, b) in actual.iter().zip(expected) {
        let a = *a as f64;
        let b = *b as f64;
        num += (a - b) * (a - b);
        den += b * b;
    }
    Ok(if den > 0.0 {
        (num / den).sqrt()
    } else {
        num.sqrt()
    })
}
