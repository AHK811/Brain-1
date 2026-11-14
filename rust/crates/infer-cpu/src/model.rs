use crate::attention::{attention_cache_gqa, attention_causal_gqa};
use crate::golden::load_arrays;
use crate::kernels::{rmsnorm_row, softmax_row};
use crate::matmul::linear_tiled;
use crate::profile;
use crate::rope::RopeTable;
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    pub name: String,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub context_len: usize,
    pub mlp_hidden: usize,
    pub n_experts: usize,
    pub experts_per_token: usize,
    pub router_aux_loss_coef: f32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub dropout: f32,
    pub tie_embeddings: bool,
    pub qkv_fused: bool,
    pub init_std: f32,
    pub residual_scale_layers: usize,
    pub param_budget: usize,
    pub param_tolerance: f32,
}

pub const SPEC_VERSION: &str = "2";

impl ModelSpec {
    pub fn qkv_out(&self) -> usize {
        (self.n_heads + 2 * self.n_kv_heads) * self.head_dim
    }

    pub fn kv_width(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    fn checked_param_count(&self) -> Option<usize> {
        let d = self.d_model;
        let qkv = self.n_heads.checked_add(self.n_kv_heads.checked_mul(2)?)?
            .checked_mul(self.head_dim)?;
        let embedding = self.vocab_size.checked_mul(d)?
            .checked_mul(if self.tie_embeddings { 1 } else { 2 })?;
        let experts = self.n_experts.checked_mul(3)?.checked_mul(self.mlp_hidden)?.checked_mul(d)?;
        let router = if self.n_experts > 1 { self.n_experts.checked_mul(d)? } else { 0 };
        let per_layer = qkv.checked_mul(d)?.checked_add(d.checked_mul(d)?)?
            .checked_add(d.checked_mul(2)?)?.checked_add(experts)?.checked_add(router)?;
        embedding.checked_add(self.n_layers.checked_mul(per_layer)?)?.checked_add(d)
    }

    pub fn param_count(&self) -> usize {
        self.checked_param_count().expect("model parameter count overflow")
    }

    pub fn attn_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.d_model > 0 && self.n_layers > 0 && self.n_heads > 0 && self.n_kv_heads > 0
                && self.head_dim > 0 && self.vocab_size > 0 && self.context_len > 0
                && self.mlp_hidden > 0 && self.n_experts > 0,
            "dimensions must be positive"
        );
        ensure!(
            self.n_heads.checked_mul(self.head_dim) == Some(self.d_model),
            "n_heads*head_dim must equal d_model"
        );
        ensure!(self.n_heads.is_multiple_of(self.n_kv_heads), "n_heads must be divisible by n_kv_heads");
        ensure!(self.head_dim.is_multiple_of(2), "head_dim must be even for split-half rope");
        ensure!(
            self.experts_per_token > 0 && self.experts_per_token <= self.n_experts,
            "experts_per_token must be in 1..=n_experts"
        );
        ensure!(
            self.router_aux_loss_coef.is_finite() && self.router_aux_loss_coef >= 0.0,
            "router_aux_loss_coef must be finite and nonnegative"
        );
        ensure!(self.qkv_fused, "only fused qkv is supported by spec/weights_format.md");
        ensure!(self.vocab_size <= 65536, "vocab must fit uint16 shards");
        ensure!(self.rms_norm_eps.is_finite() && self.rms_norm_eps > 0.0, "rms_norm_eps must be finite and positive");
        ensure!(self.rope_theta.is_finite() && self.rope_theta > 0.0, "rope_theta must be finite and positive");
        ensure!(self.dropout.is_finite() && (0.0..1.0).contains(&self.dropout), "dropout must be in [0, 1)");
        ensure!(self.init_std.is_finite() && self.init_std > 0.0, "init_std must be finite and positive");
        ensure!(self.residual_scale_layers > 0, "residual_scale_layers must be positive");
        ensure!(self.param_budget > 0, "param_budget must be positive");
        ensure!(self.param_tolerance.is_finite() && self.param_tolerance >= 0.0, "param_tolerance must be finite and nonnegative");
        let count = self.checked_param_count().context("model parameter count overflow")?;
        let drift = (count as f64 - self.param_budget as f64).abs() / self.param_budget as f64;
        ensure!(drift <= self.param_tolerance as f64, "param count drift {drift:.3} exceeds tolerance");
        Ok(())
    }

    pub fn expected_shapes(&self) -> BTreeMap<String, Vec<usize>> {
        let d = self.d_model;
        let h = self.mlp_hidden;
        let mut map: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        map.insert("token_emb.weight".to_string(), vec![self.vocab_size, d]);
        map.insert("final_norm.weight".to_string(), vec![d]);
        if !self.tie_embeddings {
            map.insert("lm_head.weight".to_string(), vec![self.vocab_size, d]);
        }
        for layer in 0..self.n_layers {
            let prefix = format!("layers.{layer}.");
            map.insert(prefix.clone() + "attn_norm.weight", vec![d]);
            map.insert(prefix.clone() + "attn_qkv.weight", vec![self.qkv_out(), d]);
            map.insert(prefix.clone() + "attn_proj.weight", vec![d, d]);
            map.insert(prefix.clone() + "mlp_norm.weight", vec![d]);
            if self.n_experts > 1 {
                map.insert(prefix.clone() + "mlp_router.weight", vec![self.n_experts, d]);
            }
            for expert in 0..self.n_experts {
                let mlp = if self.n_experts == 1 {
                    prefix.clone()
                } else {
                    format!("{prefix}experts.{expert}.")
                };
                map.insert(mlp.clone() + "mlp_gate.weight", vec![h, d]);
                map.insert(mlp.clone() + "mlp_up.weight", vec![h, d]);
                map.insert(mlp + "mlp_down.weight", vec![d, h]);
            }
        }
        map
    }
}

#[inline]
fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn rearrange(
    heads: usize,
    head_dim: usize,
    d_model: usize,
    seq: usize,
    source: &[f32],
    target: &mut [f32],
) {
    for position in 0..seq {
        for head in 0..heads {
            let src = head * seq * head_dim + position * head_dim;
            let dst = position * d_model + head * head_dim;
            target[dst..dst + head_dim].copy_from_slice(&source[src..src + head_dim]);
        }
    }
}

pub fn read_header_metadata(path: impl AsRef<Path>) -> Result<HashMap<String, String>> {
    let path = path.as_ref();
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut length = [0u8; 8];
    file.read_exact(&mut length)
        .with_context(|| format!("reading header length of {}", path.display()))?;
    let header_len = u64::from_le_bytes(length) as usize;
    if header_len == 0 || header_len > (1 << 26) {
        bail!("{}: implausible header length {header_len}", path.display());
    }
    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header)?;
    let parsed: serde_json::Value = serde_json::from_slice(&header)?;
    let map = parsed
        .get("__metadata__")
        .and_then(|value| value.as_object())
        .with_context(|| format!("{} has no __metadata__ block", path.display()))?;
    let mut out = HashMap::new();
    for (key, value) in map {
        if let Some(text) = value.as_str() {
            out.insert(key.clone(), text.to_string());
        }
    }
    Ok(out)
}

pub struct Weights {
    pub spec: ModelSpec,
    tensors: BTreeMap<String, Vec<f32>>,
    pub metadata: HashMap<String, String>,
    pub file_bytes: u64,
}

impl Weights {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let metadata = read_header_metadata(path)?;
        ensure!(
            metadata.get("brainlab.spec_version").map(String::as_str) == Some(SPEC_VERSION),
            "unsupported brainlab.spec_version {:?}; expected {SPEC_VERSION}",
            metadata.get("brainlab.spec_version")
        );
        let config = metadata
            .get("brainlab.config_json")
            .context("checkpoint metadata is missing brainlab.config_json")?;
        let spec: ModelSpec =
            serde_json::from_str(config).context("parsing brainlab.config_json")?;
        spec.validate()?;
        let declared: usize = metadata
            .get("brainlab.param_count")
            .unwrap_or(&String::new())
            .parse()
            .context("brainlab.param_count is missing or unparsable")?;
        if declared != spec.param_count() {
            bail!(
                "metadata says {declared} parameters, spec computes {}",
                spec.param_count()
            );
        }
        let mut arrays = load_arrays(path)?;
        let expected = spec.expected_shapes();
        let mut tensors = BTreeMap::new();
        for (name, shape) in &expected {
            let array = arrays
                .remove(name)
                .with_context(|| format!("{}: missing tensor {name}", path.display()))?;
            if &array.shape != shape {
                bail!(
                    "{}: {name} has shape {:?}, spec says {shape:?}",
                    path.display(),
                    array.shape
                );
            }
            tensors.insert(name.clone(), array.data);
        }
        if let Some(extra) = arrays.keys().next() {
            bail!(
                "{}: tensor {extra} is not part of spec/weights_format.md",
                path.display()
            );
        }
        Ok(Self {
            spec,
            tensors,
            metadata,
            file_bytes: std::fs::metadata(path)?.len(),
        })
    }

    pub fn tensor(&self, name: &str) -> Result<&[f32]> {
        self.tensors
            .get(name)
            .map(|values| values.as_slice())
            .with_context(|| format!("no tensor {name}"))
    }

    pub fn layer_tensor(&self, layer: usize, suffix: &str) -> Result<&[f32]> {
        self.tensor(&format!("layers.{layer}.{suffix}"))
    }

    pub fn n_tensors(&self) -> usize {
        self.tensors.len()
    }

    pub fn output_head(&self) -> Result<&[f32]> {
        self.tensor(if self.spec.tie_embeddings { "token_emb.weight" } else { "lm_head.weight" })
    }
}

pub fn route_topk(logits: &[f32], top_k: usize) -> Result<Vec<(usize, f32)>> {
    ensure!(top_k > 0 && top_k <= logits.len(), "top_k must be in 1..=number of experts");
    ensure!(logits.iter().all(|value| value.is_finite()), "router logits must be finite");
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    indices.truncate(top_k);
    let mut probabilities: Vec<f32> = indices.iter().map(|&index| logits[index]).collect();
    softmax_row(&mut probabilities);
    Ok(indices.into_iter().zip(probabilities).collect())
}

fn swiglu(
    weights: &Weights,
    prefix: &str,
    input: &[f32],
    hidden: &mut [f32],
    activated: &mut [f32],
    out: &mut [f32],
) -> Result<()> {
    let (d, h) = (weights.spec.d_model, weights.spec.mlp_hidden);
    let seq = input.len() / d;
    let gate = weights.tensor(&format!("{prefix}mlp_gate.weight"))?;
    let up = weights.tensor(&format!("{prefix}mlp_up.weight"))?;
    let down = weights.tensor(&format!("{prefix}mlp_down.weight"))?;
    let stamp = profile::start();
    linear_tiled(input, gate, hidden, seq, d, h, None)?;
    profile::record("linear_gate", stamp);
    let stamp = profile::start();
    linear_tiled(input, up, activated, seq, d, h, None)?;
    profile::record("linear_up", stamp);
    let stamp = profile::start();
    for (slot, value) in hidden.iter_mut().zip(activated.iter()) {
        *slot = silu(*slot) * *value;
    }
    profile::record("silu_mul", stamp);
    let stamp = profile::start();
    linear_tiled(hidden, down, out, seq, h, d, None)?;
    profile::record("linear_down", stamp);
    Ok(())
}

pub struct Engine<'w> {
    weights: &'w Weights,
    rope: RopeTable,
    x: Vec<f32>,
    rms: Vec<f32>,
    qkv: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attention_out: Vec<f32>,
    mixed: Vec<f32>,
    hidden: Vec<f32>,
    activated: Vec<f32>,
    cache_k: Vec<f32>,
    cache_v: Vec<f32>,
    kv_len: usize,
}

impl<'w> Engine<'w> {
    pub fn new(weights: &'w Weights) -> Result<Self> {
        let spec = &weights.spec;
        spec.validate()?;
        let d = spec.d_model;
        let kv = spec.kv_width();
        let capacity = spec.context_len;
        Ok(Self {
            weights,
            rope: RopeTable::new(capacity, spec.head_dim, spec.rope_theta)?,
            x: vec![0.0; capacity * d],
            rms: vec![0.0; capacity * d],
            qkv: vec![0.0; capacity * spec.qkv_out()],
            q: vec![0.0; capacity * d],
            k: vec![0.0; capacity * kv],
            v: vec![0.0; capacity * kv],
            attention_out: vec![0.0; capacity * d],
            mixed: vec![0.0; capacity * d],
            hidden: vec![0.0; capacity * spec.mlp_hidden],
            activated: vec![0.0; capacity * spec.mlp_hidden],
            cache_k: vec![0.0; spec.n_layers * capacity * kv],
            cache_v: vec![0.0; spec.n_layers * capacity * kv],
            kv_len: 0,
        })
    }

    pub fn kv_len(&self) -> usize {
        self.kv_len
    }

    pub fn spec(&self) -> &'w ModelSpec {
        &self.weights.spec
    }

    pub fn reset(&mut self) {
        self.kv_len = 0;
    }

    fn gather(&mut self, weights: &'w Weights, ids: &[u32]) -> Result<()> {
        let spec = &weights.spec;
        let d = spec.d_model;
        let embedding = weights.tensor("token_emb.weight")?;
        for (index, raw) in ids.iter().enumerate() {
            let id = *raw as usize;
            if id >= spec.vocab_size {
                bail!("token id {id} outside vocab {}", spec.vocab_size);
            }
            self.x[index * d..(index + 1) * d].copy_from_slice(&embedding[id * d..(id + 1) * d]);
        }
        Ok(())
    }

    fn split_qkv(&mut self, spec: &'w ModelSpec, seq: usize) {
        let (heads, hd, d) = (spec.n_heads, spec.head_dim, spec.d_model);
        let kv = spec.kv_width();
        for position in 0..seq {
            let row = position * spec.qkv_out();
            for head in 0..heads {
                let dst = head * seq * hd + position * hd;
                self.q[dst..dst + hd]
                    .copy_from_slice(&self.qkv[row + head * hd..row + head * hd + hd]);
            }
            for head in 0..spec.n_kv_heads {
                let dst = head * seq * hd + position * hd;
                let key = row + d + head * hd;
                let value = key + kv;
                self.k[dst..dst + hd].copy_from_slice(&self.qkv[key..key + hd]);
                self.v[dst..dst + hd].copy_from_slice(&self.qkv[value..value + hd]);
            }
        }
    }

    fn apply_rope(&mut self, spec: &ModelSpec, seq: usize, position: usize) -> Result<()> {
        let plane = seq * spec.head_dim;
        for head in 0..spec.n_heads {
            self.rope.apply_rows(&mut self.q[head * plane..(head + 1) * plane], position)?;
        }
        for head in 0..spec.n_kv_heads {
            self.rope.apply_rows(&mut self.k[head * plane..(head + 1) * plane], position)?;
        }
        Ok(())
    }

    fn store_kv(
        &mut self,
        spec: &'w ModelSpec,
        layer: usize,
        seq: usize,
        row: usize,
    ) -> Result<()> {
        let hd = spec.head_dim;
        if row + seq > spec.context_len {
            bail!("kv cache would exceed context_len {}", spec.context_len);
        }
        let base = layer * spec.context_len * spec.kv_width();
        for head in 0..spec.n_kv_heads {
            for position in 0..seq {
                let src = head * seq * hd + position * hd;
                let dst = base + head * spec.context_len * hd + (row + position) * hd;
                self.cache_k[dst..dst + hd].copy_from_slice(&self.k[src..src + hd]);
                self.cache_v[dst..dst + hd].copy_from_slice(&self.v[src..src + hd]);
            }
        }
        Ok(())
    }

    fn mlp(&mut self, weights: &'w Weights, layer: usize, seq: usize) -> Result<()> {
        let spec = &weights.spec;
        let (d, h) = (spec.d_model, spec.mlp_hidden);
        let mlp_norm = weights.layer_tensor(layer, "mlp_norm.weight")?;
        let stamp = profile::start();
        for position in 0..seq {
            let row = position * d;
            rmsnorm_row(
                &self.x[row..row + d],
                mlp_norm,
                &mut self.rms[row..row + d],
                spec.rms_norm_eps,
            );
        }
        profile::record("rmsnorm_mlp", stamp);
        if spec.n_experts == 1 {
            swiglu(
                weights,
                &format!("layers.{layer}."),
                &self.rms[..seq * d],
                &mut self.hidden[..seq * h],
                &mut self.activated[..seq * h],
                &mut self.mixed[..seq * d],
            )?;
        } else {
            let router = weights.layer_tensor(layer, "mlp_router.weight")?;
            let mut logits = vec![0.0; spec.n_experts];
            self.mixed[..seq * d].fill(0.0);
            for position in 0..seq {
                let row = position * d;
                let input = &self.rms[row..row + d];
                let stamp = profile::start();
                linear_tiled(input, router, &mut logits, 1, d, spec.n_experts, None)?;
                let routes = route_topk(&logits, spec.experts_per_token)?;
                profile::record("moe_router", stamp);
                for (expert, probability) in routes {
                    swiglu(
                        weights,
                        &format!("layers.{layer}.experts.{expert}."),
                        input,
                        &mut self.hidden[..h],
                        &mut self.activated[..h],
                        &mut self.attention_out[..d],
                    )?;
                    for index in 0..d {
                        self.mixed[row + index] += probability * self.attention_out[index];
                    }
                }
            }
        }
        let stamp = profile::start();
        for index in 0..seq * d {
            self.x[index] += self.mixed[index];
        }
        profile::record("residual_mlp", stamp);
        Ok(())
    }

    fn finish(&mut self, weights: &'w Weights, seq: usize, logits: &mut [f32]) -> Result<()> {
        let spec = &weights.spec;
        let d = spec.d_model;
        let final_norm = weights.tensor("final_norm.weight")?;
        let stamp = profile::start();
        for position in 0..seq {
            let row = position * d;
            rmsnorm_row(
                &self.x[row..row + d],
                final_norm,
                &mut self.rms[row..row + d],
                spec.rms_norm_eps,
            );
        }
        profile::record("rmsnorm_final", stamp);
        let output_head = weights.output_head()?;
        let stamp = profile::start();
        linear_tiled(
            &self.rms[..seq * d],
            output_head,
            &mut logits[..seq * spec.vocab_size],
            seq,
            d,
            spec.vocab_size,
            None,
        )?;
        profile::record("linear_logits", stamp);
        Ok(())
    }

    fn finish_row(
        &mut self,
        weights: &'w Weights,
        position: usize,
        logits_row: &mut [f32],
    ) -> Result<()> {
        let spec = &weights.spec;
        let d = spec.d_model;
        let final_norm = weights.tensor("final_norm.weight")?;
        let stamp = profile::start();
        let begin = position * d;
        rmsnorm_row(
            &self.x[begin..begin + d],
            final_norm,
            &mut self.rms[..d],
            spec.rms_norm_eps,
        );
        profile::record("rmsnorm_final", stamp);
        let output_head = weights.output_head()?;
        let stamp = profile::start();
        linear_tiled(
            &self.rms[..d],
            output_head,
            logits_row,
            1,
            d,
            spec.vocab_size,
            None,
        )?;
        profile::record("linear_logits", stamp);
        Ok(())
    }

    pub fn kv_cache(&self) -> (&[f32], &[f32]) {
        (&self.cache_k, &self.cache_v)
    }

    pub fn kv_cache_shape(&self) -> [usize; 4] {
        let spec = &self.weights.spec;
        [spec.n_layers, spec.n_kv_heads, spec.context_len, spec.head_dim]
    }

    fn check_sequence(&self, seq: usize) -> Result<usize> {
        let spec = &self.weights.spec;
        ensure!(
            seq > 0 && seq <= spec.context_len,
            "sequence length {seq} outside 1..{}",
            spec.context_len
        );
        Ok(spec.vocab_size)
    }

    pub fn forward(&mut self, ids: &[u32], logits: &mut [f32]) -> Result<()> {
        self.forward_cached(ids, logits, false)
    }

    pub fn forward_cached(
        &mut self,
        ids: &[u32],
        logits: &mut [f32],
        store_cache: bool,
    ) -> Result<()> {
        let weights = self.weights;
        let seq = ids.len();
        let vocab = self.check_sequence(seq)?;
        ensure!(
            logits.len() >= seq * vocab,
            "logits buffer needs {} floats",
            seq * vocab
        );
        self.run_layers(ids, store_cache)?;
        self.finish(weights, seq, logits)
    }

    /// Fills the KV cache for the whole prompt and emits logits for the final position only.
    ///
    /// Generation needs one row of logits to pick the first token, so the all-rows path allocates
    /// `seq * vocab` floats (64 MiB at seq=1024, vocab=16384, fp32) to throw all but one row away.
    pub fn prefill_last(&mut self, ids: &[u32], last_logits: &mut [f32]) -> Result<()> {
        let weights = self.weights;
        let seq = ids.len();
        let vocab = self.check_sequence(seq)?;
        ensure!(
            last_logits.len() >= vocab,
            "logits row needs {vocab} floats"
        );
        self.run_layers(ids, true)?;
        self.finish_row(weights, seq - 1, last_logits)
    }

    fn run_layers(&mut self, ids: &[u32], store_cache: bool) -> Result<()> {
        let weights = self.weights;
        let spec = &weights.spec;
        let (d, heads, hd) = (spec.d_model, spec.n_heads, spec.head_dim);
        let seq = ids.len();
        self.check_sequence(seq)?;
        self.reset();
        let stamp = profile::start();
        self.gather(weights, ids)?;
        profile::record("embed_gather", stamp);
        for layer in 0..spec.n_layers {
            let attn_norm = weights.layer_tensor(layer, "attn_norm.weight")?;
            let stamp = profile::start();
            for position in 0..seq {
                let row = position * d;
                rmsnorm_row(
                    &self.x[row..row + d],
                    attn_norm,
                    &mut self.rms[row..row + d],
                    spec.rms_norm_eps,
                );
            }
            profile::record("rmsnorm_attn", stamp);
            let qkv_weight = weights.layer_tensor(layer, "attn_qkv.weight")?;
            let stamp = profile::start();
            linear_tiled(
                &self.rms[..seq * d],
                qkv_weight,
                &mut self.qkv[..seq * 3 * d],
                seq,
                d,
                3 * d,
                None,
            )?;
            profile::record("linear_qkv", stamp);
            let stamp = profile::start();
            self.split_qkv(spec, seq);
            for head in 0..heads {
                let begin = head * seq * hd;
                self.rope
                    .apply_rows(&mut self.q[begin..begin + seq * hd], 0)?;
                self.rope
                    .apply_rows(&mut self.k[begin..begin + seq * hd], 0)?;
            }
            profile::record("rope_and_split", stamp);
            let stamp = profile::start();
            attention_causal(
                &self.q[..heads * seq * hd],
                &self.k[..heads * seq * hd],
                &self.v[..heads * seq * hd],
                &mut self.attention_out[..heads * seq * hd],
                1,
                heads,
                seq,
                hd,
                spec.attn_scale(),
            )?;
            profile::record("attention_causal", stamp);
            let stamp = profile::start();
            let attention_out = std::mem::take(&mut self.attention_out);
            rearrange(heads, hd, d, seq, &attention_out, &mut self.mixed);
            self.attention_out = attention_out;
            profile::record("rearrange_heads", stamp);
            let proj = weights.layer_tensor(layer, "attn_proj.weight")?;
            let stamp = profile::start();
            linear_tiled(
                &self.mixed[..seq * d],
                proj,
                &mut self.rms[..seq * d],
                seq,
                d,
                d,
                None,
            )?;
            profile::record("linear_proj", stamp);
            let stamp = profile::start();
            for index in 0..seq * d {
                self.x[index] += self.rms[index];
            }
            if store_cache {
                self.store_kv(spec, layer, seq, 0)?;
            }
            profile::record("residual_and_store", stamp);
            self.mlp(weights, layer, seq)?;
        }
        if store_cache {
            self.kv_len = seq;
        }
        Ok(())
    }

    pub fn prefill(&mut self, ids: &[u32], last_logits: &mut [f32]) -> Result<()> {
        let weights = self.weights;
        let spec = &weights.spec;
        let seq = ids.len();
        let mut scratch = vec![0.0f32; seq * spec.vocab_size];
        self.forward_cached(ids, &mut scratch, true)?;
        ensure!(
            last_logits.len() >= spec.vocab_size,
            "logits row needs {} floats",
            spec.vocab_size
        );
        let begin = (seq - 1) * spec.vocab_size;
        last_logits[..spec.vocab_size].copy_from_slice(&scratch[begin..begin + spec.vocab_size]);
        Ok(())
    }

    pub fn decode(&mut self, token: u32, logits_row: &mut [f32]) -> Result<()> {
        let weights = self.weights;
        let spec = &weights.spec;
        let (d, heads, hd) = (spec.d_model, spec.n_heads, spec.head_dim);
        ensure!(
            self.kv_len < spec.context_len,
            "kv cache full at {}",
            spec.context_len
        );
        ensure!(
            logits_row.len() >= spec.vocab_size,
            "logits row needs {} floats",
            spec.vocab_size
        );
        let position = self.kv_len;
        let stamp = profile::start();
        self.gather(weights, std::slice::from_ref(&token))?;
        profile::record("decode_embed_gather", stamp);
        for layer in 0..spec.n_layers {
            let attn_norm = weights.layer_tensor(layer, "attn_norm.weight")?;
            let stamp = profile::start();
            rmsnorm_row(
                &self.x[..d],
                attn_norm,
                &mut self.rms[..d],
                spec.rms_norm_eps,
            );
            profile::record("decode_rmsnorm_attn", stamp);
            let qkv_weight = weights.layer_tensor(layer, "attn_qkv.weight")?;
            let stamp = profile::start();
            linear_tiled(
                &self.rms[..d],
                qkv_weight,
                &mut self.qkv[..3 * d],
                1,
                d,
                3 * d,
                None,
            )?;
            profile::record("decode_linear_qkv", stamp);
            let stamp = profile::start();
            for head in 0..heads {
                let plane = head * hd;
                self.q[plane..plane + hd].copy_from_slice(&self.qkv[plane..plane + hd]);
                self.k[plane..plane + hd].copy_from_slice(&self.qkv[d + plane..d + plane + hd]);
                self.v[plane..plane + hd]
                    .copy_from_slice(&self.qkv[2 * d + plane..2 * d + plane + hd]);
                self.rope
                    .apply_rows(&mut self.q[plane..plane + hd], position)?;
                self.rope
                    .apply_rows(&mut self.k[plane..plane + hd], position)?;
            }
            self.split_qkv_to_cache(spec, layer, position)?;
            profile::record("decode_rope_and_cache", stamp);
            let cache_base = layer * spec.context_len * d;
            let stamp = profile::start();
            attention_cache(
                &self.q[..heads * hd],
                &self.cache_k[cache_base..cache_base + spec.context_len * d],
                &self.cache_v[cache_base..cache_base + spec.context_len * d],
                &mut self.attention_out[..heads * hd],
                heads,
                position + 1,
                spec.context_len,
                hd,
                spec.attn_scale(),
            )?;
            profile::record("decode_attention_cache", stamp);
            let stamp = profile::start();
            let attention_out = std::mem::take(&mut self.attention_out);
            rearrange(heads, hd, d, 1, &attention_out, &mut self.mixed);
            self.attention_out = attention_out;
            profile::record("decode_rearrange", stamp);
            let proj = weights.layer_tensor(layer, "attn_proj.weight")?;
            linear_tiled(&self.mixed[..d], proj, &mut self.rms[..d], 1, d, d, None)?;
            for index in 0..d {
                self.x[index] += self.rms[index];
            }
            self.mlp(weights, layer, 1)?;
        }
        self.kv_len = position + 1;
        let final_norm = weights.tensor("final_norm.weight")?;
        let stamp = profile::start();
        rmsnorm_row(
            &self.x[..d],
            final_norm,
            &mut self.rms[..d],
            spec.rms_norm_eps,
        );
        profile::record("decode_rmsnorm_final", stamp);
        let output_head = weights.output_head()?;
        let stamp = profile::start();
        linear_tiled(
            &self.rms[..d],
            output_head,
            logits_row,
            1,
            d,
            spec.vocab_size,
            None,
        )?;
        profile::record("decode_linear_logits", stamp);
        Ok(())
    }

    fn split_qkv_to_cache(&mut self, spec: &'w ModelSpec, layer: usize, row: usize) -> Result<()> {
        let (heads, hd, d) = (spec.n_heads, spec.head_dim, spec.d_model);
        if row >= spec.context_len {
            bail!("kv cache overflow at position {row}");
        }
        let base = layer * spec.context_len * d + row * hd;
        for head in 0..heads {
            let src = head * hd;
            let dst = base + head * spec.context_len * hd;
            self.cache_k[dst..dst + hd].copy_from_slice(&self.k[src..src + hd]);
            self.cache_v[dst..dst + hd].copy_from_slice(&self.v[src..src + hd]);
        }
        Ok(())
    }
}
