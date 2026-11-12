# Weights format

`brainlab-33m` ships as one single-file [safetensors](https://huggingface.co/docs/safetensors) checkpoint. Every backend in this repo loads the same file. Nothing else is a source of weights.

## Rules that are not negotiable

1. **No biases anywhere.** Norms are the only 1-D tensors.
2. **Every 2-D tensor is `[out, in]`, row-major.** A linear layer computes `y[o] = sum_i x[i] * W[o*in + i]`, which is PyTorch `nn.Linear` semantics (`x @ W.T`). A backend that wants `x @ W` must transpose at load time, not at inner-loop time.
3. **Fused QKV.** One `[3*d_model, d_model]` matrix per layer, laid out as `q` rows, then `k` rows, then `v` rows. Row `d_model + j` is key head weight `j`.
4. **Head layout is `[B, T, H, hd]` after reshape**, heads interleaved inside the row, i.e. `q[t, h*head_dim + c]`.
5. **RoPE uses the split-half convention** with `cos`/`sin` of length `head_dim`; never store rope tables in the checkpoint, they are derived from `context_len` and `rope_theta`.
6. **Embeddings are tied.** There is no `lm_head.weight`. The logits are `x @ token_emb.weight.T`.
7. **`layer index is 0-based, ascending**, and `layers.0` is closest to the embedding.

## Tensor table

`d = 512`, `h = 1344`, `V = 16384`, `L = 8`. `dtype` is the on-disk type when `--dtype f32`; `f16` exports use the same names and shapes with `F16`.

| name | shape | dtype | count |
|---|---|---|---|
| `token_emb.weight` | `[V, d]` | `F32` | 8388608 |
| `layers.0..7.attn_norm.weight` | `[d]` | `F32` | 512 each |
| `layers.0..7.attn_qkv.weight` | `[3d, d]` | `F32` | 786432 each |
| `layers.0..7.attn_proj.weight` | `[d, d]` | `F32` | 262144 each |
| `layers.0..7.mlp_norm.weight` | `[d]` | `F32` | 512 each |
| `layers.0..7.mlp_gate.weight` | `[h, d]` | `F32` | 688128 each |
| `layers.0..7.mlp_up.weight` | `[h, d]` | `F32` | 688128 each |
| `layers.0..7.mlp_down.weight` | `[d, h]` | `F32` | 688128 each |
| `final_norm.weight` | `[d]` | `F32` | 512 |

Total: `33,300,992` parameters. `spec/model_config.toml` declares the budget `33000000` with tolerance `0.02`; `brainlab.spec.config.ModelSpec.validate()` fails the build if the count drifts outside it.

## `__metadata__`

safetensors reserves a metadata map. It must contain exactly these string keys:

| key | value |
|---|---|
| `brainlab.spec_version` | `"1"` |
| `brainlab.name` | `"brainlab-33m"` |
| `brainlab.config_json` | compact JSON of the `[model]`-equivalent dict from `spec/model_config.toml` |
| `brainlab.param_count` | decimal integer string |
| `brainlab.source` | `"python/brainlab/export/to_safetensors.py"` |
| `brainlab.producer_versions` | JSON of `{"torch": ..., "safetensors": ...}` |

A loader rejects the file if `param_count` disagrees with the shapes it reads, or if `config_json` disagrees with `spec/model_config.toml`. Silent config drift is the cheapest way to make every parity number meaningless.

## Forward pass contract

For a token batch `ids` of shape `[B, T]` with `T <= context_len`:

```
x = token_emb[ids]                          # [B,T,d]
for i in 0..L:
    x = x + attn(rmsnorm(x, attn_norm[i]), i)     # rope applied to q,k; scale = head_dim**-0.5
    x = x + mlp(rmsnorm(x, mlp_norm[i]))          # swiglu: down(silu(gate(u)) * up(u))
x = rmsnorm(x, final_norm)
logits = x @ token_emb.T                    # [B,T,V]
```

`rmsnorm` is exactly:

```
x32 = float32(x)
y   = x32 * rsqrt(mean(x32^2, axis=-1) + eps)
out = cast(x32_ish * float32(weight), input_dtype)
```

The variance is a plain mean of squares in **fp32**, never a sum in fp16, and never `1e-5` when the spec says `1e-6`.

## Raw bridge format

Mojo and the CUDA bench do not parse safetensors, and there is no mature direct Mojo-to-Rust bridge, so they exchange plain arrays with Python. `python/brainlab/export/raw_io.py` writes, per workload:

```
<out_dir>/<name>.f32.bin     # C-contiguous little-endian float32
<out_dir>/<name>.f16.bin     # IEEE754 binary16, same layout
<out_dir>/manifest.json      # {"name": {"dtype": "f32", "shape": [rows, cols], "nbytes": ...}}
```

The manifest is for Python; native binaries receive paths, row counts and widths as argv. Byte order is always little-endian, alignment is always 16-byte on read (`posix_memalign` / `std::alloc` with `Layout`), and no file contains a header.

## Golden tensors

`python/brainlab/export/make_golden.py` writes `tests/golden/<op>_<dtype>.safetensors` from `seed = 1234`. Each file holds the inputs under `x`, `w`, ... and the PyTorch fp32 reference output under `y`. They are gitignored because a single `4096x512` fp32 triple is 25 MB; regenerate with `just golden`. Generation is seeded and deterministic, so a regenerated file is byte-comparable to the one that produced a published number.
