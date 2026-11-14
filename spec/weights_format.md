# Weights format

`brainlab-moe-105m` ships as one single-file [safetensors](https://huggingface.co/docs/safetensors) checkpoint. Every backend in this repo loads the same file. Nothing else is a source of weights.

`brainlab.spec_version` is `"2"`. A version-1 file is not a version-2 file: the qkv row count, the head layout, the existence of `lm_head.weight` and the expert naming all changed, so a loader rejects the version rather than guessing.

## Rules that are not negotiable

1. **No biases anywhere.** Norms and routers are the only tensors with a leading dimension below `d_model`.
2. **Every 2-D tensor is `[out, in]`, row-major.** A linear layer computes `y[o] = sum_i x[i] * W[o*in + i]`, which is PyTorch `nn.Linear` semantics (`x @ W.T`). A backend that wants `x @ W` must transpose at load time, not at inner-loop time.
3. **Fused QKV, grouped.** One `[d_model + 2*kv_dim, d_model]` matrix per layer, laid out as all `q` rows, then all `k` rows, then all `v` rows. With `d_model = 512`, `kv_dim = 128` the row blocks are `0..511`, `512..639`, `640..767`. Row `d_model + j` is key head weight `j`.
4. **Head layout is `[B, T, H, hd]` after reshape**, heads interleaved inside the row, i.e. `q[t, h*head_dim + c]`. Query heads number `n_heads = 8`; key and value heads number `n_kv_heads = 2`, so query head `q` reads KV head `q / (n_heads / n_kv_heads)`. A backend that treats the qkv row blocks as three equal blocks is wrong.
5. **RoPE uses the split-half convention** with `cos`/`sin` of length `head_dim`; never store rope tables in the checkpoint, they are derived from `context_len` and `rope_theta`.
6. **Embeddings are untied.** `lm_head.weight` is present and is not `token_emb.weight`. The logits are `x @ lm_head.weight.T`.
7. **Sparse experts are named by index.** `layers.<i>.experts.<e>.mlp_gate.weight` with `e` in `0..n_experts`. A dense model (`n_experts = 1`) uses `layers.<i>.mlp_gate.weight` and stores no router.
8. **Layer and expert indices are 0-based, ascending**, and `layers.0` is closest to the embedding.

## Tensor table

`d = 512`, `kv = 128`, `qkv = 768`, `h = 1344`, `V = 32768`, `L = 8`, `E = 4`. `dtype` is the on-disk type when `--dtype f32`; `f16` exports use the same names and shapes with `F16`.

| name | shape | dtype | count |
|---|---|---|---|
| `token_emb.weight` | `[V, d]` | `F32` | 16777216 |
| `lm_head.weight` | `[V, d]` | `F32` | 16777216 |
| `layers.0..7.attn_norm.weight` | `[d]` | `F32` | 512 each |
| `layers.0..7.attn_qkv.weight` | `[qkv, d]` | `F32` | 393216 each |
| `layers.0..7.attn_proj.weight` | `[d, d]` | `F32` | 262144 each |
| `layers.0..7.mlp_norm.weight` | `[d]` | `F32` | 512 each |
| `layers.0..7.mlp_router.weight` | `[E, d]` | `F32` | 2048 each |
| `layers.0..7.experts.0..3.mlp_gate.weight` | `[h, d]` | `F32` | 688128 each |
| `layers.0..7.experts.0..3.mlp_up.weight` | `[h, d]` | `F32` | 688128 each |
| `layers.0..7.experts.0..3.mlp_down.weight` | `[d, h]` | `F32` | 688128 each |
| `final_norm.weight` | `[d]` | `F32` | 512 |

Total stored parameters: `104,882,688`. `spec/model_config.toml` declares the budget `105000000` with tolerance `0.02`; `brainlab.spec.config.ModelSpec.validate()` fails the build if the count drifts outside it. Every name and shape in this table is derived from the same `ModelSpec.tensor_shapes()` the loader uses, so the table, the file and the check cannot drift apart.

Only `experts_per_token = 2` of the 4 experts run per token, so stored parameters and per-token FLOPs are different numbers. A benchmark that reports utilization against `104,882,688` overstates the work a token actually does; use the active-parameter count from the workload table.

## `__metadata__`

safetensors reserves a metadata map. It must contain exactly these string keys:

| key | value |
|---|---|
| `brainlab.spec_version` | `"2"` |
| `brainlab.name` | `"brainlab-moe-105m"` |
| `brainlab.config_json` | compact JSON of the `[model]`-equivalent dict from `spec/model_config.toml` |
| `brainlab.param_count` | decimal integer string, must equal the shape sum |
| `brainlab.source` | `"python/brainlab/export/to_safetensors.py"` |
| `brainlab.producer_versions` | JSON of `{"torch": ..., "safetensors": ...}` |

A loader rejects the file if `param_count` disagrees with the shapes it reads, or if `config_json` disagrees with `spec/model_config.toml`. Silent config drift is the cheapest way to make every parity number meaningless.

## Forward pass contract

For a token batch `ids` of shape `[B, T]` with `T <= context_len`:

```
x = token_emb[ids]                          # [B,T,d]
for i in 0..L:
    x = x + attn(rmsnorm(x, attn_norm[i]), i)     # rope on q,k; scale = head_dim**-0.5; gqa sdpa
    x = x + moe(rmsnorm(x, mlp_norm[i]))          # top-2 of 4 swiglu experts, softmax over the two scores
x = rmsnorm(x, final_norm)
logits = x @ lm_head.weight.T               # [B,T,V]
```

`rmsnorm` is exactly:

```
x32 = float32(x)
y   = x32 * rsqrt(mean(x32^2, axis=-1) + eps)
out = cast(x32_ish * float32(weight), input_dtype)
```

The variance is a plain mean of squares in **fp32**, never a sum in fp16, and never `1e-5` when the spec says `1e-6`.

The router is also fp32, always: `scores = x_fp32 @ router.T`, the top `experts_per_token` chosen by a **stable descending** sort of those scores, the gate weights a softmax over the selected scores only. Ties therefore break by expert index, which is what lets a second backend reproduce the first one's routing bit for bit. The load-balancing auxiliary loss is a training-only quantity and never enters `logits`.

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

The `forward_*` goldens are the only tensors that pin the grouped-head, untied, sparse-expert model end to end: they hold `ids` and the fp32 `logits` from the same `model.safetensors` the native backends load. The single-op goldens stay multi-head (`attention` operands all have `n_heads` heads) because that op is a kernel test, not a model test; GQA grouping is checked where it is actually used.
