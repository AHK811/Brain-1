# brain-lab

One 33M-parameter decoder-only transformer, described once in `spec/`, implemented several ways, and
measured against a PyTorch reference by one harness.

The point is not to build a language model. The point is to find out what it costs - in accuracy and in
bandwidth - to implement the *same* model in Python, Rust, Triton, CUDA and Mojo. Parity is established
before any speed number is believed, because a fast wrong kernel is worthless.

## The model

`spec/model_config.toml` is the source of truth: `d_model=512`, 8 layers, 8 heads of 64, vocab 16384,
context 1024, SwiGLU hidden 1344, pre-norm RMSNorm (eps 1e-6), split-half RoPE (theta 1e4), tied
embeddings, fused QKV as one `[3d, d]` matrix, no biases anywhere. **33,300,992 parameters.**

| file | contract |
|---|---|
| `spec/model_config.toml` | architecture and hyperparameters |
| `spec/weights_format.md` | tensor names, shapes, `[out, in]` order, metadata keys, raw bridge format |
| `spec/tokenizer_spec.md` | byte-level BPE, id layout, pre-tokenizer, training tie-break, `BLTK` binary |
| `spec/datapack_format.md` | packed uint16 shards, manifests, document split, file-level train/val stride |

## Layout

```
python/brainlab/     reference implementation: model, train, data, export, eval, tok
rust/crates/         tokenizer, datapack, infer-cpu (+ CLI), pyo3-bindings
triton_kernels/      Triton kernels (named so the installed `triton` package is not shadowed)
cuda/                hand-written CUDA rmsnorm, used as the bandwidth ceiling
mojo/                Mojo kernels + a raw-CLI driver
tests/golden/        generated reference tensors (gitignored) + cross-language fixtures
tests/parity/        the tests that decide whether any of the above is true
bench/harness/       one harness, every backend, identical inputs
bench/results/       JSON runs and tables
scripts/             setup and verification entry points
docs/                decisions.md, experiment_log.md
```

## Quickstart

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
just golden -- --model --raw
just test
just bench-rmsnorm
just table
```

On Colab or Kaggle, everything the runtime needs is installed by one script, because the container resets
between sessions:

```bash
!git clone <this repo> /content/brain-lab && cd /content/brain-lab
!bash scripts/setup_colab.sh
!bash scripts/check_gpu.sh
!just bench-rmsnorm --tag t4
```

Rust is not buildable in this Windows checkout (no MSVC linker, and the bundled GNU `dlltool` has no
`as`). The Linux lane is WSL:

```bash
just wsl-check      # cargo check --workspace --all-targets
just wsl-test       # cargo test --workspace --release  (parity against tests/golden)
just wsl-bench      # per-key JSON into bench/results/rust/
just table-rust     # fold those rows into the same markdown table as the Python rows
just wsl-verify     # fmt + clippy -D warnings + test, tee'd into logs/ with the commit id
just wsl-clone      # a clean clone must import, and no ignore rule may hide source
just wsl-cli        # the exit codes the docs promise (leak fails, ppl helps) really happen
```

Token splits are gated too: `brainlab leak` hashes every 50-token span of 100 sampled val sequences against
the whole train stream and exits nonzero on a hit, which is the check that makes the file-level val stride
(D025) more than an intention.

## How a backend is measured

1. `python/brainlab/export/make_golden.py` writes deterministic operands and the PyTorch reference output
   for every workload in `python/brainlab/spec/workloads.py`, plus the tolerances for that op and dtype.
2. Each backend loads the same tensors - Python and Triton from safetensors, CUDA and Mojo from the raw
   `.bin` bridge, Rust from safetensors - and returns its output, timings and hardware description.
3. A backend never reports whether it is correct. `bench/harness/run_all.py` compares the bytes against the
   reference using the workload's `atol`, `rtol` and relative-L2 budget and sets `parity_pass`.
4. Timing is warmup, then at least 20 reps, CUDA synchronize before the stop, median of the distribution,
   with min, p90 and MAD% kept in the row. Every row labels its hardware.

`justfile` names the entry points; `docs/decisions.md` records why each choice was made and what would have
to change for it to be wrong.

## Current status

Measured on one host (WSL2, 4 vCPU, AVX2+FMA, torch 2.14.1+cpu, rust 1.99.1), 40 reps, median, all rows
passing parity - the full table and its caveats are in `docs/experiment_log.md`:

| backend | parity | speed |
|---|---|---|
| `pytorch_eager` | defines the reference, so `rel_l2 = 0` is self-consistency not corroboration | 1.4 GB/s at 4096x512 - about six passes over the row |
| `pytorch_compile` | passes, `rel_l2 5.9e-08` | 11.8 GB/s |
| `rust/crates/infer-cpu` | passes, `rel_l2 5.1e-08` | 16.3 GB/s, and 13 GFLOP/s on causal attention after FMA |
| `triton_kernels` | written, unmeasured | needs a GPU |
| `cuda/rmsnorm.cu` | written, unmeasured | needs a GPU |
| `mojo/` | written, unmeasured | needs a GPU |

Python suite: 63 passed, 7 skipped (every skip is a GPU row, a Linux-only Rust binary, or a rust-trained
tokenizer cross-check that needs `BRAINLAB_TOKENIZER_DIR`).

GPU rows cannot be produced on this host. A number from a machine with no GPU is not a measurement, so those
cells stay empty until they run on a T4:

```bash
!bash scripts/setup_colab.sh && !bash scripts/check_gpu.sh
!just bench-rmsnorm --tag t4 && !just table
```
