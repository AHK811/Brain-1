<!--
  Brain-1 | Repository README
  Repository: https://github.com/AHK811/Brain-1
  Keep architecture values synchronized with spec/model_config.toml.
-->

<div align="center">

# Brain-1

### A small language model. A serious model-engineering experiment.

**One model specification. Multiple implementations. Correctness before performance.**

[![Repository](https://img.shields.io/badge/GitHub-AHK811%2FBrain--1-181717?style=flat-square&logo=github)](https://github.com/AHK811/Brain-1)
[![Status](https://img.shields.io/badge/status-research%20%2F%20experimental-d97706?style=flat-square)](#project-status)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-reference%20implementation-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Rust](https://img.shields.io/badge/Rust-CPU%20inference-000000?style=flat-square&logo=rust&logoColor=white)](https://www.rust-lang.org/)

[![Repository views](https://hits.dwyl.com/AHK811/Brain-1.svg?style=flat-square)](https://hits.dwyl.com/AHK811/Brain-1)
[![Unique views](https://hits.dwyl.com/AHK811/Brain-1.svg?style=flat-square&show=unique)](https://hits.dwyl.com/AHK811/Brain-1)
[![Stars](https://img.shields.io/github/stars/AHK811/Brain-1?style=flat-square)](https://github.com/AHK811/Brain-1/stargazers)
[![Forks](https://img.shields.io/github/forks/AHK811/Brain-1?style=flat-square)](https://github.com/AHK811/Brain-1/network/members)
[![Contributors](https://img.shields.io/github/contributors/AHK811/Brain-1?style=flat-square)](https://github.com/AHK811/Brain-1/graphs/contributors)

</div>

---

## Overview

**Brain-1** is an experimental language-model and systems-engineering project built around a 33.3-million-parameter, decoder-only Transformer.

The project has two connected goals:

1. Develop and study a compact language-model pipeline, with an explicit architecture, tokenizer, data format, training configuration, evaluation path, and weight format.
2. Understand how the same mathematical model behaves across different implementation technologies. PyTorch acts as the reference; alternative implementations and kernels are checked against shared reference outputs before their performance is compared.

This is a research codebase, not a hosted AI service or a claim of production-grade assistant capability. The emphasis is on inspectable specifications, reproducible experiments, numerical parity, and measured engineering decisions.

## At a glance

| Component | Current design |
|---|---|
| Model | `brainlab-33m` |
| Architecture | Decoder-only causal Transformer |
| Parameters | **33,300,992** |
| Width | 512 |
| Transformer blocks | 8 |
| Attention | 8 heads × 64 dimensions; fused QKV; causal attention |
| MLP | SwiGLU, hidden width 1,344 |
| Vocabulary | 16,384 tokens |
| Context length | 1,024 tokens |
| Normalization | Pre-norm RMSNorm, ε = 1e-6 |
| Position encoding | Split-half RoPE, θ = 10,000 |
| Embeddings | Tied input and output embeddings |
| Biases | No linear-layer biases |
| Canonical specification | [`spec/model_config.toml`](spec/model_config.toml) |

The configuration file is the source of truth. The table above is a human-readable summary, not a second configuration system.

## Architecture

```text
                         Token IDs
                             │
                             ▼
                  Tied Token Embeddings
                       [16,384 × 512]
                             │
                             ▼
                 ┌───────────────────────┐
                 │  Transformer Block ×8│
                 │                       │
                 │  RMSNorm              │
                 │      │                │
                 │      ▼                │
                 │  Fused QKV            │
                 │      │                │
                 │  RoPE + Causal        │
                 │  Self-Attention       │
                 │      │                │
                 │  Output Projection    │
                 │      │                │
                 │  Residual Connection  │
                 │      │                │
                 │  RMSNorm              │
                 │      │                │
                 │  SwiGLU MLP           │
                 │      │                │
                 │  Residual Connection  │
                 └───────────┬───────────┘
                             │
                             ▼
                        Final RMSNorm
                             │
                             ▼
                  Tied Language-Model Head
                             │
                             ▼
                       Next-token logits
```

### Design choices

- **RMSNorm** keeps normalization lightweight.
- **RoPE** provides position information through rotary transformations.
- **SwiGLU** supplies a gated feed-forward block.
- **Fused QKV** keeps the attention projection contract explicit across implementations.
- **Tied embeddings** share the input embedding matrix with the output projection.
- **A single model specification** defines tensor shapes, architecture values, and interoperability contracts.

The current model uses standard multi-head attention. Grouped-query attention, larger context windows, and alternative model scales are potential experiments—not features of the current specification.

## What is implemented

| Area | Implementation | Status |
|---|---|---|
| Model reference | PyTorch Transformer, model specification, weight export | Implemented |
| Training and evaluation | Python training, checkpoint, evaluation, and generation components | Experimental; validate data/tokenizer inputs before a full run |
| Tokenization | Byte-level BPE with Python reference and Rust implementation | Implemented; cross-language fixtures available |
| Data packing | Packed `uint16` token shards with manifests | Implemented; corpus-specific validation remains important |
| CPU inference | Rust inference engine, including cached decoding | Implemented; parity tests are recorded in the experiment log |
| Python bindings | Rust / PyO3 bindings | Available |
| GPU kernels | CUDA and Triton operator kernels | Experimental; GPU measurements are pending |
| Mojo kernels | Mojo kernel experiments and driver | Experimental |
| Benchmark system | Shared workload definitions, reference tensors, parity checks, and timing harness | Implemented for the workloads currently defined |

**Important distinction:** Rust contains a full CPU inference path. CUDA, Triton, and Mojo should currently be understood as kernel-level work, not as complete independent training stacks.

## Repository structure

```text
Brain-1/
│
├── spec/                       # Canonical model, weights, tokenizer and datapack contracts
│   ├── model_config.toml
│   ├── weights_format.md
│   ├── tokenizer_spec.md
│   └── datapack_format.md
│
├── python/brainlab/            # PyTorch reference, training, data, export and evaluation
├── rust/crates/                # Tokenizer, datapack, CPU inference and PyO3 bindings
├── triton_kernels/             # Triton operator kernels
├── cuda/                       # Hand-written CUDA kernels
├── mojo/                       # Mojo kernel experiments and driver
│
├── tests/
│   ├── parity/                 # Cross-implementation correctness tests
│   └── golden/                 # Reference tensors and fixtures
│
├── bench/
│   ├── harness/                # Shared backend and timing harness
│   └── results/                # Benchmark outputs and reports
│
├── scripts/                   # Setup, environment checks and verification scripts
├── notebooks/                  # Hosted-runtime notebooks (Kaggle training lane)
├── docs/                       # Engineering decisions and experiment log
├── justfile                    # Common development commands
└── pyproject.toml
```

## Getting started

### Requirements

- Python **3.11 or newer**
- PyTorch **2.4 or newer**
- NumPy and Safetensors
- [`just`](https://github.com/casey/just) for the documented task commands
- Rust toolchain for Rust components (the pinned toolchain is described in `rust-toolchain.toml`)
- CUDA-capable hardware and the relevant toolchain for GPU-specific experiments

### 1. Clone the repository

```bash
git clone https://github.com/AHK811/Brain-1.git
cd Brain-1
```

### 2. Create a Python environment

```bash
python -m venv .venv
```

**Linux / WSL:**

```bash
source .venv/bin/activate
```

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install the Python package

Install PyTorch for the machine you will train on, then install this project in the same environment.

CPU:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

NVIDIA GPU on Windows or Linux, with a compatible NVIDIA driver:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[dev]"
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

Use a fresh environment when switching from a CPU-only PyTorch installation. If the CUDA check fails,
check the driver and use the wheel channel recommended by the [PyTorch installer](https://pytorch.org/get-started/locally/).
PyTorch training needs neither a separate CUDA toolkit nor Triton, Mojo, Rust, or `just`.

Optional extras are declared in `pyproject.toml`:

```bash
python -m pip install -e ".[gpu]"   # Triton dependencies
python -m pip install -e ".[data]"  # Hugging Face Hub tooling
```

### 4. Inspect available tasks

```bash
just --list
```

### 5. Run the reference checks and benchmark workflow

```bash
just golden -- --model --raw
just test
just bench-rmsnorm
just table
```

These commands exercise the reference/golden workflow and the benchmark harness. They are not evidence that a complete pretraining run has been performed.

Earlier verification recorded **85 passed, 7 skipped** on this host and **51 passed, 19 skipped** on a fresh
clone without golden tensors. Run the suite for current counts. GPU, external-binary and tokenizer checks
need their respective runtime or fixtures; missing golden-tensor checks explain how to generate them.

### CPU and NVIDIA laptop training

Both devices use the same model, packed data and checkpoint format. After preparing data, a conservative
starting point for a roughly 15 GB GPU is one sequence per micro-batch with gradient checkpointing:

```bash
python -m brainlab.train.train --device cuda --dtype auto --micro-batch 1 --accum 32 --seq-len 1024 --grad-ckpt --eval-batches 8 --save-every 50 --out checkpoints/run1
```

The matching CPU command keeps the same effective batch and data cursor:

```bash
python -m brainlab.train.train --device cpu --dtype fp32 --micro-batch 1 --accum 32 --seq-len 1024 --eval-batches 8 --save-every 50 --out checkpoints/run1
```

`--device auto` chooses CUDA when available, otherwise CPU. Explicit CUDA fails if unavailable rather than
silently falling back. `--dtype auto` uses FP32 on CPU, native BF16 on supported CUDA devices, otherwise FP16
with GradScaler. Master parameters and optimizer updates stay FP32; `--dtype fp16` preserves the original
FP16 recipe. No memory fit or GPU speed has been measured on this host. Inspect `mem_gb` in the training log
and test a short run in a separate output directory before committing to the full schedule.

For a quick CPU smoke test, use `--steps 4 --seq-len 32 --micro-batch 1 --accum 1 --eval-batches 1`
and a separate output directory. That is not the full training recipe. Training data must have at least
one micro-batch and validation must hold two disjoint evaluation windows.

To move an ongoing run, copy its packed data, tokenizer, run settings and completed `.pt` checkpoint to the
other machine; add `--resume path/to/step-000050.pt` to the same command with the new device. Only load
trusted checkpoints. Keep sequence length, micro-batch, accumulation, seed, evaluation windows and planned
total steps unchanged; new checkpoints reject mismatched settings. Paths may change, but data/tokenizer
contents must not. Cross-device or precision changes are not bitwise-reproducible. The scaler is restored
when applicable, or initialized when entering FP16 from a non-FP16 run. Automatic resume chooses the newest
numbered checkpoint or `last.pt` by step, never `best.pt`. A forced shutdown can lose work since the last save.

### Hosted notebook environments

The [`Kaggle training notebook`](notebooks/kaggle_train.ipynb) provides CPU/CUDA selection, an explicit corpus
or synthetic smoke route, a short probe, training, checkpoint restore, export and evaluation. Enable internet
for clone/install and select an accelerator when using `DEVICE_REQUEST="cuda"`. Synthetic byte-tokenized data
is only a pipeline check, not meaningful language-model training. To import a command-line run, set
`RESUME_FROM`, `PACKED_DIR` and the original tokenizer/settings in its configuration cell. Saved settings are
checked; the first import cannot retrospectively verify data/tokenizer hashes. Subsequent notebook resumes
check the recorded content hashes. Older checkpoints lacking required run settings are rejected.

Hosted setup uses `scripts/setup_colab.sh` and preserves a working preinstalled PyTorch installation.
Set `BRAINLAB_DEVICE=cuda` to require CUDA, or `cpu` for CPU-only operation; required setup failures stop the run.
`BRAINLAB_PYTHON` selects the interpreter explicitly. `scripts/check_gpu.sh` writes a JSON capability report
and exits nonzero without usable CUDA; that does not prevent CPU training.

Keep active checkpoints on local runtime storage and copy completed saves to persistent storage. Colab
runtime storage is disposable, so restore from a Drive backup after a reset. Kaggle outputs must be saved
and explicitly attached to a later session before restoring. Neither platform guarantees automatic recovery.

### Before a long run

A parity-green tree is not evidence that the model learns. In order:

1. Fresh clone, `pip install -e .`, `just golden`, `just test` — the suite must pass with nothing carried over.
2. `just prepare` on the real corpus, then read `prepare_report.json` for `val_fraction` and `doc_sep`, and run `brainlab leak` against the packed splits.
3. `just overfit`: one batch of data, many steps, loss should approach 0. If it plateaus near `ln(vocab_size)`, the model, the optimizer, or the loss has a bug no parity test can see.
4. One short GPU training run with `--dtype auto`; use `python -m brainlab.eval.perplexity --amp` for a separate FP16 whole-model evaluation comparison, not an isolated head-precision test.
5. `just wsl-resume`: kill a run mid-flight, resume it, and require the loss curve to match on the same device and settings.
6. Only then the full run, ending with `python -m brainlab.eval.perplexity` over the whole validation set.

### Rust on Windows

The repository's documented native Rust verification lane is WSL2/Linux. The current Windows setup has toolchain/linker constraints; use WSL2 for the Rust workspace checks described by the `justfile`:

```bash
just wsl-check
just wsl-test
just wsl-bench
just table-rust
just wsl-verify
just wsl-clone      # a clean clone must import, and no ignore rule may hide source
just wsl-cli        # the exit codes the docs promise (leak fails, ppl helps) really happen
just wsl-resume     # kill a run mid-flight, resume it, and require the loss curve to match
```

## Training configuration

The checked-in configuration describes the current experiment recipe:

| Setting | Configured value |
|---|---:|
| Sequence length | 1,024 |
| Micro-batch size | 8 sequences |
| Gradient accumulation | 4 steps |
| Tokens per optimizer step | 32,768 |
| Total training steps | 8,000 |
| Configured token budget | 262,144,000 tokens |
| Warm-up | 200 steps |
| Schedule | Cosine decay |
| Compute precision | FP16 autocast |
| Master weights | FP32 |
| Optimizers | Muon + AdamW parameter groups |
| Gradient clipping | 1.0 |

These values describe the configuration—not a verified training result. The repository does not currently publish a validated pretrained checkpoint. Before a full run, verify dataset acquisition, official train/validation split handling, tokenizer/model vocabulary agreement, shard freshness, and resume reproducibility. See the specification files and engineering notes before changing the pipeline.

## Correctness before speed

A benchmark is useful only when the implementations are doing equivalent work. Brain-1 uses a reference-driven process:

1. The PyTorch reference produces deterministic inputs and expected outputs for defined workloads.
2. Those inputs are exported in the formats required by the relevant backends.
3. Each backend runs the same workload with the same declared dtype and shape.
4. The harness—not the backend—compares the output against the reference using workload-specific absolute, relative, and relative-L2 tolerances.
5. Timing is collected separately from correctness, with warm-up, repeated measurements, hardware metadata, and distribution statistics.

The goal is to prevent a fast but numerically incorrect implementation from being presented as an optimization.

### Recorded host results

The experiment log documents the following FP32 RMSNorm measurement for a `4096 × 512` workload on one WSL2 host (4 vCPU, AVX2 + FMA; 40 repetitions; median):

| Backend | Parity / relative L2 | Throughput |
|---|---:|---:|
| Rust CPU | Pass / `5.12e-08` | **16.3 GB/s** |
| PyTorch compile | Pass / `5.92e-08` | 11.8 GB/s |
| PyTorch eager | Reference implementation | 1.4 GB/s |

These are historical, single-host measurements—not universal performance claims. PyTorch eager defines the reference for this comparison, so its zero reference error is self-consistency rather than independent corroboration. GPU measurements remain pending in the documented environment. Reproduce results on your own hardware before drawing performance conclusions.

See [`docs/experiment_log.md`](docs/experiment_log.md) for methodology, hardware details, additional workloads, and caveats. See [`docs/decisions.md`](docs/decisions.md) for the reasoning behind implementation choices.

## Project status

<details>
<summary><strong>Current maturity and limitations</strong></summary>

- This is an experimental research repository. Interfaces and implementation details may change.
- The project does not claim production readiness, instruction-following quality, or assistant-level capabilities.
- The committed configuration is not proof of a completed long-duration pretraining run.
- No validated pretrained checkpoint is currently published in the repository.
- CUDA, Triton, and Mojo work is at the kernel/experiment level; their performance must be measured on suitable hardware.
- Dataset and tokenizer compatibility must be checked before training. Do not assume an arbitrary corpus can be passed directly into the current pipeline.
- Benchmark numbers must always be read with their hardware, dtype, workload, and parity result.

</details>

## Roadmap

The next engineering priorities are:

- Harden dataset ingestion, including Parquet-capable sources and preservation of official dataset splits.
- Enforce exact tokenizer/model vocabulary compatibility before training or generation.
- Make data preparation idempotent and verify shard manifests and checksums.
- ~~Strengthen checkpoint resume so optimizer, RNG, data-order, and best-validation state are reproducible.~~ Done: the checkpoint carries the data cursor and best score, the config's peak LR wins over the saved one, and `just wsl-resume` requires a killed-and-resumed loss curve to match an uninterrupted run.
- Add a clean end-to-end smoke test: data → tokenizer → packed shards → forward/backward → checkpoint → resume → generation.
- Run and publish GPU parity and performance measurements on a documented device.
- Expand benchmark coverage only when the workload and correctness contract are explicit.

## Contributing

Contributions are welcome once they preserve the project's central rule: **correctness must be established before performance is reported.**

For a useful issue or pull request, include:

- The exact commit, operating system, hardware, and toolchain versions.
- The command used to reproduce the behavior.
- Expected and actual behavior, including logs or a minimal test where possible.
- For performance work, workload shape, dtype, warm-up, repetitions, median, and parity result.

Useful entry points:

- [Open an issue](https://github.com/AHK811/Brain-1/issues)
- [Open a pull request](https://github.com/AHK811/Brain-1/pulls)
- [View contributors](https://github.com/AHK811/Brain-1/graphs/contributors)

### Contributors

The contributor panel below is generated from the repository's GitHub contribution history and updates independently of this README.

<p align="center">
  <a href="https://github.com/AHK811/Brain-1/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=AHK811/Brain-1" alt="Brain-1 contributors" />
  </a>
</p>

## Repository activity and visitors

The badges at the top use a third-party counter for this repository. The linked counter page provides total and unique hit counts for the project badge.

**Visitor privacy:** these counters provide aggregate counts. They do not identify GitHub accounts or reveal the names of people viewing the repository. For GitHub's own repository traffic analytics, open **Insights → Traffic**; GitHub's traffic API provides page-view totals and unique-view counts for the most recent 14-day window, not a list of visitor identities.

- [Open repository traffic insights](https://github.com/AHK811/Brain-1/graphs/traffic)
- [Open the external hit-counter dashboard](https://hits.dwyl.com/AHK811/Brain-1)


---


<div align="center">

**Brain-1** · Research, implementation, and measurement.

[GitHub repository](https://github.com/AHK811/Brain-1) · [Issues](https://github.com/AHK811/Brain-1/issues) · [Discussions](https://github.com/AHK811/Brain-1/discussions)

</div>
