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

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

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

The current suite is **85 passed, 7 skipped** on this host. Every skip is a GPU row, a Linux-only Rust
binary, or a rust-trained tokenizer cross-check that needs `BRAINLAB_TOKENIZER_DIR`. On a fresh clone with
no golden tensors it is 51 passed, 19 skipped — the tensor suites explain how to generate them instead of
failing.

### GPU notebook environments

For a disposable Colab or Kaggle environment, the repository includes setup and GPU-check scripts:

```bash
!git clone https://github.com/AHK811/Brain-1.git /content/Brain-1
%cd /content/Brain-1
!bash scripts/setup_colab.sh
!bash scripts/check_gpu.sh
!just bench-rmsnorm --tag t4
```

GPU commands require a compatible runtime. A CPU-only environment cannot produce meaningful CUDA, Triton, or GPU-throughput results.

`setup_colab.sh` installs the pinned Rust toolchain but not `just`, so the `just ...` recipes above need
`cargo install just` first — or call the module directly, which is what every recipe wraps:

```bash
!cargo install just                                # once per container
!.venv/bin/python -m brainlab.eval.perplexity --model checkpoints/brainlab-33m.safetensors --amp
!brainlab bench --tag t4                           # same harness, no just required
```

`--out` defaults to `checkpoints/`, which lives inside the cloned directory and is deleted when the runtime
detaches. Point training at mounted storage instead, or a long run ends when the cell does:

```bash
!just train --out /content/drive/MyDrive/brainlab-ckpt --steps 8000   # Colab
!just train --out /kaggle/working/brainlab-ckpt --steps 8000          # Kaggle
```

Resume is safe from that same directory: the checkpoint carries the data cursor and the best score, so a
disconnect continues the run instead of replaying its first batches (see `docs/decisions.md`, D028). A
step-by-step Kaggle notebook ships as [`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb).

Kaggle notebooks ship with **Internet off**, which makes the clone cell fail with `git` exit 128 and no
message, because `%%bash` captures stderr and discards it on failure. Turn Internet on for the clone and the
install; after that the notebook needs no network, since the corpus route reads attached Kaggle Datasets of
plain text rather than downloading from Hugging Face.

### Before a long run

A parity-green tree is not evidence that the model learns. In order:

1. Fresh clone, `pip install -e .`, `just golden`, `just test` — the suite must pass with nothing carried over.
2. `just prepare` on the real corpus, then read `prepare_report.json` for `val_fraction` and `doc_sep`, and run `brainlab leak` against the packed splits.
3. `just overfit`: one batch of data, many steps, loss should approach 0. If it plateaus near `ln(vocab_size)`, the model, the optimizer, or the loss has a bug no parity test can see.
4. One short run on the GPU with `--amp`: `brainlab-ppl --amp` prints the logits dtype, which is how to check whether the tied output head really runs in fp16.
5. `just wsl-resume`: kill a run mid-flight, resume it, and require the loss curve to match.
6. Only then the full run, ending with `brainlab-ppl` over the whole validation set.

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
