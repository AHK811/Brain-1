# BrainLab 33M Model Audit

**Audit date:** 2026-10-26  
**Scope:** model specification, parameter/weight inventory, implementation languages, correctness evidence, CPU measurements, training evidence, and next steps.  
**Evidence:** repository files and benchmark artifacts in this workspace, plus the latest benchmark summary supplied by the project owner. Measurements from different runs are kept distinct.

## Executive assessment

BrainLab is a well-specified, cross-language transformer and inference-engine experiment. The 33.3M-parameter architecture is implemented as a PyTorch reference/training model and a Rust CPU inference engine, with CUDA, Triton, and Mojo kernel experiments. The shared specification, safetensors contract, golden comparisons, and cached-decode regression test provide a good foundation for systems research.

The Rust forward pass has reported close parity with the PyTorch reference: relative L2 error is approximately 1.04e-6 to 1.11e-6 on the three tested shapes. Its recorded throughput is approximately 220 to 293 tokens/second on a four-vCPU WSL2 host. A first profile reports 12.07 cached decode tokens/second at a 1,000-token prefill prefix, but it is a single profiling sample rather than a stable serving benchmark. These data establish correctness against the reference and provide initial Rust performance numbers. They do **not** establish that Rust is faster than Python/PyTorch for a complete forward pass: a matched full-model Python timing is absent.

The model's language quality is not established. The Rust parity benchmark uses deterministic golden weights created from seeded initialization. The separate smoke checkpoint was trained for eight steps on synthetic data over 8,192 tokens. Neither is evidence of a capable general-purpose language model. CUDA, Triton, and Mojo GPU measurements also remain unverified on this host.

## 1. Model specification

| Property | Value |
|---|---:|
| Model name | `brainlab-33m` |
| Parameters | 33,300,992 (33.30M) |
| Model family | Decoder-only causal Transformer |
| Layers | 8 |
| Hidden width (`d_model`) | 512 |
| Query heads | 8 |
| Head width | 64 |
| MLP | SwiGLU, hidden width 1,344 |
| Vocabulary | 16,384 tokens |
| Maximum context | 1,024 tokens |
| Normalization | Pre-norm RMSNorm, epsilon 1e-6 |
| Position encoding | Split-half RoPE, theta 10,000 |
| Embeddings | Input/output weights tied |
| QKV | Fused projection |
| Biases | None |
| Dropout | 0.0 in the model spec |

### Forward pass, in plain language

Tokens are converted to vectors using the embedding matrix. Each of the eight blocks normalizes the vectors, computes causal self-attention using fused QKV projections and RoPE, adds a residual connection, then applies a second normalization and SwiGLU MLP before another residual connection. A final RMSNorm is followed by an output projection that reuses the token embedding weights. Causal attention prevents each position from using future tokens.

The implementation currently uses ordinary multi-head attention with 8 key/value heads (one per query head); the spec has no grouped-query attention setting. This is a reasonable compact baseline and a clear contract for comparing implementations.

## 2. Parameter and weight inventory

| Parameter group | Count | Share |
|---|---:|---:|
| Tied token embedding / output matrix | 8,388,608 | 25.19% |
| Attention fused QKV projections (8 layers) | 6,291,456 | 18.89% |
| Attention output projections (8 layers) | 2,097,152 | 6.30% |
| SwiGLU gate, up, and down projections (8 layers) | 16,515,072 | 49.59% |
| RMSNorm scale vectors | 8,704 | 0.03% |
| **Total** | **33,300,992** | **100.00%** |

The MLP projection matrices account for about half of the parameters. The tied embedding and output matrix accounts for another quarter. Normalization parameters are a negligible fraction of the total.

### Serialized weights and precision

- The canonical weight interchange format is safetensors, with names, shapes, configuration metadata, and parameter-count validation.
- The audited model files contain 58 tensors, all stored as float32.
- A 33,300,992-parameter float32 tensor payload is 133,203,968 bytes (about 127.03 MiB). The safetensors file is about 133.21 MB / 127.04 MiB after its header and metadata.
- At two bytes per parameter, a hypothetical fp16 weight payload would be about 66.60 MB / 63.52 MiB, before file metadata. That is not the format used for the reported Rust full-model timings.
- Rust widens fp16 inputs to fp32. Therefore, Rust fp16 parity tests do not imply fp16 memory traffic or fp16 throughput.

### Two different weight sets must not be conflated

1. `tests/golden/model.safetensors` is made by constructing the model from a fixed seed and exporting its initialized weights. The Rust forward benchmark uses this deterministic golden model. It is suitable for repeatable numerical parity and timing; it is not a trained language model.
2. `checkpoints-smoke/brainlab-33m.safetensors` is a separate smoke-training export. The workspace training log records eight steps on synthetic data, 8,192 training tokens, and a final synthetic validation loss of about 4.968 (perplexity about 143.72 over 7,168 validation tokens). Those values verify the training/export path only; they are not a meaningful language-quality result.

## 3. Implementation languages and source footprint

The following counts are a snapshot of source lines in the listed implementation directories. They exclude tests, benchmark harnesses, scripts, documentation, generated files, and dependencies. The percentages describe source footprint only; they are **not** runtime, performance, or project-effort percentages.

| Language / implementation area | Files | Source lines | Share of counted lines | Current role |
|---|---:|---:|---:|---|
| Rust (`rust/crates`) | 22 | 4,490 | 56.91% | CPU inference, tokenizer, datapack tools, bindings |
| Python (`python/brainlab`) | 32 | 2,595 | 32.89% | PyTorch reference, training, data, export, eval, tokenizer |
| CUDA (`cuda`) | 1 | 301 | 3.81% | Hand-written GPU kernel experiment |
| Triton (`triton_kernels`) | 5 | 278 | 3.52% | GPU kernel experiments |
| Mojo (`mojo`) | 4 | 226 | 2.86% | Kernel experiments and raw-file driver |
| **Total** | **64** | **7,890** | **100.00%** | |

Python's model code is Python orchestration over PyTorch tensor operations; the heavy operators run in PyTorch's native backends. Rust is the only implementation with reported full-model CPU timings in the supplied results. CUDA, Triton, and Mojo are written, but their GPU rows are still empty pending a compatible GPU run.

## 4. Correctness and verification

### Reported gate status

| Check | Status from `logs/` | Interpretation |
|---|---|---|
| `cargo check --workspace --all-targets` | Clean, zero warnings | Pass |
| `cargo test --workspace --release` | 19 passed (12 infer-cpu, 7 tokenizer), 0 failed | Pass, parity against `tests/golden` |
| `cargo clippy --all-targets -- -D warnings` | clean under `-D warnings` | Pass; the profile stamp is a zero-sized `NoStamp` struct, which is what removed the 26 unit-binding findings |
| `cargo fmt --all --check` | clean | Pass |
| `pytest` | 85 passed, 7 skipped | All executed tests passed; skips are GPU rows, the Linux-only Rust binary and the rust-trained tokenizer cross-check |
| `brainlab self-test` | Prefill/decode match; worst decode absolute error 1.6e-6 | Cache correctness evidence for the tested case |
| `scripts/wsl_clone_check.sh` | 32/32 modules import from a clean clone; wheel carries `brainlab.data` | The repository is self-sufficient, not just the working tree |
| `scripts/wsl_cli_check.sh` | leak hit exits 1, vacuous span exits 1, interpreter status propagates | The exit codes the docs promise are the exit codes the tools produce |
| `scripts/wsl_resume_equiv.sh` | 8 steps uninterrupted vs SIGTERM-then-resume: worst loss difference 0.000e+00, same `best.pt` | Resume replays neither data nor schedule nor best score |
| `scripts/wsl_overfit_check.sh` | loss 9.7041 (uniform) to 0.1100 at step 10 and 0.0028 at step 40 on one batch | The model, loss and optimizer can actually learn, which parity tests alone cannot show |
| fresh clone without goldens | 0 failed, 51 passed, 19 skipped | Tensor-dependent suites skip and name `just golden` instead of raising |
| notebook syntax check (CI step) | `notebooks/kaggle_train.ipynb`: 32 cells, every non-magic cell compiles | Pass, but it is a parse gate only - it cannot tell you the run converges |
| `notebooks/kaggle_train.ipynb` on a Kaggle T4 | **not executed on any accelerator** | All 17 Python cells ran in order on this host against a temporary stand-in for `/kaggle/working`, with the accelerator cells stubbed and the batch shrunk to finish in minutes. That pass and the follow-up found six defects in the notebook itself - see D032, D033 and D034. No GPU cell has ever executed, so every T4 number it will print is still unmeasured |

`logs/commit.txt` names the commit each gate run was produced from, so a green log can be tied to a tree.

The seven PyO3 `useless_conversion` diagnostics appear to be macro-span false positives, so the allow is
scoped at the affected crate root (decision D019); the PyO3 `anyhow` feature allows idiomatic anyhow context
propagation instead of a generic helper.

### Source provenance

The workspace currently has no Git repository. `logs/commit.txt` records `no-git`, Rust/Cargo versions, and a SHA-256 code fingerprint over Rust source plus `Cargo.lock`. That fingerprint is useful as a fallback, but it omits Python, specs, benchmark harness/configuration, and repository history. Initialize Git before the final profile/gate cycle, then commit the clean source state and attach benchmark logs to that commit. Keep generated golden tensors, checkpoints, virtual environments, and bulky benchmark binaries out of source control; track source, specs, scripts, concise result JSON, and the final log files.

### Full-model Rust parity

| Workload | Max absolute error | Relative L2 error | Relative L2 as percent |
|---|---:|---:|---:|
| 1 × 512 | 3.81e-6 | 1.04e-6 | 0.000104% |
| 1 × 1,024 | 4.23e-6 | 1.08e-6 | 0.000108% |
| 4 × 1,024 | 4.62e-6 | 1.11e-6 | 0.000111% |

These are close numerical matches within the workload's tolerance. They verify the Rust forward calculation against the deterministic PyTorch reference weights and inputs; they do not measure trained-model quality.

### Lower-level native measurements

The following kernel rows are from the supplied WSL2, 4-vCPU, AVX2+FMA summary, reported as 60 repetitions / 12 warmups with median timings. All rows passed the stated parity checks. These are kernel timings, not whole-model response times.

| Workload | Max abs error | Relative L2 | Median | Reported rate |
|---|---:|---:|---:|---:|
| RMSNorm fp32 4096×512 | 1.91e-6 | 5.12e-8 | 1.77 ms | 9.5 GB/s |
| RMSNorm fp32 1024×512 | 1.91e-6 | 5.13e-8 | 0.204 ms | 20.6 GB/s |
| RMSNorm fp32 2000×384 | 1.91e-6 | 4.99e-8 | 0.373 ms | 16.5 GB/s |
| RMSNorm fp32 1×512 | 0 | 0 | 0.0002 ms | Latency-bound; bandwidth not meaningful |
| Softmax fp32 4096×512 | 5.96e-8 | 3.39e-7 | 29.5 ms | 0.57 GB/s |
| Matmul fp32 1024×512×1536 | 2.03e-6 | 2.65e-7 | 70.5 ms | 22.9 GFLOP/s |
| Matmul fp32 2048×512×512 | 2.92e-6 | 2.64e-7 | 28.8 ms | 37.2 GFLOP/s |
| Matmul fp32 512×1344×512 | 1.91e-6 | 3.00e-7 | 55.8 ms | 12.6 GFLOP/s |
| Attention fp32 2×8×512×64 | 1.19e-6 | 3.67e-7 | 41.8 ms | 12.9 GFLOP/s (causal count) |

The attention rate was 25.7 GFLOP/s in the first version of this page. `Workload.flops` counted the full
`s x s` score matrix; `attention_causal` loops `for source in 0..=target` and never computes the masked half,
so the useful-work count is `2*b*h*seq*(seq+1)*hd` and the rate halves. Decision D023 records the convention.

The owner-reported standalone RMSNorm 4096×512 timing is 1.77 ms. The experiment log also contains a prior same-harness Rust result of 1.03 ms for that shape and warns of substantial cross-run variation. Do not use these values to claim a language speedup unless the backends were measured in the same run under the same settings.

Softmax is the weakest measured kernel at about 0.57 GB/s. The supplied profiling analysis attributes this to scalar exponentials and lack of row parallelism, rather than simply to insufficient memory bandwidth. The proposed row parallelism and vectorized exponential are reasonable experiments, each gated by parity.

### Previously recorded, same-run Python/Rust RMSNorm comparison

The experiment log also records an earlier single-host run that timed Rust, PyTorch eager, and PyTorch compile together. These are **RMSNorm-only** figures from that run, not full-model measurements and not the later standalone 1.77 ms Rust run.

| Shape | Rust | PyTorch compile | PyTorch eager | Rust speedup vs compile | Rust speedup vs eager |
|---|---:|---:|---:|---:|---:|
| 4096×512 | 1.0306 ms | 1.4236 ms | 12.4282 ms | 1.38× (27.6% less time) | 12.06× (91.7% less time) |
| 1024×512 | 0.1676 ms | 0.3589 ms | 0.9686 ms | 2.14× (53.3% less time) | 5.78× (82.7% less time) |
| 2000×384 | 0.3470 ms | 0.8714 ms | 1.6914 ms | 2.51× (60.2% less time) | 4.87× (79.5% less time) |

This is useful evidence that the Rust RMSNorm kernel beat both PyTorch paths in that particular run. It does not establish that a Rust implementation of the entire model is faster, and the known run-to-run spread means the paired-run method should be retained for any refreshed claim.

## 5.1 Latest profile and machine-throughput update

### FMA-chain throughput probe

The `flops` probe runs 16 independent AVX2 FMA accumulators with calibrated iterations, interleaves thread
counts across repeats and reports median/min/max per count. The committed `logs/flops.json` records:

| Rayon threads | median | best repeat | worst repeat |
|---:|---:|---:|---:|
| 1 | 11.6 | 16.5 | 8.7 |
| 2 | 21.2 | 30.7 | 19.1 |
| 3 | 27.2 | 31.7 | 16.0 |
| 4 | 26.5 | 31.8 | 12.3 |

Its own `worst_repeat_spread_ratio` is 2.6, and `gflops_peak` in that artifact is **26.5 GFLOP/s** - not the
78 GFLOP/s an earlier session reached with the same binary. At the reported 2,496 MHz the datasheet arithmetic
gives 159.7 GFLOP/s (one FMA unit per core) or 319.5 (two), so the VM is issuing a fraction of what the ISA
allows, and which fraction depends on the session.

**Percentages of ceiling have been removed from this page** (D024). Against the 26.5 GFLOP/s ceiling measured
in this very artifact, the batch-4 `linear_proj` rate computes to 163%, which is not a hardware result - it
means the probe and the profile did not experience the same clock state. Until the ceiling probe runs inside
the same process and artifact as the profile it divides, the absolute GFLOP/s rates below are the measurement
and any percentage against them is a guess. Softmax at 0.43 GFLOP/s is memory-and-`exp()` bound and was never
usefully interpreted as a fraction of a compute ceiling anyway.

### Forward profile: batch 4, sequence 1,024

From the committed `logs/profile_forward_fp32_4x1024.json` (4 threads, 3 reps): wall 14,826.3 ms per run,
14,824.1 ms accounted for by the instrumented operations.

| Operation | ms per run | Share of instrumented op time | GFLOP/s on useful work |
|---|---:|---:|---:|
| Causal attention | 4,394.4 | 29.6% | 7.83 |
| Logits projection | 3,059.0 | 20.6% | 22.46 |
| QKV projection | 1,930.1 | 13.0% | 26.70 |
| MLP up projection | 1,499.6 | 10.1% | 30.07 |
| MLP gate projection | 1,499.0 | 10.1% | 30.09 |
| MLP down projection | 1,140.4 | 7.7% | 39.54 |
| SiLU and gate/up multiply | 673.6 | 4.5% | - |
| Attention output projection | 397.9 | 2.7% | 43.17 |
| All remaining measured ops | ~230 | ~1.6% | - |

The six `linear` sites are **64.3%** of the instrumented pass here (65.3% at 1x1024); causal attention is
29.6% (28.6% at batch 1). An earlier version of this page reported 75.5% and 21.4% from a different session;
the ordering of work did not change, but the split did, so both sessions are now stated instead of one being
presented as the answer. Amdahl's law on the current number: making every projection 1.5x faster while
attention and elementwise work hold still predicts `1 / (0.357 + 0.643 / 1.5) = 1.30x` on the whole forward,
not 1.5x.

Useful FLOPs at 1,024 tokens are arithmetic: `qkv` 1.611 GFLOP (`2*1024*512*1536`), `proj` 0.537, each of
`gate`/`up`/`down` 1.409 (`2*1024*512*1344`), `logits` **17.18** (`2*1024*512*16384` - 34.4 was an earlier
arithmetic error that applied the factor of two twice), and causal attention 1.075 per layer. Eight layers
plus the output projection is 68.18 GFLOP of linear work and 8.60 GFLOP of attention per 1x1024 forward.

`forward_batch` walks batch rows sequentially, so batch 4 issues four 1,024-row GEMMs rather than one
4,096-row GEMM; that is what lifts the linear rate from 8.98 GFLOP/s at batch 1 to 28.63 at batch 4, and it is
why stage 2 (real batched GEMMs) should be measured before more single-row tiling. Decision D021 documents
manual `Instant` timers instead of `perf` under WSL2; read these as software timing measurements, not
hardware-counter attribution.

### First cached-decode profile

The committed 1x1024 artifact prefills a 1,000-token prefix and runs 24 decode steps. It reports
**99.16 ms per generated token** (**10.1 tokens/second**), of which 87.87 ms is accounted for by the
instrumented operations. Per-step shares: decode QKV 38.7%, MLP down 17.3%, cached attention 16.9%, MLP gate
9.9%, decode logits 8.9%, MLP up 7.9%. An earlier session measured 30.89 ms/token on the same code, a 3.2x
gap that is the VM, not the engine - see the spread in the ceiling table above.

This is a bottleneck sample, not a serving number: one 24-token decode under profiling instrumentation at a
long prefix. Repeat it without instrumentation, with interleaved rounds, and report the latency distribution
across several context lengths and batch levels.

The full-forward benchmark computes logits for every position, while generation needs only the last prompt
row during prefill and one row per decode step; D022 added `Engine::prefill_last` for that, and the 16.9% /
20.6% logits share above is what it removes.

## 6. Full-model timing and response-time interpretation

### Rust CPU measurements

The `bench/results/rust/forward_fp32_*.json` rows are committed so every number here can be checked without
rerunning anything. They specify 3 repetitions and 1 warmup, which differs from the 60-rep / 12-warmup kernel
table above, so the two must not be merged.

| Workload | Median full-forward time | min - max | Time per input token | Aggregate throughput | Parity relative L2 |
|---|---:|---:|---:|---:|---:|
| 1 × 512 | 2,712 ms | 2,195 - 2,828 | 5.297 ms/token | 188.8 tokens/s | 1.09e-6 |
| 1 × 1,024 | 6,471 ms | 6,470 - 7,134 | 6.319 ms/token | 158.3 tokens/s | 1.13e-6 |
| 4 × 1,024 | 34,974 ms | 26,282 - 43,960 | 8.539 ms/token | 117.1 tokens/s | 1.18e-6 |

Parity is the reliable column: relative L2 against the PyTorch goldens stays near 1.1e-6 across shapes,
which is what the row is for. **The timing columns do not support a batching claim.** An earlier version of
this page reported 222.2 → 292.9 tokens/s from batch 1 to 4 (a "31.8% gain") on a different session; these
committed rows go the other way, 158.3 → 117.1 tokens/s, with a 26.3-44.0 s spread inside a single row. That
is a 1.35x contradiction between sessions on the same binary, so batch scaling on this VM is **unmeasured**,
not negative and not positive. Measuring it requires the interleaved, equal-thread, many-round protocol of
D020 - which also explains why stage 2 (one 4,096-row GEMM instead of four 1,024-row GEMMs) has to be timed in
the same run as its baseline.

### What these timings are—and are not

- They are full-sequence forward-pass timings for the Rust CPU engine, including producing logits for the tested sequence workload.
- They are **not** first-token latency (TTFT) for an interactive request. A serving TTFT measurement should time prompt prefill and the first output token in a defined way.
- The first profiled cached-decode sample is above: 99.16 ms/token at a 1,000-token prefix (an earlier session reported 30.89 on the same code). It is a single profiled sample under instrumentation, not a steady-state serving number.
- They are **not** p50/p95 service latency under concurrent requests.
- There is no same-host, same-precision PyTorch full-model timing in the supplied artifacts. Therefore a whole-model Rust-versus-Python speed percentage cannot be computed honestly.

## 7. What the model is strong and weak at

### Strengths

- Small and tractable enough for learning systems work and testing multiple implementations.
- A single written architecture/weight contract reduces silent differences between languages.
- Strong reported numerical parity for kernels and full forward computation.
- Cached decoding has a dedicated regression test; it caught bugs that ordinary forward parity missed.
- The project measures dispersion and refuses to publish unsupported GPU figures.
- Batch 4 shows a measurable throughput gain on the current CPU path.

### Limitations

- 33.3M parameters is a compact research model, not a frontier-scale general assistant.
- The only training result is an eight-step synthetic smoke test; there is no meaningful capability or generalization evidence.
- The context window is 1,024 tokens and the vocabulary is 16,384 tokens.
- Full-model CPU timings use only 3 reps / 1 warmup in the saved JSON, limiting confidence in their dispersion. The new profile's batch-4 run has only 2 reps, and the first decode profile is a single profiled sequence.
- Whole-model Python-vs-Rust timing, stable cached decode speed, GPU backend performance, and user-facing latency percentiles are missing.
- The measured Rust path widens fp16 weights/inputs to fp32, so it does not realize fp16 memory-traffic savings.

## 8. Prioritized next work

1. **Initialize Git and close the verification gate.** Create the repository now, keep generated artifacts ignored, then run post-edit format, Clippy, and release tests and save final logs with the resulting commit SHA. The currently saved Clippy log still shows profile-related unit-binding failures.
2. **Finish the matched full-model benchmark.** The `--rounds` interleaving and thread-equality groundwork is implemented and smoke-tested (five rounds, 1.3% MAD). Now compare PyTorch eager, PyTorch compile, and Rust on the same host, weights, fp32 precision, batch/context shapes, and round policy. Record median and p90/MAD in one run.
3. **Turn decode profiling into a stable serving benchmark.** Report prompt prefill, time to first output token, steady-state cached decode tokens/second, and p50/p95 latency across stated context and batch/concurrency levels. Add a last-position-logits path for inference while keeping all-position logits for training/parity.
4. **Improve the measured dominant paths.** Try batched matrix multiplication across batch rows and tiled/blocked linear kernels first, followed by causal attention. Re-profile and measure whole-forward and decode impact. Do not spend time tuning RMSNorm, RoPE, or residuals while they remain near 1% of this profile.
5. **Run GPU backends on real hardware.** Measure CUDA, Triton, Mojo, and PyTorch on the same GPU and inputs; preserve parity gates and include precision and GPU/driver details.
6. **Prove learning on real data.** Run a repeatable training job on a documented real dataset and report held-out loss over enough tokens/steps to make the trend meaningful. Add task evaluations and generation examples before making quality claims.

## Final audit verdict

**Engineering status:** strong foundation for a cross-language inference and kernel benchmark.  
**Numerical status:** Rust parity is convincing on the tested deterministic cases; the final post-edit verification rerun is still pending in the supplied status.  
**Performance status:** initial Rust CPU numbers and first operation/decode profiles are available; a fair full-model language comparison and stable serving-response measurements are not.  
**Model-quality status:** unknown. The current smoke training and random-initialized golden weights do not demonstrate useful LLM capability.  
**Recommended next milestone:** finish the verification gate, reconcile the profile timing note, then produce the matched Python/PyTorch-versus-Rust benchmark and a stable decode measurement. Run a real-data validation experiment as the separate model-quality milestone.
