# Experiment log

Numbers are only meaningful with the hardware and the seed that produced them. Every entry states both,
and every published speed number is paired with the parity result that made it worth reading.

---

## 2026-10-26 - the spec, the reference and the first real bugs

**Host:** Windows 11 (10.0.26200), x64, no usable CUDA device on this machine.
**Seed:** schedule seed 0, golden seed 1234. **Torch:** see `tests/golden/index.json → producer`.

Wrote `spec/` first and kept no code there: `model_config.toml` (33,300,992 parameters, validated against
the declared budget of 33,000,000 ± 2%), `weights_format.md`, `tokenizer_spec.md`, and later
`datapack_format.md` once the packer had two implementations to agree.

The Python reference trains on CPU. An 8-step smoke run over 8,192 synthetic tokens with Muon + AdamW
reached loss 5.253935 and exported `checkpoints-smoke/brainlab-33m.safetensors`. Nothing about that is a
result; it is the plumbing check that the export, eval and generate paths all load the same file.

**Bugs found by writing the tests first:**

- `TokenShards` computed each window's `+1` target inside a single shard, so the last sequence of every
  shard overflowed and the batch became ragged. Fixed by gathering tokens across shard boundaries with one
  flat index space (`slice_tokens`), rather than dropping the window.
- `split_digit_run` had no `start += 3` and spun forever. Caught by a test that timed out, not by review.
- `generate` crashed on `token id 12778 outside vocab 258`. Not a tokenizer bug: a 16384-wide head decoded
  against a char-level smoke vocab. The fix is honest reporting - `decode_ids(strict=False)` appends
  `[N undecodable ids]` and the CLI warns, because silently replacing them with `?` would hide exactly the
  mismatch this repo is supposed to surface.
- Warmup on an 8-step run demanded more steps than existed; the schedule now clamps warmup to
  `min(configured, max(1, total/4))` and logs when it does.

## 2026-10-26 - Rust: parity first, and the toolchain is the actual obstacle

**Host for Rust:** WSL2 Linux, `rustc` from `rust-toolchain.toml`, `CARGO_TARGET_DIR` on ext4.
Windows cannot link Rust here: no MSVC, and the bundled GNU `dlltool` fails without `as.exe`.

`cargo check --workspace --all-targets` eventually came back clean; the list below is what it took, because
each of these is a way for a Rust and a PyTorch implementation to quietly disagree.

- **Online-softmax accumulation.** The new-maximum branch multiplied the running accumulator by the
  correction factor but never re-added the current row's values, so attention output was wrong only when a
  larger score arrived late. Caught by the fp32 attention golden, not by inspection.
- **SiLU.** Written as the tanh approximation. It is not: the spec says `x / (1 + exp(-x))`. The
  approximation is inside fp32 tolerance for small inputs and quietly wrong for the tails.
- **Prefill never populated the KV cache**, so decode read zeros. Fixed by wiring `store_kv` into
  `forward_cached`; the first fix still left two bugs (below).

**Parity against the PyTorch goldens (fp32, `cargo test --workspace --release`):**

| test | result |
|---|---|
| `rmsnorm_matches_golden` | passes on all 8 rmsnorm workloads, `max_abs_err = 0` on fp32 |
| `softmax_matches_golden` | passes |
| `attention_matches_golden` | passes on 3 shapes |
| `matmul_matches_golden` | passes on 3 shapes |
| `forward_matches_golden` | passes on 1x512, 1x1024, 4x1024 |
| `rope_matches_closed_form`, `transpose_roundtrips`, `sumsq_matches_naive`, `linear_equals_matmul` | pass |

**Two golden-format bugs the Rust loader exposed:**

1. `make_golden` left fp32 operands as numpy **float64**, so `rmsnorm_fp32_*.safetensors` stored `F64` and
   the reference had been computed in float64. Rust rejected the file, which was the correct behaviour, and
   the real bug was on the producing side. Now operands are cast to the workload dtype before saving, and
   `test_every_golden_tensor_uses_its_declared_dtype` fails the build if it drifts back.
2. `index.json` recorded the matmul output shape as `(m, k, n)` - the workload's dims - instead of
   `(m, n)`. The Rust loader validated shapes against the index and refused. Same class of bug: metadata
   that no single implementation reads.

## 2026-10-26 - the KV cache bug that only a decode-vs-forward test can find

Full-forward parity passed while cached decode was broken, because a full forward never reads the cache.
Two separate defects:

- the decode store wrote each head's row at `layer*ctx*d + kv_len*d + head*ctx*hd`, mixing a `d` stride
  into an `hd`-strided plane. With 8 heads it walked off the end of the allocation and panicked
  (`range start index 4197888 out of range for slice of length 4194304`).
- **both** store helpers advanced `Engine::kv_len` themselves, once per layer. Layer 1 therefore wrote one
  row past layer 0, layer 2 two rows past, and `decode`'s attention read a growing wrong length. The
  self-test reported `worst_decode_abs_error = 3.018` against a prefill that matched perfectly - the
  signature of a cache-layout bug rather than a math bug.

Fix: the write row is an explicit argument; `kv_len` is assigned exactly once per token, after the layer
loop. Regression cover: `cached_decode_matches_full_forward` in `cargo test` (asserts the last-row logits of
a cached decode equal a fresh full forward within 1e-2 for every step, and that `kv_len()` advances by one),
plus `decode_matches_full_forward` in `brainlab self-test`.

## 2026-10-26 - harness integrity findings

- The Mojo driver takes no `--dtype`; handing it an `f16` raw file and reading it as `f32` would produce a
  confident, meaningless number. Native fp32-only kernels are now filtered by `backend.dtypes` in the
  harness, so fp16 rows are skipped rather than mislabelled.
- The same reasoning restricts `rust_cpu` timing rows to fp32: the Rust engine widens fp16 goldens to fp32,
  so its traffic is 4 bytes/element while `traffic_bytes` for an fp16 workload says 2. GB/s would be
  overstated by 2x. fp16 *parity* for Rust is still checked, by `cargo test`, where no bandwidth claim is
  made.
- `report.py` used to pick up any `*.json` in `bench/results`, including `env.json` written by the setup
  script. It now excludes the non-result files explicitly.
- The harness, not the backend, decides parity: `worst_element_over_budget = max(|a-b| - (atol + rtol|b|))`
  and relative-L2 against the workload's budget. `test_parity_verdict_belongs_to_the_harness_not_the_backend`
  pins that behaviour, including that NaN output fails and that a shape mismatch is reported as such.

## 2026-10-26 - tokenizer parity across languages without a shared runtime

The Python reference writes `tests/golden/tokenizer_parity.json` (vocab 272, 14 seeded merges, 16 documents,
including the adversarial set: intralingual hyphens, `1234567`, CJK, combining-vs-precomposed accents,
44-character runs, tabs and CRLF). `cargo test -p brainlab-tokenizer` rebuilds the same vocabulary from hex
and must reproduce pre-tokenization, 32-byte word chunking, token ids and the decode roundtrip.

Two rules had to be written into the spec because the implementations disagreed until they were:
word types are indexed in **lexicographic byte order** (not first appearance), and digit-run splitting
applies to **ASCII digits only** (Unicode `Nd` tables differ between runtimes, and agreement beats coverage).

`pre_tokenize("state-of-the-art x; y")` yields `["state-of-the-art", " ", "x", ";", " ", "y"]` in both
implementations; an assertion that whitespace could glue onto preceding punctuation was wrong and was
removed rather than the code being bent to match it.

## 2026-10-26 - FMA in the dot product: 3.2x on compute-bound, nothing on bandwidth-bound

**Host:** WSL2, 4 vCPU with AVX2+FMA, `--threads 4`, 60 reps / 12 warmup, median. The reference is the
PyTorch fp32 goldens; every row passed parity before and after.

| workload | before (mul+add) | after (FMA) |
|---|---|---|
| matmul_fp32_1024x512x1536 | 164.49 ms, 9.79 GFLOP/s | 51.98 ms, 30.99 GFLOP/s |
| matmul_fp32_512x1344x512 | 104.37 ms, 6.75 GFLOP/s | 30.13 ms, 23.38 GFLOP/s |
| attention_fp32_2x8x512x64 | 133.49 ms, 4.03 GFLOP/s | 41.36 ms, 13.01 GFLOP/s |
| attention_fp32_4x8x256x64 | 70.50 ms, 3.82 GFLOP/s | 21.02 ms, 12.82 GFLOP/s |
| softmax_fp32_4096x512 | 71.77 ms | 43.25 ms |
| rmsnorm_fp32_4096x512 | 1.083 ms, 15.5 GB/s | 1.063 ms, 15.8 GB/s |

The attention rows originally read 8.04 / 25.96 and 7.61 / 25.54 GFLOP/s. They were restated on 2026-11-08
because `Workload.flops` counted the whole `s x s` score matrix; `attention_causal` loops
`for source in 0..=target`, so it only ever does the triangle. The millisecond columns are the original
measurements and the 3.2x factor is unchanged.

The change is `_mm256_fmadd_ps` in `matmul::dot` behind `is_x86_feature_detected!("fma")`, with the AVX-only
path kept as a fallback. It was worth making because 9.8 GFLOP/s on four AVX2 cores is a few percent of what
they can issue: `mul` then `add` doubles the instruction count per term and rounds twice.

The finding that matters more than the speedup: **rmsnorm did not move.** It is bandwidth-bound, so cheaper
arithmetic cannot help it; its only levers are passes-over-the-row and thread count. Saying "FMA made our
kernels up to 3.5x faster" without adding that the headline kernel gained 2% would be exactly the kind of
number this repo exists to refuse.

Parity was re-verified before keeping the change: `cargo test --workspace --release` still 17/17, with
`max_abs_err` unchanged in kind (1.9e-06 to 2.9e-06 against matmul's atol of 1e-03).

## 2026-10-26 - first-week RMSNorm table, one host

**Host:** WSL2 Ubuntu on the Windows machine - 4 vCPU, 7 GB RAM, AVX2+FMA, python 3.14.4,
torch 2.14.1+cpu, rust 1.99.1. Harness `bench/harness/run_all.py`, `--reps 40 --warmup 8`, threads 4 for
both torch (`OMP_NUM_THREADS`) and Rust (`--threads 4`), median of the per-rep distribution. Run tag
`wsl-single-host`, file `bench/results/20261026T144731Z_rmsnorm.json`. All 12 rows passed parity.

| shape | backend | parity | max_abs_err | rel_l2 | ms | GB/s |
|---|---|---|---|---|---|---|
| 4096x512 | rust_cpu | PASS | 1.91e-06 | 5.12e-08 | 1.0306 | 16.3 |
| 4096x512 | pytorch_compile | PASS | 1.91e-06 | 5.92e-08 | 1.4236 | 11.8 |
| 4096x512 | pytorch_eager | PASS | 0 | 0 | 12.4282 | 1.4 |
| 1024x512 | rust_cpu | PASS | 1.91e-06 | 5.13e-08 | 0.1676 | 25.0 |
| 1024x512 | pytorch_compile | PASS | 1.91e-06 | 5.89e-08 | 0.3589 | 11.7 |
| 1024x512 | pytorch_eager | PASS | 0 | 0 | 0.9686 | 4.3 |
| 2000x384 (non-power-of-two) | rust_cpu | PASS | 1.91e-06 | 4.99e-08 | 0.3470 | 17.7 |
| 2000x384 (non-power-of-two) | pytorch_compile | PASS | 1.91e-06 | 5.96e-08 | 0.8714 | 7.1 |
| 2000x384 (non-power-of-two) | pytorch_eager | PASS | 0 | 0 | 1.6914 | 3.6 |
| 1x512 | rust_cpu | PASS | 0 | 0 | 0.0002 | 30.7 |
| 1x512 | pytorch_eager | PASS | 0 | 0 | 0.0478 | 0.1 |
| 1x512 | pytorch_compile | PASS | 0 | 0 | 0.0510 | 0.1 |

`% of peak` is empty on purpose: this is a 4-vCPU VM and no memory-bandwidth ceiling was measured on it, so
any percentage would be an invented denominator (D016).

**What the table actually says.**

- `pytorch_eager` cannot disagree with the goldens - it *generated* them - so its `rel_l2 = 0` is
  self-consistency, not corroboration. The independent cross-language signal is `rust_cpu` at
  `rel_l2 ~ 5.1e-08` and `torch.compile` at `5.9e-08`. Both differ from the reference by roughly one ULP on
  one element (`max_abs_err` 1.91e-06 at magnitude ~16), which is what fp32 reassociation looks like.
- One genuinely reassuring accident: the goldens were written on Windows by **torch 2.13.0+cpu / numpy
  2.5.2**, and eager on WSL reported `rel_l2 = 0` and `max_abs_err = 0` for every rmsnorm shape while
  running **torch 2.14.1+cpu / numpy 2.5.4** on a different OS and a different core count. The fp32 eager
  reference is stable across those minor versions, which is what makes a committed `index.json` tolerance
  set trustworthy. It is not a substitute for regenerating goldens after a major bump.
- Eager loses by 8.7x against `compile` and 12x against Rust on `4096x512` because it materializes about six
  passes over the row (`x*x`, mean, `+eps`, `rsqrt`, two multiplies), each allocating 8 MB. That is a
  memory-traffic story, not an arithmetic story, which is consistent with FMA changing nothing here.
- Rust beats `torch.compile` by 1.4x on the large shapes and 2.1x on `2000x384`. The non-power-of-two shape
  is the interesting one: both implementations mask the tail, but Triton-style pow2 blocking is where a
  less careful kernel would fall off a cliff, so this shape stays in the suite permanently.
- The `1x512` row is latency-bound and its GB/s is meaningless - 0.2 ms is close to the clock resolution,
  and 6 KB of traffic fits in L1 several times over. It is kept because a one-token decode step is a real
  serving shape, not because of its bandwidth.
- Run-to-run noise on this VM is not negligible: eager measured 1.42 ms then 0.97 ms for `1024x512` across
  two invocations (a 1.46x spread). Medians of 40 reps inside one run are stable; comparisons *between*
  runs on a shared VM are not. Any published number therefore comes from a single run that contains every
  backend it compares.

**Still empty:** Triton, CUDA and Mojo rows. They are written and their tests are in the suite, but this
host has no CUDA device, so filling those cells requires a Colab or Kaggle T4 - which is also where
`% of peak` becomes a real column, against T4's 320 GB/s.

## 2026-10-26 - dispersion is the real limit on this VM, and softmax is the weak kernel

**Host:** same WSL2 VM. Every native row keeps `ms_min`, `ms_median`, `ms_p90`, `ms_max` from 60 reps.

| key | median ms | p90/median | max/median |
|---|---|---|---|
| rmsnorm_fp32_2000x384 | 0.373 | 3.16 | 7.32 |
| softmax_fp32_4096x512 | 29.53 | 2.67 | 4.25 |
| rmsnorm_fp32_1024x512 | 0.204 | 1.40 | 10.19 |
| matmul_fp32_1024x512x1536 | 70.49 | 1.63 | 2.63 |
| forward_fp32_4x1024 | 13983 | 1.03 | 1.03 |

Short kernels are noisy here and long ones are not, which is the expected relationship when interference is
fixed-size and the workload is not. Consequences, applied:

- A published number comes from one run that contains every backend it compares. Cross-run comparisons on
  this VM are not valid: `rmsnorm_fp32_4096x512` measured 1.03 ms inside the single-host harness run and
  1.77 ms in a standalone native run of the same binary - a 1.7x spread with identical parity.
- `rmsnorm_fp32_1x512` has `max/median` of 89. Its 0.2 ms median is at the resolution of the clock, so its
  GB/s is reported but not interpreted.
- `p90/median > 2` on `2000x384` and `softmax` is a warning label, not a result. On a T4 the same columns
  are the reason to take the median rather than the mean.

**The weakest kernel in the crate is softmax.** `kernels::softmax_rows` walks rows sequentially (there is no
`softmax_rows_par`, unlike rmsnorm) and calls scalar `exp()` per element. `softmax_fp32_4096x512` reaches
only 0.57 GB/s while rmsnorm reaches 9-20 GB/s. 2M libm exponentials at ~20 cycles each is about 17.9 ms at
this clock, which matches the observed `ms_min` - so the kernel is exp-limited, not memory-limited, and two
independent fixes apply: parallelize the rows and use a vectorized exponential. Both must be re-verified
against the golden, because a vectorized `exp` changes rounding.

## 2026-10-26 - the whole 33M model in Rust, against the PyTorch reference


**Host:** same WSL2 VM, 4 rayon threads, `brainlab forward --model tests/golden/model.safetensors`.
Each row is one full forward over the whole batch, timed as the median of the reps.

| shape | parity | max_abs_err | rel_l2 | ms | tokens/s |
|---|---|---|---|---|---|
| 1x512 | PASS | 3.81e-06 | 1.04e-06 | 2324.4 | 220.3 |
| 1x1024 | PASS | 4.23e-06 | 1.08e-06 | 4608.0 | 222.2 |
| 4x1024 | PASS | 4.62e-06 | 1.11e-06 | 13983.3 | 292.9 |

Parity here means the loaded `safetensors` file, the fused-QKV unpacking, split-half RoPE, SwiGLU, the
residual stream and the tied output projection all agree with PyTorch on 16384-wide logits over up to 4096
positions - `rel_l2` around one part in a million, with the workload's atol of 1e-03. Two notes:

- Batching 4 sequences raises throughput 33% (220 to 293 tokens/s) because each linear layer amortizes its
  weight reads over four rows. That is the whole argument for the row-blocked `linear`, and it shows up as a
  bandwidth effect rather than a compute one.
- Doubling the sequence length from 512 to 1024 roughly doubles time even though attention work quadruples
  per head, because attention is still a small fraction of total flops at `d_model=512`.
- `brainlab self-test` on the same weights: `prefill_matches_full_forward: true`,
  `decode_matches_full_forward: true`, `worst_decode_abs_error: 0.0` over 8 cached decode steps - the
  regression gate that caught the KV cache layout bug.

## 2026-10-26 - the host ceiling, then the profile that reorders the work

**Host:** same WSL2 VM. `logs/flops.json` (pure `_mm256_fmadd_ps` chains, 16 independent accumulators,
calibrated iteration count, best of 3, `--sweep 1..nproc`) and `logs/profile_forward_fp32_*.json`
(`brainlab profile`, `Instant` accumulators behind the `profile` cargo feature, 4 threads).

### The ceiling is around 55-78 GFLOP/s, not the datasheet's 319

| threads | run A | run B |
|---|---|---|
| 1 | 31.0 | 29.9 |
| 2 | 53.0 | 54.8 |
| 3 | 78.0 | 45.2 |
| 4 | 78.0 | 21.4 |

4 logical CPUs at 2496 MHz would be 319.5 GFLOP/s with two FMA units per core, or 159.7 with one. Measured
throughput is far below both, and the shape of the curve - 1→2 gives ~1.8x, then nothing at 3 and 4 - says
the WSL VM exposes **two physical cores with SMT**, so the single-thread figure (~30) is roughly what one
core does and the two-core aggregate (~55) is the usable ceiling.

The second run is also the honest footnote: `threads_4` came in at 21.4 after reporting 78.0, a 3.6x swing
on a benchmark with no I/O and no allocation in the loop. That is this VM, not the code. The probe now
interleaves thread counts across repeats and reports median/min/max per count (`worst_repeat_spread_ratio`)
so the instability is visible in the artifact instead of being hidden by a single lucky sample. Until the
repeat run settles, **no percentage of ceiling on this host is quotable** - which is also why step 5 has to
interleave backends inside one run rather than compare runs. See "% of ceiling is withdrawn" below.

### Forward pass time share, `1x1024` and `4x1024`

The table below is recomputed from the artifacts that ship in `logs/` (`profile_forward_fp32_1x1024.json`,
`profile_forward_fp32_4x1024.json`; 4 threads, 2 reps). An earlier draft quoted a different session in which
`linear_logits` took 1309.9 ms; the ranking survived but the shares did not, so the committed artifacts are
now the only source for these numbers.

| op | 1x1024 ms | share | 4x1024 ms | share |
|---|---|---|---|---|
| attention_causal | 3330.4 | 28.6% | 4394.4 | 29.6% |
| linear_logits | 1966.8 | 16.9% | 3059.0 | 20.6% |
| linear_qkv | 1536.1 | 13.2% | 1930.1 | 13.0% |
| linear_up | 1378.8 | 11.9% | 1499.6 | 10.1% |
| linear_gate | 1263.9 | 10.9% | 1499.0 | 10.1% |
| linear_down | 1016.0 | 8.7% | 1140.4 | 7.7% |
| silu_mul | 596.5 | 5.1% | 673.6 | 4.5% |
| linear_proj | 430.4 | 3.7% | 397.9 | 2.7% |
| rope + split_qkv | 58.9 | 0.5% | 116.9 | 0.8% |
| residual + KV store | 23.3 | 0.2% | 50.0 | 0.3% |
| norms, rearrange, embed gather | ~13 | 0.1% | ~47 | 0.3% |

Shares are of `forward_accounted_ms_per_run`, 11,626.4 ms and 14,824.1 ms. **The six `linear` sites total
65.3% (1x1024) and 64.3% (4x1024); causal attention is 28.6% and 29.6%; everything else is 6.1%, of which
`silu_mul` alone is 5.1%.** Linear work still comes first in both sessions, which is the finding that sets the
order of work. Where linear and attention split 76/21 in one session and 65/29 in another, the split itself is
not a measurement worth quoting twice.

Useful FLOPs at 1024 tokens are arithmetic, not measurement, so they are worth pinning down: `qkv` is
`2*1024*512*1536 = 1.611` GFLOP, `proj` 0.537, each of `gate`/`up`/`down` `2*1024*512*1344 = 1.409`, and
`logits` `2*1024*512*16384 = **17.18** GFLOP. The output projection is 17.2 GFLOP, not 34.4 - the earlier
figure counted a multiply and an accumulate as two FLOPs on top of the `2*` already in the formula. Causal
attention is `2*8*1024*1025*64 = 1.075` GFLOP per layer, the `1025` being the triangle rather than the square.
One 1x1024 forward is therefore **68.18 GFLOP of linear work and 8.60 GFLOP of attention**.

Dividing the shipped artifacts by those counts:

| block | 1x1024 | 4x1024 |
|---|---|---|
| all `linear` | 8.98 GFLOP/s | 28.63 GFLOP/s |
| `linear_logits` alone | 8.73 GFLOP/s | 22.46 GFLOP/s |
| `attention_causal` | 2.58 GFLOP/s | 7.83 GFLOP/s |

The "~26 GFLOP/s for linear at m=1024" in the previous draft is not reproducible from any artifact: the
1x1024 pass is 8.98 GFLOP/s and only the batch-4 pass reaches 28.6. Batch is the whole difference, because
four rows share one weight sweep, and it is the reason stage 2 (real batched GEMMs) is worth more than micro-
tiling a single row.

**`% of ceiling` is withdrawn from this page.** The batch-4 linear rates (26.7 GFLOP/s on `qkv`, 43.2 on
`proj`) sit at or above the 26.5 GFLOP/s that `logs/flops.json` measured in the *same* session, and that
probe reports `worst_repeat_spread_ratio = 2.6` against its own repeats while other sessions reached 78. A
denominator that moves 3x cannot turn a rate into a percentage. Going forward, quote `% of ceiling` only from
an artifact that contains its own ceiling probe measured in the same run, and only when that probe's spread is
under ~20%. Until then the GFLOP/s above are the measurement, and the only comparison that carries weight is
same shape, same thread count, interleaved rounds against PyTorch.

So the order of work is unchanged: the GEMMs first (65% of the pass), then attention (29%, at roughly a third
of the linear rate). Which is why the week-1 bandwidth table and the forward-pass priority are *different*
questions - rmsnorm is the right kernel to benchmark in isolation and the wrong one to optimise inside a
model.

### What the profile caught about itself

The decode table initially accounted for 76.7 ms of an 82.8 ms step. The missing 6.1 ms was the logits
projection: `finish()` was instrumented, but `decode` has its own inline final norm + logits `linear` and
those two calls were not. Added as `decode_rmsnorm_final` / `decode_linear_logits`, and the output now
carries `forward_accounted_ms_per_run` / `accounted_ms_per_step` beside the totals so an incomplete table
cannot be read as a complete one. The first 1x1024 profile attempt also failed honestly - `kv cache full at
1024` - because the decode phase prefilled the entire context before asking for 24 tokens; the prefix is now
`min(seq, context_len - decode_steps)` and is reported in the output as `prefill_prefix`.

---

## 2026-11-08 - a review found nine defects, six of them in what we measure rather than in the math

**Host:** Windows 11 (10.0.26200) with the WSL2 lane rebuilt from scratch: the VM's home directory had been
wiped, so `cargo`, `~/.rustup` and the python venv were all gone. `scripts/wsl_setup.sh` now only reaches for
`apt-get` when the venv tooling is genuinely missing, because it used to abort on a locked dpkg even though
python3-venv was already installed.

**A clone of the pushed repository could not train, evaluate or generate.** `.gitignore` line 24 said `data/`,
and an unanchored pattern matches that directory name at any depth, so `python/brainlab/data/` - the corpus,
prepare, pack and shard modules - was invisible to git. Everything that reads tokens imports it. The rule is
now `/data/`, the six modules are tracked, and `scripts/wsl_clone_check.sh` guards the class of bug instead of
the instance: it fails if an ignore rule hides any source file, if a module under `python/` cannot be imported
from a clean clone, or if the built wheel omits the package. Verified: 265 files in the clone, wheel carries
`brainlab/data/{__init,corpus,fetch,pack,prepare,shards}.py`.

**Three numbers in our own documents were wrong.** The output projection is 17.18 GFLOP at 1024 tokens, not
34.4. "`linear` ~26 GFLOP/s at m=1024" was 8.98 GFLOP/s in the artifact we ship, and 28.6 only at batch 4.
`Workload.flops` counted causal attention as a full `s x s` score matrix, so the two attention rows in the FMA
table were overstated by 2x (25.96 and 25.54 GFLOP/s become 13.01 and 12.82); reading `attention_causal`
settled it - the inner loop is `for source in 0..=target`, so the kernel never touches the masked half and the
restated rate is the honest one. The formulas are now pinned by `test_flop_formulas_count_useful_work`, and
every percentage against a measured ceiling has been withdrawn from this page until the ceiling probe lives in
the same artifact as the profile it divides.

**The optimizer documentation contradicted the optimizer.** D017 claimed `attn.proj` and `mlp.down` go to
AdamW; `split_params` has always put every block matrix in Muon, which is what Muon is for. The code stays and
the decision text now says so, with `tests/test_optimizer_groups.py` asserting the exact member names of each
group and that the three groups partition the model - a doc can drift, a failing test cannot be ignored.

**Muon used to swallow its own failures.** `if not bool(torch.isfinite(update).all()): continue` forced a
device sync on every matrix every step and, when it fired, silently left that parameter frozen while the rest
of the model moved. It is now a device-side counter (`nonfinite_updates`, read once per logging window), the
poisoned update is replaced with zeros, and the training loop treats any nonzero count as divergence.

**The training loop could not tell you that it had died.** Divergence was `if gnorm != gnorm`, NaN-only, inside
the `eval_every` block - so a run that went inf at step 51 kept stepping until step 250 and then saved a
checkpoint under the ordinary name. It now runs every step and reads the GradScaler: consecutive refused
gradient sets, a collapsed loss scale, or a non-finite loss/norm outside a legitimate skip, each saved as
`diverged-<step>.pt` with the reason printed. Loss is accumulated on device so a step costs one sync instead
of `accum` of them, and the run header records `git rev-parse HEAD` with `+dirty` when it does not match.

Testing that path found a failure mode none of the checks covered. `--lr 60` on the packed fixture did not blow
up: within a few steps the loss returned to `9.704061` and stayed there to the byte, `val_ppl` became exactly
`16384.000` - the vocabulary size, the uniform distribution - and `grad_norm` became exactly `0.0`. Every
number was finite, so the loop ran all 40 steps and exited 0 on a dead model. So `divergence_reason` also fires
after `max_consecutive_zero_grads` (3) steps of exactly-zero gradient norm, and the predicate is a pure
function covered by `tests/test_train_guardrails.py`, including the case that must *not* trip: a transient inf
gradient while the loss scale is still finding its level.

**`best.pt` was chosen on the same sequences we reported.** Val is now two disjoint windows: the first
`eval_batches` for loss tracking, the next for selection, and the run refuses to start if the val split cannot
supply both rather than quietly reusing one. `eval_batches` went 16 -> 64 and gained a `--eval-batches` flag
for smoke runs. Reported numbers are explicitly not from either window: the final line points at
`brainlab-ppl` over the whole val directory. `brainlab leak` (python/brainlab/data/leakage.py) hashes every
50-token span of 100 sampled val sequences against the entire train stream and exits nonzero on a hit; it also
reports source files shared between the two splits, which is the datapack stride split's own invariant.
`tests/test_leakage_check.py` plants a document in both splits and requires the checker to find it, because a
leakage check that always says "clean" is worse than none.

### Checklist verification found the measurement method was the weakest link

Re-checking each promised fix against the pushed tree surfaced a trap of its own: **exit codes measured
inline through Git Bash into `wsl.exe` read back as 0 regardless of the real status** - `python -c
"import sys; sys.exit(3)"` reported `0`, so did the leakage CLI on a run that printed `leaked: True`. The
first pass therefore "proved" the leakage gate was broken when it was the probe that was broken. Inside a
repo script run through `scripts/wsl_run.sh`, the same commands report 3 and 1 correctly.
`scripts/wsl_cli_check.sh` is now the permanent form of that check (`just wsl-cli`): a leak hit exits 1, a
span wider than a sequence exits 1 instead of reporting a vacuous clean, and the interpreter's own status
propagates. Verified against `42b5167`: 63 passed / 7 skipped, 19 cargo tests, clippy clean under
`-D warnings`, 0 warning lines in `logs/clippy.log`, no `.py` file anywhere under `python/` ignored, and the
attention entries in `tests/golden/index.json` carrying the triangle counts (537,919,488 for 2x8x512x64).
