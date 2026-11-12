# Decisions

Each entry is one decision, the alternative that was rejected, and the thing that would have to become
true for the decision to be revisited. Nothing here is a tutorial.

## D001 - One spec, many implementations

`spec/` is the source of truth and contains no code. The 33M model is built once in PyTorch and rebuilt in
Rust, Triton, CUDA and Mojo against that same description, then measured against the PyTorch reference.

**Rejected:** four partial models that each look reasonable alone. That produces four artifacts and zero
comparable numbers, because every backend quietly differs in eps, RoPE convention or dtype accumulation.

**Revisit if:** the point becomes shipping one fast model rather than comparing implementations.

## D002 - safetensors is the only weight contract

Every backend loads the same `model.safetensors`. `__metadata__` carries `config_json`, `param_count`,
`spec_version`, `source` and producer versions; a loader that finds `param_count` inconsistent with the
shapes it read fails instead of guessing (see `spec/weights_format.md`).

**Rejected:** `.pt` checkpoints shared across languages, and per-backend weight copies. `.pt` is a pickle:
unpickling untrusted files is not an option and pinning torch versions across containers is fragile.

**Revisit if:** a backend needs a fused layout (int4, transposed) - then export a derived file, never edit
the canonical one.

## D003 - Raw `.bin` + `manifest.json` bridge for Mojo and CUDA

Mojo and the CUDA bench exchange flat little-endian arrays with Python through files. Native binaries
receive paths, row counts and widths as argv and print JSON on stdout.

**Rejected:** calling Rust from Mojo per operation, or a pyo3 round-trip inside a benchmark loop. Per-op
calls across a language boundary put interpreter overhead inside the measurement and make the number
meaningless. There is no mature direct Mojo-to-Rust bridge.

**Revisit if:** a stable C ABI is needed for a serving path rather than a benchmark.

## D004 - Coarse language boundaries only

Python talks to Rust through the pyo3 extension (`brainlab_rs`) for tokenizer and packing work, and through
subprocess JSON for benchmarking. Mojo, CUDA and Triton are reached only through files and commands.

**Rejected:** a fine-grained FFI where Python calls into Rust for each kernel launch.

## D005 - The kernels directory is `triton_kernels/`, not `triton/`

A top-level `triton/` directory shadows the installed `triton` package on `sys.path`, which breaks the
real Triton import with a confusing error. The deviation from the original layout is deliberate.

## D006 - fp16 with GradScaler on T4, Muon in fp32

T4 (compute capability 7.5) has no bfloat16. Autocast is therefore fp16 with a scaler, and Muon's
Newton-Schulz orthogonalization runs in fp32 because an iterative matrix product underflows in fp16 and
there is no bf16 fallback.

**Revisit on:** A100/L4 or newer, where bf16 removes the scaler entirely and the comparison changes.

## D007 - Parity is decided by the harness, not by the backend

A backend returns tensors, timings and hardware info. It never returns a verdict. `run_all.parity_metrics`
computes `worst_element_over_budget = max(|a-b| - (atol + rtol|b|))` and relative L2 error against the
tolerances attached to the workload, and only that comparison sets `parity_pass`.

**Rejected:** each backend self-reporting accuracy - a fast wrong kernel is the exact failure mode this
repo exists to catch.

## D008 - Tolerances live in the workload table

`python/brainlab/spec/workloads.py` carries `Tol(atol, rtol, rel_l2)` per (op, dtype) and the same numbers
travel into `index.json`, so the Rust, CUDA and Mojo checkers read the budget from the golden index rather
than each hardcoding their own.

## D009 - Golden tensors are generated, not committed

A single `4096x512` fp32 triple is ~25 MB; the full set is ~540 MB. They are gitignored and regenerated
with `just golden`, seeded and deterministic, so a regenerated file is byte-comparable to the one that
produced a published number. The repo stays small on purpose.

## D010 - Reference tensors are stored in their declared dtype

`make_golden` casts operands to `f32`/`f16` before saving. An earlier version left `fp32` operands as
numpy float64, so the file named `rmsnorm_fp32_*` actually stored `F64`, the Rust loader rejected it, and
the PyTorch reference had been computed in float64 - which would have made every fp32 comparison look
better than it is. `tests/parity/test_golden_integrity.py::test_every_golden_tensor_uses_its_declared_dtype`
now fails the build if that class of drift returns.

## D011 - File-level stride split for train/val

The split is `index % round(1/val_ratio) == 0` over file paths sorted by their POSIX-relative name
(`spec/datapack_format.md`).

**Rejected:** a seeded RNG permutation. It needs bit-identical RNG semantics across numpy and Rust, and it
changes the whole split when one file is added. The stride rule is order-independent and reproducible
without shared RNG state.

## D012 - Manifests are compared parsed, not byte-wise

Rust writes `serde_json::to_string_pretty`; Python writes `indent=2, sort_keys=True`. The `.bin` files are
byte-compared (they have no formatting freedom); the JSON is compared field by field.

## D013 - Tokenizer rules that had to be written down

Word classes are whitespace / alphanumeric / other, with `-` counted as alphanumeric only between two
alphanumeric code points, so `state-of-the-art` stays one word. Digit runs split into groups of three only
for ASCII digits - Unicode `Nd` tables differ between runtimes and agreement matters more than coverage.
Word types are indexed in lexicographic byte order rather than first-appearance order so the vocab does not
depend on file read order. Each rule exists in the spec before it exists in either implementation, and the
cross-language fixture in `tests/golden/tokenizer_parity.json` is the arbiter.

## D014 - Parity fixtures cross languages through files

The Python tokenizer writes `tokenizer_parity.json`; `cargo test` reads it and rebuilds the same vocabulary
from hex tokens plus merge pairs. Rust cannot import Python and Python should not have to run Rust, so the
contract is a file that both sides must reproduce.

## D015 - Windows builds nothing; WSL is the Linux lane

The Windows toolchain cannot link Rust here: no MSVC, and the bundled GNU `dlltool` fails without `as.exe`.
`scripts/wsl_setup.sh` installs `build-essential` plus rustup, `wsl_check.sh` runs `cargo check`,
`wsl_build.sh` runs build/test/clippy, and `wsl_bench.sh` writes per-key JSON into `bench/results/rust/`,
which is on the Windows filesystem, so `just table-rust` can fold those rows into the same table as the
Python rows. GPU rows still have to come from a real GPU.

## D016 - `% of peak` is a GPU column

The memory-bound yardstick is achieved GB/s against the device's published peak (T4: 320 GB/s, overridable
with `--peak-gbs`). CPU rows report GB/s plus a hardware label (`cpu x N cores avx2`, rayon thread count)
and leave `% of peak` empty, because quoting a DDR peak that was never measured on this host would be a
made-up denominator.

## D017 - Muon and AdamW split by parameter class

Every matrix inside a block goes to Muon - `attn.qkv`, `attn.proj`, `mlp.gate`, `mlp.up`, `mlp.down` - because
orthogonalising the update is exactly what square-ish block matrices want; embeddings, the tied output
projection and the 1-D norms go to AdamW. An earlier version of this page claimed `attn_proj` and `mlp_down`
were AdamW params, which was never what the code did: `split_params` has always keyed on `blocks.` and put all
five matrices per layer in one bucket. The code was right and the text was wrong.
`tests/test_optimizer_groups.py` now pins the exact member names of each group and asserts that the three
groups partition the model, so the two cannot drift apart again.
GradScaler with two optimizers requires `unscale_` for both, then clip, then `step` for both, then one
`update`. Getting that order wrong produces inf/nan losses that look like a kernel bug.

## D018 - KV cache: one layout, and the row counter advances once per token

The cache is `[layer][head][context_len][head_dim]`, and the write row is passed to the store functions
explicitly. `Engine::kv_len` is updated **after** the layer loop, never inside it.

Two bugs lived here and both were invisible to the parity tests that only check a full forward: the store
helpers used to advance `kv_len` themselves, so layer 1 wrote one row past layer 0 and every layer after
that wrote further still; and the decode store used a `kv_len * d` stride term instead of `kv_len * hd`,
which eventually indexed past the end of the allocation and panicked. A full forward never touches the
cache, so the only test that catches either mistake is one that compares cached decode against a fresh full
forward after every step. `cargo test` now has `cached_decode_matches_full_forward`, and
`brainlab self-test` reports `decode_matches_full_forward` against a 1e-2 budget.

## D019 - `clippy -D warnings` is part of the gate, with one crate-level allow

`scripts/wsl_verify.sh` runs `cargo fmt --check`, `cargo clippy --workspace --all-targets --release --
-D warnings` and `cargo test --workspace --release` into `logs/`, and records the toolchain plus a
`code_fingerprint` (sha256 over every `.rs` file and `Cargo.lock`) in `logs/commit.txt` - there is no git
repo here, so numbers are tied to code by content hash instead of by commit.

`brainlab-pyo3` carries `#![allow(clippy::useless_conversion)]`. Every hit was on a `#[pyfunction]`
signature column, including on a function whose body contains no `?` at all, which is what the `#[pyfunction]`
macro-generated wrapper's `PyErr -> PyErr` conversion looks like from the lint's side. Verified by removing
the `?` from one function and seeing the count unchanged. Suppressed at the crate root because it is the
attribute macro, not this crate's code, that produces the pattern.

Rejected: keeping the lint alive by rewriting every `#[pyfunction]` into an inner `anyhow::Result` helper
plus a delegating wrapper. That is seven extra functions of real code to satisfy a false positive, and it
would make the actual error handling harder to read.

## D020 - matched comparisons need equal threads and interleaved rounds

`run_all.py` now sets `torch.set_num_threads(threads)` on CPU and refuses to stay quiet if
`torch.get_num_threads()` disagrees; the result document records `threads`, `torch_threads`,
`RAYON_NUM_THREADS` and `OMP_NUM_THREADS` so a reviewer can see the equality rather than assume it. The
Rust backend receives the same number through `--threads`.

`--rounds N` interleaves backends round-robin (Rust, eager, compile, repeat) and pools each backend's
per-round medians into median/min/p90/**MAD%**. MAD% is what decides whether a winner exists: a gap smaller
than the spread is reported as a tie. Row output prints `rounds=N MAD=X%` inline for that reason.

## D021 - manual op timers rather than `perf`

Per-operation timing inside the Rust engine is `std::time::Instant` accumulators behind the `profile` cargo
feature, not `perf record`. `perf` needs a kernel PMU, and under WSL2 it usually reports nothing useful;
a sampled profiler that silently returns zero is worse than a coarse timer that says what it measured. The
timers are `#[cfg]`-compiled out of the shipping build, so the default binary has no measurement overhead.

## D022 - Generation prefills logits for one row, not for the whole prompt

`Engine::prefill_last(ids, last_row)` runs the same per-layer pass with the same KV-cache writes, then
applies the final RMSNorm and the tied output projection to the **last position only**. `Engine::prefill`
(whole-prompt logits) stays, because that is what the parity tests and any training-time evaluation need.

The reason is a size check, not a hunch: at `context_len=1024`, `vocab=16384`, fp32, the discarded scratch
buffer is `(1024-1) * 16384 * 4 = 67,013,376` bytes for a single prompt - about 2x the model's own
non-embedding weights, allocated to throw away 1023 of its 1024 rows. `brainlab self-test` reports
`logits_scratch_bytes_avoided` alongside the timing of both paths so the saving is visible, not asserted.

The acceptance test is `prefill_last_row_matches_the_all_row_path`: for sequence lengths 1, 7, 64 and 257 it
requires the **KV cache to be byte-identical** to the all-row path, `kv_len()` to equal the prompt length,
the last-row logits to match both the all-row prefill and a full forward within 1e-4, and the argmax token
to be the same. Cache equality matters more than logit equality: a path that emits the right row but
poisons the cache produces a correct first token and then quietly wrong ones, which is the failure mode
that already bit this engine once (D018).

## D023 - FLOPs count useful work, and the formulas are tested

`Workload.flops` is the denominator for every GFLOP/s row in the harness, so it describes the work the
operation *requires*, not what a kernel happens to spend. For causal attention that is the triangle:
`2*b*h*seq*(seq+1)*hd`, not the square `4*b*h*seq*seq*hd`. The old formula counted the masked half, which made
attention look twice as efficient as it is - and `attention_causal` was already looping `for source in
0..=target`, so the kernel never did that work in the first place. A backend that computes and discards the
masked half will now show a lower rate, which is the correct outcome: wasted cycles belong in the numerator's
inefficiency, not in the definition of the operation.

`test_flop_formulas_count_useful_work` pins matmul, causal attention and the 17.18 GFLOP output projection
against hand-written arithmetic. Two documents had quoted 34.4 GFLOP for it by applying `2*` twice.

## D024 - no percentage of ceiling unless the probe is in the same artifact

The FMA ceiling probe on this VM has reported 26.5, 55 and 78 GFLOP/s in different sessions, and
`logs/flops.json` carries `worst_repeat_spread_ratio = 2.6` against itself. Dividing measured rates by a
denominator that moves 3x produces numbers like "108% of ceiling", which is what the shipped batch-4 profile
does if you use the ceiling from the same run. So: quote absolute GFLOP/s; quote `% of ceiling` only when the
ceiling was measured inside the same artifact and its own spread is under ~20%; and treat any claim whose gap
is smaller than the spread as unmeasured rather than as a result. This is D020's equal-threads rule applied to
the denominator instead of the comparison.

## D025 - loss tracking, model selection and reported numbers use three different val subsets

`best.pt` was being chosen on the same sequences that the tracking curve reported, so the reported curve was
already the minimum over the selection criterion. Training now reads two disjoint windows of the val stream
(`offset=0` and `offset=eval_batches*eval_batch`) and refuses to start if the val split cannot supply both,
instead of silently reusing one; `eval_batches` is 64, with `--eval-batches` for smoke runs. Neither window is
a reported number - the run ends by pointing at `brainlab-ppl`, which scores the whole val directory once.

`brainlab leak` gates the split itself: every 50-token span of 100 sampled val sequences is hashed against the
train stream, plus source files shared between splits. It exits nonzero on a hit, and
`tests/test_leakage_check.py` plants a document into both splits to prove the checker finds it. A leakage
check that only ever returns "clean" is worse than no check.

## D026 - a stalled or poisoned optimizer must be visible in the log

Divergence detection used to be `if gnorm != gnorm` inside the `eval_every` block: NaN-only, and only checked
every 250 steps, so an inf run kept burning compute and checkpointed under an ordinary filename. It now runs
every step and reads what GradScaler actually reports - consecutive steps where the scale refused to grow, a
collapsed scale, or a non-finite loss or gradient norm outside a legitimate skip - and saves
`diverged-<step>.pt` with the reason. Inf gradients during the first scaled steps are normal and are not
treated as divergence. Exactly-zero gradient norm for `max_consecutive_zero_grads` steps is divergence too:
over-training the model into the uniform distribution (ppl = vocab size) produces perfectly finite statistics,
and a check that only looks for NaN lets a dead run finish with exit code 0.

The same rule applies to Muon. `if not bool(torch.isfinite(update).all()): continue` cost a device sync per
matrix per step and, when it fired, left that parameter frozen while everything else moved: the silent worst
case. The count is now a device-side tensor (`nonfinite_updates`, read once per logging window and reported in
every train record), the poisoned update is zeroed rather than skipped, and any nonzero count is divergence.
The run header also carries `git rev-parse HEAD` with `+dirty` appended, so a published curve can be traced to
the code that drew it.

## D027 - ignore rules are checked against source, not trusted

`data/` in `.gitignore` matched `python/brainlab/data/`, so six modules that every token-reading entry point
imports were absent from the published repository: a fresh clone could not train, evaluate or generate. An
unanchored pattern is a directory-name match at any depth, which is easy to write and invisible in
`git status`. `scripts/wsl_clone_check.sh` now fails the gate if an ignore rule hides any source file, if any
module under `python/` cannot be imported from a clean clone of committed HEAD, or if the built wheel omits a
package - so this class of mistake is caught by a command rather than by a reader.

## D028 - a checkpoint has to carry the run's position, not just its weights

Resuming used to restore the model, the optimizers and the scaler, and nothing else. Three separate
consequences followed from that, all silent:

1. `data = cycle(batcher)` always started at batch 0, so every Colab or Kaggle disconnect re-trained on the
   first batches of the shuffle. A run that resumed five times had seen those batches six times.
2. `best_val` reset to `inf`, so the first post-resume eval wrote `best.pt` unconditionally - usually with a
   worse model than the one it replaced.
3. `scale_lr` stored its peak learning rate inside each param group, which is exactly the part that
   `Optimizer.state_dict()` serialises. A resumed run therefore re-adopted the old peak and ignored the config
   or `--lr` it was started with.

Now the checkpoint's `cfg` carries `best_val` and `start_batch`, `ShardBatcher.batches(start_batch)` and
`resume_cursor` place a fresh iterator where the dead one stopped, and the peaks are captured from the config
**before** `load_checkpoint` and passed explicitly to `scale_lr`.

The acceptance test is not a unit test: `scripts/wsl_resume_equiv.sh` (`just wsl-resume`) trains 8 steps
uninterrupted, then runs the same command, sends SIGTERM once `step-000004.pt` exists, resumes, and requires
every loss to match and `best.pt` to hold the same step and value. The first version of that script trained the
short leg with `--steps 4`, which changes `warmup_steps = total_steps // 4` and made the two curves differ for
a reason that had nothing to do with resume; both legs must request the same `total_steps`.

## D029 - sampling, split shape and tokenizer inputs are all explicit

`keep_top_p` subtracted the sorted **logits** from a cumulative **probability** and scattered the survivors
back into the sorted buffer instead of the original layout. At p=0.5 on a mixed-sign row it kept 5 of 8 tokens
where the nucleus is 1, and on an all-negative row it masked everything and fell back to greedy without
saying so. It now compares probabilities to probabilities, scatters into `torch.full_like(logits, -inf)`, and
rejects `p <= 0` instead of returning an all-masked row. `tests/test_generation_sampling.py` checks the kept
set against an independent nucleus implementation and asserts the surviving logits stay in their own columns.

The file-level val stride needs roughly `1 / val_ratio` files to hit its target; with 2 files and
`val_ratio=0.005` it silently produced a 50% val split. `split_files` now raises when the achieved fraction
exceeds 3x the request and says how many files the ratio needs, and `prepare` reports `val_fraction`.

Document boundaries were blank lines, hard-coded, so a paragraph-formatted corpus emitted one `<eos>` per
paragraph. `--doc-sep` now selects `blank`, `line`, `file`, or a literal separator string that is stripped.

`prepare --tokenizer-corpus DIR` refuses to pack when any val-split file appears in the directory the BPE
merges were trained on. A tokenizer trained across the split leaks vocabulary into the evaluation set, which
no parity test can see.

## D030 - one Rust pin, read from rust-toolchain.toml

`setup_colab.sh` defaulted to `1.85.0` while `rust-toolchain.toml` pinned `1.99.1`. The code uses
`usize::is_multiple_of`, which is still unstable on 1.85 (`unsigned_is_multiple_of`, tracking issue 128101),
and `rustc +1.85.0` fails with `error[E0658]` on it - verified on this machine, not inferred. Two pins that
disagree mean the Colab lane either fails to build or silently builds with a third version. The setup script
now reads the channel out of `rust-toolchain.toml`, with `BRAINLAB_RUST_TOOLCHAIN` still overriding it.

## D031 - CI runs the whole gate on a fresh checkout

`.github/workflows/ci.yml` installs CPU torch, generates the goldens, runs fmt, `clippy -D warnings`,
`cargo test --workspace --release` and `pytest`. Two reasons: the repository is a set of cross-language
parity claims, and a push from this machine cannot assume the local tree is representative - the working tree
lost its `.git` directory once during development. Tests that need golden tensors now skip with a message that
names `just golden` instead of failing with `FileNotFoundError`: the index is committed, the 540 MB of tensors
are not, so a fresh clone is a legitimate place to run the non-tensor tests.

## D032 - the Kaggle lane drives the pipeline from Python, not from `just`

`notebooks/kaggle_train.ipynb` shells out to `python -m brainlab...` for every step that needs a value computed
in an earlier cell. Three reasons, all found by running the cells rather than by reading them.

A `%%bash` cell cannot see a Python name, so `"$PY"` and `${DEVICE}` in a bash cell silently expand to empty
strings and the harness picks up whatever interpreter the container defaults to. Passing state through the
kernel's environment instead would have made the notebook depend on install order. `setup_colab.sh` also
creates its venv partway through, so the interpreter is resolved *after* the install cell, not before it.

`just` is not installed by any script in the repository, which made the Colab instructions in the README
unrunnable on a fresh container. The recipes are thin wrappers around `python -m`, so the notebook does without
and the README now says how to get `just` if you want it.

Interruption is the part worth writing down: `train.py` installs no signal handler and writes `last.pt` only
when the loop finishes. A killed run therefore resumes from the newest `step-NNNNNN.pt`, and a cell that
suggested `--resume last.pt` after a timeout would hand back a path that does not exist and restart the run
from step 0. `latest_cursor` picks the highest-numbered step file and deliberately ignores `best.pt`, whose
cursor does not describe a position in the data.
