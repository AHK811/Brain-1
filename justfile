set shell := ["bash", "-c"]
set windows-shell := ["bash", "-c"]

py := if os() == "windows" { ".venv/Scripts/python.exe" } else { ".venv/bin/python" }
cargo := "cargo"
goldendir := "tests/golden"
results := "bench/results"
dtf := "fp32"

default:
    @just --list

setup:
    python3 -m venv .venv || py -3 -m venv .venv
    {{py}} -m pip install --upgrade pip
    {{py}} -m pip install -e ".[dev]"
    rustup show
    cargo --version

setup-colab:
    bash scripts/setup_colab.sh

check-gpu:
    bash scripts/check_gpu.sh

venv:
    @echo {{py}}

golden *OPS:
    {{py}} -m brainlab.export.make_golden {{OPS}}

prepare *ARGS:
    {{py}} -m brainlab.data.prepare {{ARGS}}

overfit:
    MSYS_NO_PATHCONV=1 wsl.exe bash -c "bash /mnt/d/BRAIN-LAB/scripts/wsl_run.sh /mnt/d/BRAIN-LAB/scripts/wsl_overfit_check.sh"

build:
    cd rust && {{cargo}} build --workspace --release

test:
    {{py}} -m pytest -q

test-rust:
    cd rust && {{cargo}} test --workspace --release

test-gpu:
    {{py}} -m pytest -q -m gpu

parity:
    {{py}} -m pytest -q tests/parity

bench-rmsnorm:
    {{py}} -m bench.harness.run_all --ops rmsnorm --dtypes fp32 fp16

bench-matched OPS=rmsnorm ROUNDS=30:
    {{py}} -m bench.harness.run_all --ops {{OPS}} --dtypes fp32 --rounds {{ROUNDS}} --reps 8 --warmup 3

bench-dtype DT=fp32:
    {{py}} -m bench.harness.run_all --ops rmsnorm softmax attention matmul --dtypes {{DT}}

bench-all:
    {{py}} -m bench.harness.run_all --ops rmsnorm softmax attention matmul forward --dtypes fp32 fp16

table:
    {{py}} -m bench.harness.report --latest

table-file FILE:
    {{py}} -m bench.harness.report --file {{FILE}}

tokenizer-train CORPUS=data/corpus VAL=data/tokenizer:
    cd rust && {{cargo}} run --release -p brainlab-tokenizer --bin train-bpe -- --corpus {{CORPUS}} --out {{VAL}} --vocab-size 16384

pack CORPUS=data/corpus TOK=data/tokenizer OUT=data/packed:
    cd rust && {{cargo}} run --release -p brainlab-datapack --bin pack -- --corpus {{CORPUS}} --tokenizer {{TOK}} --out {{OUT}}

py-ext:
    cd rust && {{cargo}} build -p brainlab-pyo3 --release

cuda ARCH=sm_75:
    make -C cuda ARCH={{ARCH}}

cuda-check ARCH=sm_75:
    make -C cuda ARCH={{ARCH}} check

mojo:
    bash scripts/check_mojo.sh

data:
    {{py}} -m brainlab.data.fetch

train *ARGS:
    {{py}} -m brainlab.train.train {{ARGS}}

export SRC=checkpoints/last.pt DST=checkpoints/brainlab-33m.safetensors DT=f32:
    {{py}} -m brainlab.export.to_safetensors --checkpoint {{SRC}} --out {{DST}} --dtype {{DT}}

eval CKPT=checkpoints/brainlab-33m.safetensors:
    {{py}} -m brainlab.eval.perplexity --model {{CKPT}}

gen PROMPT="hello" CKPT=checkpoints/brainlab-33m.safetensors TOK=data/tokenizer:
    {{py}} -m brainlab.eval.generate --model {{CKPT}} --tokenizer-dir {{TOK}} --prompt "{{PROMPT}}"

rust-gen CKPT=checkpoints/brainlab-33m.safetensors TOK=data/tokenizer:
    cd rust && {{cargo}} run --release -p brainlab-infer-cpu --bin brainlab -- generate --model {{CKPT}} --tokenizer {{TOK}}

rust-info CKPT=checkpoints/brainlab-33m.safetensors:
    cd rust && {{cargo}} run --release -p brainlab-infer-cpu --bin brainlab -- info --model {{CKPT}}

wsl-check:
    MSYS_NO_PATHCONV=1 wsl.exe bash -c "BRAINLAB_ROOT=/mnt/d/BRAIN-LAB bash /mnt/d/BRAIN-LAB/scripts/wsl_check.sh"

wsl-verify *ARGS:
    MSYS_NO_PATHCONV=1 wsl.exe bash -c "bash /mnt/d/BRAIN-LAB/scripts/wsl_run.sh /mnt/d/BRAIN-LAB/scripts/wsl_verify.sh {{ARGS}}"

wsl-clone:
    MSYS_NO_PATHCONV=1 wsl.exe bash -c "bash /mnt/d/BRAIN-LAB/scripts/wsl_run.sh /mnt/d/BRAIN-LAB/scripts/wsl_clone_check.sh"

wsl-cli:
    MSYS_NO_PATHCONV=1 wsl.exe bash -c "bash /mnt/d/BRAIN-LAB/scripts/wsl_run.sh /mnt/d/BRAIN-LAB/scripts/wsl_cli_check.sh"

wsl-resume:
    MSYS_NO_PATHCONV=1 wsl.exe bash -c "bash /mnt/d/BRAIN-LAB/scripts/wsl_run.sh /mnt/d/BRAIN-LAB/scripts/wsl_resume_equiv.sh"

wsl-flops *ARGS:
    MSYS_NO_PATHCONV=1 wsl.exe bash -c "bash /mnt/d/BRAIN-LAB/scripts/wsl_run.sh /mnt/d/BRAIN-LAB/scripts/wsl_flops.sh {{ARGS}}"

wsl-profile:
    MSYS_NO_PATHCONV=1 wsl.exe bash -c "bash /mnt/d/BRAIN-LAB/scripts/wsl_run.sh /mnt/d/BRAIN-LAB/scripts/wsl_profile.sh"

profile-file FILE:
    {{py}} -m bench.harness.report --profile {{FILE}}

wsl-build:
    MSYS_NO_PATHCONV=1 wsl.exe bash -c "BRAINLAB_ROOT=/mnt/d/BRAIN-LAB bash /mnt/d/BRAIN-LAB/scripts/wsl_build.sh build"

wsl-test:
    MSYS_NO_PATHCONV=1 wsl.exe bash -c "BRAINLAB_ROOT=/mnt/d/BRAIN-LAB bash /mnt/d/BRAIN-LAB/scripts/wsl_build.sh test"

wsl-bench *ARGS:
    MSYS_NO_PATHCONV=1 wsl.exe bash -c "bash /mnt/d/BRAIN-LAB/scripts/wsl_run.sh /mnt/d/BRAIN-LAB/scripts/wsl_bench.sh {{ARGS}}"

wsl-python-setup:
    MSYS_NO_PATHCONV=1 wsl.exe bash -c "WITH_PYTHON=1 BRAINLAB_ROOT=/mnt/d/BRAIN-LAB bash /mnt/d/BRAIN-LAB/scripts/wsl_setup.sh"

wsl-harness *ARGS:
    MSYS_NO_PATHCONV=1 wsl.exe bash -c "bash /mnt/d/BRAIN-LAB/scripts/wsl_run.sh /mnt/d/BRAIN-LAB/scripts/wsl_harness.sh {{ARGS}}"

pack-rust CORPUS=data/corpus TOK=data/tokenizer OUT=data/rust-packed-smoke:
    cd rust && {{cargo}} run --release -p brainlab-datapack --bin pack -- --corpus {{CORPUS}} --tokenizer {{TOK}} --out {{OUT}} --seq-len 8 --shard-tokens 64 --val-ratio 0.5

table-rust:
    {{py}} -m bench.harness.report --merge-dir bench/results/rust

datapack-fixture:
    {{py}} -m brainlab.tok.parity_fixture --datapack tests/golden/datapack_fixture
    MSYS_NO_PATHCONV=1 wsl.exe bash -c "bash /mnt/d/BRAIN-LAB/scripts/wsl_run.sh /mnt/d/BRAIN-LAB/scripts/wsl_datapack_fixture.sh"

fixture-tokenizer:
    {{py}} -m brainlab.tok.parity_fixture

fmt:
    cd rust && {{cargo}} fmt
    {{py}} -m ruff format python tests bench triton_kernels || true

lint:
    cd rust && {{cargo}} clippy --workspace --all-targets
    {{py}} -m ruff check python tests bench triton_kernels || true

clean:
    cd rust && {{cargo}} clean
    rm -rf .pytest_cache .ruff_cache **/__pycache__
    rm -f {{goldendir}}/*.safetensors {{goldendir}}/*.bin

distclean: clean
    rm -rf {{results}}/*.json data checkpoints


