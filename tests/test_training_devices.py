from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace

import pytest
import torch

from brainlab.data.pack import PackConfig, pack_and_write
from brainlab.model.gpt import TransformerLM
from brainlab.spec.config import load_model_spec
from brainlab.train.checkpoint import latest_checkpoint, load_payload, rotate, save_checkpoint
from brainlab.train.train import resolve_device, resolve_dtype, train


@pytest.fixture
def tiny_spec():
    spec = replace(
        load_model_spec(), d_model=16, n_layers=1, n_heads=2, head_dim=8,
        mlp_hidden=64, vocab_size=320, context_len=32, residual_scale_layers=1,
    )
    return replace(spec, param_budget=spec.param_count)


@pytest.fixture
def training_args(tmp_path, monkeypatch, tiny_spec):
    monkeypatch.setattr("brainlab.train.train.load_model_spec", lambda: tiny_spec)
    for split in ("train", "val"):
        pack_and_write(
            [[(i + j) % 256 for j in range(32)] for i in range(12)],
            tmp_path / split, split, PackConfig(seq_len=8, shard_tokens=256, eos_id=257),
        )
    return [
        "--train-dir", str(tmp_path / "train"), "--val-dir", str(tmp_path / "val"),
        "--steps", "2", "--seq-len", "8", "--micro-batch", "1", "--accum", "2",
        "--eval-every", "1", "--eval-batches", "1", "--save-every", "1",
        "--log-every", "1", "--grad-ckpt",
    ]


def test_auto_cpu_and_explicit_cuda_failure(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")
    assert resolve_dtype("auto", torch.device("cpu")) == "fp32"
    with pytest.raises(SystemExit, match="CUDA is unavailable"):
        resolve_device("cuda")
    for dtype in ("fp16", "bf16"):
        with pytest.raises(SystemExit, match="CPU training requires"):
            resolve_dtype(dtype, torch.device("cpu"))
    with pytest.raises(SystemExit, match="training supports"):
        resolve_device("meta")


@pytest.mark.parametrize("major,supported,expected", [(7, False, "fp16"), (8, True, "bf16"), (8, False, "fp16")])
def test_cuda_precision_selection(monkeypatch, major, supported, expected):
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (major, 0))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: supported)
    device = torch.device("cuda:0")
    assert resolve_dtype("auto", device) == expected
    assert resolve_dtype("fp32", device) == "fp32"
    assert resolve_dtype("fp16", device) == "fp16"
    if expected == "fp16":
        with pytest.raises(SystemExit, match="native BF16"):
            resolve_dtype("bf16", device)


def test_cuda_index_is_checked(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(SystemExit, match="not visible"):
        resolve_device("cuda:1")


def test_checkpointed_backward_matches_eager(tiny_spec):
    eager = TransformerLM(tiny_spec)
    checkpointed = TransformerLM(tiny_spec, gradient_checkpoint=True)
    checkpointed.load_state_dict(eager.state_dict())
    ids = torch.arange(8).reshape(1, 8)
    for model in (eager, checkpointed):
        _, loss = model(ids, ids)
        loss.backward()
    for first, second in zip(eager.parameters(), checkpointed.parameters(), strict=True):
        torch.testing.assert_close(first.grad, second.grad, rtol=0, atol=0)


def test_latest_checkpoint_uses_last_step_not_name(tmp_path, tiny_spec):
    model = TransformerLM(tiny_spec)
    assert latest_checkpoint(tmp_path) is None
    save_checkpoint(tmp_path / "last.pt", model, 2)
    save_checkpoint(tmp_path / "step-000003.pt", model, 3)
    (tmp_path / "step-broken.pt").write_bytes(b"ignored")
    assert latest_checkpoint(tmp_path).name == "step-000003.pt"
    save_checkpoint(tmp_path / "last.pt", model, 4)
    assert latest_checkpoint(tmp_path).name == "last.pt"


def test_rotation_honors_keep_after_many_saves(tmp_path):
    for step in range(7):
        (tmp_path / f"step-{step:06d}.pt").touch()
    assert len(rotate(tmp_path, 2)) == 5
    assert sorted(path.name for path in tmp_path.glob("*.pt")) == ["step-000005.pt", "step-000006.pt"]


def test_cpu_training_resume_and_batch_contract(tmp_path, training_args):
    full = tmp_path / "full"
    resumed = tmp_path / "resumed"
    assert train([*training_args, "--device", "cpu", "--out", str(full)]) == 0
    cursor = full / "step-000001.pt"
    assert train([*training_args, "--device", "cpu", "--resume", str(cursor), "--out", str(resumed)]) == 0
    expected = load_payload(full / "last.pt")
    actual = load_payload(resumed / "last.pt")
    assert actual["step"] == 2
    assert actual["cfg"]["start_batch"] == 4
    assert actual["cfg"]["best_val"] == expected["cfg"]["best_val"]
    for name, weight in expected["model"].items():
        torch.testing.assert_close(actual["model"][name], weight, rtol=0, atol=0)
    for flag, value, field in (("--accum", "1", "accum_steps"), ("--steps", "3", "total_steps")):
        with pytest.raises(SystemExit, match=f"resume {field}"):
            train([*training_args, flag, value, "--device", "cpu", "--resume", str(cursor), "--out", str(resumed)])
    with pytest.raises(SystemExit, match="does not exist"):
        train([*training_args, "--device", "cpu", "--resume", str(tmp_path / "missing.pt")])


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required for cross-device training")
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_cpu_cuda_cpu_training_resume(tmp_path, training_args, dtype):
    device = torch.device("cuda")
    if dtype == "bf16" and resolve_dtype("auto", device) != "bf16":
        pytest.skip("native BF16 hardware required")
    cpu = tmp_path / "cpu"
    cuda = tmp_path / "cuda"
    back = tmp_path / "back"
    args = [*training_args, "--steps", "3"]
    assert train([*args, "--device", "cpu", "--out", str(cpu)]) == 0
    assert train([
        *args, "--device", "cuda", "--dtype", dtype,
        "--resume", str(cpu / "step-000001.pt"), "--out", str(cuda),
    ]) == 0
    gpu_payload = load_payload(cuda / "last.pt")
    assert gpu_payload["step"] == 3
    assert bool(gpu_payload["scaler"]) == (dtype == "fp16")
    assert all(value.dtype == torch.float32 for value in gpu_payload["model"].values())
    assert all(torch.isfinite(value).all() for value in gpu_payload["model"].values())
    assert train([
        *args, "--device", "cpu", "--resume", str(cuda / "step-000002.pt"), "--out", str(back),
    ]) == 0
    assert load_payload(back / "last.pt")["cfg"]["start_batch"] == 6
