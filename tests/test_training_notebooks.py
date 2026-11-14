from __future__ import annotations

import ast
import io
import json
import shutil
import subprocess
import sys
import tokenize
from pathlib import Path

import pytest
import torch

from brainlab.spec.config import load_model_spec
from brainlab.train.checkpoint import CHECKPOINT_FORMAT, latest_checkpoint
from brainlab.train.train import parse_args

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ("kaggle_train.ipynb", "colab_train.ipynb")


@pytest.fixture(params=NOTEBOOKS)
def notebook(request):
    path = ROOT / "notebooks" / request.param
    if not path.is_file():
        if request.param == "colab_train.ipynb":
            pytest.skip("Local-only ignored notebook is not present in this checkout")
        pytest.fail(f"Missing tracked notebook: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def cell_source(notebook, index):
    return "".join(notebook["cells"][index]["source"])


@pytest.fixture
def helpers(notebook):
    namespace = {"__name__": "notebook_helper_test"}
    exec(compile(cell_source(notebook, 1), "notebook-config", "exec"), namespace)
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        definitions = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), f"notebook-cell-{index}", "exec"), namespace)
    namespace.update(PY=Path(sys.executable), REPO=ROOT, DEVICE="cpu")
    return namespace


def test_all_cells_and_embedded_python_compile(notebook):
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        tree = ast.parse(source)
        compile(tree, f"cell-{index}", "exec")
        assert cell["outputs"] == []
        assert cell["execution_count"] is None
        assert not any(token.type == tokenize.COMMENT for token in tokenize.generate_tokens(io.StringIO(source).readline))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "py":
                if node.args and isinstance(node.args[0], ast.Constant):
                    compile(node.args[0].value, f"cell-{index}-subprocess", "exec")


def test_config_and_training_arguments(helpers, tmp_path):
    assert helpers["MICRO_BATCH"] * helpers["ACCUM"] * helpers["SEQ_LEN"] == 32768
    assert helpers["TOTAL_STEPS"] == 8000
    assert helpers["SEED"] == 0
    assert helpers["DTYPE"] == "auto"
    assert helpers["GRAD_CKPT"] is True
    command = helpers["train_command"](tmp_path, tmp_path / "train", tmp_path / "val", 17, 64, 2, 16, 5, 3, 4, 1)
    args = parse_args([str(value) for value in command[4:]])
    assert (args.steps, args.seq_len, args.micro_batch, args.accum) == (17, 64, 2, 16)
    assert (args.eval_every, args.eval_batches, args.save_every, args.log_every) == (5, 3, 4, 1)
    assert args.device == "cpu" and args.dtype == "auto" and args.grad_ckpt


def test_checked_run_preserves_stderr_and_none_cwd(helpers):
    result = helpers["run"]([sys.executable, "-c", "print('actual child')"], cwd=None)
    assert result.stdout.strip() == "actual child"
    with pytest.raises(subprocess.CalledProcessError) as error:
        helpers["run"]([sys.executable, "-c", "import sys; sys.stderr.write('required failure'); sys.exit(7)"])
    assert error.value.returncode == 7
    assert "required failure" in error.value.stderr


def test_clone_and_reuse_real_local_checkout(helpers, tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git is required for a real local clone")
    calls = []
    actual_run = helpers["run"]

    def recorded_run(args, **kwargs):
        calls.append((list(args), kwargs))
        return actual_run(args, **kwargs)

    helpers["run"] = recorded_run
    destination = tmp_path / "checkout with spaces"
    revision = helpers["ensure_checkout"](destination, ROOT.as_uri())
    assert revision
    assert helpers["ensure_checkout"](destination, str(tmp_path / "absent remote")) == revision
    clones = [kwargs for args, kwargs in calls if args[:2] == ["git", "clone"]]
    assert clones == [{"cwd": None}]
    assert not any("pull" in args for args, _ in calls)
    invalid = tmp_path / "not a checkout"
    invalid.mkdir()
    (invalid / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Not a Brain-1 checkout"):
        helpers["ensure_checkout"](invalid, ROOT.as_uri())
    assert (invalid / "keep.txt").read_text(encoding="utf-8") == "keep"
    with pytest.raises(subprocess.CalledProcessError) as error:
        helpers["ensure_checkout"](tmp_path / "failed clone", str(tmp_path / "missing source"))
    assert error.value.stderr


def test_backup_replaces_same_size_changes_and_filters_incomplete_files(helpers, tmp_path):
    source, destination = tmp_path / "local", tmp_path / "backup"
    helpers["verify_contract"](source, {"run": "one"})
    for name in ("best.pt", "last.pt", "step-000001.pt", "train_log.jsonl", "train_stdout.log", "scored.safetensors"):
        (source / name).write_bytes(b"old!")
    for name in ("step-999999.pt.tmp", "last.pt.partial", "diverged-999999.pt", "brainlab-33m.safetensors"):
        (source / name).write_bytes(b"incomplete")
    helpers["backup_run"](source, destination)
    assert not (destination / "scored.safetensors").exists()
    for name in ("last.pt", "best.pt", "train_log.jsonl"):
        (source / name).write_bytes(b"new!")
    changed = helpers["backup_run"](source, destination, include_exports=True)
    assert {"last.pt", "best.pt", "train_log.jsonl", "scored.safetensors"} <= set(changed)
    assert (destination / "last.pt").read_bytes() == b"new!"
    assert helpers["backup_run"](source, destination, include_exports=True) == []
    assert not list(destination.glob("*.partial"))
    assert not any("diverged" in path.name or ".tmp" in path.name for path in destination.iterdir())
    assert not (destination / "brainlab-33m.safetensors").exists()
    (destination / "notebook_run.json").write_text('{"run":"other"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="another or unverified run"):
        helpers["backup_run"](source, destination)


def metadata_checkpoint(path, step):
    torch.save({"format": CHECKPOINT_FORMAT, "step": step, "model": {}, "spec": {}}, path)


def test_latest_and_restore_use_shared_cursor_and_preserve_logs(helpers, tmp_path):
    source, destination = tmp_path / "saved", tmp_path / "restored"
    helpers["verify_contract"](source, {"run": "one"})
    metadata_checkpoint(source / "step-000007.pt", 7)
    metadata_checkpoint(source / "last.pt", 3)
    (source / "diverged-999999.pt").write_bytes(b"not a cursor")
    (source / "last.pt.partial").write_bytes(b"not a cursor")
    log = '{"event":"config","seed":0}\n'
    (source / "train_log.jsonl").write_text(log, encoding="utf-8")
    assert helpers["latest_cursor"](source) == latest_checkpoint(source) == source / "step-000007.pt"
    metadata_checkpoint(source / "last.pt", 9)
    assert helpers["latest_cursor"](source) == latest_checkpoint(source) == source / "last.pt"
    restored = helpers["restore_run"](source, destination)
    assert restored == destination / "last.pt"
    assert helpers["latest_cursor"](destination) == restored
    assert (destination / "train_log.jsonl").read_text(encoding="utf-8") == log
    assert not (destination / "last.pt.partial").exists()
    assert helpers["restore_run"](source, destination) == restored
    assert helpers["restore_run"](restored, destination) == restored
    (destination / "train_log.jsonl").write_text(log + '{"event":"train"}\n', encoding="utf-8")
    before = (destination / "train_log.jsonl").read_bytes()
    with pytest.raises(RuntimeError, match="Conflicting existing local run"):
        helpers["restore_run"](source, destination)
    assert (destination / "train_log.jsonl").read_bytes() == before
    explicit = helpers["restore_run"](source / "step-000007.pt", tmp_path / "explicit")
    assert explicit.name == "step-000007.pt"
    assert not (explicit.parent / "last.pt").exists()
    for name in ("diverged-999999.pt", "last.pt.partial"):
        with pytest.raises(RuntimeError, match="Not a completed checkpoint"):
            helpers["restore_run"](source / name, tmp_path / "invalid")
    assert helpers["latest_cursor"](tmp_path / "absent") is None


def test_resume_contract_rejects_changed_inputs(helpers, tmp_path):
    contract = {"micro_batch": 1, "accum": 32, "seq_len": 1024, "seed": 0, "total_steps": 8000,
                "eval_batches": 64, "data_sha256": "data", "tokenizer_sha256": "tok", "spec_sha256": "spec"}
    helpers["verify_contract"](tmp_path, contract)
    helpers["verify_contract"](tmp_path, dict(contract))
    for key in contract:
        with pytest.raises(RuntimeError, match="contract changed"):
            helpers["verify_contract"](tmp_path, {**contract, key: "changed"})
    missing = tmp_path / "unverified"
    missing.mkdir()
    metadata_checkpoint(missing / "last.pt", 1)
    with pytest.raises(RuntimeError, match="no notebook provenance"):
        helpers["verify_contract"](missing, contract)
    restored = helpers["restore_run"](missing / "last.pt", tmp_path / "target")
    with pytest.raises(RuntimeError, match="no notebook provenance"):
        helpers["verify_contract"](restored.parent, contract)


def test_cli_checkpoint_import_validates_saved_settings(helpers, tmp_path):
    source = tmp_path / "cli"
    source.mkdir()
    contract = {"micro_batch": 1, "accum": 32, "seq_len": 1024, "seed": 0, "total_steps": 8000,
                "eval_batches": 8, "train_sequences": 20, "val_sequences": 20}
    cfg = {**contract, "accum_steps": 32, "optimizer": "muon", "eval_batch_size": 1}
    payload = {"format": CHECKPOINT_FORMAT, "step": 1, "model": {},
               "spec": load_model_spec().to_dict(), "cfg": cfg}
    torch.save(payload, source / "step-000001.pt")
    restored = helpers["restore_run"](source / "step-000001.pt", tmp_path / "imported")
    helpers["verify_contract"](restored.parent, contract, restored)
    assert json.loads((restored.parent / "notebook_run.json").read_text()) == contract
    for key in ("micro_batch", "accum_steps", "seq_len", "optimizer"):
        payload["cfg"] = {**cfg, key: "wrong"}
        torch.save(payload, source / "step-000001.pt")
        with pytest.raises(RuntimeError, match=f"Cannot verify CLI checkpoint {key}"):
            helpers["verify_cli_resume"](source / "step-000001.pt", contract)
    payload["cfg"] = {"seed": 0}
    torch.save(payload, source / "step-000001.pt")
    with pytest.raises(RuntimeError, match="Cannot verify CLI checkpoint"):
        helpers["verify_cli_resume"](source / "step-000001.pt", contract)


def test_backup_retention_preserves_special_and_unrelated_files(helpers, tmp_path):
    source, destination = tmp_path / "local", tmp_path / "drive"
    helpers["KEEP"] = 2
    helpers["verify_contract"](source, {"run": "retention"})
    for step in range(5):
        (source / f"step-{step:06d}.pt").write_bytes(b"checkpoint")
    for name in ("best.pt", "last.pt"):
        (source / name).write_bytes(b"checkpoint")
    helpers["backup_run"](source, destination)
    unrelated = destination / "notes.txt"
    unrelated.write_text("keep")
    helpers["backup_run"](source, destination)
    assert sorted(path.name for path in destination.glob("step-*.pt")) == ["step-000003.pt", "step-000004.pt"]
    assert unrelated.read_text() == "keep"
    assert (destination / "best.pt").exists() and (destination / "last.pt").exists()


def test_actual_packing_imports_and_validation_lengths(helpers, tmp_path):
    helpers["pack_overfit"](tmp_path, 8, 0)
    prepared = helpers["inspect_data"](tmp_path, 8, None)
    assert (prepared["train"], prepared["val"]) == (1, 7)
    assert helpers["eval_capacity"](1, 7, 1, 64) == 3
    assert helpers["eval_capacity"](4, 7, 2, 64) == 1
    with pytest.raises(RuntimeError, match="Validation cannot hold"):
        helpers["eval_capacity"](4, 3, 2, 64)


def test_prepare_cell_refuses_nonempty_output(notebook, helpers, tmp_path):
    helpers.update(WORK=tmp_path, RUN_NAME="one", PACKED_DIR=None)
    data = tmp_path / "data" / "one"
    data.mkdir(parents=True)
    sentinel = data / "keep.bin"
    sentinel.write_bytes(b"preserve")
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        exec(compile(cell_source(notebook, 13), "prepare-cell", "exec"), helpers)
    assert sentinel.read_bytes() == b"preserve"
    helpers["PACKED_DIR"] = data
    exec(compile(cell_source(notebook, 13), "prepare-cell", "exec"), helpers)
    assert sentinel.read_bytes() == b"preserve"


def test_leak_gate_actual_check_and_invalid_reports(helpers, tmp_path):
    helpers["pack_overfit"](tmp_path, 8, 0)
    proc = helpers["run"]([sys.executable, "-m", "brainlab.data.leakage", "--train-dir", tmp_path / "train",
                            "--val-dir", tmp_path / "val", "--seq-len", 8, "--span", 8, "--sequences", 7, "--json"],
                           cwd=ROOT, check=False)
    report = helpers["validate_leak"](proc, False, False, 8, 7, 8)
    assert not report["leaked"]
    leaked = {**report, "matching_windows": 1, "val_sequences_with_match": 1, "leaked": True}
    matched = subprocess.CompletedProcess([], 1, json.dumps(leaked), "")
    assert helpers["validate_leak"](matched, True, True, 8, 7, 8)["leaked"]
    for synthetic, allowed in ((False, True), (True, False)):
        with pytest.raises(RuntimeError, match="Validation matches"):
            helpers["validate_leak"](matched, synthetic, allowed, 8, 7, 8)
    invalid = ((3, report), (0, leaked), (0, {}), (0, None), (0, {**report, "val_windows": 0}))
    for code, payload in invalid:
        with pytest.raises(RuntimeError, match="Invalid leakage report"):
            helpers["validate_leak"](subprocess.CompletedProcess([], code, json.dumps(payload), "bad"), True, True, 8, 7, 8)
    shared = {**leaked, "shared_source_files": ["same.txt"]}
    with pytest.raises(RuntimeError, match="source files overlap"):
        helpers["validate_leak"](subprocess.CompletedProcess([], 1, json.dumps(shared), ""), True, True, 8, 7, 8)


def test_authoritative_latest_header_and_disjoint_windows(helpers, tmp_path):
    valid = {"event": "config", "eval_track_window": [0, 3], "eval_select_window": [3, 6], "val_sequences": 7}
    newer = {**valid, "dtype": "fp32"}
    (tmp_path / "train_stdout.log").write_text("not JSON", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        helpers["read_rows"](tmp_path)
    (tmp_path / "train_log.jsonl").write_text(json.dumps(valid) + "\n" + json.dumps(newer) + "\n", encoding="utf-8")
    assert helpers["latest_header"](helpers["read_rows"](tmp_path)) == newer
    for track, select in (([], []), ([0, 0], [1, 2]), ([0, 3], [2, 4]), ([0, 3], [3, 8])):
        with pytest.raises(RuntimeError):
            helpers["latest_header"]([{**valid, "eval_track_window": track, "eval_select_window": select}])
    with pytest.raises(RuntimeError, match="No authoritative config"):
        helpers["latest_header"]([{"event": "train"}])


def test_budget_timer_and_final_backup_with_real_child(helpers, tmp_path):
    source, backup = tmp_path / "run", tmp_path / "backup"
    helpers["verify_contract"](source, {"run": "timer"})
    (source / "step-000001.pt").write_bytes(b"preserve")
    calls = []

    def copy():
        calls.append(helpers["time"].monotonic())
        helpers["backup_run"](source, backup)

    command = [sys.executable, "-u", "-c", "import threading; print('waiting'); threading.Event().wait(60)"]
    outcome = helpers["run_budgeted"](command, source, 0.01, copy, 0.05)
    assert outcome["budget_hit"]
    assert len(calls) >= 2
    assert (backup / "step-000001.pt").read_bytes() == b"preserve"
    assert not (source / "last.pt").exists()
    assert (backup / "train_stdout.log").read_bytes() == (source / "train_stdout.log").read_bytes()


def test_error_and_keyboard_interrupt_cleanup_with_real_children(helpers, tmp_path):
    calls = []
    command = [sys.executable, "-c", "import sys; sys.exit(9)"]
    with pytest.raises(subprocess.CalledProcessError) as error:
        helpers["run_budgeted"](command, tmp_path, 1, lambda: calls.append("final"), 10)
    assert error.value.returncode == 9 and calls == ["final"]
    stopped = []
    actual_stop = helpers["stop_child"]

    def record_stop(child):
        code = actual_stop(child)
        stopped.append(child.poll())
        return code

    def interrupt_once():
        calls.append("backup")
        if len(calls) == 2:
            raise KeyboardInterrupt

    helpers["stop_child"] = record_stop
    command = [sys.executable, "-c", "import threading; threading.Event().wait(60)"]
    with pytest.raises(KeyboardInterrupt):
        helpers["run_budgeted"](command, tmp_path, 1, interrupt_once, 0.05)
    assert calls == ["final", "backup", "backup"]
    assert stopped and all(code is not None for code in stopped)


def test_normal_exit_always_runs_final_backup(helpers, tmp_path):
    calls = []
    outcome = helpers["run_budgeted"]([sys.executable, "-c", "print('completed')"], tmp_path, 1,
                                       lambda: calls.append("final"), 10)
    assert outcome == {"budget_hit": False, "returncode": 0}
    assert calls == ["final"]
    assert "completed" in (tmp_path / "train_stdout.log").read_text(encoding="utf-8")
