from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from brainlab.export.raw_io import write_raw_bundle
from brainlab.export.to_safetensors import save_model
from brainlab.kernels.torch_ops import attention_ref, matmul_ref, rmsnorm_ref, softmax_rows
from brainlab.model.gpt import TransformerLM
from brainlab.spec.config import ModelSpec, load_model_spec, repo_root
from brainlab.spec.workloads import WORKLOADS, Workload, select

TORCH_DTYPE = {"fp32": torch.float32, "fp16": torch.float16}
NP_DTYPE = {"fp32": np.float32, "fp16": np.float16}
MODEL_FILE = "model.safetensors"
INDEX_FILE = "index.json"


def output_name(workload: Workload) -> str:
    return "logits" if workload.op == "forward" else "y"


def entropy(workload: Workload, index: int, salt: int) -> list[int]:
    return [workload.seed, index, salt, *workload.dims]


def operand_arrays(workload: Workload, index: int, spec: ModelSpec) -> dict[str, np.ndarray]:
    def draw(salt: int, shape: tuple[int, ...]) -> np.ndarray:
        return np.random.default_rng(entropy(workload, index, salt)).standard_normal(shape)

    op = workload.op
    if op == "rmsnorm":
        rows, cols = workload.dims
        operands = {"x": draw(1, (rows, cols)), "w": draw(2, (cols,))}
    elif op == "softmax":
        operands = {"x": draw(1, workload.dims)}
    elif op == "matmul":
        m, k, n = workload.dims
        operands = {"a": draw(1, (m, k)), "b": draw(2, (k, n)) / math.sqrt(k)}
    elif op == "attention":
        operands = {name: draw(salt, workload.dims) for name, salt in (("q", 1), ("k", 2), ("v", 3))}
    elif op == "forward":
        ids = np.random.default_rng(entropy(workload, index, 1)).integers(0, spec.vocab_size, size=workload.dims)
        operands = {"ids": ids.astype(np.uint16)}
    else:
        raise ValueError(f"unknown op {op}")
    if workload.op != "forward":
        operands = {k: v.astype(NP_DTYPE[workload.dtype]) for k, v in operands.items()}
    return operands


def reference(workload: Workload, operands: dict[str, np.ndarray], model: TransformerLM | None) -> np.ndarray:
    tensors = {k: torch.from_numpy(np.asarray(v)) for k, v in operands.items()}
    out_dtype = TORCH_DTYPE[workload.dtype]
    if workload.op == "rmsnorm":
        out = rmsnorm_ref(tensors["x"], tensors["w"], workload.eps)
    elif workload.op == "softmax":
        out = softmax_rows(tensors["x"])
    elif workload.op == "matmul":
        out = matmul_ref(tensors["a"], tensors["b"])
    elif workload.op == "attention":
        out = attention_ref(tensors["q"], tensors["k"], tensors["v"])
    elif workload.op == "forward":
        if model is None:
            raise ValueError("forward workloads need a model; pass --model")
        with torch.no_grad():
            out = model(tensors["ids"].long())
    else:
        raise ValueError(f"unknown op {workload.op}")
    return out.to(dtype=out_dtype).contiguous().numpy()


def build(
    workload: Workload,
    index: int,
    out_dir: Path,
    spec: ModelSpec,
    model: TransformerLM | None,
    write_raw: bool,
) -> dict[str, object]:
    operands = operand_arrays(workload, index, spec)
    out = reference(workload, operands, model)
    name = output_name(workload)
    arrays = dict(operands)
    arrays[name] = out
    tensors = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in arrays.items()}
    path = out_dir / workload.golden_file
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = workload_meta(workload, spec, name)
    save_file(tensors, str(path), metadata=meta)
    if write_raw and workload.op != "forward":
        write_raw_bundle(raw_dir(out_dir, workload), arrays, workload.dtype, meta)
    record = dict(meta)
    record["file"] = workload.golden_file
    record["bytes"] = path.stat().st_size
    record["output"] = name
    return record


def workload_meta(workload: Workload, spec: ModelSpec, out_name: str) -> dict[str, str]:
    shapes = {k: "x".join(str(d) for d in v) for k, v in workload.operand_shapes().items()}
    shapes[out_name] = "x".join(str(d) for d in workload.output_shapes(spec.vocab_size)[out_name])
    return {
        "brainlab.op": workload.op,
        "brainlab.dtype": workload.dtype,
        "brainlab.key": workload.key,
        "brainlab.eps": repr(workload.eps),
        "brainlab.seed": str(workload.seed),
        "brainlab.shapes": json.dumps(shapes, separators=(",", ":")),
        "brainlab.atol": repr(workload.tol.atol),
        "brainlab.rtol": repr(workload.tol.rtol),
        "brainlab.rel_l2": repr(workload.tol.rel_l2),
        "brainlab.traffic_bytes": str(workload.traffic_bytes),
        "brainlab.flops": repr(workload.flops),
    }


def raw_dir(out_dir: Path, workload: Workload) -> Path:
    return out_dir / "raw" / workload.key


def build_all(
    ops: Iterable[str] | None,
    dtypes: Iterable[str] | None,
    out_dir: Path,
    seed: int | None,
    with_model: bool,
    write_raw: bool,
) -> dict[str, object]:
    spec = load_model_spec()
    workloads = select(ops, dtypes)
    if not with_model:
        workloads = tuple(w for w in workloads if w.op != "forward")
    if not workloads:
        raise SystemExit("no workloads selected")
    out_dir.mkdir(parents=True, exist_ok=True)

    model: TransformerLM | None = None
    if with_model:
        torch.manual_seed(seed if seed is not None else 0)
        model = TransformerLM(spec).eval()
        save_model(model, out_dir / MODEL_FILE, dtype=torch.float32)

    records = []
    for index, workload in enumerate(WORKLOADS):
        if workload not in workloads:
            continue
        if seed is not None:
            workload = workload.with_seed(seed)
        if workload.op == "forward":
            workload = workload.with_model(spec.param_count, spec.vocab_size)
        records.append(build(workload, index, out_dir, spec, model, write_raw))

    index_doc = {
        "spec": spec.to_dict(),
        "param_count": spec.param_count,
        "model_file": MODEL_FILE if with_model else None,
        "producer": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "workloads": records,
    }
    (out_dir / INDEX_FILE).write_text(json.dumps(index_doc, indent=2, sort_keys=True), encoding="utf-8")
    return index_doc


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="brainlab-golden", description="Generate deterministic PyTorch reference tensors.")
    parser.add_argument("--ops", nargs="+", default=None)
    parser.add_argument("--dtypes", nargs="+", default=None, choices=["fp32", "fp16"])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model", action="store_true", help="also export model.safetensors and forward goldens")
    parser.add_argument("--raw", action="store_true", help="also write .bin bundles for mojo, cuda and rust CLIs")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    if args.list:
        from brainlab.spec.workloads import describe

        print(describe())
        return 0

    out_dir = args.out or (repo_root() / "tests" / "golden")
    doc = build_all(args.ops, args.dtypes, out_dir, args.seed, args.model, args.raw)
    total = sum(int(r["bytes"]) for r in doc["workloads"])
    for record in doc["workloads"]:
        print(f'{record["file"]:52s} {record["bytes"]:>12,d} B  atol={record["brainlab.atol"]}')
    print(f'{len(doc["workloads"])} workloads -> {out_dir} ({total/1e6:.1f} MB, index.json, model={"yes" if doc["model_file"] else "no"})')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
