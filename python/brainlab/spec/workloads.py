from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Sequence

ITEMSIZE = {"fp32": 4, "fp16": 2, "bf16": 2}
OP_NAMES = ("rmsnorm", "softmax", "attention", "matmul", "forward")


@dataclasses.dataclass(frozen=True)
class Tol:
    atol: float
    rtol: float
    rel_l2: float


TOLERANCES: dict[tuple[str, str], Tol] = {
    ("rmsnorm", "fp32"): Tol(1e-5, 1e-4, 1e-5),
    ("rmsnorm", "fp16"): Tol(1e-2, 1e-2, 2e-3),
    ("softmax", "fp32"): Tol(1e-5, 1e-4, 1e-5),
    ("softmax", "fp16"): Tol(1e-2, 1e-2, 2e-3),
    ("attention", "fp32"): Tol(1e-4, 1e-4, 1e-5),
    ("attention", "fp16"): Tol(5e-2, 2e-2, 5e-3),
    ("matmul", "fp32"): Tol(1e-3, 1e-4, 1e-5),
    ("matmul", "fp16"): Tol(2e-1, 2e-2, 5e-3),
    ("forward", "fp32"): Tol(1e-3, 1e-3, 1e-4),
    ("forward", "fp16"): Tol(5e-1, 5e-2, 1e-2),
}


@dataclasses.dataclass(frozen=True)
class Workload:
    op: str
    dtype: str
    dims: tuple[int, ...]
    eps: float = 1e-6
    seed: int = 1234
    params: int = 0
    vocab: int = 0

    @property
    def key(self) -> str:
        return f"{self.op}_{self.dtype}_{'x'.join(str(d) for d in self.dims)}"

    @property
    def golden_file(self) -> str:
        return f"{self.key}.safetensors"

    @property
    def itemsize(self) -> int:
        return ITEMSIZE[self.dtype]

    @property
    def tol(self) -> Tol:
        return TOLERANCES[(self.op, self.dtype)]

    def elements(self) -> int:
        return math.prod(self.dims)

    def operand_shapes(self) -> dict[str, tuple[int, ...]]:
        if self.op == "rmsnorm":
            rows, cols = self.dims
            return {"x": (rows, cols), "w": (cols,)}
        if self.op == "softmax":
            rows, cols = self.dims
            return {"x": (rows, cols)}
        if self.op == "matmul":
            m, k, n = self.dims
            return {"a": (m, k), "b": (k, n)}
        if self.op == "attention":
            batch, heads, seq, hd = self.dims
            return {"q": (batch, heads, seq, hd), "k": (batch, heads, seq, hd), "v": (batch, heads, seq, hd)}
        if self.op == "forward":
            batch, seq = self.dims
            return {"ids": (batch, seq)}
        raise ValueError(f"unknown op {self.op}")

    def output_shapes(self, vocab: int | None = None) -> dict[str, tuple[int, ...]]:
        if self.op in {"rmsnorm", "softmax"}:
            return {"y": self.dims}
        if self.op == "matmul":
            m, _, n = self.dims
            return {"y": (m, n)}
        if self.op == "attention":
            return {"y": self.dims}
        if self.op == "forward":
            v = vocab if vocab is not None else self.vocab
            if not v:
                raise ValueError("forward workload needs a vocab size to size logits")
            return {"logits": (self.dims[0], self.dims[1], v)}
        raise ValueError(f"unknown op {self.op}")

    def with_model(self, params: int, vocab: int) -> Workload:
        if self.op != "forward":
            return self
        return dataclasses.replace(self, params=params, vocab=vocab)

    @property
    def traffic_bytes(self) -> int:
        it = self.itemsize
        if self.op == "rmsnorm":
            rows, cols = self.dims
            return (2 * rows * cols + cols) * it
        if self.op == "softmax":
            rows, cols = self.dims
            return 2 * rows * cols * it
        if self.op == "matmul":
            m, k, n = self.dims
            return (m * k + k * n + m * n) * it
        if self.op == "attention":
            batch, heads, seq, hd = self.dims
            return 4 * batch * heads * seq * hd * it
        batch, seq = self.dims
        return (self.params + batch * seq * self.vocab) * it

    @property
    def flops(self) -> float:
        if self.op == "rmsnorm":
            rows, cols = self.dims
            return rows * cols * 4.0
        if self.op == "softmax":
            rows, cols = self.dims
            return rows * cols * 6.0
        if self.op == "matmul":
            m, k, n = self.dims
            return 2.0 * m * k * n
        if self.op == "attention":
            batch, heads, seq, hd = self.dims
            return 2.0 * batch * heads * seq * (seq + 1) * hd
        return 2.0 * self.params * self.dims[0] * self.dims[1]

    @property
    def is_memory_bound(self) -> bool:
        return self.op in {"rmsnorm", "softmax"} or (self.op == "attention" and self.dims[2] <= 1024)

    @property
    def intensity(self) -> float:
        return self.flops / max(self.traffic_bytes, 1)

    def with_seed(self, seed: int) -> Workload:
        return dataclasses.replace(self, seed=seed)


def _w(op: str, dims: tuple[int, ...], dtypes: Sequence[str] = ("fp32", "fp16")) -> tuple[Workload, ...]:
    return tuple(Workload(op=op, dtype=dt, dims=dims) for dt in dtypes)


WORKLOADS: tuple[Workload, ...] = (
    *_w("rmsnorm", (4096, 512)),
    *_w("rmsnorm", (1024, 512)),
    *_w("rmsnorm", (2000, 384)),
    *_w("rmsnorm", (1, 512)),
    *_w("softmax", (4096, 512)),
    *_w("softmax", (1024, 1024)),
    *_w("softmax", (2000, 384)),
    *_w("attention", (2, 8, 512, 64)),
    *_w("attention", (1, 8, 1024, 64)),
    *_w("attention", (4, 8, 256, 64)),
    *_w("matmul", (1024, 512, 1536)),
    *_w("matmul", (2048, 512, 512)),
    *_w("matmul", (512, 1344, 512)),
    *_w("forward", (1, 512), ("fp32",)),
    *_w("forward", (1, 1024), ("fp32",)),
    *_w("forward", (4, 1024), ("fp32",)),
)

_BY_KEY = {w.key: w for w in WORKLOADS}


def get(key: str) -> Workload:
    return _BY_KEY[key]


def select(ops: Iterable[str] | None = None, dtypes: Iterable[str] | None = None) -> tuple[Workload, ...]:
    want_ops = set(ops or OP_NAMES)
    want_dt = set(dtypes or ITEMSIZE)
    return tuple(w for w in WORKLOADS if w.op in want_ops and w.dtype in want_dt)


def describe(workloads: Iterable[Workload] | None = None) -> str:
    rows = []
    for w in workloads or WORKLOADS:
        rows.append(f"{w.key:34s} dims={str(w.dims):20s} tol(atol={w.tol.atol:g}, rtol={w.tol.rtol:g}, rel_l2={w.tol.rel_l2:g)}")
    return "\n".join(rows)
