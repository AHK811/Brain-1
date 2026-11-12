from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TOKEN_DTYPE = {"uint16": np.dtype(np.uint16), "uint32": np.dtype(np.uint32)}


@dataclass(frozen=True)
class PackConfig:
    seq_len: int
    shard_tokens: int
    eos_id: int
    dtype: str = "uint16"

    def __post_init__(self) -> None:
        if self.shard_tokens % self.seq_len:
            raise ValueError(f"shard_tokens {self.shard_tokens} must be a multiple of seq_len {self.seq_len}")
        if self.dtype not in TOKEN_DTYPE:
            raise ValueError(f"unsupported dtype {self.dtype}")
        limit = int(np.iinfo(TOKEN_DTYPE[self.dtype]).max)
        if self.eos_id > limit:
            raise ValueError(f"eos id {self.eos_id} does not fit {self.dtype}")


def pack_stream(docs: Sequence[Sequence[int]], cfg: PackConfig) -> np.ndarray:
    parts: list[np.ndarray] = []
    total = 0
    for doc in docs:
        arr = np.asarray(doc, dtype=np.int64)
        parts.append(arr)
        parts.append(np.asarray([cfg.eos_id], dtype=np.int64))
        total += arr.size + 1
    if total == 0:
        return np.zeros(0, dtype=TOKEN_DTYPE[cfg.dtype])
    stream = np.concatenate(parts)
    usable = stream.size - (stream.size % cfg.seq_len)
    return stream[:usable].astype(TOKEN_DTYPE[cfg.dtype])


def split_shards(stream: np.ndarray, cfg: PackConfig) -> list[np.ndarray]:
    if stream.size == 0:
        return []
    limit = int(np.iinfo(TOKEN_DTYPE[cfg.dtype]).max)
    if int(stream.max(initial=0)) > limit:
        raise ValueError(f"token id above {cfg.dtype} range; tokenizer vocab is inconsistent with pack config")
    return [stream[start : start + cfg.shard_tokens] for start in range(0, stream.size, cfg.shard_tokens)]


def write_shards(
    shards: Iterable[np.ndarray],
    out_dir: Path,
    split: str,
    cfg: PackConfig,
    source_files: Sequence[str] = (),
    start_index: int = 0,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for offset, shard in enumerate(shards):
        index = start_index + offset
        bin_path = out_dir / f"{index:06d}.bin"
        bin_path.write_bytes(np.asarray(shard, dtype=TOKEN_DTYPE[cfg.dtype]).tobytes(order="C"))
        manifest = {
            "n_tokens": int(shard.size),
            "seq_len": cfg.seq_len,
            "dtype": cfg.dtype,
            "split": split,
            "byte_order": "little",
            "row_major": True,
            "source_files": list(source_files),
            "sha256": _sha256(bin_path),
        }
        (out_dir / f"{index:06d}.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        written.append(bin_path)
    return written


def pack_and_write(
    docs: Sequence[Sequence[int]],
    out_dir: Path,
    split: str,
    cfg: PackConfig,
    source_files: Sequence[str] = (),
) -> list[Path]:
    return write_shards(split_shards(pack_stream(docs, cfg), cfg), out_dir, split, cfg, source_files)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
