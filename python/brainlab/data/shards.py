from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from brainlab.data.pack import PackConfig, _sha256, split_shards, write_shards

TOKEN_DTYPE = {"uint16": np.dtype(np.uint16), "uint32": np.dtype(np.uint32)}
MANIFEST_SUFFIX = ".json"


@dataclass(frozen=True)
class ShardInfo:
    bin_path: Path
    n_tokens: int
    seq_len: int
    token_dtype: str
    split: str
    source_files: tuple[str, ...]
    loss_weights_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_sequences(self) -> int:
        return self.n_tokens // self.seq_len

    def open(self) -> np.memmap:
        return np.memmap(self.bin_path, dtype=TOKEN_DTYPE[self.token_dtype], mode="r")


def read_manifest(manifest_path: Path) -> ShardInfo:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    bin_path = manifest_path.with_suffix(".bin")
    n_tokens = int(raw["n_tokens"])
    seq_len = int(raw["seq_len"])
    dtype = str(raw["dtype"])
    if dtype not in TOKEN_DTYPE:
        raise ValueError(f"{manifest_path}: unsupported token dtype {dtype}")
    expected = n_tokens * TOKEN_DTYPE[dtype].itemsize
    if not bin_path.is_file():
        raise FileNotFoundError(f"{manifest_path}: missing {bin_path}")
    actual = bin_path.stat().st_size
    if actual != expected:
        raise ValueError(f"{bin_path}: manifest says {expected} bytes, file has {actual}")
    weights_path = None
    if "loss_weights" in raw:
        entry = raw["loss_weights"]
        weights_path = manifest_path.with_suffix(".weights.bin")
        if entry.get("dtype") != "float32" or entry.get("alignment") != "token":
            raise ValueError(f"{manifest_path}: loss weights require token-aligned float32")
        if not weights_path.is_file() or weights_path.stat().st_size != n_tokens * 4:
            raise ValueError(f"{manifest_path}: missing or mis-sized loss weights")
        if _sha256(weights_path) != entry.get("sha256"):
            raise ValueError(f"{manifest_path}: loss weights checksum mismatch")
    metadata = raw.get("metadata", {})
    if metadata.get("data_format") == "chat_sft" and weights_path is None:
        raise ValueError(f"{manifest_path}: chat_sft requires loss weights")
    return ShardInfo(
        bin_path=bin_path,
        n_tokens=n_tokens,
        seq_len=seq_len,
        token_dtype=dtype,
        split=str(raw.get("split", "unknown")),
        source_files=tuple(raw.get("source_files", ())),
        loss_weights_path=weights_path,
        metadata=metadata,
    )


def write_weighted_shards(
    tokens: np.ndarray,
    loss_weights: np.ndarray,
    out_dir: Path,
    split: str,
    cfg: PackConfig,
    metadata: dict[str, Any],
    source_files: Sequence[str] = (),
) -> list[Path]:
    tokens = np.asarray(tokens)
    weights = np.asarray(loss_weights, dtype="<f4")
    if tokens.ndim != 1 or weights.shape != tokens.shape:
        raise ValueError("loss weights must align with the flat token stream")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("loss weights must be finite and non-negative")
    if tokens.size == 0 or tokens.size % cfg.seq_len:
        raise ValueError("weighted stream must contain whole packed sequences")
    if np.any(tokens < 0) or np.any(tokens > np.iinfo(TOKEN_DTYPE[cfg.dtype]).max):
        raise ValueError("token ids do not fit the shard dtype")
    if list(Path(out_dir).glob("*.json")) or list(Path(out_dir).glob("*.bin")):
        raise ValueError("write weighted shards to an empty directory")
    paths = write_shards(split_shards(tokens, cfg), out_dir, split, cfg, source_files)
    for index, path in enumerate(paths):
        start = index * cfg.shard_tokens
        weights_path = path.with_suffix(".weights.bin")
        weights_path.write_bytes(weights[start : start + cfg.shard_tokens].tobytes())
        manifest_path = path.with_suffix(".json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["metadata"] = metadata
        manifest["loss_weights"] = {
            "dtype": "float32",
            "alignment": "token",
            "sha256": _sha256(weights_path),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return paths


def discover(shard_dir: Path) -> tuple[ShardInfo, ...]:
    manifests = sorted(Path(shard_dir).glob(f"*{MANIFEST_SUFFIX}"))
    if not manifests:
        raise FileNotFoundError(f"no shard manifests under {shard_dir}; run the datapack step")
    shards = tuple(read_manifest(p) for p in manifests)
    seq_lens = {s.seq_len for s in shards}
    if len(seq_lens) > 1:
        raise ValueError(f"mixed seq_len in {shard_dir}: {sorted(seq_lens)}")
    for shard in shards:
        if shard.n_sequences == 0:
            raise ValueError(f"{shard.bin_path}: shorter than one sequence")
    return shards


class TokenShards:
    def __init__(self, shard_dir: Path, seq_len: int | None = None, offset: int = 0) -> None:
        self.shards = discover(Path(shard_dir))
        self.seq_len = seq_len or self.shards[0].seq_len
        for shard in self.shards:
            if shard.seq_len != self.seq_len:
                raise ValueError(f"{shard.bin_path}: seq_len {shard.seq_len} != {self.seq_len}")
        self._maps: list[np.memmap] = []
        self._starts: list[int] = []
        running = 0
        for shard in self.shards:
            self._starts.append(running)
            self._maps.append(shard.open())
            running += int(shard.n_tokens)
        self._total_tokens = running
        self._n_sequences = max((running - 1) // self.seq_len, 0)
        self._offset = offset
        self._weight_maps = [
            np.memmap(shard.loss_weights_path, dtype="<f4", mode="r")
            if shard.loss_weights_path is not None else None
            for shard in self.shards
        ]
        if any(m is not None for m in self._weight_maps) and any(m is None for m in self._weight_maps):
            raise ValueError("cannot mix weighted and unweighted shards")
        for weights in self._weight_maps:
            if weights is not None and (not np.isfinite(weights).all() or np.any(weights < 0)):
                raise ValueError("loss weights must be finite and non-negative")

    @property
    def has_loss_weights(self) -> bool:
        return self._weight_maps[0] is not None

    @cached_property
    def provenance(self) -> dict[str, Any]:
        digest = hashlib.sha256()
        for shard in self.shards:
            digest.update(shard.bin_path.with_suffix(".json").read_bytes())
            digest.update(_sha256(shard.bin_path).encode("ascii"))
        return {"sha256": digest.hexdigest(), "shards": len(self.shards)}

    def __len__(self) -> int:
        return max(self._n_sequences - self._offset, 0)

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    def slice_tokens(self, start: int, stop: int) -> np.ndarray:
        return self._slice(self._maps, start, stop)

    def loss_weights(self, index: int) -> np.ndarray | None:
        if not self.has_loss_weights:
            return None
        flat = (index + self._offset) * self.seq_len
        return self._slice(self._weight_maps, flat + 1, flat + self.seq_len + 1).copy()

    def _slice(self, maps: Sequence[np.ndarray], start: int, stop: int) -> np.ndarray:
        parts: list[np.ndarray] = []
        remaining_start = start
        for shard_id, shard in enumerate(self.shards):
            shard_begin = self._starts[shard_id]
            shard_end = shard_begin + int(shard.n_tokens)
            if remaining_start >= shard_end or remaining_start >= stop:
                continue
            local_begin = remaining_start - shard_begin
            local_end = min(stop, shard_end) - shard_begin
            parts.append(np.asarray(maps[shard_id][local_begin:local_end]))
            remaining_start = shard_begin + local_end
        if sum(int(p.size) for p in parts) != stop - start:
            raise IndexError(f"tokens {start}:{stop} outside the packed stream")
        return parts[0] if len(parts) == 1 else np.concatenate(parts)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        if index < 0:
            index %= len(self)
        if index >= len(self):
            raise IndexError(f"sequence {index} out of range for {len(self)}")
        flat = (index + self._offset) * self.seq_len
        chunk = self.slice_tokens(flat, flat + self.seq_len + 1)
        targets = chunk[1:].astype(np.int64, copy=True)
        weights = self.loss_weights(index)
        if weights is not None:
            targets[weights == 0] = -1
        return chunk[:-1].astype(np.int64, copy=False), targets

    def tokens(self) -> np.ndarray:
        return np.concatenate([np.asarray(m) for m in self._maps])


class ShardBatcher:
    def __init__(
        self,
        shards: TokenShards,
        batch_size: int,
        shuffle_seed: int | None = None,
        drop_last: bool = True,
    ) -> None:
        self.shards = shards
        self.batch_size = batch_size
        self.shuffle_seed = shuffle_seed
        self.drop_last = drop_last
        self.epoch = 0

    def indices(self) -> np.ndarray:
        order = np.arange(len(self.shards))
        if self.shuffle_seed is not None:
            rng = np.random.default_rng([self.shuffle_seed, self.epoch])
            rng.shuffle(order)
        return order

    def batches(self, start_batch: int = 0) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if start_batch < 0:
            raise ValueError(f"start_batch {start_batch} is negative")
        order = self.indices()
        for start in range(start_batch * self.batch_size, len(order), self.batch_size):
            chosen = order[start : start + self.batch_size]
            if len(chosen) < self.batch_size and self.drop_last:
                break
            pairs = [self.shards[int(i)] for i in chosen]
            yield tuple(np.stack(items) for items in zip(*pairs, strict=True))
        self.epoch += 1

    def __iter__(self) -> Iterator[tuple[np.ndarray, ...]]:
        return self.batches()

    def __len__(self) -> int:
        usable = len(self.shards) // self.batch_size if self.drop_last else -(-len(self.shards) // self.batch_size)
        return max(usable, 0)


def load_shard_dir(shard_dir: Path, seq_len: int | None = None) -> TokenShards:
    return TokenShards(Path(shard_dir), seq_len)
