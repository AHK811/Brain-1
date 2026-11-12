from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
    return ShardInfo(
        bin_path=bin_path,
        n_tokens=n_tokens,
        seq_len=seq_len,
        token_dtype=dtype,
        split=str(raw.get("split", "unknown")),
        source_files=tuple(raw.get("source_files", ())),
    )


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

    def __len__(self) -> int:
        return max(self._n_sequences - self._offset, 0)

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    def slice_tokens(self, start: int, stop: int) -> np.ndarray:
        parts: list[np.ndarray] = []
        remaining_start = start
        for shard_id, shard in enumerate(self.shards):
            shard_begin = self._starts[shard_id]
            shard_end = shard_begin + int(shard.n_tokens)
            if remaining_start >= shard_end or remaining_start >= stop:
                continue
            local_begin = remaining_start - shard_begin
            local_end = min(stop, shard_end) - shard_begin
            parts.append(np.asarray(self._maps[shard_id][local_begin:local_end]))
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
        return chunk[:-1].astype(np.int64, copy=False), chunk[1:].astype(np.int64, copy=False)

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

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        order = self.indices()
        for start in range(0, len(order), self.batch_size):
            chosen = order[start : start + self.batch_size]
            if len(chosen) < self.batch_size and self.drop_last:
                break
            pairs = [self.shards[int(i)] for i in chosen]
            yield np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])
        self.epoch += 1

    def __len__(self) -> int:
        usable = len(self.shards) // self.batch_size if self.drop_last else -(-len(self.shards) // self.batch_size)
        return max(usable, 0)


def load_shard_dir(shard_dir: Path, seq_len: int | None = None) -> TokenShards:
    return TokenShards(Path(shard_dir), seq_len)
