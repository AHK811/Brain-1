"""Reads the fixed-length token shards written by
Datasets.Processing.packing.ShardWriter, for training.

Since every stored sequence is already a fixed length (packed, not
padded), this deliberately does NOT go through CausalCollator -- there's
no padding to compute, so a plain stack is both correct and faster.

Each stored sequence is `meta["seq_len"]` tokens; __getitem__ splits it
into input_ids=[:-1] and labels=[1:], each `meta["seq_len"] - 1` tokens --
i.e. the model's actual usable context length is one less than the stored
seq_len. Datasets.Processing.packing.build_shards_from_texts already
accounts for this (it stores context_length + 1 per sequence so the
*post-shift* length matches the model's context_length exactly); if shards
were written directly via ShardWriter instead, this class's `.context_length`
property tells you what you actually get after the shift.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedDataset(Dataset):
    def __init__(self, shard_dir: str | Path):
        self.shard_dir = Path(shard_dir)
        meta_path = self.shard_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No meta.json in {self.shard_dir} -- was this directory written by "
                f"Datasets.Processing.packing.ShardWriter?"
            )
        self.meta = json.loads(meta_path.read_text())
        self.seq_len = self.meta["seq_len"]  # stored length, BEFORE the next-token shift
        self.context_length = self.seq_len - 1  # what the model actually trains on
        self.dtype = np.dtype(self.meta["dtype"])

        self._mmaps = [
            np.memmap(self.shard_dir / name, dtype=self.dtype, mode="r").reshape(-1, self.seq_len)
            for name in self.meta["shard_files"]
        ]
        self._shard_lengths = [m.shape[0] for m in self._mmaps]
        self._cum_lengths = np.cumsum([0] + self._shard_lengths)
        total = int(self._cum_lengths[-1])
        if total != self.meta["total_sequences"]:
            raise ValueError(
                f"meta.json claims {self.meta['total_sequences']} sequences but the "
                f"shard files on disk contain {total} -- shards may be missing, "
                f"truncated, or from a different run. Refusing to silently proceed."
            )

    def __len__(self) -> int:
        return int(self._cum_lengths[-1])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0:
            idx += len(self)
        if not (0 <= idx < len(self)):
            raise IndexError(idx)
        shard_idx = int(np.searchsorted(self._cum_lengths, idx, side="right") - 1)
        row_idx = idx - int(self._cum_lengths[shard_idx])
        tokens = torch.from_numpy(self._mmaps[shard_idx][row_idx].astype(np.int64))

        input_ids = tokens[:-1].clone()
        labels = tokens[1:].clone()
        return {"input_ids": input_ids, "labels": labels}
