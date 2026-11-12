from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from brainlab.data.shards import TokenShards, discover
from brainlab.util import configure_stdout

BASE = np.uint64(911382323)
BLOCK = 1 << 20


def rolling_hashes(tokens: np.ndarray, span: int) -> np.ndarray:
    stream = np.asarray(tokens, dtype=np.uint64) + np.uint64(1)
    count = stream.size - span + 1
    if count <= 0:
        return np.zeros(0, dtype=np.uint64)
    acc = np.zeros(count, dtype=np.uint64)
    weight = np.uint64(1)
    with np.errstate(over="ignore"):
        for offset in range(span):
            acc += stream[offset : offset + count] * weight
            weight *= BASE
    return acc


def stream_hashes(shards: TokenShards, span: int, block: int = BLOCK) -> np.ndarray:
    parts: list[np.ndarray] = []
    total = shards.total_tokens
    start = 0
    while start < total:
        stop = min(start + block, total)
        begin = max(start - (span - 1), 0)
        parts.append(rolling_hashes(shards.slice_tokens(begin, stop), span))
        start = stop
    if not parts:
        return np.zeros(0, dtype=np.uint64)
    return np.sort(np.concatenate(parts))


def check(
    train_dir: Path,
    val_dir: Path,
    seq_len: int | None = None,
    sequences: int = 100,
    span: int = 50,
    seed: int = 0,
) -> dict[str, object]:
    if span < 2:
        raise ValueError("span must be at least 2 tokens")
    train = TokenShards(train_dir, seq_len)
    val = TokenShards(val_dir, seq_len)
    width = train.seq_len
    train_shards = discover(Path(train_dir))
    val_shards = discover(Path(val_dir))
    shared = sorted({name for shard in train_shards for name in shard.source_files} & {
        name for shard in val_shards for name in shard.source_files
    })
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(val), size=min(sequences, len(val)), replace=False)
    hashes: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    offsets: list[np.ndarray] = []
    for local, index in enumerate(picks):
        ids, _ = val[int(index)]
        found = rolling_hashes(ids, span)
        hashes.append(found)
        owners.append(np.full(found.size, int(index), dtype=np.int64))
        offsets.append(np.arange(found.size, dtype=np.int64))
    if not hashes:
        raise SystemExit("no val sequences to sample")
    sampled = np.concatenate(hashes)
    if sampled.size == 0:
        raise SystemExit(f"span {span} does not fit the {width}-token sequences, so nothing was compared")
    owner = np.concatenate(owners)
    offset = np.concatenate(offsets)
    train_hashes = stream_hashes(train, span)
    if train_hashes.size == 0:
        raise SystemExit("train shard dir holds no windows")
    place = np.searchsorted(train_hashes, sampled)
    clipped = np.minimum(place, train_hashes.size - 1)
    matched = np.nonzero(train_hashes[clipped] == sampled)[0]

    examples = []
    for handle in matched[:5]:
        position = int(offset[handle])
        ids, _ = val[int(owner[handle])]
        examples.append(
            {
                "val_sequence": int(owner[handle]),
                "position": position,
                "tokens": [int(t) for t in ids[position : position + span]],
            }
        )
    return {
        "span": span,
        "seq_len": width,
        "val_sequences_sampled": int(picks.size),
        "val_windows": int(sampled.size),
        "train_windows": int(train_hashes.size),
        "matching_windows": int(matched.size),
        "val_sequences_with_match": int(len(set(owner[matched].tolist()))),
        "shared_source_files": shared,
        "examples": examples,
        "leaked": bool(matched.size) or bool(shared),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="brainlab-leak", description="Search val sequences for train spans.")
    parser.add_argument("--train-dir", type=Path, default=Path("data/packed/train"))
    parser.add_argument("--val-dir", type=Path, default=Path("data/packed/val"))
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--sequences", type=int, default=100)
    parser.add_argument("--span", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    configure_stdout()
    report = check(args.train_dir, args.val_dir, args.seq_len, args.sequences, args.span, args.seed)
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        for name, value in report.items():
            if name != "examples":
                print(f"{name}: {value}")
        for example in report["examples"]:
            print(f"  seq {example['val_sequence']} @{example['position']}: {example['tokens'][:12]}...")
    return 1 if report["leaked"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
