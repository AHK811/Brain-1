from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from brainlab.data.shards import ShardBatcher, TokenShards
from brainlab.export.to_safetensors import load_model
from brainlab.spec.config import load_model_spec, load_raw_config
from brainlab.util import configure_stdout


@torch.no_grad()
def measure(
    model: torch.nn.Module,
    batcher: ShardBatcher,
    device: torch.device,
    max_batches: int | None = None,
    amp: bool = False,
) -> dict[str, float]:
    model.eval()
    total = 0.0
    counted = 0
    batches = 0
    for index, (x, y) in enumerate(batcher):
        if max_batches is not None and index >= max_batches:
            break
        ids = torch.from_numpy(np.ascontiguousarray(x)).to(device)
        targets = torch.from_numpy(np.ascontiguousarray(y)).to(device)
        with torch.autocast(device_type=device.type, enabled=amp, dtype=torch.float16):
            _, loss = model(ids, targets)
        n = int(np.count_nonzero(y != -1))
        total += float(loss.detach().float()) * n
        counted += n
        batches += 1
    if counted == 0:
        raise SystemExit("no labelled tokens to evaluate")
    mean = total / counted
    return {
        "loss": mean,
        "perplexity": math.exp(mean),
        "bits_per_token": mean / math.log(2.0),
        "tokens": float(counted),
        "batches": float(batches),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="brainlab-ppl")
    parser.add_argument("--model", type=Path, required=True, help="safetensors checkpoint")
    parser.add_argument("--data-dir", type=Path, default=Path("data/packed/val"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--batches", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    configure_stdout()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    spec = load_model_spec()
    raw = load_raw_config()
    model = load_model(args.model, device=str(device), dtype=torch.float32, spec=spec)
    seq_len = args.seq_len or int(raw["batch"]["seq_len"])
    shards = TokenShards(args.data_dir, seq_len)
    batcher = ShardBatcher(shards, min(args.batch, len(shards)) or 1, shuffle_seed=None)
    result = measure(model, batcher, device, args.batches, amp=False)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"loss {result['loss']:.4f}  ppl {result['perplexity']:.2f}  bits/token {result['bits_per_token']:.4f}")
        print(f"{int(result['tokens']):,} tokens over {int(result['batches'])} batches on {device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
