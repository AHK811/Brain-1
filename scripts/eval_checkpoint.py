"""
scripts/eval_checkpoint.py

Report loss/perplexity on a small held-out token memmap or text file.

Usage:
  PYTHONPATH=. python scripts/eval_checkpoint.py \\
      --config configs/model/brain_0_3_35m.yaml \\
      --checkpoint path/to/ckpt.pt \\
      --tokens data/val.tokens.uint16 \\
      --ctx 512 --batches 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from brain.core.config import BrainConfig
from brain.eval.metrics import perplexity_from_loss
from brain.model.brain_model import BrainForCausalLM


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokens", required=True, help="uint16 memmap of token ids")
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--batches", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()

    cfg = BrainConfig.from_yaml(args.config)
    try:
        model = BrainForCausalLM.load_checkpoint(args.checkpoint)
    except Exception:
        model = BrainForCausalLM(cfg)
        payload = torch.load(args.checkpoint, map_location="cpu")
        state = payload.get("model_state_dict", payload)
        model.load_state_dict(state, strict=False)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    n = Path(args.tokens).stat().st_size // 2
    tokens = np.memmap(args.tokens, dtype=np.uint16, mode="r", shape=(n,))
    losses = []
    with torch.no_grad():
        for _ in range(args.batches):
            starts = np.random.randint(0, n - args.ctx - 1, size=args.batch_size)
            x = np.stack([tokens[s : s + args.ctx] for s in starts]).astype(np.int64)
            x_t = torch.from_numpy(x).to(device)
            out = model(x_t, labels=x_t)
            loss = out["loss"] if isinstance(out, dict) else out.loss
            losses.append(float(loss))
    mean_loss = sum(losses) / len(losses)
    print(f"mean_loss={mean_loss:.4f}  ppl={perplexity_from_loss(mean_loss):.2f}")


if __name__ == "__main__":
    main()
