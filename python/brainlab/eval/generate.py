from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

import torch

from brainlab.data.prepare import byte_tokenizer
from brainlab.export.to_safetensors import load_model
from brainlab.spec.config import load_model_spec
from brainlab.tok.reference import Tokenizer
from brainlab.util import configure_stdout


def keep_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    threshold = logits.topk(k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < threshold, float("-inf"))


def keep_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    if not 0.0 < p <= 1.0:
        raise ValueError(f"top_p must be in (0,1], got {p}")
    if p == 1.0:
        return logits
    sorted_values, sorted_indices = logits.sort(dim=-1, descending=True)
    probabilities = sorted_values.softmax(dim=-1)
    remove = (probabilities.cumsum(dim=-1) - probabilities) >= p
    kept = sorted_values.masked_fill(remove, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, sorted_indices, kept)


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    max_new: int,
    temperature: float,
    top_k: int,
    top_p: float,
    eos_id: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[int], dict[str, float]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    context = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    limit = model.spec.context_len
    started = time.perf_counter()
    first_token_ms: float | None = None
    produced: list[int] = []
    for step in range(max_new):
        window = context[:, -limit:] if context.shape[1] > limit else context
        logits = model(window).to(dtype=dtype)[:, -1, :]
        if first_token_ms is None:
            first_token_ms = (time.perf_counter() - started) * 1000.0
        if temperature <= 0:
            next_id = int(logits.argmax(dim=-1).item())
        else:
            scores = keep_top_p(keep_top_k(logits / temperature, top_k), top_p)
            probs = scores.softmax(dim=-1)
            if not bool(torch.isfinite(probs).all()) or float(probs.sum()) == 0:
                next_id = int(logits.argmax(dim=-1).item())
            else:
                next_id = int(torch.multinomial(probs.cpu(), 1, generator=generator).item())
        if next_id == eos_id:
            break
        produced.append(next_id)
        context = torch.cat([context, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
    elapsed = time.perf_counter() - started
    stats = {
        "ttft_ms": first_token_ms or elapsed * 1000.0,
        "total_ms": elapsed * 1000.0,
        "generated": float(len(produced)),
        "decode_tokens_per_sec": len(produced) / max(elapsed, 1e-9),
    }
    return produced, stats


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="brainlab-generate")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", default="The model")
    parser.add_argument("--tokenizer-dir", type=Path, default=None)
    parser.add_argument("--charlevel", action="store_true")
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    configure_stdout()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but CUDA is unavailable")
    tokenizer: Tokenizer
    if args.tokenizer_dir:
        tokenizer = Tokenizer.load(args.tokenizer_dir)
    elif args.charlevel:
        tokenizer = byte_tokenizer()
    else:
        raise SystemExit("pass --tokenizer-dir DIR or --charlevel")
    spec = load_model_spec()
    model = load_model(args.model, device=str(device), dtype=torch.float32, spec=spec).eval()
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        raise SystemExit("prompt tokenized to nothing")
    produced, stats = generate(
        model,
        prompt_ids,
        args.max_new,
        args.temperature,
        args.top_k,
        args.top_p,
        tokenizer.eos_id,
        args.seed,
        device,
        torch.float32,
    )
    if spec.vocab_size > tokenizer.vocab_size:
        print(
            f"warning: model head has {spec.vocab_size} rows, tokenizer has {tokenizer.vocab_size} tokens; ids above {tokenizer.vocab_size - 1} are undecodable",
            flush=True,
        )
    text = tokenizer.decode_ids(list(prompt_ids) + produced, strict=False)
    if args.json:
        print(json.dumps({"prompt": args.prompt, "text": text, "ids": produced, "stats": stats}, sort_keys=True))
    else:
        print(text)
        print(f"ttft {stats['ttft_ms']:.1f} ms  generated {int(stats['generated'])} tokens  {stats['decode_tokens_per_sec']:.1f} tok/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
