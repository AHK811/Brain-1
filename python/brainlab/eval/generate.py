from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from brainlab.data.prepare import byte_tokenizer
from brainlab.eval.constraints import ConstraintError, JsonSchemaConstraint, RegexConstraint, parse_json
from brainlab.eval.tools import ToolError, ToolRunner, safe_cli_tools
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


class GenerationError(ValueError):
    pass


def sample_token(
    logits: torch.Tensor,
    allowed: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: torch.Generator,
) -> int:
    if not math.isfinite(temperature) or temperature < 0 or top_k < 0 or not 0 < top_p <= 1:
        raise GenerationError("invalid sampling parameters")
    scores = logits.to(dtype=torch.float64).masked_fill(~allowed, float("-inf"))
    if bool(torch.isnan(scores).any()) or bool(torch.isposinf(scores).any()):
        raise GenerationError("non-finite allowed logits")
    if not bool(torch.isfinite(scores).any()):
        raise GenerationError("no allowed token has a finite logit")
    if temperature == 0:
        return int(scores.argmax(dim=-1).item())
    scores = (scores - scores.max()) / temperature
    scores = keep_top_p(keep_top_k(scores, top_k), top_p)
    probs = scores.softmax(dim=-1)
    if not bool(torch.isfinite(probs).all()) or float(probs.sum()) <= 0:
        raise GenerationError("invalid sampling probabilities")
    return int(torch.multinomial(probs.cpu(), 1, generator=generator).item())


def control_id(tokenizer: Tokenizer, spelling: str) -> int:
    lookup = getattr(tokenizer, "special_id", None)
    if lookup is None:
        raise GenerationError(f"tokenizer has no registered control {spelling}")
    try:
        token_id = lookup(spelling)
    except (KeyError, ValueError) as exc:
        raise GenerationError(f"tokenizer has no registered control {spelling}") from exc
    if token_id is None or not 0 <= token_id < tokenizer.vocab_size:
        raise GenerationError(f"tokenizer has no registered control {spelling}")
    return int(token_id)


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
    *,
    tokenizer: Tokenizer | None = None,
    constraint: RegexConstraint | None = None,
    tools: ToolRunner | None = None,
    reason_by_default: bool = False,
) -> tuple[list[int], dict[str, Any]]:
    """Return continuation IDs including tool frames; generated counts only model tokens.

    reason_by_default seeds a formatting control, not trained reasoning. Constraints
    start fresh for every call and apply to ordinary output, not tool frames.
    """
    if max_new < 0 or not math.isfinite(temperature) or temperature < 0 or top_k < 0 or not 0 < top_p <= 1:
        raise GenerationError("invalid generation budget or sampling parameters")
    if (constraint is not None or tools is not None or reason_by_default) and tokenizer is None:
        raise GenerationError("constraints, tools and reasoning format require a tokenizer")
    context_ids = list(prompt_ids)
    if not context_ids:
        raise GenerationError("prompt tokenized to nothing")
    if tokenizer is not None:
        if eos_id != tokenizer.eos_id:
            raise GenerationError("eos_id does not match tokenizer")
        if any(not 0 <= token < tokenizer.vocab_size for token in context_ids):
            raise GenerationError("prompt contains undecodable token ids")
    grammar = constraint.fresh() if constraint is not None else None
    runner = tools.fresh() if tools is not None else None
    controls = {}
    if runner is not None:
        controls = {name: control_id(tokenizer, f"<|{name}|>") for name in (
            "call_start", "call_end", "response_start", "response_end",
        )}
    produced: list[int] = []
    if reason_by_default:
        thought = control_id(tokenizer, "<|thought_start|>")
        context_ids.append(thought)
        produced.append(thought)
    limit = model.spec.context_len
    if len(context_ids) > limit:
        raise GenerationError("prompt exceeds model context limit")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    started = time.perf_counter()
    first_token_ms: float | None = None
    cached_forward = getattr(model, "forward_cached", None)
    cache = None
    processed = 0
    sampled = 0
    inserted = len(produced)
    call_grammar: RegexConstraint | None = None
    finish_reason = "length"
    for step in range(max_new):
        if len(context_ids) >= limit:
            finish_reason = "context"
            break
        ids = context_ids[processed:] if cached_forward is not None else context_ids
        context = torch.tensor([ids], dtype=torch.long, device=device)
        if cached_forward is not None:
            output, cache = cached_forward(context, cache=cache)
            if cache is None:
                raise GenerationError("forward_cached returned no cache")
            processed = len(context_ids)
        else:
            output = model(context)
        logits = output.to(dtype=dtype)[:, -1, :]
        if logits.shape[0] != 1:
            raise GenerationError("generation supports one sequence")
        if first_token_ms is None:
            first_token_ms = (time.perf_counter() - started) * 1000.0
        allowed = torch.ones_like(logits, dtype=torch.bool)
        if tokenizer is not None:
            allowed[:, tokenizer.vocab_size:] = False
            forbidden = {tokenizer.pad_id, tokenizer.eos_id}
            forbidden.update(getattr(tokenizer, "special_tokens", {}).values())
            for token in forbidden:
                if 0 <= token < logits.shape[-1]:
                    allowed[:, token] = False
            active = call_grammar if call_grammar is not None else grammar
            if active is not None:
                for token in range(min(tokenizer.vocab_size, logits.shape[-1])):
                    if token not in forbidden and not active.allows(tokenizer.token_bytes[token]):
                        allowed[:, token] = False
            if call_grammar is not None:
                if call_grammar.complete:
                    runner.validate_call(call_grammar.data)
                    allowed[:, controls["call_end"]] = True
            else:
                if grammar is None or grammar.complete:
                    if 0 <= eos_id < logits.shape[-1]:
                        allowed[:, eos_id] = True
                if runner is not None and runner.calls < runner.max_calls and step + 3 < max_new:
                    allowed[:, controls["call_start"]] = True
        next_id = sample_token(logits, allowed, temperature, top_k, top_p, generator)
        if next_id == eos_id:
            finish_reason = "eos"
            break
        sampled += 1
        context_ids.append(next_id)
        produced.append(next_id)
        if runner is not None and call_grammar is None and next_id == controls["call_start"]:
            call_grammar = runner.grammar.fresh()
        elif runner is not None and call_grammar is not None and next_id == controls["call_end"]:
            if step + 1 >= max_new or len(context_ids) + 3 >= limit:
                raise GenerationError("no token/context budget to resume after tool call")
            result = runner.dispatch(call_grammar.data)
            response = [controls["response_start"], *tokenizer.encode(result, allow_special=False), controls["response_end"]]
            if len(context_ids) + len(response) >= limit:
                raise GenerationError("tool result exceeds remaining context limit")
            context_ids.extend(response)
            produced.extend(response)
            inserted += len(response)
            call_grammar = None
        elif call_grammar is not None:
            call_grammar.advance(tokenizer.token_bytes[next_id])
        elif grammar is not None:
            grammar.advance(tokenizer.token_bytes[next_id])
    if call_grammar is not None:
        raise GenerationError(f"{finish_reason} limit left an incomplete tool call")
    if grammar is not None:
        try:
            grammar.finish()
        except ConstraintError as exc:
            raise GenerationError(f"{finish_reason}: {exc}") from exc
    elapsed = time.perf_counter() - started
    stats = {
        "ttft_ms": first_token_ms or elapsed * 1000.0,
        "total_ms": elapsed * 1000.0,
        "generated": float(sampled),
        "inserted_tokens": inserted,
        "decode_tokens_per_sec": sampled / max(elapsed, 1e-9),
        "finish_reason": finish_reason,
        "constraint_complete": grammar.complete if grammar is not None else None,
        "tool_calls": runner.calls if runner is not None else 0,
        "used_kv_cache": cached_forward is not None,
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


