from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from brainlab.data.pack import PackConfig
from brainlab.data.shards import write_weighted_shards
from brainlab.spec.config import load_raw_config
from brainlab.tok.reference import Tokenizer
from brainlab.train.objective import ObjectiveConfig, control_id, tokenizer_provenance


@dataclass(frozen=True)
class ChatFormat:
    system: str = "<|system|>"
    user: str = "<|user|>"
    assistant: str = "<|assistant|>"
    turn_end: str = "<|end|>"


def encode_chat(
    record: Mapping[str, Any],
    tokenizer: Any,
    objective: ObjectiveConfig,
    chat_format: ChatFormat = ChatFormat(),
) -> tuple[list[int], list[float]]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("chat_sft requires a non-empty messages list")
    roles = {role: control_id(tokenizer, getattr(chat_format, role)) for role in ("system", "user", "assistant")}
    turn_end = control_id(tokenizer, chat_format.turn_end)
    if len(set([*roles.values(), turn_end])) != 4:
        raise ValueError("chat role and turn-end controls must be distinct")
    ids: list[int] = []
    weights: list[float] = []
    has_assistant = False
    last_assistant = False
    for message in messages:
        if not isinstance(message, dict) or set(message) - {"role", "content"}:
            raise ValueError("messages contain only role and content")
        role = message.get("role")
        if role not in roles:
            raise ValueError(f"unsupported chat role {role!r}")
        assistant = role == "assistant"
        last_assistant = assistant
        has_assistant |= assistant
        ids.append(roles[role])
        weights.append(0.0)
        content = message.get("content")
        parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
        if not isinstance(parts, list):
            raise ValueError("content must be text or a list of typed text/thought parts")
        for part in parts:
            if not isinstance(part, dict) or set(part) != {"type", "text"} or not isinstance(part["text"], str):
                raise ValueError("content parts require exactly type and text")
            kind = part["type"]
            if kind not in {"text", "thought"} or (kind == "thought" and not assistant):
                raise ValueError("thought parts are allowed only in assistant turns")
            if kind == "thought":
                ids.append(control_id(tokenizer, objective.thought_start))
                weights.append(1.0)
            encoded = list(tokenizer.encode(part["text"], allow_special=False))
            ids.extend(encoded)
            weight = objective.thought_weight if kind == "thought" else float(assistant)
            weights.extend([weight] * len(encoded))
            if kind == "thought":
                ids.append(control_id(tokenizer, objective.thought_end))
                weights.append(1.0)
        ids.append(turn_end)
        weights.append(float(assistant))
    if not has_assistant:
        raise ValueError("chat_sft requires at least one assistant turn")
    ids.append(tokenizer.eos_id)
    weights.append(float(last_assistant))
    return ids, weights


def pack_chats(
    records: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    cfg: PackConfig,
    objective: ObjectiveConfig,
    chat_format: ChatFormat = ChatFormat(),
) -> tuple[np.ndarray, np.ndarray]:
    if cfg.eos_id != tokenizer.eos_id:
        raise ValueError("pack eos_id differs from tokenizer")
    tokens: list[int] = []
    weights: list[float] = []
    for record in records:
        ids, mask = encode_chat(record, tokenizer, objective, chat_format)
        tokens.extend(ids)
        weights.extend(mask)
    if not tokens:
        raise ValueError("no chat records to pack")
    padding = (-(len(tokens) + 1)) % cfg.seq_len + 1
    tokens.extend([tokenizer.pad_id] * padding)
    weights.extend([0.0] * padding)
    return np.asarray(tokens, dtype=np.int64), np.asarray(weights, dtype=np.float32)


def prepare_sft(
    inputs: Sequence[Path],
    tokenizer: Any,
    out_dir: Path,
    cfg: PackConfig,
    objective: ObjectiveConfig,
    split: str = "train",
    chat_format: ChatFormat = ChatFormat(),
) -> list[Path]:
    records: list[Mapping[str, Any]] = []
    sources = []
    for path in inputs:
        raw = Path(path).read_bytes()
        sources.append({"file": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
        for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("each JSONL record must be an object")
                encode_chat(record, tokenizer, objective, chat_format)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            records.append(record)
    tokens, weights = pack_chats(records, tokenizer, cfg, objective, chat_format)
    metadata = {
        "data_format": "chat_sft",
        "format_version": 1,
        "chat_format": asdict(chat_format),
        "objective": objective.as_dict(),
        "tokenizer": tokenizer_provenance(tokenizer),
        "sources": sources,
        "records": len(records),
        "labels": "assistant_only",
        "thought_scope": "typed_assistant_content_only; delimiters have ordinary weight",
    }
    return write_weighted_shards(tokens, weights, out_dir, split, cfg, metadata, [str(p) for p in inputs])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pack explicit JSONL chat SFT with assistant-only labels.")
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--shard-tokens", type=int, default=None)
    parser.add_argument("--thought-weight", type=float, default=None)
    parser.add_argument("--thought-start", default=None)
    parser.add_argument("--thought-end", default=None)
    for name, default in asdict(ChatFormat()).items():
        parser.add_argument("--" + name.replace("_", "-") + "-token", default=default)
    args = parser.parse_args(argv)
    raw = load_raw_config()
    tokenizer = Tokenizer.load(args.tokenizer_dir)
    objective_cfg = dict(raw.get("objective", {}))
    for name in ("thought_weight", "thought_start", "thought_end"):
        if getattr(args, name) is not None:
            objective_cfg[name] = getattr(args, name)
    objective = ObjectiveConfig.from_config(objective_cfg)
    seq_len = args.seq_len or int(raw["batch"]["seq_len"])
    cfg = PackConfig(seq_len, args.shard_tokens or int(raw["data"]["shard_tokens"]), tokenizer.eos_id)
    chat_format = ChatFormat(**{name: getattr(args, name + "_token") for name in asdict(ChatFormat())})
    paths = prepare_sft(args.input, tokenizer, args.out, cfg, objective, args.split, chat_format)
    print(json.dumps({"data_format": "chat_sft", "shards": len(paths), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
