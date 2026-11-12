from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

from brainlab.data.corpus import documents_from_text, read_text_dir, split_files, synthetic_docs, val_stride
from brainlab.data.pack import PackConfig, pack_stream, split_shards, write_shards
from brainlab.spec.config import load_raw_config
from brainlab.tok.reference import BYTE_TOKENS, Tokenizer


def byte_tokenizer() -> Tokenizer:
    tokens = tuple(bytes([b]) for b in range(BYTE_TOKENS)) + (b"<pad>", b"<eos>")
    return Tokenizer(token_bytes=tokens, merges=(), eos_id=len(tokens) - 1, pad_id=len(tokens) - 2)


def rust_module():
    try:
        import brainlab_rs
    except ImportError:
        return None
    return brainlab_rs


def encode_docs(tokenizer: Tokenizer, docs: Sequence[str], engine: str = "auto") -> tuple[list[list[int]], str]:
    module = rust_module() if engine != "python" else None
    if module is not None:
        blob = json.dumps({"docs": list(docs)}).encode("utf-8")
        raw = module.encode_docs(blob)
        parsed = json.loads(bytes(raw).decode("utf-8"))["docs"]
        return [list(map(int, row)) for row in parsed], "rust"
    return [list(tokenizer.encode(doc)) for doc in docs], "python"


def prepare(
    files: Sequence[tuple[str, str]],
    tokenizer: Tokenizer,
    out_dir: Path,
    cfg: PackConfig,
    val_ratio: float,
    engine: str = "auto",
) -> dict[str, object]:
    started = time.time()

    def documents(items: Sequence[tuple[str, str]]) -> list[str]:
        return [doc for _, text in items for doc in documents_from_text(text)]

    train_files, val_files = split_files(files, val_ratio)
    train_names = [name for name, _ in train_files]
    val_names = [name for name, _ in val_files]
    train_docs, used_engine = encode_docs(tokenizer, documents(train_files), engine)
    val_docs, _ = encode_docs(tokenizer, documents(val_files), engine)
    if not train_docs or not val_docs:
        raise SystemExit("every document tokenized to nothing; check the tokenizer and corpus")
    train_stream = pack_stream(train_docs, cfg)
    val_stream = pack_stream(val_docs, cfg)
    if train_stream.size == 0 or val_stream.size == 0:
        raise SystemExit(f"packing produced train={train_stream.size} val={val_stream.size} tokens")
    train_shards = split_shards(train_stream, cfg)
    val_shards = split_shards(val_stream, cfg)
    write_shards(train_shards, out_dir / "train", "train", cfg, train_names)
    write_shards(val_shards, out_dir / "val", "val", cfg, val_names)
    return {
        "files": len(files),
        "train_files": len(train_names),
        "val_files": len(val_names),
        "val_stride": val_stride(val_ratio) if val_ratio > 0 else None,
        "train_documents": len(train_docs),
        "val_documents": len(val_docs),
        "tokens_train": int(train_stream.size),
        "tokens_val": int(val_stream.size),
        "train_sequences": int(train_stream.size) // cfg.seq_len,
        "val_sequences": int(val_stream.size) // cfg.seq_len,
        "shards_train": len(train_shards),
        "shards_val": len(val_shards),
        "seq_len": cfg.seq_len,
        "shard_tokens": cfg.shard_tokens,
        "engine": used_engine,
        "seconds": round(time.time() - started, 3),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="brainlab-prepare", description="Tokenize and pack a corpus into train and val shards.")
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    parser.add_argument("--tokenizer-dir", type=Path, default=None, help="directory with tokenizer.bin or vocab.json plus merges.txt")
    parser.add_argument("--charlevel", action="store_true", help="use the 258-token byte vocab, no trained merges")
    parser.add_argument("--out", type=Path, default=Path("data/packed"))
    parser.add_argument("--synthetic", type=int, default=0, help="generate N words of deterministic text instead of reading --corpus")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--shard-tokens", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--engine", choices=("auto", "rust", "python"), default="auto")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    raw = load_raw_config()
    data_cfg = raw["data"]
    tok_cfg = raw["tokenizer"]
    seq_len = args.seq_len or int(raw["batch"]["seq_len"])
    shard_tokens = args.shard_tokens or int(data_cfg["shard_tokens"])
    val_ratio = args.val_ratio if args.val_ratio is not None else float(data_cfg["val_ratio"])
    seed = args.seed if args.seed is not None else int(raw["schedule"]["seed"])
    vocab_size = int(tok_cfg["vocab_size"])

    if args.charlevel:
        tokenizer = byte_tokenizer()
    elif args.tokenizer_dir:
        tokenizer = Tokenizer.load(args.tokenizer_dir)
    else:
        raise SystemExit("pass --tokenizer-dir DIR or --charlevel")
    if tokenizer.vocab_size > vocab_size:
        raise SystemExit(f"tokenizer vocab {tokenizer.vocab_size} exceeds configured vocab_size {vocab_size}")

    if args.synthetic:
        files = [(f"{index:06d}.txt", text) for index, text in enumerate(synthetic_docs(args.synthetic, seed=seed))]
    else:
        files = read_text_dir(args.corpus)
        if not files:
            raise SystemExit(f"no .txt files under {args.corpus}")
    cfg = PackConfig(
        seq_len=seq_len,
        shard_tokens=shard_tokens if shard_tokens % seq_len == 0 else (shard_tokens // seq_len) * seq_len,
        eos_id=tokenizer.eos_id,
    )
    report = prepare(files, tokenizer, args.out, cfg, val_ratio, args.engine)
    report["tokenizer_vocab"] = tokenizer.vocab_size
    report["corpus_words"] = sum(len(text.split()) for _, text in files)
    (args.out / "prepare_report.json").parent.mkdir(parents=True, exist_ok=True)
    (args.out / "prepare_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
