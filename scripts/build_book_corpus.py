"""
scripts/build_book_corpus.py

Ties the real pieces together into one corpus-building run:

    raw .txt files (e.g. from gutenberg_importer.py)
      -> clean (strip Gutenberg boilerplate, normalize, quality-filter)
      -> deduplicate (exact hash + MinHash near-dup)
      -> tokenize + pack + shard (Datasets.Processing.packing)
      -> ready for brain.data.packed_dataset.PackedDataset

Usage:
    PYTHONPATH=. python3 scripts/build_book_corpus.py \\
        --input-dir /path/to/raw_txt_files \\
        --tokenizer data/brain_tokenizer.json \\
        --output-dir data/book_corpus_shards \\
        --context-length 4096

Run scripts/train_tokenizer.py first if you don't have a trained tokenizer
yet -- point it at a sample of your cleaned corpus, not the raw Gutenberg
files (boilerplate would otherwise pollute the vocabulary).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from Datasets.Processing.cleaning import clean_text, is_low_quality
from Datasets.Processing.deduplication import MinHashLSH, exact_hash
from Datasets.Processing.packing import build_shards_from_texts
from brain.tokenizer.tokenizer import BrainTokenizer


def load_and_clean_corpus(
    input_dir: Path,
    strip_gutenberg: bool,
    min_chars: int,
) -> list[str]:
    exact_seen: set[str] = set()
    lsh = MinHashLSH()
    kept: list[str] = []
    stats = {"total": 0, "low_quality": 0, "exact_dup": 0, "near_dup": 0, "kept": 0}

    for path in sorted(input_dir.rglob("*.txt")):
        stats["total"] += 1
        raw = path.read_text(encoding="utf-8", errors="ignore")
        text = clean_text(raw, strip_gutenberg=strip_gutenberg, normalize=True)

        if is_low_quality(text, min_chars=min_chars):
            stats["low_quality"] += 1
            continue
        h = exact_hash(text)
        if h in exact_seen:
            stats["exact_dup"] += 1
            continue
        exact_seen.add(h)
        if lsh.is_duplicate_and_add(str(path), text):
            stats["near_dup"] += 1
            continue

        kept.append(text)
        stats["kept"] += 1

    print("Corpus cleaning summary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a packed training corpus from raw text files.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory of raw .txt files (recursive).")
    parser.add_argument("--tokenizer", required=True, type=Path, help="Path to a trained BrainTokenizer JSON.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Where to write the packed shards.")
    parser.add_argument("--context-length", required=True, type=int, help="Model's context length.")
    parser.add_argument("--strip-gutenberg", action="store_true", help="Strip Project Gutenberg boilerplate.")
    parser.add_argument("--min-chars", type=int, default=500, help="Quality filter: minimum document length.")
    parser.add_argument("--sequences-per-shard", type=int, default=50_000)
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.tokenizer} ...")
    tokenizer = BrainTokenizer.from_file(args.tokenizer)
    print(f"Vocab size: {tokenizer.vocab_size}")

    print(f"Scanning {args.input_dir} ...")
    texts = load_and_clean_corpus(args.input_dir, args.strip_gutenberg, args.min_chars)
    if not texts:
        raise SystemExit("No documents survived cleaning -- nothing to pack. Check --input-dir and --min-chars.")

    print(f"Packing {len(texts)} documents at context_length={args.context_length} ...")
    meta = build_shards_from_texts(
        texts, tokenizer, args.output_dir,
        context_length=args.context_length,
        sequences_per_shard=args.sequences_per_shard,
    )
    print("Done.")
    print(f"  Shards: {meta['num_shards']}")
    print(f"  Total sequences: {meta['total_sequences']:,}")
    print(f"  Total tokens: {meta['total_sequences'] * meta['seq_len']:,}")
    print(f"  Written to: {args.output_dir}")


if __name__ == "__main__":
    main()
