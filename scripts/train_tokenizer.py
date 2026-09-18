"""
scripts/train_tokenizer.py

CLI entry point for tokenizer training. Thin wrapper over
brain.tokenizer.tokenizer.BrainTokenizer.train() -- kept separate from the
tokenizer implementation itself so the implementation stays importable
without pulling in argparse/CLI concerns (same separation-of-concerns
principle as the model/API boundary elsewhere in this repo).

Usage:
    PYTHONPATH=. python3 scripts/train_tokenizer.py \\
        --corpus data/sample_corpus.txt \\
        --vocab-size 32768 \\
        --output data/brain_tokenizer.json
"""

import argparse

from brain.tokenizer.tokenizer import BrainTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Brain byte-level BPE tokenizer.")
    parser.add_argument("--corpus", nargs="+", required=True, help="One or more corpus text files.")
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--output", required=True, help="Path to save the trained tokenizer JSON.")
    args = parser.parse_args()

    print(f"Training on: {args.corpus}")
    print(f"Target vocab size: {args.vocab_size}")
    tok = BrainTokenizer.train(
        corpus_paths=args.corpus,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        save_path=args.output,
    )
    print(f"Saved to: {args.output}")
    print(f"Actual vocab size: {tok.vocab_size}")
    if tok.vocab_size < args.vocab_size:
        print(
            f"NOTE: actual vocab ({tok.vocab_size}) is smaller than requested "
            f"({args.vocab_size}) -- the corpus doesn't contain enough distinct "
            f"merges to reach the target. This is expected on small corpora; it "
            f"is the same behavior the original Brain Trail tokenizer had "
            f"(requested 500, got 344 on the 21-line sample corpus)."
        )


if __name__ == "__main__":
    main()
