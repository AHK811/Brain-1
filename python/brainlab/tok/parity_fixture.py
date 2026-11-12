from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from brainlab.data.corpus import synthetic_docs
from brainlab.spec.config import repo_root
from brainlab.tok.reference import BYTE_TOKENS, Tokenizer, pre_tokenize, word_chunks, write_bin

FIXTURE_VERSION = 1

SEED_MERGES: tuple[tuple[str, str], ...] = (
    ("t", "h"),
    ("t", "r"),
    ("h", "e"),
    ("th", "e"),
    ("m", "o"),
    ("d", "e"),
    ("de", "l"),
    ("mo", "del"),
    ("n", "g"),
    ("i", "n"),
    ("i", "ng"),
    ("tr", "a"),
    ("tra", "in"),
    ("the", "model"),
)


def build_tokenizer() -> Tokenizer:
    tokens: list[bytes] = [bytes([index]) for index in range(BYTE_TOKENS)]
    ids: dict[bytes, int] = {token: index for index, token in enumerate(tokens)}
    merges: list[tuple[int, int]] = []
    for left, right in SEED_MERGES:
        left_bytes = left.encode("utf-8")
        right_bytes = right.encode("utf-8")
        if left_bytes not in ids or right_bytes not in ids:
            raise ValueError(f"merge {left!r}+{right!r} references a token that was never created")
        merged = left_bytes + right_bytes
        if merged in ids:
            raise ValueError(f"merge {left!r}+{right!r} already exists")
        merges.append((ids[left_bytes], ids[right_bytes]))
        ids[merged] = len(tokens)
        tokens.append(merged)
    tokens.append(b"<pad>")
    tokens.append(b"<eos>")
    return Tokenizer(token_bytes=tuple(tokens), merges=tuple(merges), eos_id=len(tokens) - 1, pad_id=len(tokens) - 2)


def adversarial_docs() -> list[str]:
    return [
        "The model trains, and the gradient flows.",
        "state-of-the-art 1234567 -42 3.14159",
        "naïve résumé, 日本語 text, €uro and Ωmega.",
        "a\n\nb",
        "   leading and trailing   ",
        "tabs\tbetween\twords and\r\nwindows newlines",
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa (44 a's)",
        "!!!!????;;;;,,,, ---- ==== ###$$$",
        "MiXeD CaSe Is PreSeRvEd By ThIs ToKeNiZeR",
        "«quotes» ‘smart’ “curly” —dash…ellipsis",
        "1 22 333 4444 55555 666666",
        "combining é and precomposed é differ",
    ]


def fixture_docs(count: int = 40, seed: int = 0) -> list[str]:
    return synthetic_docs(count * 12, seed=seed) + adversarial_docs()


def build_fixture() -> dict[str, object]:
    tokenizer = build_tokenizer()
    docs = fixture_docs()
    return {
        "version": FIXTURE_VERSION,
        "vocab_size": tokenizer.vocab_size,
        "eos_id": tokenizer.eos_id,
        "pad_id": tokenizer.pad_id,
        "tokens_hex": [token.hex() for token in tokenizer.token_bytes],
        "merges": [[int(left), int(right)] for left, right in tokenizer.merges],
        "docs": docs,
        "pre_tokenize": [pre_tokenize(doc) for doc in docs],
        "word_chunks_hex": [[chunk.hex() for chunk in word_chunks(word)] for doc in docs for word in pre_tokenize(doc)],
        "ids": [tokenizer.encode(doc) for doc in docs],
    }


def write_fixture(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_fixture(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_against_file(payload: dict[str, object], path: Path) -> list[str]:
    stored = load_fixture(path)
    problems: list[str] = []
    for key in ("version", "vocab_size", "eos_id", "pad_id", "tokens_hex", "merges", "docs", "pre_tokenize", "ids"):
        if stored.get(key) != payload.get(key):
            problems.append(key)
    return problems


def default_fixture_path() -> Path:
    return repo_root() / "tests" / "golden" / "tokenizer_parity.json"


DATAPACK_CONFIG = {"seq_len": 16, "shard_tokens": 48, "val_ratio": 0.25}


def datapack_corpus() -> list[tuple[str, str]]:
    docs = fixture_docs(count=40, seed=3)
    files: list[tuple[str, str]] = []
    for index in range(5):
        chosen = [docs[(index * 7 + offset * 3) % len(docs)] for offset in range(3)]
        text = "\n\n".join(chosen) + "\n\ntrailing paragraph that keeps this shard comfortably longer than one sequence of tokens.\n"
        files.append((f"group-{index // 2}/file-{index}.txt", text))
    return files


def write_datapack_inputs(root: Path) -> dict[str, object]:
    corpus = root / "corpus"
    for name, text in datapack_corpus():
        path = corpus / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    write_bin(build_tokenizer(), root / "tokenizer" / "tokenizer.bin")
    return dict(DATAPACK_CONFIG)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="write or verify the cross-language tokenizer parity fixture")
    parser.add_argument("--out", type=Path, default=default_fixture_path())
    parser.add_argument("--verify", action="store_true", help="fail if the stored fixture differs from this run")
    parser.add_argument("--datapack", type=Path, default=None, help="write corpus and tokenizer inputs for the datapack fixture")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.datapack:
        config = write_datapack_inputs(args.datapack)
        print(json.dumps({"root": str(args.datapack), "files": len(datapack_corpus()), **config}))
        return 0
    payload = build_fixture()
    if args.verify:
        if not args.out.is_file():
            print(f"missing fixture: {args.out}; run without --verify to write it")
            return 1
        problems = verify_against_file(payload, args.out)
        print(json.dumps({"fixture": str(args.out), "drift": problems}))
        return 1 if problems else 0
    write_fixture(args.out, payload)
    print(
        json.dumps(
            {
                "fixture": str(args.out),
                "vocab_size": payload["vocab_size"],
                "merges": len(payload["merges"]),
                "docs": len(payload["docs"]),
                "bytes": args.out.stat().st_size,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
