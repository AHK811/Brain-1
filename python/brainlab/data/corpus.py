from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

NO_VAL = (1 << 64) - 1

WORD_LIST = (
    "the model trains because the gradient flows through every normalized residual branch",
    "a kernel is only useful once its output matches the reference within a stated tolerance",
    "memory bound operations reward bandwidth and punish redundant passes over the row",
    "the tokenizer splits text into bytes then merges the most frequent adjacent pairs",
    "attention costs grow with the square of the sequence length so causal masking matters",
    "weights travel between languages as a single safetensors file and never as copies",
    "the benchmark records hardware driver versions and medians because shared gpus are noisy",
    "mixed precision in half float needs a scaler because the tensor cores dislike accumulation",
    "a small language model can still write fluent nonsense about optimization if hurried",
    "measure parity first then measure speed otherwise the fast number describes a wrong result",
)
_CONNECTORS = ("however", "therefore", "and", "or", "because", "although", "since", "when", "if", "so")
_SYMBOLS = (",", ".", ";", ":", "!", "?", '"', "'", "(", ")", "-", "--", "*", "#", "%", "$", "[", "]", "{", "}", "/")
_NUMBERS = ("1", "42", "1024", "3.14159", "0.001", "1789", "256", "65536", "1000000")
_UNICODE = ("naïve", "résumé", "日本語", "Ünïcödé", "Ωmega", "café", "ξ", "€uro", "—dash", "‘quote’")


def synthetic_docs(total_tokens: int, seed: int = 0, doc_sentences: tuple[int, int] = (2, 12)) -> list[str]:
    rng = np.random.default_rng([seed, total_tokens])
    docs: list[str] = []
    produced = 0
    while produced < total_tokens:
        count = int(rng.integers(doc_sentences[0], doc_sentences[1] + 1))
        sentences = []
        for _ in range(count):
            base = WORD_LIST[int(rng.integers(len(WORD_LIST)))]
            words = base.split()
            inserted = int(rng.integers(0, max(len(words), 1)))
            if rng.random() < 0.55:
                pool = _CONNECTORS if rng.random() < 0.5 else _NUMBERS if rng.random() < 0.5 else _UNICODE
                words.insert(inserted, pool[int(rng.integers(len(pool)))])
            sentence = " ".join(words)
            if rng.random() < 0.3:
                sentence += " " + _SYMBOLS[int(rng.integers(len(_SYMBOLS)))] + " " + sentence[:24]
            sentences.append(sentence.strip())
        doc = ". ".join(sentences).strip() + "."
        docs.append(doc)
        produced += len(doc.split()) + 1
    return docs


DOC_SEPS = ("blank", "line", "file")


def documents_from_text(text: str, min_chars: int = 1, doc_sep: str = "blank") -> list[str]:
    if doc_sep == "file":
        blocks: list[str] = [text]
    elif doc_sep == "line":
        blocks = text.splitlines()
    elif doc_sep == "blank":
        blocks = re.split(r"\n[ \t\r\n\v\f]*\n", text)
    else:
        blocks = text.split(doc_sep)
    return [block.strip() for block in blocks if len(block.strip()) >= max(min_chars, 1)]


def read_text_dir(root: Path, suffixes: Sequence[str] = (".txt",)) -> list[tuple[str, str]]:
    out = []
    for name, path in text_file_paths(root, suffixes):
        out.append((name, path.read_text(encoding="utf-8", errors="replace")))
    return out


def text_file_paths(root: Path, suffixes: Sequence[str] = (".txt",)) -> list[tuple[str, Path]]:
    root = Path(root)
    paths = sorted((p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes), key=str)
    return [(path.relative_to(root).as_posix(), path) for path in paths]


def val_stride(val_ratio: float) -> int:
    if val_ratio <= 0.0:
        return NO_VAL
    return max(1, math.floor(1.0 / val_ratio + 0.5))


def split_files(files: Iterable[tuple[str, str]], val_ratio: float) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    ordered = sorted(files, key=lambda item: item[0])
    if not ordered:
        return [], []
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0,1), got {val_ratio}")
    stride = val_stride(val_ratio)
    val_idx = {index for index in range(len(ordered)) if stride != NO_VAL and index % stride == 0}
    train_files = [item for index, item in enumerate(ordered) if index not in val_idx]
    val_files = [item for index, item in enumerate(ordered) if index in val_idx]
    if not train_files or (val_ratio > 0 and len(ordered) > 1 and not val_files):
        raise ValueError(f"split produced train={len(train_files)} val={len(val_files)} from {len(ordered)} files")
    if val_ratio > 0 and len(ordered) > 1:
        achieved = len(val_files) / len(ordered)
        if achieved > 3.0 * val_ratio:
            raise ValueError(
                f"{len(ordered)} files give {achieved:.1%} val, wanted {val_ratio:.1%}; "
                f"a file-level stride needs at least {math.ceil(1.0 / val_ratio)} files, "
                f"so shard the corpus into more files or accept the larger split explicitly"
            )
    return train_files, val_files


def stable_token_count(texts: Iterable[str]) -> int:
    return sum(len(t.split()) for t in texts)


def corpus_digest(files: Sequence[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for name, text in sorted(files, key=lambda item: item[0]):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8", errors="replace"))
    return digest.hexdigest()
