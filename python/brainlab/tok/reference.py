from __future__ import annotations

import json
import struct
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC = b"BLTK"
FORMAT_VERSION = 1
BYTE_TOKENS = 256
MAX_WORD_BYTES = 32


def is_alnum(ch: str) -> bool:
    group = unicodedata.category(ch)[0]
    return group in {"L", "N"}


def is_digit(ch: str) -> bool:
    return "0" <= ch <= "9"


def char_class(text: str, index: int) -> str:
    ch = text[index]
    if ch.isspace():
        return "w"
    if is_alnum(ch):
        return "a"
    if ch == "-" and 0 < index < len(text) - 1 and is_alnum(text[index - 1]) and is_alnum(text[index + 1]):
        return "a"
    return "o"


def split_digit_run(word: str) -> list[str]:
    if len(word) <= 3 or not all(is_digit(ch) for ch in word):
        return [word]
    head = len(word) % 3
    groups = [word[:head]] if head else []
    start = head
    while start < len(word):
        groups.append(word[start : start + 3])
        start += 3
    return groups


def pre_tokenize(text: str) -> list[str]:
    words: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        group = char_class(text, index)
        end = index + 1
        while end < length and char_class(text, end) == group:
            end += 1
        run = text[index:end]
        if group == "a":
            words.extend(split_digit_run(run))
        else:
            words.append(run)
        index = end
    return words


def word_chunks(word: str, max_bytes: int = MAX_WORD_BYTES) -> list[bytes]:
    chunks: list[bytes] = []
    current = bytearray()
    for ch in word:
        raw = ch.encode("utf-8")
        if len(raw) > max_bytes:
            raise ValueError(f"code point U+{ord(ch):04X} exceeds max_word_bytes")
        if current and len(current) + len(raw) > max_bytes:
            chunks.append(bytes(current))
            current = bytearray()
        current.extend(raw)
    if current:
        chunks.append(bytes(current))
    return chunks


def encode_chunk(chunk: bytes, ranks: dict[tuple[int, int], tuple[int, int]]) -> list[int]:
    syms = list(chunk)
    while len(syms) >= 2:
        best_rank: int | None = None
        best_pos = -1
        best_id = -1
        for pos in range(len(syms) - 1):
            hit = ranks.get((syms[pos], syms[pos + 1]))
            if hit is None:
                continue
            if best_rank is None or hit[0] < best_rank:
                best_rank, best_pos, best_id = hit[0], pos, hit[1]
        if best_rank is None:
            break
        syms[best_pos : best_pos + 2] = [best_id]
    return syms


@dataclass(frozen=True)
class Tokenizer:
    token_bytes: tuple[bytes, ...]
    merges: tuple[tuple[int, int], ...]
    eos_id: int
    pad_id: int
    max_word_bytes: int = MAX_WORD_BYTES
    _ranks: dict[tuple[int, int], tuple[int, int]] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.token_bytes) <= max(self.eos_id, self.pad_id):
            raise ValueError("special ids outside vocab")
        for index, pair in enumerate(self.merges):
            self._ranks.setdefault(pair, (index, BYTE_TOKENS + index))
        expected = tuple(range(BYTE_TOKENS))
        if tuple(self.token_bytes[:BYTE_TOKENS]) != tuple(bytes([b]) for b in expected):
            raise ValueError("ids 0..255 must be single bytes per spec/tokenizer_spec.md")

    @property
    def vocab_size(self) -> int:
        return len(self.token_bytes)

    def encode_word(self, word: str) -> list[int]:
        out: list[int] = []
        for chunk in word_chunks(word, self.max_word_bytes):
            out.extend(encode_chunk(chunk, self._ranks))
        return out

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for word in pre_tokenize(text):
            out.extend(self.encode_word(word))
        return out

    def encode_docs(self, texts: Sequence[str]) -> list[list[int]]:
        return [self.encode(text) for text in texts]

    def decode_ids(self, ids: Iterable[int], strict: bool = True) -> str:
        parts = bytearray()
        unknown = 0
        for token_id in ids:
            index = int(token_id)
            if not 0 <= index < self.vocab_size:
                if strict:
                    raise ValueError(f"token id {index} outside vocab {self.vocab_size}")
                unknown += 1
                parts += "\ufffd".encode("utf-8")
                continue
            parts.extend(self.token_bytes[index])
        if unknown:
            parts += f" [{unknown} undecodable ids]".encode("utf-8")
        return parts.decode("utf-8", errors="replace")

    def encode_decode(self, text: str) -> list[int]:
        ids = self.encode(text)
        if self.decode_ids(ids) != text:
            raise AssertionError("roundtrip changed the text; the vocab or decoder is wrong")
        return ids

    def to_numpy(self, ids: Sequence[int]) -> np.ndarray:
        return np.asarray(ids, dtype=np.uint16 if self.vocab_size <= 65536 else np.uint32)

    @classmethod
    def from_bin(cls, path: Path) -> Tokenizer:
        raw = path.read_bytes()
        magic, version, vocab_size, n_merges = struct.unpack_from("<4sHII", raw, 0)
        if magic != MAGIC:
            raise ValueError(f"{path}: bad magic {magic!r}")
        if version != FORMAT_VERSION:
            raise ValueError(f"{path}: unsupported format version {version}")
        offset = 14
        tokens: list[bytes] = []
        for _ in range(vocab_size):
            length = raw[offset]
            offset += 1
            tokens.append(bytes(raw[offset : offset + length]))
            offset += length
        merges: list[tuple[int, int]] = []
        for _ in range(n_merges):
            merges.append(struct.unpack_from("<II", raw, offset))
            offset += 8
        eos_id, pad_id = struct.unpack_from("<II", raw, offset)
        offset += 8
        if offset != len(raw):
            raise ValueError(f"{path}: {len(raw) - offset} trailing bytes")
        return cls(token_bytes=tuple(tokens), merges=tuple(merges), eos_id=int(eos_id), pad_id=int(pad_id))

    @classmethod
    def from_vocab_files(cls, vocab_path: Path, merges_path: Path) -> Tokenizer:
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        pairs: list[tuple[int, int]] = []
        for line_no, line in enumerate(merges_path.read_text(encoding="utf-8").splitlines()):
            if line_no == 0:
                if not line.startswith("#version:"):
                    raise ValueError(f"{merges_path}: first line must be a version comment")
                continue
            if not line.strip():
                continue
            left, right = line.split()
            pairs.append((int(left), int(right)))
        by_id: list[bytes] = [b""] * max(vocab.values()) + 1
        for key, token_id in vocab.items():
            by_id[int(token_id)] = key.encode("latin-1")
        if any(item == b"" for item in by_id[:BYTE_TOKENS]):
            raise ValueError(f"{vocab_path}: byte tokens 0..255 are missing")
        eos_id, pad_id = len(by_id) - 1, len(by_id) - 2
        return cls(token_bytes=tuple(by_id), merges=tuple(pairs), eos_id=eos_id, pad_id=pad_id)

    @classmethod
    def load(cls, directory: Path) -> Tokenizer:
        directory = Path(directory)
        binary = directory / "tokenizer.bin"
        if binary.is_file():
            return cls.from_bin(binary)
        return cls.from_vocab_files(directory / "vocab.json", directory / "merges.txt")


def write_bin(tokenizer: Tokenizer, path: Path) -> Path:
    body = bytearray(struct.pack("<4sHII", MAGIC, FORMAT_VERSION, tokenizer.vocab_size, len(tokenizer.merges)))
    for token in tokenizer.token_bytes:
        if len(token) > 255:
            raise ValueError("token longer than 255 bytes cannot be stored in tokenizer.bin")
        body.append(len(token))
        body.extend(token)
    for pair in tokenizer.merges:
        body.extend(struct.pack("<II", *pair))
    body.extend(struct.pack("<II", tokenizer.eos_id, tokenizer.pad_id))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(body))
    return path
