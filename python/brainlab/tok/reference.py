from __future__ import annotations

import json
import re
import struct
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import numpy as np

MAGIC = b"BLTK"
FORMAT_VERSION = 2
BYTE_TOKENS = 256
MAX_WORD_BYTES = 32
CONTROL_TOKENS = (
    "<|image_start|>",
    "<|image_end|>",
    "<|thought_start|>",
    "<|thought_end|>",
    "<|call_start|>",
    "<|call_end|>",
    "<|response_start|>",
    "<|response_end|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|turn_end|>",
)
SPECIAL_TOKENS = (*CONTROL_TOKENS, "<pad>", "<eos>")
_SPECIAL_PATTERN = re.compile("|".join(re.escape(token) for token in SPECIAL_TOKENS))


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
    special_tokens: Mapping[str, int] = field(init=False, repr=False)
    _ranks: dict[tuple[int, int], tuple[int, int]] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        special_start = BYTE_TOKENS + len(self.merges)
        special_count = self.vocab_size - special_start
        if special_count not in (2, len(SPECIAL_TOKENS)):
            raise ValueError("vocab must contain bytes, ranked merges, then the complete special suffix")
        labels = SPECIAL_TOKENS if special_count == len(SPECIAL_TOKENS) else SPECIAL_TOKENS[-2:]
        if self.token_bytes[special_start:] != tuple(label.encode("ascii") for label in labels):
            raise ValueError("special tokens are missing or out of order")
        if self.eos_id != self.vocab_size - 1 or self.pad_id != self.vocab_size - 2:
            raise ValueError("pad and eos must be the final two ids")
        if not 4 <= self.max_word_bytes <= MAX_WORD_BYTES:
            raise ValueError("max_word_bytes must be between 4 and 32")
        if self.token_bytes[:BYTE_TOKENS] != tuple(bytes([b]) for b in range(BYTE_TOKENS)):
            raise ValueError("ids 0..255 must be single bytes per spec/tokenizer_spec.md")
        protected = tuple(label.encode("ascii") for label in SPECIAL_TOKENS)
        seen = set(self.token_bytes[:BYTE_TOKENS])
        for rank, pair in enumerate(self.merges):
            token_id = BYTE_TOKENS + rank
            if len(pair) != 2 or any(type(child) is not int or not 0 <= child < token_id for child in pair):
                raise ValueError(f"merge {rank} must reference earlier ordinary ids")
            token = self.token_bytes[token_id]
            if token != self.token_bytes[pair[0]] + self.token_bytes[pair[1]]:
                raise ValueError(f"merge {rank} bytes disagree with its children")
            if len(token) > MAX_WORD_BYTES or any(label in token for label in protected):
                raise ValueError(f"merge {rank} is too long or contains protected bytes")
            if token in seen or pair in self._ranks:
                raise ValueError(f"duplicate merge {rank}")
            seen.add(token)
            self._ranks[pair] = (rank, token_id)
        object.__setattr__(
            self, "special_tokens", MappingProxyType({label: special_start + i for i, label in enumerate(labels)})
        )

    @property
    def vocab_size(self) -> int:
        return len(self.token_bytes)

    @property
    def format_version(self) -> int:
        return FORMAT_VERSION if len(self.special_tokens) == len(SPECIAL_TOKENS) else 1

    def special_id(self, text: str) -> int:
        return self.special_tokens[text]

    def encode_word(self, word: str) -> list[int]:
        out: list[int] = []
        for chunk in word_chunks(word, self.max_word_bytes):
            out.extend(encode_chunk(chunk, self._ranks))
        return out

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        out: list[int] = []
        start = 0
        if allow_special:
            for match in _SPECIAL_PATTERN.finditer(text):
                token_id = self.special_tokens.get(match.group())
                if token_id is not None:
                    out.extend(self.encode(text[start : match.start()]))
                    out.append(token_id)
                    start = match.end()
        for word in pre_tokenize(text[start:]):
            out.extend(self.encode_word(word))
        return out

    def encode_docs(self, texts: Sequence[str], allow_special: bool = False) -> list[list[int]]:
        return [self.encode(text, allow_special=allow_special) for text in texts]

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
        if len(raw) < 22:
            raise ValueError(f"{path}: tokenizer.bin is truncated")
        magic, version, vocab_size, n_merges = struct.unpack_from("<4sHII", raw, 0)
        if magic != MAGIC:
            raise ValueError(f"{path}: bad magic {magic!r}")
        if version not in (1, FORMAT_VERSION):
            raise ValueError(f"{path}: unsupported format version {version}")
        special_count = 2 if version == 1 else len(SPECIAL_TOKENS)
        if vocab_size != BYTE_TOKENS + n_merges + special_count:
            raise ValueError(f"{path}: vocab size does not match version and merge count")
        table_end = len(raw) - n_merges * 8 - 8
        if table_end < 14 + vocab_size:
            raise ValueError(f"{path}: tokenizer.bin is truncated")
        offset = 14
        tokens: list[bytes] = []
        for _ in range(vocab_size):
            if offset >= table_end:
                raise ValueError(f"{path}: truncated token table")
            length = raw[offset]
            offset += 1
            if offset + length > table_end:
                raise ValueError(f"{path}: truncated token body")
            tokens.append(bytes(raw[offset : offset + length]))
            offset += length
        if offset != table_end:
            raise ValueError(f"{path}: unexpected bytes in tokenizer.bin")
        merges = tuple(struct.iter_unpack("<II", raw[offset : offset + n_merges * 8]))
        eos_id, pad_id = struct.unpack_from("<II", raw, table_end + n_merges * 8)
        return cls(token_bytes=tuple(tokens), merges=merges, eos_id=eos_id, pad_id=pad_id)

    @classmethod
    def from_vocab_files(cls, vocab_path: Path, merges_path: Path) -> Tokenizer:
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        if not isinstance(vocab, dict) or not vocab:
            raise ValueError(f"{vocab_path}: expected a nonempty vocabulary object")
        if any(type(token_id) is not int for token_id in vocab.values()):
            raise ValueError(f"{vocab_path}: token ids must be integers")
        if set(vocab.values()) != set(range(len(vocab))):
            raise ValueError(f"{vocab_path}: ids must be unique and contiguous from zero")
        lines = merges_path.read_text(encoding="utf-8").splitlines()
        versions = {"#version: 1.0": 1, "#version: 2.0": FORMAT_VERSION}
        if not lines or lines[0] not in versions:
            raise ValueError(f"{merges_path}: unsupported or missing version comment")
        pairs: list[tuple[int, int]] = []
        for line in lines[1:]:
            if not line.strip():
                continue
            left, right = line.split()
            if not left.isascii() or not right.isascii() or not left.isdecimal() or not right.isdecimal():
                raise ValueError(f"{merges_path}: merge ids must be unsigned decimal integers")
            pairs.append((int(left), int(right)))
        by_id: list[bytes] = [b""] * len(vocab)
        for key, token_id in vocab.items():
            by_id[token_id] = key.encode("latin-1")
        tokenizer = cls(
            token_bytes=tuple(by_id), merges=tuple(pairs), eos_id=len(by_id) - 1, pad_id=len(by_id) - 2
        )
        if tokenizer.format_version != versions[lines[0]]:
            raise ValueError(f"{merges_path}: version does not match special token layout")
        return tokenizer

    @classmethod
    def load(cls, directory: Path) -> Tokenizer:
        directory = Path(directory)
        binary = directory / "tokenizer.bin"
        if binary.is_file():
            return cls.from_bin(binary)
        return cls.from_vocab_files(directory / "vocab.json", directory / "merges.txt")


def write_bin(tokenizer: Tokenizer, path: Path) -> Path:
    body = bytearray(struct.pack("<4sHII", MAGIC, tokenizer.format_version, tokenizer.vocab_size, len(tokenizer.merges)))
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
