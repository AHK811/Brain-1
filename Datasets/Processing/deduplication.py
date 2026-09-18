"""Deduplication for corpus building.

Two layers, since they catch different things:
  - exact_hash(): identical documents (e.g. the same public-domain book
    submitted to Gutenberg twice under different IDs).
  - MinHashLSH: near-duplicates (different editions/OCR passes/formatting
    of the same underlying text) that exact hashing can't see because a
    single re-typed word changes the whole hash.

Dependency-free (hashlib + numpy, both already required by this repo) --
deliberately not using a library like `datasketch`. At truly massive scale
(tens of millions+ of documents) a real distributed LSH implementation
would be worth it; at the scale a from-scratch project actually builds a
corpus for (thousands to low millions of documents), this is correct and
fast enough, and is honest about being pure-Python-loop-bound rather than
claiming performance it doesn't have.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import numpy as np

_WORD_RE = re.compile(r"\w+")


def normalized_for_hash(text: str) -> str:
    """Whitespace-collapsed, lowercased -- for exact-duplicate comparison
    only. Near-duplicate detection below uses the raw text's shingles, not
    this, since over-normalizing before shingling would make legitimately
    different texts collide.
    """
    return re.sub(r"\s+", " ", text).strip().lower()


def exact_hash(text: str) -> str:
    """SHA-256 of the whitespace/case-normalized text. Two documents with
    this same hash are byte-for-byte-equivalent after normalization.
    """
    return hashlib.sha256(normalized_for_hash(text).encode("utf-8")).hexdigest()


def _shingles(text: str, k: int = 5) -> set[str]:
    """Word k-shingles (contiguous k-word spans). Word-level (not
    character-level) because it's much cheaper for book-length documents
    and is the standard choice for this kind of corpus-level near-dup
    detection (e.g. used in C4/RedPajama-style pipelines).
    """
    words = _WORD_RE.findall(text.lower())
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


@dataclass
class MinHashLSH:
    """Streaming near-duplicate detector. Call `is_duplicate_and_add(doc_id,
    text)` once per document, in order -- the first copy of any near-duplicate
    cluster is kept (returns False), every later one is flagged (returns True)
    without being added to the index.
    """
    num_hashes: int = 128
    num_bands: int = 16          # must divide num_hashes evenly
    shingle_size: int = 5
    jaccard_threshold: float = 0.8
    seed: int = 0

    _a: np.ndarray = field(init=False, repr=False)
    _b: np.ndarray = field(init=False, repr=False)
    _signatures: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _buckets: dict[tuple, list[str]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.num_hashes % self.num_bands != 0:
            raise ValueError(
                f"num_hashes ({self.num_hashes}) must be divisible by "
                f"num_bands ({self.num_bands})"
            )
        rng = np.random.default_rng(self.seed)
        # Odd 64-bit multipliers + 64-bit XOR masks, combined with wrapping
        # (mod 2**64) unsigned arithmetic -- numpy's uint64 overflow is
        # well-defined (silent wraparound), not undefined behavior, so this
        # is deterministic and reproducible across runs/machines. This is
        # a multiply-xor mix (splitmix64-style), not literal Carter-Wegman
        # modular hashing with a Mersenne prime -- avoids the int64
        # overflow/sign issues that approach has here while giving
        # equally good uniformity for MinHash's purposes.
        self._a = (rng.integers(0, 2**63, size=self.num_hashes, dtype=np.uint64) * 2 + 1)  # force odd
        self._b = rng.integers(0, 2**63, size=self.num_hashes, dtype=np.uint64)

    def _signature(self, text: str) -> np.ndarray | None:
        shingles = _shingles(text, self.shingle_size)
        if not shingles:
            return None
        base_hashes = np.array(
            [int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")
             for s in shingles],
            dtype=np.uint64,
        )
        # (n_shingles, num_hashes): wrapping uint64 multiply + xor per hash
        # function; min over shingles per hash function gives the signature.
        hashed = (base_hashes[:, None] * self._a[None, :]) ^ self._b[None, :]
        return hashed.min(axis=0)

    def is_duplicate_and_add(self, doc_id: str, text: str) -> bool:
        sig = self._signature(text)
        if sig is None:
            return False  # nothing to hash (empty/near-empty doc) -- let the quality filter handle it
        rows_per_band = self.num_hashes // self.num_bands
        candidates: set[str] = set()
        band_keys = []
        for band_idx in range(self.num_bands):
            band = sig[band_idx * rows_per_band:(band_idx + 1) * rows_per_band]
            key = (band_idx, band.tobytes())
            band_keys.append(key)
            candidates.update(self._buckets.get(key, []))

        for cand_id in candidates:
            cand_sig = self._signatures[cand_id]
            est_jaccard = float((sig == cand_sig).mean())
            if est_jaccard >= self.jaccard_threshold:
                return True  # duplicate of cand_id -- don't add, don't index

        self._signatures[doc_id] = sig
        for key in band_keys:
            self._buckets.setdefault(key, []).append(doc_id)
        return False
