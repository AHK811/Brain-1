"""Text cleaning for corpus building.

clean_text() keeps its original signature/behavior for existing callers
(Datasets.Pipelines.ingestion_pipeline, Datasets.__init__) -- the new
Gutenberg-boilerplate stripping and unicode normalization are opt-in via
new parameters, not silently applied to every caller.
"""
from __future__ import annotations

import re
import unicodedata

# Project Gutenberg's start/end markers have used a few different exact
# wordings over the decades; this matches all the common variants (both the
# newer "*** START OF THE PROJECT GUTENBERG EBOOK <title> ***" style and the
# older "*** START OF THIS PROJECT GUTENBERG EBOOK <title> ***" / numbered
# "*** START OF THE PROJECT GUTENBERG EBOOK, <NN> ***" style).
_GUTENBERG_START_RE = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)
_GUTENBERG_END_RE = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)


def strip_gutenberg_boilerplate(text: str) -> str:
    """Cut everything up to and including the START marker, and everything
    from the END marker onward -- these wrap Project Gutenberg's legal
    boilerplate, not the book's actual content. If a marker isn't found
    (a minority of older-format texts), that side is left untouched rather
    than guessing, since guessing risks silently truncating real content.
    """
    start_match = _GUTENBERG_START_RE.search(text)
    if start_match:
        text = text[start_match.end():]
    end_match = _GUTENBERG_END_RE.search(text)
    if end_match:
        text = text[:end_match.start()]
    return text


def normalize_unicode(text: str) -> str:
    """NFKC normalization only -- deliberately does NOT rewrite smart quotes,
    em-dashes, etc. to ASCII equivalents. Those are legitimate characters a
    tokenizer should learn, not noise; over-aggressive normalization here
    would make the corpus less representative of real text, not cleaner.
    """
    return unicodedata.normalize("NFKC", text)


def clean_text(
    text: str,
    strip_gutenberg: bool = False,
    normalize: bool = False,
) -> str:
    text = (text or "").replace("\x00", " ")
    if strip_gutenberg:
        text = strip_gutenberg_boilerplate(text)
    if normalize:
        text = normalize_unicode(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_low_quality(text: str, min_chars: int = 500, min_alpha_ratio: float = 0.6) -> bool:
    """Cheap quality filter: too short, or too little alphabetic content
    (catches OCR garbage, tables-of-numbers, near-empty files) to be worth
    keeping in a text corpus. Deliberately conservative -- false negatives
    (letting some junk through) are cheaper than false positives (throwing
    away real text), since dedup/downstream filtering catches more later.
    """
    if len(text) < min_chars:
        return True
    alpha = sum(1 for c in text if c.isalpha())
    return (alpha / len(text)) < min_alpha_ratio
