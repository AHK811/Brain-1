from __future__ import annotations
def fertility(n_tokens: int, n_words: int) -> float:
    return n_tokens / max(n_words, 1)
def chars_per_token(n_chars: int, n_tokens: int) -> float:
    return n_chars / max(n_tokens, 1)
