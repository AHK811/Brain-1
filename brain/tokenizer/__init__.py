"""Brain tokenizer package: base BPE + advanced APIs.

Importing `brain.tokenizer.special_tokens` does not require the `tokenizers`
package. Importing BrainTokenizer / AdvancedTokenizer does.
"""

from brain.tokenizer.special_tokens import ALL_SPECIAL_TOKENS

__all__ = ["ALL_SPECIAL_TOKENS", "BrainTokenizer", "AdvancedTokenizer"]


def __getattr__(name: str):
    if name == "BrainTokenizer":
        from brain.tokenizer.tokenizer import BrainTokenizer
        return BrainTokenizer
    if name in ("AdvancedTokenizer", "EncodeResult", "BatchEncodeResult"):
        from brain.tokenizer import advanced as _adv
        return getattr(_adv, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
