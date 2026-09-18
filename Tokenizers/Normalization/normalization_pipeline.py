from __future__ import annotations
import re, unicodedata
def normalize(text: str, *, lower: bool = False, strip_html: bool = True) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    if strip_html:
        text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if lower:
        text = text.lower()
    return text
