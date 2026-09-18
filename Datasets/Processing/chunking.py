from __future__ import annotations
def chunk_text(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    if size <= 0: return [text]
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i+size])
        if i + size >= len(text): break
        i += max(size - overlap, 1)
    return out
