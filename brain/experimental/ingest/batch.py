"""Batch / folder ingest for corpora of mixed files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from brain.experimental.ingest.api import load_any
from brain.experimental.ingest.types import IngestResult

_DEFAULT_EXTS = {
    ".txt", ".md", ".pdf", ".docx", ".pptx", ".xlsx", ".csv",
    ".png", ".jpg", ".jpeg", ".webp", ".zip", ".mp3", ".wav", ".mp4",
}


def iter_folder(
    root: str | Path,
    *,
    recursive: bool = True,
    exts: Optional[set[str]] = None,
    limit: Optional[int] = None,
) -> Iterator[IngestResult]:
    root = Path(root)
    exts = exts or _DEFAULT_EXTS
    globber = root.rglob if recursive else root.glob
    n = 0
    for path in sorted(globber("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in exts:
            continue
        yield load_any(path)
        n += 1
        if limit is not None and n >= limit:
            break


def folder_to_prompt(
    root: str | Path,
    *,
    max_files: int = 20,
    max_chars: int = 20000,
) -> str:
    parts: list[str] = []
    used = 0
    for r in iter_folder(root, limit=max_files):
        frag = r.to_prompt(max_chars=max_chars - used)
        parts.append(frag)
        used += len(frag)
        if used >= max_chars:
            parts.append("[folder truncated]")
            break
    return "\n".join(parts)
