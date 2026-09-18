"""
load_any(path) — route by extension / MIME to the right loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from brain.experimental.ingest import loaders as L
from brain.experimental.ingest.types import ContentBlock, IngestResult, Modality

_EXT_MAP = {
    ".txt": L.load_text, ".md": L.load_text, ".py": L.load_text, ".json": L.load_text,
    ".csv": L.load_csv, ".tsv": L.load_csv,
    ".pdf": L.load_pdf,
    ".docx": L.load_docx,
    ".pptx": L.load_pptx,
    ".xlsx": L.load_xlsx, ".xlsm": L.load_xlsx,
    ".png": L.load_image, ".jpg": L.load_image, ".jpeg": L.load_image,
    ".webp": L.load_image, ".bmp": L.load_image, ".tif": L.load_image, ".tiff": L.load_image,
    ".mp3": L.load_audio, ".wav": L.load_audio, ".m4a": L.load_audio, ".flac": L.load_audio,
    ".mp4": L.load_video, ".mov": L.load_video, ".mkv": L.load_video, ".avi": L.load_video,
    ".zip": None,  # special
}


def load_any(path: str | Path, opts: Optional[dict[str, Any]] = None) -> IngestResult:
    path = Path(path)
    opts = dict(opts or {})
    if not path.exists():
        return IngestResult(source=str(path), errors=[f"not found: {path}"])

    ext = path.suffix.lower()
    if ext == ".zip":
        return _load_zip(path, opts)

    fn = _EXT_MAP.get(ext)
    if fn is None:
        # try as text
        try:
            return L.load_text(path, opts)
        except Exception as e:
            return IngestResult(
                source=str(path),
                blocks=[ContentBlock(Modality.BINARY, text=f"[unsupported binary {ext}]", source=str(path))],
                errors=[str(e)],
            )
    return fn(path, opts)


def _load_zip(path: Path, opts: dict) -> IngestResult:
    import tempfile
    import zipfile
    blocks = [ContentBlock(Modality.META, text=f"zip={path.name}", source=str(path))]
    errors: list[str] = []
    max_files = int(opts.get("max_zip_files", 30))
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")][:max_files]
            with tempfile.TemporaryDirectory() as td:
                for name in names:
                    try:
                        zf.extract(name, td)
                        inner = Path(td) / name
                        if inner.is_file():
                            sub = load_any(inner, opts)
                            blocks.extend(sub.blocks)
                            errors.extend(sub.errors)
                    except Exception as e:
                        errors.append(f"{name}: {e}")
    except Exception as e:
        errors.append(str(e))
    return IngestResult(source=str(path), blocks=blocks, errors=errors)


def blocks_to_chat_context(result: IngestResult, max_chars: int = 12000) -> str:
    return result.to_prompt(max_chars=max_chars)
