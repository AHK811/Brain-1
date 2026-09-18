"""
ASR backends for Phase C.

Order: faster-whisper → openai-whisper → stub.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class ASRResult:
    text: str
    backend: str
    language: Optional[str] = None
    segments: list[dict[str, Any]] | None = None


def transcribe(
    path: str | Path,
    *,
    backend: str = "auto",
    model_size: str = "base",
    language: Optional[str] = None,
) -> ASRResult:
    path = Path(path)
    if not path.exists():
        return ASRResult(text="", backend="none", language=None)

    order = [backend] if backend != "auto" else ["faster-whisper", "whisper", "stub"]
    errors: list[str] = []
    for name in order:
        try:
            if name in ("faster-whisper", "faster_whisper"):
                from faster_whisper import WhisperModel
                model = WhisperModel(model_size)
                segs, info = model.transcribe(str(path), language=language)
                segments = []
                texts = []
                for s in segs:
                    texts.append(s.text)
                    segments.append({"start": s.start, "end": s.end, "text": s.text})
                return ASRResult(
                    text=" ".join(texts).strip(),
                    backend="faster-whisper",
                    language=getattr(info, "language", language),
                    segments=segments,
                )
            if name == "whisper":
                import whisper
                model = whisper.load_model(model_size)
                kwargs = {}
                if language:
                    kwargs["language"] = language
                result = model.transcribe(str(path), **kwargs)
                segments = result.get("segments") or []
                return ASRResult(
                    text=(result.get("text") or "").strip(),
                    backend="whisper",
                    language=result.get("language", language),
                    segments=[{"start": s.get("start"), "end": s.get("end"), "text": s.get("text")} for s in segments],
                )
            if name == "stub":
                return ASRResult(
                    text=f"[asr-stub] install openai-whisper or faster-whisper for {path.name}",
                    backend="stub",
                )
        except Exception as e:
            errors.append(f"{name}: {e}")
            continue
    return ASRResult(text="[asr failed] " + "; ".join(errors[:2]), backend="error")
