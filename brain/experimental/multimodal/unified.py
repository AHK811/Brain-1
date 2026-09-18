"""
Unified media entry: any path → text context + optional vision/audio tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from brain.experimental.ingest import load_any, IngestResult

_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
_AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
_VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


@dataclass
class MediaBundle:
    path: str
    kind: str  # image|audio|video|document|other
    ingest: IngestResult
    prompt: str
    extra: dict[str, Any] = field(default_factory=dict)


def process_media(
    path: str | Path,
    *,
    vision_tower=None,
    audio_tower=None,
    device: str = "cpu",
    n_video_frames: int = 6,
) -> MediaBundle:
    path = Path(path)
    ext = path.suffix.lower()

    if ext in _VIDEO_EXT:
        from brain.experimental.multimodal.video import process_video
        v = process_video(path, n_frames=n_video_frames, ocr=True, asr=True)
        prompt = v.to_prompt()
        extra: dict[str, Any] = {"video": v}
        if vision_tower is not None and v.frame_paths:
            try:
                extra["vision_tokens"] = vision_tower.encode_paths(v.frame_paths[:4], device=device)
            except Exception as e:
                extra["vision_error"] = str(e)
        return MediaBundle(str(path), "video", v.to_ingest_result(), prompt, extra)

    if ext in _AUDIO_EXT:
        from brain.experimental.multimodal.audio import process_audio
        a = process_audio(path)
        prompt = f"[audio | {a.asr.backend}]\n{a.asr.text}\n"
        ingest = load_any(path)
        return MediaBundle(str(path), "audio", ingest, prompt, {"asr": a.asr})

    if ext in _IMAGE_EXT:
        ingest = load_any(path)
        prompt = ingest.to_prompt()
        extra = {}
        if vision_tower is not None:
            try:
                extra["vision_tokens"] = vision_tower.encode_paths([path], device=device)
            except Exception as e:
                extra["vision_error"] = str(e)
        return MediaBundle(str(path), "image", ingest, prompt, extra)

    ingest = load_any(path)
    return MediaBundle(str(path), "document", ingest, ingest.to_prompt(), {})
