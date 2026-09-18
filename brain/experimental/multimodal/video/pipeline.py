"""
Phase D — Video: keyframes + optional OCR + optional audio ASR.

Practical path for a ~33M LM:
  sample frames → vision tower / OCR
  extract audio track → ASR
  merge into one prompt / token stream
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from brain.experimental.ingest.types import ContentBlock, IngestResult, Modality
from brain.experimental.multimodal.audio.asr import ASRResult, transcribe
from brain.experimental.ocr.engines import get_ocr_engine


@dataclass
class VideoBundle:
    path: str
    frame_paths: list[str] = field(default_factory=list)
    frame_texts: list[str] = field(default_factory=list)  # OCR per frame
    asr: Optional[ASRResult] = None
    errors: list[str] = field(default_factory=list)

    def to_prompt(self, max_chars: int = 8000) -> str:
        parts = [f"[video] {self.path}\n"]
        if self.asr and self.asr.text:
            parts.append(f"[audio transcript | {self.asr.backend}]\n{self.asr.text}\n")
        for i, (fp, txt) in enumerate(zip(self.frame_paths, self.frame_texts)):
            parts.append(f"[frame {i} | {fp}]\n{txt}\n")
        out = "".join(parts)
        return out[:max_chars] + ("…" if len(out) > max_chars else "")

    def to_ingest_result(self) -> IngestResult:
        blocks: list[ContentBlock] = [
            ContentBlock(Modality.META, text=f"video={self.path}", source=self.path)
        ]
        if self.asr and self.asr.text:
            blocks.append(ContentBlock(
                Modality.AUDIO, text=self.asr.text, source=self.path,
                meta={"asr_backend": self.asr.backend}
            ))
        for i, (fp, txt) in enumerate(zip(self.frame_paths, self.frame_texts)):
            blocks.append(ContentBlock(
                Modality.IMAGE, text=txt, source=self.path, page=i,
                image_path=fp, meta={"frame": i}
            ))
        return IngestResult(source=self.path, blocks=blocks, errors=list(self.errors))


def sample_frames(
    path: str | Path,
    *,
    n_frames: int = 8,
    out_dir: Optional[str | Path] = None,
) -> tuple[list[str], list[str]]:
    """Return (frame_paths, errors)."""
    path = Path(path)
    errors: list[str] = []
    try:
        import cv2
    except Exception as e:
        return [], [f"opencv required for frame sampling: {e}"]

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [], [f"cannot open video: {path}"]

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        # read sequentially
        indices = list(range(n_frames))
    else:
        indices = [int(i * (total - 1) / max(n_frames - 1, 1)) for i in range(n_frames)]

    if out_dir is None:
        out_dir = Path(tempfile.mkdtemp(prefix="brain_v04_vid_"))
    else:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[str] = []
    for fi, idx in enumerate(indices):
        if total > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            errors.append(f"frame {fi} failed")
            continue
        fp = out_dir / f"{path.stem}_f{fi:03d}.jpg"
        cv2.imwrite(str(fp), frame)
        paths.append(str(fp))
    cap.release()
    return paths, errors


def process_video(
    path: str | Path,
    *,
    n_frames: int = 8,
    ocr: bool = True,
    asr: bool = True,
    asr_backend: str = "auto",
    out_dir: Optional[str | Path] = None,
) -> VideoBundle:
    path = Path(path)
    bundle = VideoBundle(path=str(path))
    frames, errs = sample_frames(path, n_frames=n_frames, out_dir=out_dir)
    bundle.frame_paths = frames
    bundle.errors.extend(errs)

    if ocr and frames:
        engine = get_ocr_engine()
        for fp in frames:
            try:
                bundle.frame_texts.append(engine.extract(fp) or "")
            except Exception as e:
                bundle.frame_texts.append("")
                bundle.errors.append(f"ocr {fp}: {e}")
    else:
        bundle.frame_texts = [""] * len(frames)

    if asr:
        # Best-effort: try ASR directly on video path (whisper can mux audio)
        try:
            bundle.asr = transcribe(path, backend=asr_backend)
        except Exception as e:
            bundle.errors.append(f"asr: {e}")
            bundle.asr = ASRResult(text="", backend="error")

    return bundle
