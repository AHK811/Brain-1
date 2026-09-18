"""Content blocks — unified representation for any ingested file."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Modality(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    BINARY = "binary"
    META = "meta"


@dataclass
class ContentBlock:
    modality: Modality
    text: str = ""                    # always fill when possible (OCR/ASR/extract)
    source: str = ""                  # path or URL
    page: Optional[int] = None
    mime: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    # Optional raw / tensor refs for multimodal path (not serialized by default)
    image_path: Optional[str] = None
    audio_path: Optional[str] = None
    video_path: Optional[str] = None
    embedding: Any = None

    def as_prompt_fragment(self) -> str:
        loc = f" ({self.source}"
        if self.page is not None:
            loc += f" p.{self.page}"
        loc += ")" if self.source else ""
        if self.modality == Modality.TABLE:
            return f"[table{loc}]\n{self.text}\n"
        if self.modality == Modality.IMAGE:
            return f"[image{loc}]\n{self.text}\n"
        if self.modality == Modality.AUDIO:
            return f"[audio{loc}]\n{self.text}\n"
        if self.modality == Modality.VIDEO:
            return f"[video{loc}]\n{self.text}\n"
        if self.modality == Modality.META:
            return f"[meta{loc}] {self.text}\n"
        return f"[text{loc}]\n{self.text}\n"


@dataclass
class IngestResult:
    source: str
    blocks: list[ContentBlock] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_prompt(self, max_chars: int = 12000) -> str:
        parts: list[str] = []
        n = 0
        for b in self.blocks:
            frag = b.as_prompt_fragment()
            if n + len(frag) > max_chars:
                parts.append("[truncated]")
                break
            parts.append(frag)
            n += len(frag)
        return "".join(parts)
