"""
Phase E — Multimodal SFT / preference data formats.

JSONL schemas
-------------
Caption / DocVQA / instruction (SFT):
{
  "id": "optional",
  "text": "user question or caption prompt",
  "answer": "assistant target",
  "images": ["path1.jpg"],          # optional
  "audio": ["path.wav"],            # optional
  "video": ["path.mp4"],            # optional
  "context_files": ["doc.pdf"],     # optional — run through ingest
  "domain": "optional",
  "use_cot": false
}

Preference (DPO):
{
  "prompt": "...",                  # or text + images like SFT
  "chosen": "...",
  "rejected": "...",
  "images": [],
  "domain": "optional"
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

from brain.experimental.sft.templates import format_chat, format_cot, format_domain_prompt
from brain.experimental.multimodal.fusion.interleave import build_interleaved_prompt


@dataclass
class MMSFTExample:
    text: str
    answer: str
    images: list[str] = field(default_factory=list)
    audio: list[str] = field(default_factory=list)
    video: list[str] = field(default_factory=list)
    context_files: list[str] = field(default_factory=list)
    domain: Optional[str] = None
    use_cot: bool = False
    id: Optional[str] = None

    def build_prompt(self, *, n_image_tokens: int = 64, ingest_context: bool = True) -> str:
        ctx_parts: list[str] = []
        if ingest_context and self.context_files:
            try:
                from brain.experimental.ingest import load_any
                for fp in self.context_files:
                    r = load_any(fp)
                    ctx_parts.append(r.to_prompt(max_chars=4000))
            except Exception as e:
                ctx_parts.append(f"[context error: {e}]")
        user = self.text
        if ctx_parts:
            user = "Context:\n" + "\n".join(ctx_parts) + "\n\nQuestion:\n" + self.text
        if self.images:
            # one interleaved block per image (simple)
            prefix = "\n".join(build_interleaved_prompt("", n_image_tokens=n_image_tokens) for _ in self.images)
            user = prefix + "\n" + user
        if self.use_cot:
            return format_cot(user, domain=self.domain or "reasoning", answer=self.answer)
        return format_domain_prompt(
            user, domain=self.domain or "genai", assistant=self.answer, add_generation_prompt=False
        )


@dataclass
class MMPreferenceExample:
    prompt: str
    chosen: str
    rejected: str
    images: list[str] = field(default_factory=list)
    domain: Optional[str] = None


def load_mm_sft_jsonl(path: str | Path) -> list[MMSFTExample]:
    rows: list[MMSFTExample] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            rows.append(MMSFTExample(
                text=o.get("text") or o.get("prompt") or "",
                answer=o.get("answer") or o.get("response") or "",
                images=list(o.get("images") or []),
                audio=list(o.get("audio") or []),
                video=list(o.get("video") or []),
                context_files=list(o.get("context_files") or []),
                domain=o.get("domain"),
                use_cot=bool(o.get("use_cot", False)),
                id=o.get("id"),
            ))
    return rows


def load_mm_preference_jsonl(path: str | Path) -> list[MMPreferenceExample]:
    rows: list[MMPreferenceExample] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            rows.append(MMPreferenceExample(
                prompt=o["prompt"] if "prompt" in o else o.get("text", ""),
                chosen=o["chosen"],
                rejected=o["rejected"],
                images=list(o.get("images") or []),
                domain=o.get("domain"),
            ))
    return rows


def write_sample_sft_jsonl(path: str | Path) -> None:
    samples = [
        {"text": "Caption this image.", "answer": "A red apple on a wooden table.",
         "images": ["examples/apple.jpg"], "domain": "genai"},
        {"text": "What is 2+2?", "answer": "4", "domain": "math", "use_cot": True},
        {"text": "Summarize the document.", "answer": "It discusses RoPE scaling.",
         "context_files": ["examples/rope_note.txt"], "domain": "nlp"},
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
