"""
Preference data format for DPO.

JSONL line schema:
{
  "prompt": "user (+ optional system) text",
  "chosen": "preferred assistant reply",
  "rejected": "worse assistant reply",
  "domain": "optional domain key"
}

Or chat-style:
{
  "messages": [{"role":"system","content":"..."},{"role":"user","content":"..."}],
  "chosen": "...",
  "rejected": "..."
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from brain.experimental.sft.templates import format_chat


@dataclass
class PreferenceExample:
    prompt: str
    chosen: str
    rejected: str
    domain: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d = {"prompt": self.prompt, "chosen": self.chosen, "rejected": self.rejected}
        if self.domain:
            d["domain"] = self.domain
        return d


def format_preference_pair(
    user: str,
    chosen: str,
    rejected: str,
    *,
    domain: Optional[str] = None,
    system: Optional[str] = None,
) -> PreferenceExample:
    """Build prompt string using Brain chat template (prompt only, no assistant)."""
    prompt = format_chat(
        user, system=system, domain=domain, add_generation_prompt=True
    )
    return PreferenceExample(prompt=prompt, chosen=chosen, rejected=rejected, domain=domain)


def load_preference_jsonl(path: str | Path) -> list[PreferenceExample]:
    path = Path(path)
    rows: list[PreferenceExample] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "prompt" in obj:
                rows.append(
                    PreferenceExample(
                        prompt=obj["prompt"],
                        chosen=obj["chosen"],
                        rejected=obj["rejected"],
                        domain=obj.get("domain"),
                    )
                )
            elif "messages" in obj:
                # flatten messages into prompt via simple join
                parts = []
                for m in obj["messages"]:
                    role = m.get("role", "user")
                    parts.append(f"{role}: {m.get('content', '')}")
                prompt = "\n".join(parts) + "\nassistant:"
                rows.append(
                    PreferenceExample(
                        prompt=prompt,
                        chosen=obj["chosen"],
                        rejected=obj["rejected"],
                        domain=obj.get("domain"),
                    )
                )
            else:
                raise ValueError(f"Invalid preference row keys: {list(obj)}")
    return rows


def iter_preference_jsonl(path: str | Path) -> Iterator[PreferenceExample]:
    for ex in load_preference_jsonl(path):
        yield ex
