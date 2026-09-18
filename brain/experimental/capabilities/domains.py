
"""Capability domains for SFT / few-shot scaffolding (v0.5)."""
from __future__ import annotations

CAPABILITY_DOMAINS = {
    "few_shot": {
        "name": "Emergent Few-Shot Learning",
        "instruction": (
            "Learn the pattern from the examples, then answer the new case. "
            "Do not ignore the demonstrated format."
        ),
    },
    "code": {
        "name": "High-Quality Code Generation",
        "instruction": (
            "Write correct, idiomatic, well-structured code. Prefer clarity, "
            "tests when useful, and match the requested language/version."
        ),
    },
    "long_form": {
        "name": "Coherent Long-Form Writing",
        "instruction": (
            "Write coherent multi-paragraph text with clear structure, "
            "consistent voice, and logical flow from start to end."
        ),
    },
    "math": {
        "name": "Advanced Mathematical Reasoning",
        "instruction": (
            "Solve step by step. Show intermediate reasoning. "
            "Check units and edge cases. Put the final answer clearly."
        ),
    },
    "commonsense": {
        "name": "Common-Sense Reasoning",
        "instruction": (
            "Use everyday physical and social knowledge. "
            "State assumptions when the question is underspecified."
        ),
    },
}

def system_preamble(*domains: str) -> str:
    parts = []
    for d in domains:
        meta = CAPABILITY_DOMAINS.get(d)
        if meta:
            parts.append(f"[{meta['name']}] {meta['instruction']}")
    return "\n".join(parts) if parts else ""
