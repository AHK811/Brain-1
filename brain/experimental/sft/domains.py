"""
brain/sft/domains.py

Domain definitions for Brain v0.3 specialization.

These drive:
  - System prompts (see templates.py)
  - Recommended data mixtures for SFT
  - Evaluation categories

The model itself is still a single generalist transformer.
Specialization comes from the data you train on + the templates above.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Domain:
    key: str
    title: str
    description: str
    # Suggested public datasets / sources for SFT mixture (examples only)
    suggested_data: tuple[str, ...]
    # Whether CoT format is especially useful
    prefer_cot: bool = False


DOMAINS: dict[str, Domain] = {
    "nlp": Domain(
        key="nlp",
        title="Natural Language Processing",
        description="Tokenization, language modeling, classification, NER, parsing, embeddings.",
        suggested_data=("FLAN-style NLP tasks", "SuperGLUE rationales", "instruction NLP tutorials"),
        prefer_cot=False,
    ),
    "reasoning": Domain(
        key="reasoning",
        title="Reasoning & Agentic AI",
        description="Multi-step reasoning, tool use, planning, self-critique.",
        suggested_data=("GSM8K / MATH CoT", "StrategyQA", "ReAct-style traces", "OpenHermes reasoning"),
        prefer_cot=True,
    ),
    "legal": Domain(
        key="legal",
        title="Legal & Cybersecurity (Legal track)",
        description="Legal concepts, contracts, compliance language (not legal advice).",
        suggested_data=("LegalBench-style tasks", "public case summaries", "regulation explainers"),
        prefer_cot=True,
    ),
    "cyber": Domain(
        key="cyber",
        title="Legal & Cybersecurity (Cyber track)",
        description="Defensive security, threat models, secure design, incident response concepts.",
        suggested_data=("OWASP explainers", "security FAQ / playbooks", "defensive CTF writeups"),
        prefer_cot=True,
    ),
    "software": Domain(
        key="software",
        title="Software Engineering",
        description="Design patterns, APIs, testing, architecture trade-offs.",
        suggested_data=("Code documentation Q&A", "design-pattern tutorials", "PR review style data"),
        prefer_cot=False,
    ),
    "finance": Domain(
        key="finance",
        title="Finance (BFSI)",
        description="Banking, markets, risk, insurance concepts, quantitative reasoning.",
        suggested_data=("Financial QA", "earnings-call style summaries", "risk/regulation primers"),
        prefer_cot=True,
    ),
    "linguistics": Domain(
        key="linguistics",
        title="Advanced Text & Linguistics",
        description="Syntax, semantics, morphology, discourse, typology.",
        suggested_data=("Linguistics textbooks Q&A", "UD / syntax exercises", "semantics tutorials"),
        prefer_cot=False,
    ),
    "math": Domain(
        key="math",
        title="Logical & Mathematical Reasoning",
        description="Proofs, algebra, calculus, discrete math, formal logic.",
        suggested_data=("GSM8K", "MATH", "ProofNet-style", "logic puzzles with steps"),
        prefer_cot=True,
    ),
    "agent": Domain(
        key="agent",
        title="Agentic Planning",
        description="Goal decomposition, tool selection, plan revision.",
        suggested_data=("ReAct / Reflexion traces", "ToolBench-style", "plan-and-execute examples"),
        prefer_cot=True,
    ),
    "code": Domain(
        key="code",
        title="Code Architecture & Generation",
        description="Writing, explaining, and refactoring code; system design.",
        suggested_data=("The Stack / StarCoder data (filtered)", "CodeAlpaca", "architecture Q&A"),
        prefer_cot=False,
    ),
    "genai": Domain(
        key="genai",
        title="Gen AI",
        description="LLMs, training, alignment, evaluation, deployment, safety.",
        suggested_data=("LLM survey Q&A", "HF course style material", "alignment / eval primers"),
        prefer_cot=False,
    ),
}


def list_domains() -> list[str]:
    return list(DOMAINS.keys())


def get_domain(key: str) -> Optional[Domain]:
    return DOMAINS.get(key.lower().strip())
