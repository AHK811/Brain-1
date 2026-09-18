"""
brain/sft/templates.py

Chat, Chain-of-Thought, and domain-specialized prompt templates.

These are pure string helpers — no torch dependency. Used by:
  - SFT data preparation
  - Inference / playground system prompts
  - Future agent loops
"""

from __future__ import annotations

from typing import Optional, Sequence

from brain.tokenizer.special_tokens import (
    SYSTEM, USER, ASSISTANT, END_OF_TURN,
    THINK, END_THINK, STEP,
    PLAN, END_PLAN,
    DOMAIN_MAP,
)

# ---------------------------------------------------------------------------
# Domain system prompts — the model learns these styles via SFT data
# ---------------------------------------------------------------------------
DOMAIN_SYSTEM_PROMPTS: dict[str, str] = {
    "nlp": (
        "You are an expert in Natural Language Processing. "
        "Explain concepts clearly, give precise definitions, and when useful "
        "provide short examples or pseudocode."
    ),
    "reasoning": (
        "You are a careful reasoning assistant. "
        "Break problems into steps, state assumptions, and conclude only after "
        "the steps support it. Prefer correct reasoning over speed."
    ),
    "legal": (
        "You are a legal research assistant (not a lawyer). "
        "Cite principles carefully, distinguish jurisdiction when relevant, "
        "and clearly mark uncertainty. Never give binding legal advice."
    ),
    "cyber": (
        "You are a cybersecurity analyst. "
        "Prioritize defensive and ethical guidance. Explain threats, mitigations, "
        "and best practices. Refuse requests for offensive exploitation."
    ),
    "software": (
        "You are a senior software engineer. "
        "Prefer clean architecture, clear interfaces, and practical trade-offs. "
        "Explain design decisions and show concise, correct code when asked."
    ),
    "finance": (
        "You are a BFSI (Banking, Financial Services, Insurance) analyst. "
        "Be precise with numbers and assumptions. Distinguish fact, estimate, "
        "and opinion. Note regulatory or risk context when relevant."
    ),
    "linguistics": (
        "You are an advanced linguistics expert. "
        "Discuss phonology, morphology, syntax, semantics, and pragmatics "
        "with accurate terminology and clear examples."
    ),
    "math": (
        "You are a mathematical reasoning assistant. "
        "Show step-by-step derivations. State theorems or definitions you use. "
        "Check edge cases and units when applicable."
    ),
    "agent": (
        "You are an agentic planner with access to tools (web, files, code, terminal) when the host provides them. "
        "First produce a short plan when the task is complex, then use tools as needed. "
        "There is no limit on tool calls while tools are available — stop only when you can answer. "
        "Revise the plan if new information appears."
    ),
    "code": (
        "You are an expert in code architecture and generation. "
        "Write correct, readable code. Prefer simple designs. "
        "Comment non-obvious logic and note complexity or failure modes."
    ),
    "genai": (
        "You are a Generative AI specialist. "
        "Explain models, training, inference, evaluation, and safety clearly. "
        "Give practical guidance for building and deploying GenAI systems."
    ),
}


def _domain_token(domain: Optional[str]) -> str:
    if not domain:
        return ""
    key = domain.lower().strip().replace(" ", "_").replace("-", "_")
    return DOMAIN_MAP.get(key, DOMAIN_MAP.get(key.split("_")[0], ""))


def format_chat(
    user: str,
    *,
    system: Optional[str] = None,
    assistant: Optional[str] = None,
    domain: Optional[str] = None,
    add_generation_prompt: bool = False,
) -> str:
    """Build a single-turn or partial chat string in Brain chat format."""
    parts: list[str] = []

    dom = _domain_token(domain)
    sys_text = system
    if domain and domain.lower() in DOMAIN_SYSTEM_PROMPTS and not sys_text:
        sys_text = DOMAIN_SYSTEM_PROMPTS[domain.lower()]

    if sys_text or dom:
        block = f"{SYSTEM}"
        if dom:
            block += f" {dom}"
        if sys_text:
            block += f" {sys_text}"
        parts.append(block + f" {END_OF_TURN}")

    parts.append(f"{USER} {user.strip()} {END_OF_TURN}")

    if assistant is not None:
        parts.append(f"{ASSISTANT} {assistant.strip()} {END_OF_TURN}")
    elif add_generation_prompt:
        parts.append(f"{ASSISTANT}")

    return "".join(parts)


def format_cot(
    user: str,
    *,
    reasoning: Optional[str] = None,
    answer: Optional[str] = None,
    system: Optional[str] = None,
    domain: Optional[str] = "reasoning",
    steps: Optional[Sequence[str]] = None,
    add_generation_prompt: bool = False,
) -> str:
    """
    Chain-of-Thought format:

      [system + domain]
      user
      assistant <|think|> step... <|/think|> final answer
    """
    parts: list[str] = []

    dom = _domain_token(domain)
    sys_text = system or DOMAIN_SYSTEM_PROMPTS.get((domain or "reasoning").lower(), "")

    if sys_text or dom:
        block = f"{SYSTEM}"
        if dom:
            block += f" {dom}"
        if sys_text:
            block += f" {sys_text}"
        parts.append(block + f" {END_OF_TURN}")

    parts.append(f"{USER} {user.strip()} {END_OF_TURN}")

    if reasoning is not None or answer is not None or steps is not None:
        asst = f"{ASSISTANT} {THINK} "
        if steps:
            asst += " ".join(f"{STEP} {s.strip()}" for s in steps)
        elif reasoning:
            asst += reasoning.strip()
        asst += f" {END_THINK} "
        if answer:
            asst += answer.strip()
        asst += f" {END_OF_TURN}"
        parts.append(asst)
    elif add_generation_prompt:
        # Encourage the model to think before answering
        parts.append(f"{ASSISTANT} {THINK}")

    return "".join(parts)


def format_domain_prompt(
    user: str,
    domain: str,
    *,
    use_cot: bool = False,
    assistant: Optional[str] = None,
    add_generation_prompt: bool = True,
) -> str:
    """Convenience wrapper: domain system prompt + optional CoT."""
    if use_cot:
        return format_cot(
            user,
            domain=domain,
            answer=assistant,
            add_generation_prompt=add_generation_prompt and assistant is None,
        )
    return format_chat(
        user,
        domain=domain,
        assistant=assistant,
        add_generation_prompt=add_generation_prompt and assistant is None,
    )


def format_plan(
    user: str,
    plan_steps: Sequence[str],
    *,
    domain: str = "agent",
    final_answer: Optional[str] = None,
) -> str:
    """Agentic planning format with explicit plan block."""
    plan_body = " ".join(f"{STEP} {s.strip()}" for s in plan_steps)
    assistant = f"{PLAN} {plan_body} {END_PLAN}"
    if final_answer:
        assistant += f" {final_answer.strip()}"
    return format_chat(user, domain=domain, assistant=assistant)
