"""
brain/tokenizer/special_tokens.py

Centralizes special-token definitions for:
  - Core tokenizer (pad/unk/bos/eos)
  - Instruction / chat format
  - Chain-of-Thought / reasoning
  - Domain tags (NLP, Legal, Code, Finance, etc.)
  - Agentic / tool-calling markers

All domain and reasoning capability is primarily driven by data + SFT.
These tokens give the model explicit structure to learn those formats.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"

CORE_SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]

# ---------------------------------------------------------------------------
# Instruction / chat
# ---------------------------------------------------------------------------
SYSTEM = "<|system|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
INSTRUCTION_SEP = "<|instruction_sep|>"
END_OF_TURN = "<|end_of_turn|>"

CHAT_SPECIAL_TOKENS = [SYSTEM, USER, ASSISTANT, INSTRUCTION_SEP, END_OF_TURN]

# ---------------------------------------------------------------------------
# Reasoning / CoT
# ---------------------------------------------------------------------------
THINK = "<|think|>"           # start internal reasoning
END_THINK = "<|/think|>"      # end internal reasoning, answer follows
STEP = "<|step|>"             # explicit reasoning step marker

REASONING_SPECIAL_TOKENS = [THINK, END_THINK, STEP]

# ---------------------------------------------------------------------------
# Agentic / tools
# ---------------------------------------------------------------------------
TOOL_CALL = "<|tool_call|>"
END_TOOL_CALL = "<|end_tool_call|>"
TOOL_RESULT = "<|tool_result|>"
PLAN = "<|plan|>"
END_PLAN = "<|/plan|>"

AGENTIC_SPECIAL_TOKENS = [TOOL_CALL, END_TOOL_CALL, TOOL_RESULT, PLAN, END_PLAN]

# ---------------------------------------------------------------------------
# Domain tags (used in system prompts / data mixture for specialization)
# ---------------------------------------------------------------------------
DOMAIN_NLP = "<|domain:nlp|>"
DOMAIN_REASONING = "<|domain:reasoning|>"
DOMAIN_LEGAL = "<|domain:legal|>"
DOMAIN_CYBER = "<|domain:cyber|>"
DOMAIN_SE = "<|domain:software|>"
DOMAIN_FINANCE = "<|domain:finance|>"
DOMAIN_LINGUISTICS = "<|domain:linguistics|>"
DOMAIN_MATH = "<|domain:math|>"
DOMAIN_AGENT = "<|domain:agent|>"
DOMAIN_CODE = "<|domain:code|>"
DOMAIN_GENAI = "<|domain:genai|>"

DOMAIN_SPECIAL_TOKENS = [
    DOMAIN_NLP,
    DOMAIN_REASONING,
    DOMAIN_LEGAL,
    DOMAIN_CYBER,
    DOMAIN_SE,
    DOMAIN_FINANCE,
    DOMAIN_LINGUISTICS,
    DOMAIN_MATH,
    DOMAIN_AGENT,
    DOMAIN_CODE,
    DOMAIN_GENAI,
]

# ---------------------------------------------------------------------------
# Full list for tokenizer training
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Multimodal (v0.4)
# ---------------------------------------------------------------------------
IMAGE_START = "<|image_start|>"
IMAGE_END = "<|image_end|>"
IMAGE_PAD = "<|image_pad|>"
AUDIO_START = "<|audio_start|>"
AUDIO_END = "<|audio_end|>"

MULTIMODAL_SPECIAL_TOKENS = [IMAGE_START, IMAGE_END, IMAGE_PAD, AUDIO_START, AUDIO_END]

ALL_SPECIAL_TOKENS = (
    CORE_SPECIAL_TOKENS
    + CHAT_SPECIAL_TOKENS
    + REASONING_SPECIAL_TOKENS
    + AGENTIC_SPECIAL_TOKENS
    + DOMAIN_SPECIAL_TOKENS
    + MULTIMODAL_SPECIAL_TOKENS
)

# Human-readable map for documentation / data pipelines
DOMAIN_MAP = {
    "nlp": DOMAIN_NLP,
    "reasoning": DOMAIN_REASONING,
    "legal": DOMAIN_LEGAL,
    "cyber": DOMAIN_CYBER,
    "cybersecurity": DOMAIN_CYBER,
    "software": DOMAIN_SE,
    "software_engineering": DOMAIN_SE,
    "finance": DOMAIN_FINANCE,
    "bfsi": DOMAIN_FINANCE,
    "linguistics": DOMAIN_LINGUISTICS,
    "math": DOMAIN_MATH,
    "logical_math": DOMAIN_MATH,
    "agent": DOMAIN_AGENT,
    "agentic": DOMAIN_AGENT,
    "code": DOMAIN_CODE,
    "code_architecture": DOMAIN_CODE,
    "genai": DOMAIN_GENAI,
    "gen_ai": DOMAIN_GENAI,
}
