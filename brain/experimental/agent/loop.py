"""
brain/agent/loop.py

Agentic loop: generate → parse tool calls → execute → append results → repeat.

There is NO fixed tool-call limit when tools are available. Stopping conditions:
  1. Model produces a turn with no <|tool_call|> (final answer)
  2. Optional soft max_turns safety (default high: 32)
  3. Optional wall-clock timeout
  4. Host cancels

Works with any generate_fn(model, prompt_ids) -> token ids / text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from brain.experimental.sft.templates import format_chat, DOMAIN_SYSTEM_PROMPTS
from brain.tokenizer.special_tokens import (
    SYSTEM, USER, ASSISTANT, END_OF_TURN, TOOL_CALL, TOOL_RESULT, PLAN, END_PLAN,
)
from brain.experimental.tools.registry import ToolRegistry, parse_tool_calls, format_tool_result


@dataclass
class AgentConfig:
    domain: str = "agent"
    system_extra: str = ""
    # Soft safety only — not a product limit when tools work
    max_turns: int = 32
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 40
    require_plan_first: bool = False


@dataclass
class AgentTurn:
    role: str  # user | assistant | tool
    content: str


@dataclass
class AgentLoop:
    registry: ToolRegistry
    # generate_fn(prompt_text: str) -> str   (decoded assistant continuation)
    generate_fn: Callable[[str], str]
    config: AgentConfig = field(default_factory=AgentConfig)

    def _system_prompt(self) -> str:
        base = DOMAIN_SYSTEM_PROMPTS.get(self.config.domain, DOMAIN_SYSTEM_PROMPTS["agent"])
        tools_block = self.registry.schemas_for_prompt()
        rules = (
            "You may call tools when they help. "
            "Emit tool calls in this exact format:\n"
            f"{TOOL_CALL}\n"
            '{"name": "tool_name", "arguments": {...}}\n'
            "<|end_tool_call|>\n"
            "You may emit multiple tool calls. After tool results arrive, continue reasoning. "
            "When you have enough information, answer the user directly without more tool calls. "
            "There is no limit on tool use when tools are available."
        )
        if self.config.require_plan_first:
            rules += f" Start with {PLAN} ... {END_PLAN} before the first tool call when the task is multi-step."
        extra = self.config.system_extra.strip()
        return f"{base}\n\nAvailable tools:\n{tools_block}\n\n{rules}" + (f"\n{extra}" if extra else "")

    def build_prompt(self, history: list[AgentTurn]) -> str:
        parts = [f"{SYSTEM} {self._system_prompt()} {END_OF_TURN}"]
        for t in history:
            if t.role == "user":
                parts.append(f"{USER} {t.content} {END_OF_TURN}")
            elif t.role == "assistant":
                parts.append(f"{ASSISTANT} {t.content} {END_OF_TURN}")
            elif t.role == "tool":
                parts.append(t.content)  # already formatted tool_result blocks
        parts.append(f"{ASSISTANT}")
        return "".join(parts)

    def run(self, user_message: str) -> dict[str, Any]:
        history: list[AgentTurn] = [AgentTurn(role="user", content=user_message)]
        trace: list[dict[str, Any]] = []

        for turn_i in range(self.config.max_turns):
            prompt = self.build_prompt(history)
            assistant_text = self.generate_fn(prompt)
            history.append(AgentTurn(role="assistant", content=assistant_text))
            calls = parse_tool_calls(assistant_text)
            trace.append({"turn": turn_i, "assistant": assistant_text, "tool_calls": calls})

            if not calls:
                # Final answer
                return {
                    "answer": assistant_text,
                    "history": history,
                    "trace": trace,
                    "turns": turn_i + 1,
                    "stopped": "final_answer",
                }

            # Execute tool calls (parallel when safe)
            results = self.registry.run_many(calls, parallel=True)
            result_blocks = []
            for result in results:
                result_blocks.append(format_tool_result(result))
                trace[-1].setdefault("results", []).append(
                    {"name": result.name, "ok": result.ok, "content": result.content,
                     "error_type": result.error_type}
                )
            history.append(AgentTurn(role="tool", content="".join(result_blocks)))

        return {
            "answer": history[-1].content if history else "",
            "history": history,
            "trace": trace,
            "turns": self.config.max_turns,
            "stopped": "max_turns",
        }
