from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from Agents.Core.task_graph import TaskGraph, TaskNode
from Agents.Core.event_bus import EventBus
from Agents.Core.state_machine import StateMachine
GenerateFn = Callable[[str], str]
@dataclass
class AgentConfig:
    max_turns: int = 16
    max_tool_calls_per_turn: int = 8
@dataclass
class AgentEngine:
    generate: Optional[GenerateFn] = None
    tool_engine: Any = None
    config: AgentConfig = field(default_factory=AgentConfig)
    bus: EventBus = field(default_factory=EventBus)
    sm: StateMachine = field(default_factory=lambda: StateMachine(state="idle"))
    history: list = field(default_factory=list)
    def run(self, goal: str) -> dict[str, Any]:
        self.history.clear()
        final = ""
        for turn in range(self.config.max_turns):
            plan_prompt = f"Goal: {goal}\nTurn {turn}. Reply with tool calls or FINAL: answer."
            text = self.generate(plan_prompt) if self.generate else f"FINAL: acknowledged: {goal}"
            self.history.append({"role": "assistant", "content": text, "turn": turn})
            if self.tool_engine is not None and "<|tool_call|>" in text:
                try:
                    from Tools.Core.tool_router import parse_tool_calls
                    calls = parse_tool_calls(text)[:self.config.max_tool_calls_per_turn]
                    results = self.tool_engine.run_many(calls, parallel=True)
                    obs = "\n".join(f"{r.name}: {r.content[:500]}" for r in results)
                    self.history.append({"role": "tool", "content": obs, "turn": turn})
                    continue
                except Exception as e:
                    self.history.append({"role": "tool", "content": str(e), "turn": turn})
            if "FINAL:" in text.upper() or self.generate is None:
                final = text.split("FINAL:")[-1].strip() if "FINAL:" in text else text
                break
            final = text
            break
        return {"goal": goal, "final": final, "history": self.history}
