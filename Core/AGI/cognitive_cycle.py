from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

@dataclass
class CognitiveCycle:
    """Perceive → Reason → Plan → Act → Reflect → Learn (one loop)."""
    generate: Optional[Callable[[str], str]] = None
    tool_run: Optional[Callable[[str, dict], Any]] = None
    memory_remember: Optional[Callable[[str, str], None]] = None
    memory_recall: Optional[Callable[[str], list[str]]] = None
    log: list = field(default_factory=list)

    def step(self, goal: str, observation: str = "") -> dict[str, Any]:
        self.log.append({"phase": "perceive", "observation": observation})
        recalled = self.memory_recall(goal) if self.memory_recall else []
        self.log.append({"phase": "recall", "items": recalled[:5]})
        prompt = f"Goal: {goal}\nObs: {observation}\nMemory: {recalled[:3]}\nDecide next action or FINAL answer."
        thought = self.generate(prompt) if self.generate else f"FINAL: progress on {goal}"
        self.log.append({"phase": "reason", "thought": thought[:500]})
        result = {"goal": goal, "thought": thought, "final": None}
        if "FINAL:" in thought.upper():
            result["final"] = thought.split("FINAL:")[-1].strip()
            self.log.append({"phase": "done", "final": result["final"]})
            if self.memory_remember and result["final"]:
                self.memory_remember(goal, result["final"])
        return result
