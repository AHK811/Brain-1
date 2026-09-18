from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
from Core.AGI.cognitive_cycle import CognitiveCycle

@dataclass
class AGIController:
    """High-level controller composing cycle + optional subsystems."""
    cycle: CognitiveCycle = field(default_factory=CognitiveCycle)
    max_cycles: int = 8

    def run(self, goal: str) -> dict[str, Any]:
        obs = ""
        final = None
        history = []
        for i in range(self.max_cycles):
            out = self.cycle.step(goal, observation=obs)
            history.append(out)
            if out.get("final"):
                final = out["final"]
                break
            obs = str(out.get("thought", ""))[:400]
        return {"goal": goal, "final": final or (history[-1].get("thought") if history else ""), "history": history}
