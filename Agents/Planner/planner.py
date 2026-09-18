from __future__ import annotations
from dataclasses import dataclass
@dataclass
class PlanStep:
    id: str
    description: str
    tool: str | None = None
@dataclass
class Plan:
    goal: str
    steps: list[PlanStep]
class Planner:
    def plan(self, goal: str) -> Plan:
        return Plan(goal=goal, steps=[PlanStep(id="s1", description=goal)])
