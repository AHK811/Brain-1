from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any
@dataclass
class Skill:
    name: str
    description: str
    handler: Callable[..., Any]
    tags: list[str] = field(default_factory=list)
@dataclass
class SkillRegistry:
    skills: dict[str, Skill] = field(default_factory=dict)
    def register(self, skill: Skill) -> None:
        self.skills[skill.name] = skill
    def match(self, query: str) -> list[Skill]:
        q = query.lower()
        return [s for s in self.skills.values() if q in s.name.lower() or q in s.description.lower()]
