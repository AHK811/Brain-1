from __future__ import annotations
from dataclasses import dataclass, field
@dataclass
class StateMachine:
    state: str = "idle"
    transitions: dict = field(default_factory=dict)
    def allow(self, frm: str, event: str, to: str) -> None:
        self.transitions[(frm, event)] = to
    def trigger(self, event: str) -> str:
        key = (self.state, event)
        if key not in self.transitions:
            raise ValueError(f"bad transition {key}")
        self.state = self.transitions[key]
        return self.state
