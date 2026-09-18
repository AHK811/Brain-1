from __future__ import annotations
from dataclasses import dataclass
@dataclass
class ReflectionResult:
    ok: bool
    issues: list
    suggestion: str
class ReflectionEngine:
    def critique(self, answer: str, goal: str="") -> ReflectionResult:
        issues=[]
        if not answer or not str(answer).strip(): issues.append("empty")
        if len(str(answer))<5: issues.append("short")
        return ReflectionResult(ok=not issues, issues=issues, suggestion="revise" if issues else "accept")
