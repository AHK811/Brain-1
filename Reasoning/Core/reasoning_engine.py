from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from Reasoning.Mathematical.equation_solver import solve_arithmetic
from Reasoning.Reflection.reflection_engine import ReflectionEngine
@dataclass
class ReasoningTrace:
    steps: list = field(default_factory=list)
    def add(self, kind, detail):
        self.steps.append({"kind":kind,"detail":detail})
class ReasoningEngine:
    def __init__(self):
        self.reflection=ReflectionEngine(); self.trace=ReasoningTrace()
    def solve_math(self, expression: str):
        try:
            val=solve_arithmetic(expression); self.trace.add("math",{"expr":expression,"value":val}); return {"ok":True,"value":val}
        except Exception as e:
            self.trace.add("math_error",str(e)); return {"ok":False,"error":str(e)}
    def reflect(self, answer: str, goal: str=""):
        r=self.reflection.critique(answer,goal); self.trace.add("reflect",r); return {"ok":r.ok,"issues":r.issues,"suggestion":r.suggestion}
