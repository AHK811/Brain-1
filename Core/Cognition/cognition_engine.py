from __future__ import annotations
from typing import Any
class CognitionEngine:
    def reason(self, question: str) -> dict[str, Any]:
        # Bridge to Reasoning package when available
        try:
            from Reasoning import ReasoningEngine
            re = ReasoningEngine()
            # try math first
            if any(c in question for c in "+-*/"):
                expr = "".join(ch for ch in question if ch in "0123456789+-*/(). ")
                if expr.strip():
                    return re.solve_math(expr.strip())
            return re.reflect(question)
        except Exception as e:
            return {"ok": False, "error": str(e), "question": question}
