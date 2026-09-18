from __future__ import annotations
from dataclasses import dataclass

@dataclass
class TokenBudget:
    max_tokens: int = 4096
    def fit(self, text: str, reserve: int = 512) -> str:
        budget = max(self.max_tokens - reserve, 256) * 4
        return text if len(text) <= budget else text[:budget] + "\n[truncated]"
