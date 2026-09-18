from __future__ import annotations
class MemoryController:
    def __init__(self):
        try:
            from Memory import MemoryEngine
            self.engine = MemoryEngine()
        except Exception:
            self.engine = None
    def remember(self, key: str, content: str) -> None:
        if self.engine:
            self.engine.remember(key, content)
    def recall(self, query: str, k: int = 5) -> list[str]:
        if self.engine:
            return self.engine.recall(query, k=k)
        return []
