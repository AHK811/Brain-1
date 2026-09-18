from __future__ import annotations
from collections import defaultdict
from typing import Any, Callable
class EventBus:
    def __init__(self):
        self._subs = defaultdict(list)
    def on(self, event: str, handler: Callable) -> None:
        self._subs[event].append(handler)
    def emit(self, event: str, **payload: Any) -> None:
        for h in self._subs.get(event, []):
            h(**payload)
