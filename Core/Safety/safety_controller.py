from __future__ import annotations
import re
_DANGER = re.compile(r"(rm\s+-rf\s+/|mkfs\.|format\s+c:)", re.I)
class SafetyController:
    def check_text(self, text: str) -> dict:
        if _DANGER.search(text or ""):
            return {"ok": False, "reason": "dangerous_pattern"}
        return {"ok": True}
    def check_tool(self, name: str) -> dict:
        if name in {"delete_file"}:
            return {"ok": False, "reason": "tool_blocked", "needs_approval": True}
        return {"ok": True}
