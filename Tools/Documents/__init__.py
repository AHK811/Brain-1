from __future__ import annotations
from pathlib import Path
from Tools.Core.tool_factory import make_tool
from Tools.Core.tool_result import ToolResult

def _ingest(path: str) -> ToolResult:
    try:
        from brain.experimental.ingest import load_any
        r = load_any(path)
        return ToolResult("doc_read", True, r.to_prompt(max_chars=12000), meta={"errors": r.errors})
    except Exception as e:
        p = Path(path)
        if p.suffix.lower() in {".txt", ".md", ".csv", ".json", ".yaml", ".yml", ".html", ".xml"}:
            return ToolResult("doc_read", True, p.read_text(encoding="utf-8", errors="ignore")[:12000])
        return ToolResult("doc_read", False, str(e), error_type="runtime")

def pdf_read(ctx=None, path: str = "", **kw): return _ingest(path)
def docx_read(ctx=None, path: str = "", **kw): return _ingest(path)
def xlsx_read(ctx=None, path: str = "", **kw): return _ingest(path)

def register(registry):
    for name, fn in [("pdf_read", pdf_read), ("docx_read", docx_read), ("xlsx_read", xlsx_read)]:
        registry.register(make_tool(name, f"Read {name.split('_')[0].upper()} via ingest", {"path":{"type":"string","required":True}}, fn, "documents", "fs:read"))
