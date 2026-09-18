from __future__ import annotations
from Tools.Core.tool_factory import make_tool
from Tools.Core.tool_result import ToolResult

def media_process(ctx=None, path: str = "", **kw) -> ToolResult:
    try:
        from brain.experimental.multimodal.unified import process_media
        b = process_media(path)
        return ToolResult("media_process", True, b.prompt[:12000], meta={"kind": b.kind})
    except Exception as e:
        return ToolResult("media_process", False, str(e), error_type="runtime")

def ocr_image(ctx=None, path: str = "", **kw) -> ToolResult:
    try:
        from brain.experimental.ocr import get_ocr_engine
        return ToolResult("ocr_image", True, get_ocr_engine().extract(path))
    except Exception as e:
        return ToolResult("ocr_image", False, str(e), error_type="runtime")

def register(registry):
    registry.register(make_tool("media_process", "Process image/audio/video/document", {"path":{"type":"string","required":True}}, media_process, "media", "fs:read"))
    registry.register(make_tool("ocr_image", "OCR an image", {"path":{"type":"string","required":True}}, ocr_image, "media", "fs:read"))
