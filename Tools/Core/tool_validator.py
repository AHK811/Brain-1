"""JSON-schema-like argument validation."""
from __future__ import annotations
from typing import Any, Optional

def validate_args(schema: dict[str, dict], arguments: dict[str, Any]) -> Optional[str]:
    if not isinstance(arguments, dict):
        return "arguments must be an object"
    for key, meta in schema.items():
        if meta.get("required") and key not in arguments:
            return f"missing required argument: {key}"
        if key not in arguments:
            continue
        val = arguments[key]
        typ = meta.get("type")
        if typ == "string" and not isinstance(val, str):
            return f"{key} must be string"
        if typ == "integer" and not isinstance(val, int):
            return f"{key} must be integer"
        if typ == "number" and not isinstance(val, (int, float)):
            return f"{key} must be number"
        if typ == "boolean" and not isinstance(val, bool):
            return f"{key} must be boolean"
        if typ == "array" and not isinstance(val, list):
            return f"{key} must be array"
        if typ == "object" and not isinstance(val, dict):
            return f"{key} must be object"
        if "enum" in meta and val not in meta["enum"]:
            return f"{key} must be one of {meta['enum']}"
    return None
