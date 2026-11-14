from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from brainlab.eval.constraints import (
    ConstraintError,
    RegexConstraint,
    finite_json,
    parse_json,
    schema_pattern,
    validate_schema,
)


class ToolError(ValueError):
    pass


@dataclass(frozen=True)
class Tool:
    callback: Callable[..., Any]
    arguments_schema: Mapping[str, Any]


class ToolRunner:
    """Only explicitly registered trusted callbacks can run; arguments are keyword arguments."""

    def __init__(
        self,
        registry: Mapping[str, Tool],
        *,
        max_calls: int = 4,
        max_call_bytes: int = 8192,
        max_result_bytes: int = 8192,
    ):
        if not registry or len(registry) > 32:
            raise ToolError("register between 1 and 32 trusted tools")
        if max_calls < 1 or max_call_bytes < 1 or max_result_bytes < 1:
            raise ToolError("tool limits must be positive")
        self.registry: dict[str, Tool] = {}
        branches = []
        for name, tool in registry.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,63}", name):
                raise ToolError(f"invalid registered tool name: {name!r}")
            if not callable(tool.callback):
                raise ToolError(f"tool {name} is not callable")
            schema = copy.deepcopy(tool.arguments_schema)
            if schema.get("type") != "object":
                raise ToolError("tool arguments must use a closed object schema")
            schema_pattern(schema)
            self.registry[name] = Tool(tool.callback, schema)
            branches.append(schema_pattern({
                "type": "object",
                "properties": {"name": {"const": name}, "arguments": schema},
                "required": ["name", "arguments"],
                "additionalProperties": False,
            }))
        self.grammar = RegexConstraint(
            rb"[ \t\r\n]*(?:" + b"|".join(branches) + rb")[ \t\r\n]*",
            max_bytes=max_call_bytes,
        )
        self.max_calls = max_calls
        self.max_call_bytes = max_call_bytes
        self.max_result_bytes = max_result_bytes
        self.calls = 0

    def fresh(self) -> ToolRunner:
        runner = copy.copy(self)
        runner.calls = 0
        return runner

    def validate_call(self, payload: bytes) -> tuple[Tool, dict[str, Any]]:
        if len(payload) > self.max_call_bytes:
            raise ToolError("tool call exceeds byte limit")
        try:
            call = parse_json(payload)
            if type(call) is not dict or set(call) != {"name", "arguments"}:
                raise ToolError("tool call requires exactly name and arguments")
            name = call["name"]
            if type(name) is not str or name not in self.registry:
                raise ToolError("tool name is not registered")
            tool = self.registry[name]
            validate_schema(call["arguments"], tool.arguments_schema)
            return tool, call["arguments"]
        except ConstraintError as exc:
            raise ToolError(f"invalid tool call: {exc}") from exc

    def dispatch(self, payload: bytes) -> str:
        if self.calls >= self.max_calls:
            raise ToolError("tool call limit reached")
        tool, arguments = self.validate_call(payload)
        self.calls += 1
        try:
            result = tool.callback(**arguments)
        except Exception as exc:
            raise ToolError(f"registered tool raised {type(exc).__name__}") from exc
        try:
            finite_json(result)
            pieces: list[str] = []
            size = 0
            encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            for piece in encoder.iterencode(result):
                size += len(piece.encode("utf-8"))
                if size > self.max_result_bytes:
                    raise ToolError("tool result exceeds byte limit")
                pieces.append(piece)
            return "".join(pieces)
        except (ValueError, TypeError, RecursionError) as exc:
            raise ToolError(f"invalid tool result: {exc}") from exc


def safe_cli_tools(*, max_calls: int = 4, max_result_bytes: int = 8192) -> ToolRunner:
    return ToolRunner(
        {"echo": Tool(lambda text: text, {
            "type": "object",
            "properties": {"text": {"type": "string", "maxLength": 1024}},
            "required": ["text"],
            "additionalProperties": False,
        })},
        max_calls=max_calls,
        max_result_bytes=max_result_bytes,
    )
