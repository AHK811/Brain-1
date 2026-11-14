from __future__ import annotations

import codecs
import json
import math
from collections.abc import Mapping
from typing import Any


class ConstraintError(ValueError):
    pass


_WS = rb"[ \t\r\n]*"
_UTF8 = (
    rb"(?:[\xc2-\xdf][\x80-\xbf]|\xe0[\xa0-\xbf][\x80-\xbf]|"
    rb"[\xe1-\xec\xee-\xef][\x80-\xbf]{2}|\xed[\x80-\x9f][\x80-\xbf]|"
    rb"\xf0[\x90-\xbf][\x80-\xbf]{2}|[\xf1-\xf3][\x80-\xbf]{3}|"
    rb"\xf4[\x80-\x8f][\x80-\xbf]{2})"
)
_ESCAPE = (
    rb'\\(?:["\\/bfnrt]|u(?:[0-9a-cA-Ce-fE-F][0-9a-fA-F]{3}|'
    rb'[dD][0-7][0-9a-fA-F]{2}|[dD][89aAbB][0-9a-fA-F]{2}'
    rb'\\u[dD][c-fC-F][0-9a-fA-F]{2}))'
)
_STRING_CHAR = rb'(?:[\x20-\x21\x23-\x5b\x5d-\x7f]|' + _UTF8 + b'|' + _ESCAPE + b')'
_INTEGER = rb'-?(?:0|[1-9][0-9]*)'
_NUMBER = _INTEGER + rb'(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?'
_TYPES = {"object", "array", "string", "integer", "number", "boolean", "null"}


def finite_json(value: Any, depth: int = 0) -> None:
    if depth > 64:
        raise ConstraintError("JSON nesting exceeds 64")
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is str:
        try:
            value.encode("utf-8")
        except UnicodeError as exc:
            raise ConstraintError("JSON contains an unpaired Unicode surrogate") from exc
        return
    if type(value) is list:
        for item in value:
            finite_json(item, depth + 1)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for key, item in value.items():
            finite_json(key, depth + 1)
            finite_json(item, depth + 1)
        return
    raise ConstraintError("value is not finite JSON")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConstraintError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(raw: bytes | str) -> Any:
    def invalid(text: str) -> None:
        raise ConstraintError(f"non-finite JSON constant: {text}")

    try:
        value = json.loads(raw, parse_constant=invalid, object_pairs_hook=_pairs)
        finite_json(value)
        return value
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ConstraintError(f"invalid JSON: {exc}") from exc


def _literal(value: Any) -> bytes:
    import regex

    finite_json(value)
    return regex.escape(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _bound(schema: Mapping[str, Any], key: str, default: int | None) -> int | None:
    value = schema.get(key, default)
    if value is not None and (type(value) is not int or not 0 <= value <= 10000):
        raise ConstraintError(f"{key} must be an integer in [0,10000]")
    return value


def _is_type(value: Any, kind: str) -> bool:
    return {
        "object": type(value) is dict,
        "array": type(value) is list,
        "string": type(value) is str,
        "integer": type(value) is int,
        "number": type(value) in (int, float),
        "boolean": type(value) is bool,
        "null": value is None,
    }[kind]


def schema_pattern(schema: Mapping[str, Any], depth: int = 0) -> bytes:
    if not isinstance(schema, Mapping) or depth > 16:
        raise ConstraintError("schema must be an object with nesting at most 16")
    kind = schema.get("type")
    if kind is not None and (not isinstance(kind, str) or kind not in _TYPES):
        raise ConstraintError("only a single supported schema type is allowed")
    if "enum" in schema or "const" in schema:
        allowed = {"type", "enum"} if "enum" in schema else {"type", "const"}
        if set(schema) - allowed:
            raise ConstraintError("enum/const support only an optional type keyword")
        values = schema["enum"] if "enum" in schema else [schema["const"]]
        if not isinstance(values, list) or not 1 <= len(values) <= 256:
            raise ConstraintError("enum must contain 1..256 values")
        if kind and any(not _is_type(value, kind) for value in values):
            raise ConstraintError("enum/const value does not match type")
        pattern = b"(?:" + b"|".join(_literal(value) for value in values) + b")"
    else:
        if kind not in _TYPES:
            raise ConstraintError("schema requires type, enum, or const")
        keys = {
            "object": {"type", "properties", "required", "additionalProperties"},
            "array": {"type", "items", "minItems", "maxItems"},
            "string": {"type", "minLength", "maxLength"},
        }.get(kind, {"type"})
        unknown = set(schema) - keys
        if unknown:
            raise ConstraintError(f"unsupported schema keywords: {sorted(unknown)}")
        if kind == "object":
            if schema.get("additionalProperties") is not False:
                raise ConstraintError("objects require additionalProperties:false")
            props = schema.get("properties", {})
            required = schema.get("required", [])
            if not isinstance(props, Mapping) or len(props) > 32 or any(type(k) is not str for k in props):
                raise ConstraintError("properties must contain at most 32 named schemas")
            if not isinstance(required, list) or any(type(k) is not str for k in required):
                raise ConstraintError("required must be a list of property names")
            if len(set(required)) != len(required) or set(required) - set(props):
                raise ConstraintError("required contains duplicate or undeclared properties")
            pairs = [(key, _literal(key) + _WS + b":" + _WS + schema_pattern(sub, depth + 1)) for key, sub in props.items()]
            choices = [] if required else [b""]
            for index, (key, pair) in enumerate(pairs):
                suffix = b""
                for other, body in pairs[index + 1:]:
                    part = _WS + b"," + _WS + body
                    suffix += part if other in required else b"(?:" + part + b")?"
                choices.append(pair + suffix)
                if key in required:
                    break
            pattern = rb"\{" + _WS + b"(?:" + b"|".join(choices) + b")" + _WS + rb"\}"
        elif kind in {"array", "string"}:
            lower_key, upper_key = ("minItems", "maxItems") if kind == "array" else ("minLength", "maxLength")
            lower, upper = _bound(schema, lower_key, 0), _bound(schema, upper_key, None)
            if lower is None or (upper is not None and upper < lower):
                raise ConstraintError("schema has inconsistent length bounds")
            if kind == "string":
                count = f"{{{lower},{'' if upper is None else upper}}}".encode()
                pattern = b'"' + _STRING_CHAR + count + b'"'
            else:
                if "items" not in schema:
                    raise ConstraintError("arrays require an items schema")
                item = schema_pattern(schema["items"], depth + 1)
                if upper == 0:
                    body = b""
                else:
                    count = f"{{{max(0, lower - 1)},{'' if upper is None else upper - 1}}}".encode()
                    body = item + b"(?:" + _WS + b"," + _WS + item + b")" + count
                    if lower == 0:
                        body = b"(?:" + body + b")?"
                pattern = rb"\[" + _WS + body + _WS + rb"\]"
        else:
            pattern = {"integer": _INTEGER, "number": _NUMBER, "boolean": rb"(?:true|false)", "null": b"null"}[kind]
    if len(pattern) > 262144:
        raise ConstraintError("compiled schema exceeds 256 KiB")
    return pattern


def validate_schema(value: Any, schema: Mapping[str, Any]) -> None:
    finite_json(value)
    kind = schema.get("type")
    if kind and not _is_type(value, kind):
        raise ConstraintError(f"value does not match schema type {kind}")
    if "const" in schema or "enum" in schema:
        values = schema.get("enum", [schema.get("const")])
        if not any(_json_equal(value, option) for option in values):
            raise ConstraintError("value does not match enum/const")
    elif kind == "object":
        if set(value) - set(schema.get("properties", {})) or set(schema.get("required", [])) - set(value):
            raise ConstraintError("object has missing or additional properties")
        for key, item in value.items():
            validate_schema(item, schema["properties"][key])
    elif kind == "array":
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", math.inf):
            raise ConstraintError("array length outside schema bounds")
        for item in value:
            validate_schema(item, schema["items"])
    elif kind == "string":
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", math.inf):
            raise ConstraintError("string length outside schema bounds")


def _json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return type(left) in (int, float) and type(right) in (int, float) and left == right
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_json_equal(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_json_equal(a, b) for a, b in zip(left, right))
    return left == right


class RegexConstraint:
    """Partial fullmatch over UTF-8 bytes; regex character classes are byte-oriented."""

    def __init__(self, pattern: str | bytes, *, timeout: float = 0.05, max_bytes: int = 65536):
        import regex

        if not math.isfinite(timeout) or timeout <= 0 or max_bytes <= 0:
            raise ConstraintError("constraint timeout and max_bytes must be positive")
        raw = pattern.encode("utf-8") if isinstance(pattern, str) else pattern
        if len(raw) > 262144:
            raise ConstraintError("regex exceeds 256 KiB")
        try:
            self.pattern = regex.compile(raw)
        except regex.error as exc:
            raise ConstraintError(f"invalid regex: {exc}") from exc
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.data = b""

    def fresh(self) -> RegexConstraint:
        import copy

        result = copy.copy(self)
        result.data = b""
        return result

    def _match(self, data: bytes):
        try:
            return self.pattern.fullmatch(data, partial=True, timeout=self.timeout)
        except TimeoutError as exc:
            raise ConstraintError("constraint regex timed out") from exc

    def allows(self, token: bytes) -> bool:
        data = self.data + token
        if not token or len(data) > self.max_bytes:
            return False
        try:
            codecs.utf_8_decode(data, "strict", False)
        except UnicodeError:
            return False
        match = self._match(data)
        return match is not None

    def advance(self, token: bytes) -> None:
        if not self.allows(token):
            raise ConstraintError("token violates constraint")
        self.data += token

    @property
    def complete(self) -> bool:
        try:
            self.data.decode("utf-8")
        except UnicodeError:
            return False
        match = self._match(self.data)
        return match is not None and not match.partial

    def finish(self) -> None:
        if not self.complete:
            raise ConstraintError("generation ended with an incomplete constraint")


class JsonSchemaConstraint(RegexConstraint):
    """Closed objects in schema property order, typed arrays/scalars, enum/const and lengths."""

    def __init__(self, schema: Mapping[str, Any], **kwargs: Any):
        import copy

        self.schema = copy.deepcopy(schema)
        super().__init__(_WS + schema_pattern(self.schema) + _WS, **kwargs)

    @property
    def complete(self) -> bool:
        if not super().complete:
            return False
        try:
            validate_schema(parse_json(self.data), self.schema)
        except ConstraintError:
            return False
        return True

    def finish(self) -> None:
        super().finish()
        validate_schema(parse_json(self.data), self.schema)
