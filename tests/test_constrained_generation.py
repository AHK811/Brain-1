from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from brainlab.eval.constraints import (
    ConstraintError,
    JsonSchemaConstraint,
    RegexConstraint,
    parse_json,
    schema_pattern,
    validate_schema,
)
from brainlab.eval.generate import GenerationError, generate
from brainlab.eval.tools import Tool, ToolError, ToolRunner, safe_cli_tools
from brainlab.tok.reference import SPECIAL_TOKENS, Tokenizer


SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"const": "search"},
        "arguments": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 32},
                "limit": {"type": "integer"},
            },
            "required": ["query", "limit"],
            "additionalProperties": False,
        },
    },
    "required": ["name", "arguments"],
    "additionalProperties": False,
}


def mask_ids(constraint: RegexConstraint, token_bytes: list[bytes]) -> list[int]:
    return [index for index, raw in enumerate(token_bytes) if not constraint.allows(raw)]


def test_regex_partial_match_allows_only_completions() -> None:
    constraint = RegexConstraint(rb"ab")
    assert constraint.allows(b"a") and not constraint.allows(b"b"), "a prefix of the pattern is still partial"
    constraint.advance(b"a")
    assert not constraint.complete
    constraint.advance(b"b")
    assert constraint.complete
    with pytest.raises(ConstraintError, match="violates constraint"):
        constraint.advance(b"c")
    with pytest.raises(ConstraintError, match="incomplete"):
        RegexConstraint(rb"ab").finish()


def test_byte_and_utf8_limits_are_enforced() -> None:
    small = RegexConstraint(rb"[a-z]*", max_bytes=4)
    small.advance(b"abcd")
    assert not small.allows(b"e"), "the byte budget is hard"
    assert small.allows(b"") is False, "an empty token can never advance the stream"
    split_codepoint = RegexConstraint(rb"(?s:.)*")
    assert not split_codepoint.allows(b"\xc3\x28"), "a lead byte followed by ASCII is malformed, not incomplete"
    split_codepoint.advance(b"\xc3")
    assert not split_codepoint.complete, "a lone UTF-8 lead byte is not a finished document"
    assert split_codepoint.allows(b"\xa9"), "a byte-level tokenizer may split a code point across tokens"
    split_codepoint.advance(b"\xa9")
    assert split_codepoint.complete
    with pytest.raises(ConstraintError, match="positive"):
        RegexConstraint(rb"a", timeout=0.0)
    with pytest.raises(ConstraintError, match="positive"):
        RegexConstraint(rb"a", max_bytes=0)
    with pytest.raises(ConstraintError, match="invalid regex"):
        RegexConstraint(rb"(?P")


def test_schema_pattern_accepts_valid_and_rejects_invalid_documents() -> None:
    pattern = schema_pattern(SEARCH_SCHEMA)
    good = b'{"name":"search","arguments":{"query":"primes","limit":3}}'
    bad = [
        b'{"name":"search","arguments":{"limit":3,"query":"primes"}}',
        b'{"name":"other","arguments":{"query":"primes","limit":3}}',
        b'{"name":"search","arguments":{"query":"primes"}}',
        b'{"name":"search","arguments":{"query":"primes","limit":3,"extra":1}}',
        b'{"arguments":{"query":"primes","limit":3},"name":"search"}',
    ]
    constraint = RegexConstraint(pattern.decode())
    assert constraint.allows(good)
    for document in bad:
        assert not constraint.allows(document), document


def test_masking_removes_every_token_that_breaks_the_grammar() -> None:
    constraint = JsonSchemaConstraint(SEARCH_SCHEMA)
    vocabulary = [
        b'{"name":"search","arguments":{"query":"ok","limit":1}}',
        b'{"name":',
        b'{"nam',
        b'{"wrong":1}',
        b'}',
        b'{"name":"search","arguments":{"query":"","limit":1}}',
    ]
    allowed = [index for index, raw in enumerate(vocabulary) if constraint.allows(raw)]
    assert allowed == [0, 1, 2], "only the complete call and its prefixes survive"
    banned = mask_ids(constraint, vocabulary)
    assert banned == [3, 4, 5]
    assert not constraint.complete, "a prefix alone never satisfies the schema"
    constraint.advance(vocabulary[1])
    constraint.advance(b'"search"')
    constraint.advance(b',"arguments"')
    constraint.advance(b':{"query"')
    constraint.advance(b':"hi"')
    constraint.advance(b',"limit"')
    constraint.advance(b":2}}")
    assert constraint.complete
    assert parse_json(constraint.data) == {"name": "search", "arguments": {"query": "hi", "limit": 2}}


def test_json_numbers_integers_booleans_and_arrays_follow_the_schema() -> None:
    schema = {
        "type": "object",
        "properties": {
            "tags": {"type": "array", "items": {"type": "string", "maxLength": 4}, "minItems": 1, "maxItems": 2},
            "score": {"type": "number"},
            "flag": {"type": "boolean"},
            "nothing": {"type": "null"},
        },
        "required": ["tags"],
        "additionalProperties": False,
    }
    constraint = JsonSchemaConstraint(schema)
    for document in (
        b'{"tags":["a"]}',
        b'{"tags":["a","b"],"score":-1.5e3,"flag":true,"nothing":null}',
    ):
        assert constraint.allows(document), document
        validate_schema(parse_json(document.decode()), schema)
    for document in (
        b'{"tags":[]}',
        b'{"tags":["a","b","c"]}',
        b'{"tags":["abcde"]}',
        b'{"tags":["a"],"score":"3"}',
        b'{"tags":["a"],"other":1}',
        b'{"tags":["a"],"flag":"true"}',
    ):
        assert not constraint.allows(document), document


def test_repeated_calls_and_invalid_tokens_never_complete_a_schema() -> None:
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"], "additionalProperties": False}
    constraint = JsonSchemaConstraint(schema)
    for token in (b"{", b'"a"', b":", b"1", b"}"):
        constraint.advance(token)
    assert constraint.complete
    for token in (b"{", b"}", b",", b"2"):
        assert not constraint.allows(token), "trailing text cannot extend a finished document"


def test_enum_and_const_reject_out_of_set_values() -> None:
    constraint = JsonSchemaConstraint({"type": "string", "enum": ["a", "bb", "c\"d"]})
    assert constraint.allows(b'"bb"') and constraint.allows(b'"c\\"d"')
    assert not constraint.allows(b'"cc"')
    assert not constraint.allows(b'"a\\u0000b"')
    with pytest.raises(ConstraintError, match="enum/const value does not match type"):
        schema_pattern({"type": "integer", "enum": [1, "two"]})


def test_unsupported_schema_keywords_are_refused() -> None:
    for schema, message in (
        ({"type": "object"}, "additionalProperties"),
        ({"type": "object", "properties": {}, "additionalProperties": True}, "additionalProperties"),
        ({"type": "string", "pattern": "^a$"}, "unsupported schema keywords"),
        ({"type": "array"}, "items"),
        ({"type": "object", "minItems": 1, "additionalProperties": False}, "unsupported schema keywords"),
        ({"type": "nested"}, "single supported schema type"),
        ({}, "requires type"),
        ({"type": "string", "maxLength": 20000}, "must be an integer in"),
    ):
        with pytest.raises(ConstraintError, match=message):
            schema_pattern(schema)


def test_non_finite_json_is_never_accepted() -> None:
    for raw in ("NaN", "Infinity", "-Infinity", '{"a":NaN}'):
        with pytest.raises(ConstraintError, match="JSON"):
            parse_json(raw)
    with pytest.raises(ConstraintError, match="duplicate JSON key"):
        parse_json('{"a":1,"a":2}')
    with pytest.raises(ConstraintError, match="surrogate"):
        from brainlab.eval.constraints import finite_json

        finite_json("\ud800")


def runner(calls: list[dict], *, max_calls: int = 8) -> ToolRunner:
    def add(*, a: int, b: int) -> int:
        calls.append({"a": a, "b": b})
        return a + b

    return ToolRunner(
        {
            "add": Tool(add, {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
                "additionalProperties": False,
            }),
            "boom": Tool(lambda: (_ for _ in ()).throw(RuntimeError("exploded")), {"type": "object", "properties": {}, "additionalProperties": False}),
            "huge": Tool(lambda: "x" * 4096, {"type": "object", "properties": {}, "additionalProperties": False}),
        },
        max_calls=max_calls,
        max_result_bytes=64,
    )


def test_tool_registry_validates_names_schemas_and_dispatch() -> None:
    calls: list[dict] = []
    tools = runner(calls)
    assert tools.dispatch(b'{"name":"add","arguments":{"a":2,"b":3}}') == "5"
    assert calls == [{"a": 2, "b": 3}], "arguments arrive as keyword arguments"
    for payload, message in (
        (b'{"name":"subtract","arguments":{"a":2,"b":3}}', "not registered"),
        (b'{"name":"add","arguments":{"a":2}}', "missing or additional"),
        (b'{"name":"add","arguments":{"a":2,"b":3},"extra":1}', "exactly name and arguments"),
        (b"[]", "exactly name and arguments"),
        (b'{"name":"add","arguments":{"a":"2","b":3}}', "does not match schema type"),
        (b'{"name":"add","arguments":{"a":2,"b":3.5}}', "does not match schema type"),
    ):
        with pytest.raises(ToolError, match=message):
            tools.dispatch(payload)
    assert tools.calls == 1, "a refused call must not consume the budget"
    for payload, message in (
        (b'{"name":"boom","arguments":{}}', "registered tool raised RuntimeError"),
        (b'{"name":"huge","arguments":{}}', "result exceeds byte limit"),
    ):
        with pytest.raises(ToolError, match=message):
            tools.dispatch(payload)
    assert tools.calls == 3, "a dispatched call that fails still costs the budget"


def test_tool_call_budget_and_grammar_are_enforced() -> None:
    calls: list[dict] = []
    tools = runner(calls, max_calls=2)
    payload = b'{"name":"add","arguments":{"a":1,"b":1}}'
    assert tools.dispatch(payload) == "2"
    assert tools.dispatch(payload) == "2"
    with pytest.raises(ToolError, match="call limit"):
        tools.dispatch(payload)
    reset = tools.fresh()
    assert reset.calls == 0
    assert reset.dispatch(payload) == "2", "a fresh runner restores the budget without re-registering"
    assert tools.grammar.allows(payload), "the grammar must admit a registered call"
    assert tools.grammar.allows(b'{"name":"subtract","arguments":{}}') is False
    assert tools.grammar.allows(b'{"name":"add","arguments":{"a":1},"x":2}') is False
    assert tools.grammar.allows(b'{"name":"add","arguments":{"a":1,"b":1.5}}') is False
    assert tools.grammar.allows(b'{"name":"add","arguments":{"a":1'), "an open object stays partial"


def test_runner_rejects_a_registry_it_cannot_constrain() -> None:
    with pytest.raises(ToolError, match="between 1 and 32"):
        ToolRunner({})
    with pytest.raises(ToolError, match="invalid registered tool name"):
        ToolRunner({"<|tool|>": Tool(lambda: 1, {"type": "object", "additionalProperties": False})})
    with pytest.raises(ToolError, match="closed object schema"):
        ToolRunner({"ok": Tool(lambda: 1, {"type": "string"})})
    with pytest.raises(ToolError, match="not callable"):
        ToolRunner({"ok": Tool(1, {"type": "object", "additionalProperties": False})})  # type: ignore[arg-type]
    with pytest.raises(ToolError, match="positive"):
        safe_cli_tools(max_calls=0)


def test_safe_cli_echo_round_trips_unicode() -> None:
    tools = safe_cli_tools()
    assert tools.dispatch('{"name":"echo","arguments":{"text":"héllo"}}'.encode()) == json.dumps(
        "héllo", ensure_ascii=False
    )
    with pytest.raises(ToolError, match="does not match schema type"):
        tools.dispatch(b'{"name":"echo","arguments":{"text":5}}')
    with pytest.raises(ToolError, match="missing or additional"):
        tools.dispatch(b'{"name":"echo","arguments":{}}')


class StubModel:
    """One forward per decode step; each step scores a fixed set of token ids."""

    def __init__(self, vocab_size: int, context_len: int, steps: list[dict[int, float]]) -> None:
        self.spec = SimpleNamespace(context_len=context_len, vocab_size=vocab_size)
        self.steps = steps
        self.forwards = 0

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        logits = torch.full((1, ids.shape[1], self.spec.vocab_size), -1000.0)
        for token, score in self.steps[self.forwards].items():
            logits[0, -1, token] = score
        self.forwards += 1
        return logits


def control_tokenizer() -> Tokenizer:
    tokens = tuple(bytes([b]) for b in range(256)) + tuple(label.encode("ascii") for label in SPECIAL_TOKENS)
    return Tokenizer(token_bytes=tokens, merges=(), eos_id=len(tokens) - 1, pad_id=len(tokens) - 2)


def add_tools(calls: list[dict], *, max_calls: int = 4) -> ToolRunner:
    def add(*, a: int, b: int) -> int:
        calls.append({"a": a, "b": b})
        return a + b

    return ToolRunner(
        {"add": Tool(add, {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
            "additionalProperties": False,
        })},
        max_calls=max_calls,
    )


def run(model: StubModel, tokenizer: Tokenizer, max_new: int, *, prompt: list[int] | None = None, **kwargs):
    return generate(
        model, prompt if prompt is not None else tokenizer.encode("q"), max_new, 0.0, 0, 1.0,
        tokenizer.eos_id, 0, torch.device("cpu"), torch.float32, tokenizer=tokenizer, **kwargs,
    )


CALL = b'{"name":"add","arguments":{"a":2,"b":3}}'


def test_tool_loop_pauses_executes_and_resumes_with_the_result() -> None:
    tokenizer = control_tokenizer()
    calls: list[dict] = []
    special = tokenizer.special_id
    script = [special("<|call_start|>"), *CALL, special("<|call_end|>"), *b"sum is 5", tokenizer.eos_id]
    produced, stats = run(StubModel(tokenizer.vocab_size, 4096, [{t: 100.0} for t in script]), tokenizer, 64,
                          tools=add_tools(calls, max_calls=4))
    assert calls == [{"a": 2, "b": 3}], "the model's JSON arrives as keyword arguments"
    assert stats["tool_calls"] == 1 and stats["finish_reason"] == "eos"
    frame = [special("<|response_start|>"), ord("5"), special("<|response_end|>")]
    start = produced.index(frame[0])
    assert produced[start : start + 3] == frame, "the result is reinserted between the response controls"
    assert stats["inserted_tokens"] == 3
    assert stats["generated"] == len(script) - 1, "inserted frames are not model tokens"
    tail = produced[start + 3 :]
    assert tokenizer.decode_ids(tail, strict=False) == "sum is 5"
    assert produced[:start] == [special("<|call_start|>"), *CALL, special("<|call_end|>")]


def test_generation_refuses_to_end_inside_a_tool_call() -> None:
    tokenizer = control_tokenizer()
    calls: list[dict] = []
    special = tokenizer.special_id
    script = [{special("<|call_start|>"): 100.0}, *[{byte: 100.0} for byte in b'{"name"']]
    with pytest.raises(GenerationError, match="incomplete tool call"):
        run(StubModel(tokenizer.vocab_size, 4096, script), tokenizer, len(script), tools=add_tools(calls))
    assert calls == [], "a truncated call never dispatches"


def test_a_tool_call_at_the_last_step_reports_the_missing_budget() -> None:
    tokenizer = control_tokenizer()
    calls: list[dict] = []
    special = tokenizer.special_id
    script = [{special("<|call_start|>"): 100.0}, *[{byte: 100.0} for byte in CALL],
              {special("<|call_end|>"): 100.0}]
    with pytest.raises(GenerationError, match="budget to resume"):
        run(StubModel(tokenizer.vocab_size, 4096, script), tokenizer, len(script), tools=add_tools(calls))
    assert calls == []


def test_the_call_control_is_masked_once_the_tool_budget_is_spent() -> None:
    tokenizer = control_tokenizer()
    calls: list[dict] = []
    special = tokenizer.special_id
    script = [{special("<|call_start|>"): 100.0}, *[{byte: 100.0} for byte in CALL],
              {special("<|call_end|>"): 100.0}, {ord("z"): 100.0}, {tokenizer.eos_id: 100.0}]
    produced, stats = run(StubModel(tokenizer.vocab_size, 4096, script), tokenizer, len(script),
                          tools=add_tools(calls, max_calls=1))
    assert stats["tool_calls"] == 1 and calls == [{"a": 2, "b": 3}]
    assert produced[-1] == ord("z"), "generation resumes as ordinary text after the one call"
    steps = [{special("<|call_start|>"): 100.0}, *[{byte: 100.0} for byte in CALL],
             {special("<|call_end|>"): 100.0}, {special("<|call_start|>"): 100.0}, {tokenizer.eos_id: 100.0}]
    produced, stats = run(StubModel(tokenizer.vocab_size, 4096, steps), tokenizer, len(steps) + 4,
                          tools=add_tools([], max_calls=1))
    assert stats["tool_calls"] == 1, "the second call cannot open once the budget is spent"
    assert produced[-2] != special("<|call_start|>"), "call_start is masked out of the vocabulary"
    assert stats["finish_reason"] == "eos"


def test_grammar_masking_overrides_a_louder_illegal_token() -> None:
    tokenizer = control_tokenizer()
    illegal, legal = ord("x"), ord("7")
    steps = [{illegal: 100.0, legal: 50.0}] * 2 + [{illegal: 100.0}]
    model = StubModel(tokenizer.vocab_size, 4096, list(steps))
    produced, stats = run(model, tokenizer, 5, constraint=RegexConstraint(rb"[0-9]{2}"))
    assert produced == [legal, legal], "the illegal token is masked before softmax"
    assert stats["constraint_complete"] is True and stats["finish_reason"] == "eos"
    free = StubModel(tokenizer.vocab_size, 4096, list(steps))
    unmasked, _ = run(free, tokenizer, 3)
    assert unmasked[0] == illegal, "without the constraint the same logits win with the illegal token"


def test_reason_by_default_seeds_only_a_formatting_control() -> None:
    tokenizer = control_tokenizer()
    thought = tokenizer.special_id("<|thought_start|>")
    end = tokenizer.special_id("<|thought_end|>")
    script = [{end: 100.0}, {tokenizer.eos_id: 100.0}]
    produced, stats = run(StubModel(tokenizer.vocab_size, 4096, script), tokenizer, 4, reason_by_default=True)
    assert produced == [thought, end]
    assert stats["generated"] == 1, "the seeded control is not a model sample"


def test_generation_validates_its_inputs_before_sampling() -> None:
    tokenizer = control_tokenizer()
    model = StubModel(tokenizer.vocab_size, 4096, [{tokenizer.eos_id: 100.0}])
    with pytest.raises(GenerationError, match="require a tokenizer"):
        generate(model, [1], 4, 0.0, 0, 1.0, tokenizer.eos_id, 0, torch.device("cpu"), torch.float32,
                 constraint=RegexConstraint(rb"a"))
    with pytest.raises(GenerationError, match="tokenized to nothing"):
        run(model, tokenizer, 4, prompt=[])
    with pytest.raises(GenerationError, match="exceeds model context"):
        run(StubModel(tokenizer.vocab_size, 4, [{1: 100.0}]), tokenizer, 4, prompt=[1, 2, 3, 4, 5])
    with pytest.raises(GenerationError, match="undecodable"):
        run(model, tokenizer, 4, prompt=[tokenizer.vocab_size + 5])
    with pytest.raises(GenerationError, match="sampling"):
        generate(model, [1], 4, -1.0, 0, 1.0, tokenizer.eos_id, 0, torch.device("cpu"), torch.float32)
    with pytest.raises(GenerationError, match="no allowed token"):
        run(StubModel(tokenizer.vocab_size, 4096, [{}]), tokenizer, 4, constraint=RegexConstraint(rb"(?!x)x"))
