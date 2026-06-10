"""Tests for tools.py module.

Covers ToolResult dataclass, Tools registry, today_date, and the helper
functions used for schema generation. All paths exercised with pure tests
and mocks only where necessary for error simulation.

Note: get_tools_for_model now returns xai_sdk.chat.tool protos (not OpenAI dict wrappers).
"""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from python_chat.tools import (
    ToolResult,
    Tools,
    get_description,
    get_property_type,
    today_date,
)
from xai_sdk.proto import chat_pb2

# --- Tests for today_date ---


def test_today_date_returns_yyyy_mm_dd_string() -> None:
    """today_date returns a string in YYYY-MM-DD format for current date."""
    result = today_date()
    assert isinstance(result, str)
    # Validate format roughly (exact date changes daily)
    assert len(result) == 10
    assert result[4] == "-" and result[7] == "-"
    # Should be parseable
    datetime.strptime(result, "%Y-%m-%d")


# --- Tests for ToolResult ---


@pytest.mark.parametrize(
    ("content", "expected_preview_len"),
    [
        ("short", 5),
        ("x" * 150, 103),  # 100 + "..."
    ],
)
def test_tool_result_str_and_repr_truncate_long_content(
    content: str, expected_preview_len: int
) -> None:
    """__str__ and __repr__ truncate content preview to ~100 chars + ellipsis."""
    tr = ToolResult(
        content_for_model="model sees this",
        content=content,
        content_type="text",
        tool_call_id="tool_1",
    )

    s = str(tr)
    r = repr(tr)

    assert "ToolResult:" in s
    assert "Model Content: model sees this" in s
    assert "Type: text" in s
    assert "..." in s or len(content) <= 100

    assert "ToolResult(content_for_model=" in r
    assert "content_type='text'" in r
    # preview in repr is truncated
    assert len([line for line in r.splitlines()]) == 1  # single line


def test_tool_result_str_repr_with_image_bytes() -> None:
    """Special case for binary content (e.g. images) still produces safe previews in __str__ (attrs provides __repr__)."""
    tr = ToolResult(
        content_for_model="image generated",
        content=b"\x89PNG\r\n" + b"x" * 200,
        content_type="image",
        tool_call_id="tool_1",
    )
    s = str(tr)
    r = repr(tr)
    assert "Type: image" in s
    assert "Content Preview:" in s
    assert "..." in s
    # attrs-generated __repr__ includes the raw fields (no custom preview logic)
    assert "content=" in r and "content_type='image'" in r


# --- Tests for helper functions (public) ---


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        (str, "string"),
        (int, "integer"),
        (float, "number"),
        (bool, "boolean"),
        (list, "array"),
        (dict, "object"),
        ("str", "string"),
        ("int", "integer"),
        (type(None), "NoneType"),  # edge, though not typical
    ],
)
def test_get_property_type_maps_annotations(annotation: Any, expected: str) -> None:
    """get_property_type returns JSON Schema type names (string/integer/number/boolean/array/object)."""
    assert get_property_type(annotation) == expected


def test_get_property_type_empty_annotation_defaults() -> None:
    """Empty annotation falls back to string."""
    import inspect

    assert get_property_type(inspect.Parameter.empty) == "string"


@pytest.mark.parametrize(
    ("docstring", "param_name", "expected"),
    [
        ("foo: the first param\nbar: second", "foo", "the first param"),
        ("  baz :   with spaces  ", "baz", "with spaces"),
        ("no colon here", "missing", ""),
        (None, "x", ""),
        ("alpha: one\nbeta: two", "gamma", ""),
    ],
)
def test_get_description_extracts_from_docstring(
    docstring: str | None, param_name: str, expected: str
) -> None:
    """Parses 'name: desc' lines from Google/NumPy style docstrings (first match)."""
    assert get_description(docstring, param_name) == expected


# --- Tests for Tools class ---


def test_tools_init_starts_empty() -> None:
    """New Tools registry has no tools."""
    t = Tools()
    assert t.tools == {}


def test_tools_register_tool_populates_registry_and_schema() -> None:
    """register_tool stores func + builds JSON schema (for xai tool protos) from signature+docstring."""

    def sample_tool(query: str, count: int = 5) -> str:
        """Search for things.

        query: the search query string
        count: how many results
        """
        return "ok"

    tools = Tools()
    tools.register_tool(sample_tool, "Performs a sample search")

    assert "sample_tool" in tools.tools
    entry = tools.tools["sample_tool"]
    assert entry["function"] is sample_tool
    tool_proto = entry["tool"]
    assert tool_proto.function.name == "sample_tool"
    assert tool_proto.function.description == "Performs a sample search"

    params = json.loads(tool_proto.function.parameters)
    assert "query" in params["properties"]
    assert params["properties"]["query"]["type"] == "string"
    assert params["properties"]["query"]["description"] == "the search query string"
    assert params["properties"]["count"]["type"] == "integer"
    assert "count" not in params["required"]
    assert "query" in params["required"]
    assert params["additionalProperties"] is False


def test_tools_register_tool_no_params() -> None:
    """Tool with zero params still registers correctly (required empty)."""

    def ping() -> str:
        """Ping the service."""
        return "pong"

    tools = Tools()
    tools.register_tool(ping, "Health check")
    tool_proto = tools.tools["ping"]["tool"]
    params = json.loads(tool_proto.function.parameters)
    assert params["properties"] == {}
    assert params["required"] == []


def test_tools_remove_tool_removes_if_present() -> None:
    """remove_tool deletes by name; idempotent if absent."""

    def dummy() -> None:
        pass

    tools = Tools()
    tools.register_tool(dummy, "d")
    assert "dummy" in tools.tools
    tools.remove_tool(dummy)
    assert "dummy" not in tools.tools
    # no error on second remove
    tools.remove_tool(dummy)


def test_tools_get_tools_for_model_returns_xai_tool_protos() -> None:
    """get_tools_for_model returns xai_sdk.chat.tool(...) protos (plus any additional server tools)."""

    def echo(msg: str) -> str:
        return msg

    tools = Tools()
    tools.register_tool(echo, "echoes input")
    formatted = tools.get_tools_for_model()
    assert len(formatted) == 1
    # The returned objects are the xai tool protos; they expose .name
    tool0 = formatted[0]
    # xai tool proto nests the function definition; .function.name holds the tool name
    assert getattr(getattr(tool0, "function", None), "name", None) == "echo"


def test_tools_handle_tool_calls_executes_and_returns_responses() -> None:
    """Happy path: parses args, calls func, builds tool response messages, collects results."""

    def add(a: int, b: int = 0) -> int:
        """Add two numbers.

        a: first
        b: second
        """
        return a + b

    def returns_tool_result(x: str) -> ToolResult:
        return ToolResult(
            content_for_model="special",
            content=x,
            content_type="text",
            tool_call_id="call_2",
        )

    tools = Tools()
    tools.register_tool(add, "adds")
    tools.register_tool(returns_tool_result, "returns special")

    # Build fake message like the one from xai stream (post-collect tool_calls on response)
    tool_calls = [
        chat_pb2.ToolCall(
            id="call_1",
            function=chat_pb2.FunctionCall(
                name="add",
                arguments='{"a": 2, "b": 3}',
            ),
        ),
        chat_pb2.ToolCall(
            id="call_2",
            function=chat_pb2.FunctionCall(
                name="returns_tool_result",
                arguments='{"x": "hi"}',
            ),
        ),
    ]

    results = tools.handle_tool_calls(tool_calls)

    assert len(results) == 2
    assert results[0].tool_call_id == "call_1"
    assert results[0].content_for_model == "5"
    assert results[0].content == 5
    assert results[0].content_type == "text"

    assert isinstance(results[1], ToolResult)
    assert results[1].content == "hi"
    assert results[1].tool_call_id == "call_2"


def test_tools_handle_tool_calls_honors_max_turns() -> None:
    """Ensure that when a tool call is registered with max_turns, it is only called that many times."""

    tool_function = MagicMock(return_value="ok")

    tools = Tools()
    tools.register_tool(tool_function, name="tool_function", max_turns=1)

    # Build fake message like the one from xai stream (post-collect tool_calls on response)
    tool_calls = [
        chat_pb2.ToolCall(
            id="call_1",
            function=chat_pb2.FunctionCall(
                name="tool_function",
                arguments="{}",
            ),
        ),
        chat_pb2.ToolCall(
            id="call_2",
            function=chat_pb2.FunctionCall(
                name="tool_function",
                arguments="{}",
            ),
        ),
    ]

    results = tools.handle_tool_calls(tool_calls)

    assert tool_function.call_count == 1
    assert len(results) == 1
    assert results[0].tool_call_id == "call_1"
    assert results[0].content_for_model == "ok"
    assert results[0].content == "ok"
    assert results[0].content_type == "text"


def test_tools_handle_tool_calls_handles_missing_tool_and_exceptions() -> None:
    """Covers unknown tool name (skipped? wait no, if not in dict) and exception during call."""

    def boom(x: str) -> str:
        raise ValueError("intentional")

    tools = Tools()
    tools.register_tool(boom, "will fail")

    tool_calls = [
        chat_pb2.ToolCall(
            id="call_bad",
            function=chat_pb2.FunctionCall(name="boom", arguments='{"x": "1"}'),
        ),
        chat_pb2.ToolCall(
            id="call_unknown",
            function=chat_pb2.FunctionCall(name="ghost", arguments="{}"),
        ),
    ]

    with patch("python_chat.tools.print"):
        results = tools.handle_tool_calls(tool_calls)

    assert len(results) == 1
    assert "Error executing tool boom" in results[0].content_for_model
    assert results[0].content is None


def test_tools_handle_tool_calls_with_no_tool_calls() -> None:
    """Empty tool_calls list returns empty list."""
    tools = Tools()
    results = tools.handle_tool_calls([])
    assert results == []


def test_tools_return_additional_tools() -> None:
    """get_tools_for_model can include additional tools beyond the registered ones."""
    tools = Tools()

    def dummy() -> None:
        pass

    tools.register_tool(dummy, "does nothing")

    additional = [
        {"type": "extra_tool"},
    ]

    formatted = tools.get_tools_for_model(additional_tools=additional)
    assert len(formatted) == 2
    # Client-side tools are returned as xai_sdk.chat.tool protos (not dicts)
    assert hasattr(formatted[0], "function")
    assert formatted[0].function.name == "dummy"
    # Additional tools (e.g. server-side like web_search) are passed through as-is
    assert formatted[1] == {"type": "extra_tool"}
