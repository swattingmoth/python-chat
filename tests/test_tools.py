"""Tests for tools.py module.

Covers ToolResult dataclass, stateless Tools helpers, today_date, and helper
functions used for schema generation.
"""

from __future__ import annotations

import inspect
import json
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from python_chat.tools import (
    ToolResult,
    Tools,
    get_description,
    get_property_type,
    today_date,
)
from xai_sdk.proto import chat_pb2


def test_today_date_returns_yyyy_mm_dd_string() -> None:
    """today_date returns a string in YYYY-MM-DD format for current date."""
    result = today_date()
    assert isinstance(result, str)
    assert len(result) == 10
    assert result[4] == "-" and result[7] == "-"
    datetime.strptime(result, "%Y-%m-%d")


@pytest.mark.parametrize(
    ("content", "expected_preview_len"),
    [
        ("short", 5),
        ("x" * 150, 103),
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
    assert len([line for line in r.splitlines()]) == 1


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
        (type(None), "NoneType"),
    ],
)
def test_get_property_type_maps_annotations(annotation: Any, expected: str) -> None:
    """get_property_type returns JSON Schema type names."""
    assert get_property_type(annotation) == expected


def test_get_property_type_empty_annotation_defaults() -> None:
    """Empty annotation falls back to string."""
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
    """Parses 'name: desc' lines from docstrings."""
    assert get_description(docstring, param_name) == expected


def test_register_tool_returns_new_list_with_schema() -> None:
    """register_tool builds schema and returns a new active tool list."""

    def sample_tool(query: str, count: int = 5) -> str:
        """Search for things.

        query: the search query string
        count: how many results
        """
        return "ok"

    tools: list[Any] = []
    updated = Tools.register_tool(tools, sample_tool, "Performs a sample search")

    assert tools == []
    assert len(updated) == 1
    entry = updated[0]
    assert entry["name"] == "sample_tool"
    assert entry["function"] is sample_tool

    tool_proto = entry["tool"]
    assert tool_proto.function.name == "sample_tool"
    assert tool_proto.function.description == "Performs a sample search"

    params = json.loads(tool_proto.function.parameters)
    assert "query" in params["properties"]
    assert params["properties"]["query"]["type"] == "string"
    assert params["properties"]["query"]["description"] == "the search query string"
    assert params["properties"]["count"]["type"] == "integer"
    assert "query" in params["required"]
    assert "count" not in params["required"]


def test_register_tool_replaces_existing_name() -> None:
    """register_tool replaces an existing entry with the same name."""

    def a(msg: str) -> str:
        return msg

    def b(msg: str) -> str:
        return msg.upper()

    tools = Tools.register_tool([], a, "a")
    tools = Tools.register_tool(tools, b, "b", name="a")

    assert len(tools) == 1
    assert tools[0]["name"] == "a"
    assert tools[0]["function"] is b


def test_remove_tool_returns_new_list_without_target() -> None:
    """remove_tool removes by function name and is idempotent."""

    def one() -> str:
        return "1"

    def two() -> str:
        return "2"

    tools = Tools.register_tool([], one, "one")
    tools = Tools.register_tool(tools, two, "two")

    removed = Tools.remove_tool(tools, one)
    assert len(removed) == 1
    assert removed[0]["name"] == "two"

    again = Tools.remove_tool(removed, one)
    assert again == removed


def test_get_tools_for_model_includes_additional_tools() -> None:
    """get_tools_for_model returns xai tool protos plus server-side tools."""

    def echo(msg: str) -> str:
        return msg

    tools = Tools.register_tool([], echo, "echo")
    additional = [{"type": "extra_tool"}]

    formatted = Tools.get_tools_for_model(tools, additional_tools=additional)
    assert len(formatted) == 2
    assert formatted[0].function.name == "echo"
    assert formatted[1] == {"type": "extra_tool"}


def test_handle_tool_calls_executes_registered_tools() -> None:
    """handle_tool_calls executes calls and returns ToolResult entries."""

    def add(a: int, b: int = 0) -> int:
        return a + b

    def returns_tool_result(x: str) -> ToolResult:
        return ToolResult(
            content_for_model="special",
            content=x,
            content_type="text",
            tool_call_id="call_2",
        )

    active_tools = Tools.register_tool([], add, "adds")
    active_tools = Tools.register_tool(active_tools, returns_tool_result, "special")

    tool_calls = [
        chat_pb2.ToolCall(
            id="call_1",
            function=chat_pb2.FunctionCall(name="add", arguments='{"a": 2, "b": 3}'),
        ),
        chat_pb2.ToolCall(
            id="call_2",
            function=chat_pb2.FunctionCall(
                name="returns_tool_result", arguments='{"x": "hi"}'
            ),
        ),
    ]

    results = Tools.handle_tool_calls(active_tools, tool_calls)

    assert len(results) == 2
    assert results[0].tool_call_id == "call_1"
    assert results[0].content_for_model == "5"
    assert results[0].content == 5
    assert results[1].tool_call_id == "call_2"
    assert results[1].content == "hi"


def test_handle_tool_calls_honors_max_turns() -> None:
    """A tool registered with max_turns should be called only up to the limit."""

    tool_function = MagicMock(return_value="ok")

    active_tools = Tools.register_tool(
        [], tool_function, name="tool_function", max_turns=1
    )

    tool_calls = [
        chat_pb2.ToolCall(
            id="call_1",
            function=chat_pb2.FunctionCall(name="tool_function", arguments="{}"),
        ),
        chat_pb2.ToolCall(
            id="call_2",
            function=chat_pb2.FunctionCall(name="tool_function", arguments="{}"),
        ),
    ]

    results = Tools.handle_tool_calls(active_tools, tool_calls)

    assert tool_function.call_count == 1
    assert len(results) == 1
    assert results[0].tool_call_id == "call_1"


def test_handle_tool_calls_handles_exceptions_and_unknown_tools() -> None:
    """Unknown tools are skipped; exceptions produce an error ToolResult."""

    def boom(x: str) -> str:
        raise ValueError("intentional")

    active_tools = Tools.register_tool([], boom, "will fail")

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

    results = Tools.handle_tool_calls(active_tools, tool_calls)

    assert len(results) == 1
    assert "Error executing tool boom" in results[0].content_for_model
    assert results[0].content is None


def test_handle_tool_calls_with_no_tool_calls_returns_empty() -> None:
    """Empty tool_calls list returns empty list."""
    assert Tools.handle_tool_calls([], []) == []
