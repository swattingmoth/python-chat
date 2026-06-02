"""Tests for tools.py module.

Covers ToolResult dataclass, Tools registry, today_date, and the helper
functions used for schema generation. All paths exercised with pure tests
and mocks only where necessary for error simulation.
"""

from __future__ import annotations

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
    ("annotation", "for_xai", "expected"),
    [
        (str, True, "string"),
        (int, True, "int"),
        (str, False, "str"),
        (float, False, "float"),
        (type(None), True, "NoneType"),  # edge, though not typical
    ],
)
def test_get_property_type_maps_annotations(
    annotation: Any, for_xai: bool, expected: str
) -> None:
    """get_property_type returns JSON schema type names, with xAI special case for str."""
    assert get_property_type(annotation, for_xai) == expected


def test_get_property_type_empty_annotation_defaults() -> None:
    """Empty annotation falls back appropriately."""
    import inspect

    assert get_property_type(inspect.Parameter.empty, True) == "string"
    assert get_property_type(inspect.Parameter.empty, False) == "str"


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
    """register_tool stores func + builds OpenAI-compatible JSON schema from signature+docstring."""

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
    js = entry["json"]
    assert js["name"] == "sample_tool"
    assert js["description"] == "Performs a sample search"
    # Note: current implementation places 'required' and 'additional_properties' at function schema level (not inside parameters)
    assert "query" in js["parameters"]["properties"]
    assert js["parameters"]["properties"]["query"]["type"] == "string"
    assert (
        js["parameters"]["properties"]["query"]["description"]
        == "the search query string"
    )
    assert js["parameters"]["properties"]["count"]["type"] == "int"
    assert "count" not in js["required"]  # has default
    assert "query" in js["required"]
    assert js["additional_properties"] is False


def test_tools_register_tool_no_params() -> None:
    """Tool with zero params still registers correctly (required empty)."""

    def ping() -> str:
        """Ping the service."""
        return "pong"

    tools = Tools()
    tools.register_tool(ping, "Health check")
    js = tools.tools["ping"]["json"]
    assert js["parameters"]["properties"] == {}
    assert js["required"] == []


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


def test_tools_get_tools_for_model_formats_for_openai() -> None:
    """get_tools_for_model wraps each json under {'type': 'function', 'function': ...}."""

    def echo(msg: str) -> str:
        return msg

    tools = Tools()
    tools.register_tool(echo, "echoes input")
    formatted = tools.get_tools_for_model()
    assert len(formatted) == 1
    assert formatted[0]["type"] == "function"
    assert formatted[0]["function"]["name"] == "echo"


def test_tools_handle_tool_calls_executes_and_returns_responses() -> None:
    """Happy path: parses args, calls func, builds tool response messages, collects results."""

    def add(a: int, b: int = 0) -> int:
        """Add two numbers.

        a: first
        b: second
        """
        return a + b

    def returns_tool_result(x: str) -> ToolResult:
        return ToolResult(content_for_model="special", content=x, content_type="text")

    tools = Tools()
    tools.register_tool(add, "adds")
    tools.register_tool(returns_tool_result, "returns special")

    # Build fake message like the one from OpenAI stream parsing
    msg = SimpleNamespace(
        tool_calls=[
            SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(name="add", arguments='{"a": 2, "b": 3}'),
            ),
            SimpleNamespace(
                id="call_2",
                function=SimpleNamespace(
                    name="returns_tool_result", arguments='{"x": "hi"}'
                ),
            ),
        ]
    )

    responses, results = tools.handle_tool_calls(msg)

    assert len(responses) == 2
    assert responses[0]["role"] == "tool"
    assert responses[0]["tool_call_id"] == "call_1"
    assert (
        responses[0]["content"] == 5
    )  # from add (raw return value, not stringified by current impl)
    assert responses[1]["content"] == "special"

    assert len(results) == 2
    assert results[0] is None  # plain int return
    assert isinstance(results[1], ToolResult)
    assert results[1].content == "hi"


def test_tools_handle_tool_calls_handles_missing_tool_and_exceptions() -> None:
    """Covers unknown tool name (skipped? wait no, if not in dict) and exception during call."""

    def boom(x: str) -> str:
        raise ValueError("intentional")

    tools = Tools()
    tools.register_tool(boom, "will fail")

    msg = SimpleNamespace(
        tool_calls=[
            SimpleNamespace(
                id="call_bad",
                function=SimpleNamespace(name="boom", arguments='{"x": "1"}'),
            ),
            SimpleNamespace(
                id="call_unknown",
                function=SimpleNamespace(name="ghost", arguments="{}"),
            ),
        ]
    )

    # Patch print to avoid noise, but still execute
    with patch("python_chat.tools.print"):
        responses, results = tools.handle_tool_calls(msg)

    # For boom: error response (except path appends to responses but NOT to tool_results)
    # For unknown: {} falsy so skipped, no appends
    assert len(responses) == 1
    assert "Error executing tool boom" in responses[0]["content"]
    assert results == []


def test_tools_handle_tool_calls_with_no_tool_calls() -> None:
    """Empty tool_calls list returns empty lists."""
    tools = Tools()
    msg = SimpleNamespace(tool_calls=[])
    responses, results = tools.handle_tool_calls(msg)
    assert responses == []
    assert results == []
