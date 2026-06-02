"""Tests for context.py module.

Covers the ModelContext singleton, model switching, tool delegation,
validation, and error cases. Uses mocks for OpenAI client; reset() between tests.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Generator
from unittest.mock import MagicMock

import pytest

from python_chat.api import Models
from python_chat.context import ModelContext
from python_chat.tools import ToolResult, Tools


@pytest.fixture(autouse=True)
def reset_context() -> Generator[None, None, None]:
    """Ensure clean singleton state before and after every context test."""
    ModelContext.reset()
    yield
    ModelContext.reset()


def test_model_context_create_and_current_returns_same_instance() -> None:
    """create initializes singleton; current returns it; repeated create returns same."""
    fake_client = MagicMock()
    ctx1 = ModelContext.create(fake_client, "/tmp/imgs")
    ctx2 = ModelContext.current()
    ctx3 = ModelContext.create(MagicMock(), "/other")  # should ignore, return existing

    assert ctx1 is ctx2 is ctx3
    assert ctx1.client is fake_client
    assert ctx1.image_path == "/tmp/imgs"


def test_model_context_current_raises_if_not_initialized() -> None:
    """current() before any create raises the documented exception."""
    with pytest.raises(Exception, match="has not been initialized"):
        ModelContext.current()


def test_model_context_properties_and_model_name_setter() -> None:
    """Direct property access and model_name setter with validation."""
    fake_client = MagicMock()
    ctx = ModelContext.create(fake_client, "/p")

    assert ctx.model_name == Models.QUESTIONS
    assert ctx.client is fake_client
    assert ctx.image_path == "/p"
    assert isinstance(ctx.tools, Tools)

    # change model
    ctx.model_name = Models.COMPLEX_QUESTIONS
    assert ctx.model_name == Models.COMPLEX_QUESTIONS

    # invalid raises
    with pytest.raises(ValueError, match="Invalid model name"):
        ctx.model_name = "grok-9000"


def test_model_context_register_remove_tool_delegates_to_tools() -> None:
    """register_tool / remove_tool / get_tools_for_model delegate to internal Tools."""
    fake_client = MagicMock()
    ctx = ModelContext.create(fake_client, "/p")

    def my_tool(x: str) -> str:
        return x

    ctx.register_tool(my_tool, "does x")
    assert "my_tool" in ctx.tools.tools

    formatted = ctx.get_tools_for_model()
    assert len(formatted) == 1
    assert formatted[0]["function"]["name"] == "my_tool"

    ctx.remove_tool(my_tool)
    assert "my_tool" not in ctx.tools.tools
    assert ctx.get_tools_for_model() == []


def test_model_context_handle_tool_calls_delegates() -> None:
    """handle_tool_calls forwards to Tools and returns its result."""
    fake_client = MagicMock()
    ctx = ModelContext.create(fake_client, "/p")

    def echo(s: str) -> str:
        return s

    ctx.register_tool(echo, "echo")

    msg = SimpleNamespace(
        tool_calls=[
            SimpleNamespace(
                id="c1", function=SimpleNamespace(name="echo", arguments='{"s": "hi"}')
            )
        ]
    )

    responses, results = ctx.handle_tool_calls(msg)
    assert responses[0]["content"] == "hi"
    assert results[0] is None


def test_model_context_use_model_temporarily_switches_and_restores() -> None:
    """Context manager switches model_name for duration, restores on exit (even on error)."""
    fake_client = MagicMock()
    ctx = ModelContext.create(fake_client, "/p")
    original = ctx.model_name

    with ctx.use_model(Models.IMAGES):
        assert ctx.model_name == Models.IMAGES

    assert ctx.model_name == original

    # exception path still restores
    try:
        with ctx.use_model(Models.COMPLEX_QUESTIONS):
            assert ctx.model_name == Models.COMPLEX_QUESTIONS
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    assert ctx.model_name == original


def test_model_context_use_model_validates_on_entry() -> None:
    """Invalid model in use_model raises before entering."""
    fake_client = MagicMock()
    ctx = ModelContext.create(fake_client, "/p")

    with pytest.raises(ValueError):
        with ctx.use_model("bad-model"):
            pass  # never reached
