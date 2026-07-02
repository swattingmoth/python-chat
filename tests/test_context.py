"""Tests for context.py module.

Covers the ModelContext singleton, model switching, tool delegation,
validation, and error cases. Uses mocks for xAI Client; reset() between tests.
"""

from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from xai_sdk.proto import chat_pb2

from python_chat.api import Models
from python_chat.context import ModelContext
from python_chat.tools import Tools


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
    assert ctx.tools is Tools

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

    active_tools = ctx.register_tool([], my_tool, "does x")
    assert len(active_tools) == 1
    assert active_tools[0]["name"] == "my_tool"

    formatted = ctx.get_tools_for_model(active_tools)
    assert len(formatted) == 1
    assert formatted[0].function.name == "my_tool"

    active_tools = ctx.remove_tool(active_tools, my_tool)
    assert active_tools == []
    assert ctx.get_tools_for_model(active_tools) == []


def test_model_context_handle_tool_calls_delegates() -> None:
    """handle_tool_calls forwards to Tools and returns a list of ToolResult."""
    fake_client = MagicMock()
    ctx = ModelContext.create(fake_client, "/p")

    def echo(s: str) -> str:
        return s

    active_tools = ctx.register_tool([], echo, "echo")

    tool_calls = [
        chat_pb2.ToolCall(
            id="c1",
            function=chat_pb2.FunctionCall(
                name="echo",
                arguments='{"s": "hi"}',
            ),
        )
    ]

    results = ctx.handle_tool_calls(active_tools, tool_calls)

    assert len(results) == 1
    assert results[0].content == "hi"
    assert results[0].tool_call_id == "c1"


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


def test_model_context_complex_questions_adds_additional_tools() -> None:
    """Switching to COMPLEX_QUESTIONS model adds the native web_search server tool."""
    fake_client = MagicMock()
    ctx = ModelContext.create(fake_client, "/p")

    assert ctx.get_tools_for_model([]) == []  # no additional tools for QUESTIONS

    ctx.model_name = Models.COMPLEX_QUESTIONS
    tools = ctx.get_tools_for_model([])
    # For COMPLEX we now return both native server-side tools.
    assert len(tools) == 2
    assert any(tool.HasField("web_search") for tool in tools)
    assert any(tool.HasField("code_execution") for tool in tools)

    ctx.model_name = Models.QUESTIONS
    assert ctx.get_tools_for_model([]) == []


def test_setters_and_session_short_circuit_paths() -> None:
    ctx = ModelContext.create(MagicMock(), "c:/tmp")
    ctx.set_log_queue(None)
    ctx.set_user_id("u")
    assert ctx.user_id == "u"

    # Existing session for same mode returns cached id.
    ctx._session_id = 123
    ctx._session_mode = "Question"
    assert ctx.ensure_session("Question") == 123


def test_ensure_session_none_when_missing_dependencies() -> None:
    ctx = ModelContext.create(MagicMock(), "c:/tmp")

    ctx._persistence_client = None
    assert ctx.ensure_session("Question") is None

    ctx._persistence_client = SimpleNamespace(enabled=False)  # type: ignore[assignment]
    assert ctx.ensure_session("Question") is None

    ctx._persistence_client = SimpleNamespace(enabled=True, rpc_client=object())  # type: ignore[assignment]
    ctx._user_id = None
    assert ctx.ensure_session("Question") is None


def test_ensure_session_success_and_exception(monkeypatch: Any) -> None:
    ctx = ModelContext.create(MagicMock(), "c:/tmp")
    ctx._persistence_client = SimpleNamespace(enabled=True, rpc_client=object())  # type: ignore[assignment]
    ctx._user_id = "user-1"

    create_chat_session = MagicMock(return_value=777)
    monkeypatch.setattr(
        "python_chat.context.db.create_chat_session", create_chat_session
    )

    assert ctx.ensure_session("Question") == 777

    create_chat_session.side_effect = RuntimeError("db")
    ctx._session_id = None
    assert ctx.ensure_session("Question") is None


def test_complete_session_paths(monkeypatch: Any) -> None:
    ctx = ModelContext.create(MagicMock(), "c:/tmp")

    ctx._persistence_client = None
    ctx.complete_session()

    ctx._persistence_client = SimpleNamespace(enabled=True, rpc_client=object())  # type: ignore[assignment]
    ctx._session_id = None
    ctx.complete_session()

    complete_chat_session = MagicMock()
    monkeypatch.setattr(
        "python_chat.context.db.complete_chat_session", complete_chat_session
    )

    ctx._session_id = 9
    ctx.complete_session()
    assert complete_chat_session.called

    complete_chat_session.side_effect = RuntimeError("fail")
    ctx.complete_session()


def test_enqueue_log_event_enqueues_when_queue_present() -> None:
    enqueue = MagicMock()
    ctx = ModelContext.create(MagicMock(), "c:/tmp")
    ctx._log_queue = SimpleNamespace(enqueue=enqueue)  # type: ignore[assignment]

    ctx.enqueue_log_event({"event": "x"})
    enqueue.assert_called_once()
