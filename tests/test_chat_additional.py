from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, Generator
from unittest.mock import MagicMock

from xai_sdk.chat import assistant
from xai_sdk.proto import chat_pb2
import pytest

from python_chat.chat import ChatInterface
from python_chat.tools import ToolResult


@pytest.fixture
def mock_context() -> MagicMock:
    ctx = MagicMock()
    ctx.get_tools_for_model.return_value = []
    ctx.ensure_session.return_value = None
    ctx.persistence_client = None
    ctx.enqueue_log_event.return_value = None
    return ctx


def test_default_completer_yields_stream_pairs() -> None:
    fake_stream = [
        (
            SimpleNamespace(
                content="a",
                tool_calls=[],
                cost_usd=0.0,
                usage=SimpleNamespace(total_tokens=1),
            ),
            SimpleNamespace(content="a", tool_calls=[]),
        )
    ]

    class _FakeChat:
        def stream(self) -> Generator[tuple[Any, Any], None, None]:
            for item in fake_stream:
                yield item

    model_context = MagicMock()
    model_context.client.chat.create.return_value = _FakeChat()

    iface = ChatInterface(model_context)
    out = list(iface._default_completer("model", [], []))
    assert out == fake_stream


def test_default_tool_handler_uses_provided_active_tools() -> None:
    model_context = MagicMock()
    iface = ChatInterface(model_context)

    calls = [chat_pb2.ToolCall(id="id")]
    result = iface._default_tool_handler([], calls)

    assert result == []


def test_persist_message_handles_guard_and_success_and_exception(
    monkeypatch: Any,
) -> None:
    model_context = MagicMock()
    iface = ChatInterface(model_context)

    # Guards
    assert iface._persist_message(session_id=None, role="user", content="x") is None

    model_context.persistence_client = None
    assert iface._persist_message(session_id=1, role="user", content="x") is None

    model_context.persistence_client = SimpleNamespace(enabled=False)
    assert iface._persist_message(session_id=1, role="user", content="x") is None

    # Success
    rpc_client = object()
    model_context.persistence_client = SimpleNamespace(
        enabled=True, rpc_client=rpc_client
    )
    create_chat_message = MagicMock(return_value=123)
    monkeypatch.setattr("python_chat.chat.db.create_chat_message", create_chat_message)

    assert iface._persist_message(session_id=1, role="assistant", content="ok") == 123

    # Exception fallback
    create_chat_message.side_effect = RuntimeError("db fail")
    assert iface._persist_message(session_id=1, role="assistant", content="ok") is None


def test_persist_tool_call_guard_and_exception(monkeypatch: Any) -> None:
    model_context = MagicMock()
    iface = ChatInterface(model_context)

    iface._persist_tool_call(
        message_id=None,
        tool_name="t",
        status="ok",
        input_args=None,
        output_result=None,
        error_message=None,
        latency_ms=None,
    )

    model_context.persistence_client = SimpleNamespace(
        enabled=True, rpc_client=object()
    )
    create_tool_call = MagicMock(side_effect=RuntimeError("tool fail"))
    monkeypatch.setattr("python_chat.chat.db.create_tool_call", create_tool_call)

    iface._persist_tool_call(
        message_id=1,
        tool_name="t",
        status="ok",
        input_args={"a": 1},
        output_result={"b": 2},
        error_message=None,
        latency_ms=10,
    )

    assert create_tool_call.called


def test_chat_invalid_tool_call_json_uses_raw_args(mock_context: MagicMock) -> None:
    tc = chat_pb2.ToolCall()
    tc.id = "call_bad"
    tc.type = chat_pb2.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL
    tc.function.name = "today_date"
    tc.function.arguments = "{bad-json"

    call_count = 0

    def completer(
        model: str, messages: Sequence[chat_pb2.Message], tools: list[Any]
    ) -> Generator[tuple[Any, Any], None, None]:
        nonlocal call_count
        del model, messages, tools
        call_count += 1
        if call_count == 1:
            yield (
                SimpleNamespace(
                    content="",
                    tool_calls=[tc],
                    cost_usd=0.0,
                    usage=SimpleNamespace(total_tokens=0),
                ),
                SimpleNamespace(content=None, tool_calls=[tc]),
            )
        else:
            yield (
                SimpleNamespace(
                    content="done",
                    tool_calls=[],
                    cost_usd=0.0,
                    usage=SimpleNamespace(total_tokens=0),
                ),
                SimpleNamespace(content="done", tool_calls=[]),
            )

    handler = MagicMock(
        return_value=[
            ToolResult(content_for_model="ok", content="ok", tool_call_id="call_bad")
        ]
    )

    iface = ChatInterface(mock_context, completer=completer, tool_handler=handler)
    spy_persist = MagicMock()
    iface._persist_tool_call = spy_persist  # type: ignore[method-assign]

    _ = list(iface.chat("x", [], "Question"))

    assert spy_persist.called
    assert spy_persist.call_args.kwargs["input_args"] == {"raw": "{bad-json"}


def test_chat_image_mode_invalid_image_bytes_do_not_crash(
    mock_context: MagicMock,
) -> None:
    tc = chat_pb2.ToolCall()
    tc.id = "call_img"
    tc.type = chat_pb2.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL
    tc.function.name = "generate_image"
    tc.function.arguments = "{}"

    call_count = 0

    def completer(
        model: str, messages: Sequence[chat_pb2.Message], tools: list[Any]
    ) -> Generator[tuple[Any, Any], None, None]:
        nonlocal call_count
        del model, messages, tools
        call_count += 1
        if call_count == 1:
            yield (
                SimpleNamespace(
                    content="attempting image",
                    tool_calls=[tc],
                    cost_usd=0.0,
                    usage=SimpleNamespace(total_tokens=0),
                ),
                SimpleNamespace(content="attempting image", tool_calls=[tc]),
            )
        else:
            yield (
                SimpleNamespace(
                    content="done",
                    tool_calls=[],
                    cost_usd=0.0,
                    usage=SimpleNamespace(total_tokens=0),
                ),
                SimpleNamespace(content="done", tool_calls=[]),
            )

    handler = MagicMock(
        return_value=[
            ToolResult(
                content_for_model="img",
                content=b"not-a-real-image",
                content_type="image",
                tool_call_id="call_img",
            )
        ]
    )

    iface = ChatInterface(mock_context, completer=completer, tool_handler=handler)

    turns = list(iface.chat("draw", [], "Generate Image"))
    assert turns
    assert iface.image is None


def test_message_to_dict_includes_tool_calls() -> None:
    msg = assistant("text")
    tool_call = chat_pb2.ToolCall(id="call_1")
    msg.tool_calls.extend([tool_call])

    result = {
        "role": "assistant",
        "content": "text",
        "tool_calls": [{"id": "call_1"}],
    }

    from python_chat.chat import message_to_dict

    parsed = message_to_dict(msg)
    assert parsed["role"] == result["role"]
    assert parsed["content"] == result["content"]
    assert parsed["tool_calls"][0]["id"] == "call_1"
