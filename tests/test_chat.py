"""Tests for chat.py module.

Covers the two pure choice functions and ChatInterface with dependency injection
for high coverage without real API calls or side effects.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, Generator
from unittest.mock import MagicMock

import pytest
from xai_sdk.chat import assistant, tool_result, user
from xai_sdk.proto import chat_pb2

from python_chat.api import Models
from python_chat.chat import (
    ChatCompleter,
    ChatInterface,
    SessionRuntime,
    ToolHandler,
    get_model_for_choice,
    get_system_message_for_choice,
    message_to_dict,
)
from python_chat.context import ModelContext
from python_chat.tools import ToolResult

# --- Helpers for realistic fake streams (xai-sdk style) ---
# The chat() loop consumes Generator[tuple[Response, Chunk], ...] directly.
# Both response and chunk must expose .content (for token yields + final append) and
# .tool_calls (code collects client tools from chunks; extends from last_response).
# Use SimpleNamespace to simulate without importing heavy xai runtime types in tests.


def _make_tool_call(id: str, name: str, arguments: str = "{}") -> chat_pb2.ToolCall:
    """Create a real protobuf ToolCall (client-side) for use in fake responses/chunks.

    Real protos are required on responses because chat.py does
    assistant_message.tool_calls.extend(last_response.tool_calls) where the
    target repeated field validates that items are chat_pb2.ToolCall messages.
    """
    tc = chat_pb2.ToolCall()
    tc.id = id
    tc.type = chat_pb2.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _make_chunk(content: str | None = None, tool_calls: list[Any] | None = None) -> Any:
    """Create a chunk with .content (token) and .tool_calls (may be empty).

    For tool_calls we accept either real protos or SimpleNamespace (the loop only
    reads .type via get_tool_call_type and .function.name for the Calling metadata).
    """
    return SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
        cost_usd=0.0,
        usage=SimpleNamespace(total_tokens=0),
    )


def _make_response(content: str = "", tool_calls: list[Any] | None = None) -> Any:
    """Create a (partial/final) response carrying .content and .tool_calls.

    tool_calls here should be real chat_pb2.ToolCall when simulating tool turns
    (see _make_tool_call) so that extend onto assistant proto succeeds.
    """
    return SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
        cost_usd=0.0,
        usage=SimpleNamespace(total_tokens=0),
    )


def _make_runtime() -> SessionRuntime:
    return {
        "user_id": None,
        "access_token": None,
        "session_id": None,
        "session_mode": None,
        "selected_choice": "Question",
        "model_name": Models.QUESTIONS,
        "request_counter": 0,
        "active_tools": [],
        "chat_history": [],
        "current_image_path": None,
    }


def make_text_only_stream(text: str) -> list[tuple[Any, Any]]:
    """Simple single response stream (no tools). Yields (resp, chunk) pairs.

    Responses carry accumulating content so that token-yield snapshots and the
    post-loop last_response.content produce the expected full text in history dicts.
    """
    tokens = [text[: len(text) // 2], text[len(text) // 2 :]]
    pairs: list[tuple[Any, Any]] = []
    partial = ""
    for tok in tokens:
        partial += tok
        pairs.append((_make_response(content=partial), _make_chunk(content=tok)))
    return pairs


def make_tool_call_stream(
    tool_name: str = "today_date",
    arguments: str = "{}",
    preceding_text: str | None = None,
    tool_call_id: str = "call_123",
) -> list[tuple[Any, Any]]:
    """Stream that ends with a (client-side) tool call on the final response.

    Optionally includes preceding assistant text content. Tool calls are placed on
    *both* the response and chunk of the terminating pair because chat.py reads
    chunk.tool_calls (to decide client-side handling) and last_response.tool_calls
    (to attach to assistant history entry).
    """
    pairs: list[tuple[Any, Any]] = []
    tc = _make_tool_call(tool_call_id, tool_name, arguments)
    if preceding_text:
        # Colocate the preceding text + tool_calls on the *same* (final) response/chunk pair.
        # This matches real xai-sdk stream behavior (last Response carries full .content
        # *and* .tool_calls). It ensures the post-stream append uses last_response.content
        # containing the text, so it persists in self.chat_history (the test asserts this).
        pairs.append(
            (
                _make_response(content=preceding_text, tool_calls=[tc]),
                _make_chunk(content=preceding_text, tool_calls=[tc]),
            )
        )
    else:
        pairs.append(
            (
                _make_response(content="", tool_calls=[tc]),
                _make_chunk(content=None, tool_calls=[tc]),
            )
        )
    return pairs


def make_multi_tool_call_stream() -> list[tuple[Any, Any]]:
    """Stream with two parallel client-side tool calls (on final response)."""
    tc0 = _make_tool_call("call_a", "today_date", "{}")
    tc1 = _make_tool_call("call_b", "other", '{"x":1}')
    tcs = [tc0, tc1]
    return [
        (
            _make_response(content="", tool_calls=tcs),
            _make_chunk(content=None, tool_calls=tcs),
        )
    ]


def create_responder(
    streams: list[list[tuple[Any, Any]]],
    completer_calls: (
        list[tuple[str, Sequence[chat_pb2.Message], list[Any]]] | None
    ) = None,
) -> ChatCompleter:
    """Stateful fake completer that yields successive (response, chunk) streams.

    Returns something assignable to ChatCompleter (model, Sequence[chat_pb2.Message], tools).
    The messages arg is ignored (fakes are stateful by call order).
    """
    call_idx = 0

    def completer(
        model: str, messages: Sequence[chat_pb2.Message], tools: list[Any]
    ) -> Generator[tuple[Any, Any], None, None]:
        nonlocal call_idx
        if completer_calls is not None:
            completer_calls.append((model, messages, tools))
        stream = streams[min(call_idx, len(streams) - 1)]
        call_idx += 1
        for pair in stream:
            yield pair

    return completer


def make_mock_model_context(**overrides: Any) -> MagicMock:
    """Build a MagicMock spec'd to the ModelContext interface with sane defaults.

    Using `spec=ModelContext` ensures the double only exposes attributes/methods
    that actually exist on `ModelContext`, catching drift if the real class changes.
    Pass keyword overrides to customize attributes/return values per test.
    """
    ctx = MagicMock(spec=ModelContext)
    ctx.model_name = Models.QUESTIONS
    ctx.image_path = "/tmp/images"
    ctx.user_id = None
    ctx.session_id = None
    ctx.persistence_client = None
    ctx.log_queue = None
    ctx.tools = None
    ctx.get_tools_for_model.return_value = []
    ctx.ensure_session.return_value = None
    ctx.ensure_session_for.return_value = (1, None)
    ctx.enqueue_log_event.return_value = None
    ctx.enqueue_log_event_for.return_value = None
    for key, value in overrides.items():
        setattr(ctx, key, value)
    return ctx


@pytest.fixture
def mock_context() -> MagicMock:
    """ModelContext double conforming to the ModelContext interface for chat()."""
    return make_mock_model_context()


# --- Tests for pure functions ---


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        ("Question", Models.QUESTIONS),
        ("Complex Question", Models.COMPLEX_QUESTIONS),
        ("Generate Image", Models.QUESTIONS),
        ("Unknown Mode", Models.QUESTIONS),
    ],
)
def test_get_model_for_choice_returns_correct_model_for_each_choice(
    choice: str, expected: str
) -> None:
    """get_model_for_choice maps UI choice to the right model constant."""
    assert get_model_for_choice(choice) == expected


@pytest.mark.parametrize(
    ("choice", "is_image_prompt"),
    [
        ("Generate Image", True),
        ("Question", False),
        ("Complex Question", False),
    ],
)
def test_get_system_message_for_choice_returns_image_prompt_when_generate_image_else_default(
    choice: str, is_image_prompt: bool
) -> None:
    """Specialized prompt only for image mode; everything else uses the default SYSTEM_MESSAGE."""
    msg = get_system_message_for_choice(choice)
    if is_image_prompt:
        assert "generate images" in msg.lower()
        assert "content guidelines" in msg.lower()
    else:
        from python_chat.chat import SYSTEM_MESSAGE

        assert msg == SYSTEM_MESSAGE


# --- Tests for ChatInterface ---


def test_get_history_returns_serialized_messages_without_tool_entries(
    mock_context: MagicMock,
) -> None:
    """get_history converts proto messages to UI dicts and strips internal tool entries."""
    iface = ChatInterface(mock_context)
    history = [
        user("hello"),
        assistant("hi"),
        tool_result("tool output", tool_call_id="call_1"),
    ]

    result = iface.get_history(history)

    assert result == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_get_history_includes_message_and_metadata_flags(
    mock_context: MagicMock,
) -> None:
    """get_history adds the live message plus the pending/thinking metadata branches."""
    iface = ChatInterface(mock_context)
    history = [user("hello")]

    result = iface.get_history(
        history,
        message=assistant("current reply"),
        calling_function="weather",
        thinking=True,
    )

    assert result[0] == {"role": "user", "content": "hello"}
    assert result[1] == {"role": "assistant", "content": "current reply"}
    assert result[2] == {
        "role": "assistant",
        "content": "",
        "metadata": {"title": "Calling weather...", "status": "pending"},
    }
    assert result[3] == {
        "role": "assistant",
        "content": "",
        "metadata": {"title": "Thinking"},
    }


def test_chat_interface_init_sets_history_and_accepts_injections(
    mock_context: MagicMock,
) -> None:
    """Constructor initializes state and allows overriding completer/tool_handler for testability."""
    custom_completer: ChatCompleter = MagicMock()
    custom_handler: ToolHandler = MagicMock()

    iface = ChatInterface(
        mock_context, completer=custom_completer, tool_handler=custom_handler
    )

    assert iface.modelContext is mock_context
    assert iface._completer is custom_completer
    assert iface._tool_handler is custom_handler


def test_chat_does_nothing_and_returns_history_when_message_empty(
    mock_context: MagicMock,
) -> None:
    """Early return path when no user message is provided."""
    iface = ChatInterface(mock_context)
    runtime = _make_runtime()
    prior_history = [user("prior")]
    runtime["chat_history"] = prior_history

    result = list(iface.chat("", "Question", runtime=runtime))

    chat_result = result[0]
    returned_runtime = chat_result["runtime"]

    assert chat_result["history"] == [{"role": "user", "content": "prior"}]
    assert chat_result["image_path"] is None
    # Runtime is returned unchanged; history was never appended to.
    assert returned_runtime["chat_history"] == prior_history


def test_chat_streams_single_simple_response_and_updates_history(
    mock_context: MagicMock,
) -> None:
    """Happy path: one user turn, model streams text tokens, history and yields are correct."""
    chunks = make_text_only_stream("Hello there, how can I help?")
    responder = create_responder([chunks])

    iface = ChatInterface(mock_context, completer=responder)

    runtime = _make_runtime()
    yields = list(iface.chat("Hi", "Question", runtime=runtime))

    # First yield: user appended + thinking placeholder assistant entry
    assert len(yields) >= 2
    first_history = yields[0]["history"]
    assert first_history[-2]["role"] == "user"
    thinking = first_history[-1]
    assert thinking["role"] == "assistant"
    assert "Thinking" in thinking.get("metadata", {}).get("title", "")

    # Last yield should have the full assistant response
    final_history = yields[-1]["history"]
    final_img = yields[-1]["image_path"]
    assert final_img is None
    assistant_msgs = [h for h in final_history if h["role"] == "assistant"]
    assert len(assistant_msgs) == 1
    assert "Hello there" in assistant_msgs[0]["content"]

    # Internal history should match (user + assistant, no tool entries)
    final_runtime = yields[-1]["runtime"]
    chat_history = final_runtime["chat_history"]
    assert len(chat_history) == 2
    assert message_to_dict(chat_history[0])["role"] == "user"
    assert message_to_dict(chat_history[1])["role"] == "assistant"


def test_chat_multiple_turns_accumulates_history_correctly(
    mock_context: MagicMock,
) -> None:
    """Second chat() call continues from previous history (multi-turn conversation)."""
    chunks1 = make_text_only_stream("First answer.")
    chunks2 = make_text_only_stream("Second answer following up.")

    completer_calls: list[tuple[str, Sequence[chat_pb2.Message], list[Any]]] = []

    iface = ChatInterface(
        mock_context,
        completer=create_responder([chunks1, chunks2], completer_calls=completer_calls),
    )
    runtime = _make_runtime()

    yields = list(iface.chat("First question", "Question", runtime=runtime))
    output_runtime = yields[-1]["runtime"]
    yields2 = list(iface.chat("Follow up?", "Question", runtime=output_runtime))

    # After two turns we should have 4 entries in internal history
    final_runtime = yields2[-1]["runtime"]
    chat_history = final_runtime["chat_history"]
    assert len(chat_history) == 4
    roles = [message_to_dict(e)["role"] for e in chat_history]
    assert roles == ["user", "assistant", "user", "assistant"]

    # The last yield of second turn contains the latest assistant message
    last_hist = yields2[-1]["history"]
    assert "Second answer" in last_hist[-1]["content"]

    assert len(completer_calls) == 2
    messages1 = completer_calls[0][1]
    assert len(messages1) == 2
    assert messages1[0].role == chat_pb2.MessageRole.ROLE_SYSTEM
    assert messages1[1].role == chat_pb2.MessageRole.ROLE_USER

    messages2 = completer_calls[1][1]
    assert len(messages2) == 4
    assert messages2[0].role == chat_pb2.MessageRole.ROLE_SYSTEM
    assert messages2[1].role == chat_pb2.MessageRole.ROLE_USER
    assert messages2[2].role == chat_pb2.MessageRole.ROLE_ASSISTANT
    assert messages2[3].role == chat_pb2.MessageRole.ROLE_USER


def test_chat_handles_tool_call_and_continues_for_non_image_tool(
    mock_context: MagicMock,
) -> None:
    """Tool call path: stream yields content then final response carries tool_calls; handler invoked, loop continues for final answer."""
    tool_stream = make_tool_call_stream()
    final_stream = make_text_only_stream("Today is 2025-09-18.")

    tool_res = ToolResult(
        content_for_model="2025-09-18",
        content="2025-09-18",
        tool_call_id="call_123",
    )
    handler: ToolHandler = MagicMock(return_value=[tool_res])

    iface = ChatInterface(
        mock_context,
        completer=create_responder([tool_stream, final_stream]),
        tool_handler=handler,
    )
    runtime = _make_runtime()

    yields = list(iface.chat("What day is it?", "Question", runtime=runtime))

    handler.assert_called_once()  # type: ignore[attr-defined]
    # History: user, assistant(with tool_calls), tool(result), assistant(final)
    final_runtime = yields[-1]["runtime"]
    chat_history = final_runtime["chat_history"]
    assert len(chat_history) == 4
    assert message_to_dict(chat_history[0])["role"] == "user"
    assert message_to_dict(chat_history[1])["role"] == "assistant"
    assert len(getattr(chat_history[1], "tool_calls", [])) == 1
    assert message_to_dict(chat_history[2])["role"] == "tool"
    assert message_to_dict(chat_history[2])["content"] == "2025-09-18"
    assert message_to_dict(chat_history[3])["role"] == "assistant"
    assert "2025-09-18" in message_to_dict(chat_history[3])["content"]

    # Tool role lives only in internal history (for model context); yields contain
    # user/assistant (+ metadata asst entries for Thinking/Calling). Verify a Calling
    # metadata entry was produced for the tool turn.
    calling_found = any(
        any("Calling" in (h.get("metadata", {}) or {}).get("title", "") for h in hist)
        for hist in (h["history"] for h in yields)
    )
    assert calling_found


def test_chat_tool_call_to_generate_image_sets_image_and_stops(
    mock_context: MagicMock,
) -> None:
    """Image generation tool path: special result type sets self.image_path, yields it, and breaks without extra completion."""
    # Model "says" something then calls the (fake) image tool
    tool_call_id = "call_img"
    img_stream = make_tool_call_stream(
        tool_name="generate_image",
        preceding_text="Calling the image generation tool.",
        tool_call_id=tool_call_id,
    )

    sample_image_path = "sampleimage.png"
    tool_result = ToolResult(
        content_for_model="Image generated successfully",
        content=sample_image_path,
        tool_call_id=tool_call_id,
        content_type="image",
    )
    handler: ToolHandler = MagicMock(return_value=[tool_result])

    iface = ChatInterface(
        mock_context,
        completer=create_responder([img_stream]),
        tool_handler=handler,
    )
    runtime = _make_runtime()

    yields = list(iface.chat("Draw a red square", "Generate Image", runtime=runtime))

    handler.assert_called_once()  # type: ignore[attr-defined]
    # Image must be populated
    final_runtime = yields[-1]["runtime"]
    assert final_runtime["current_image_path"] == sample_image_path

    # Final yield must carry the image
    final_img = yields[-1]["image_path"]
    assert final_img is final_runtime["current_image_path"]
    # The preceding text from model should be present (check internal history after consumption
    # as it is mutated by appends that happen after the tool_result yield snapshot).
    assert any(
        "Calling the image generation tool" in (message_to_dict(h).get("content") or "")
        for h in final_runtime["chat_history"]
        if message_to_dict(h).get("role") == "assistant"
    )
    # Tool entry present (in final internal state after append/extend that occur after the last yield)
    assert any(
        message_to_dict(h).get("role") == "tool" for h in final_runtime["chat_history"]
    )


def test_chat_catches_exception_and_yields_generic_error(
    mock_context: MagicMock,
) -> None:
    """Exception path in the main chat loop produces a friendly error message."""

    def exploding_completer(
        model: str, messages: Sequence[chat_pb2.Message], tools: list[Any]
    ) -> Generator[tuple[Any, Any], None, None]:
        raise RuntimeError("boom from test")

    iface = ChatInterface(mock_context, completer=exploding_completer)
    runtime = _make_runtime()

    yields = list(iface.chat("Trigger error", "Question", runtime=runtime))

    final_runtime = yields[-1]["runtime"]
    assert len(final_runtime["chat_history"]) == 2
    assert (
        "error processing your request"
        in message_to_dict(final_runtime["chat_history"][-1])["content"].lower()
    )

    # Last yield contains the error
    last_hist = yields[-1]
    assert "error" in last_hist["history"][-1]["content"].lower()


def test_chat_handles_multiple_parallel_tool_calls(
    mock_context: MagicMock,
) -> None:
    """Chat loop correctly surfaces multiple parallel client-side tool calls to the handler.

    (Replaces prior _collect_stream delta accumulation test; xai-sdk provides complete
    tool_calls on the final Response, which chat() normalizes and passes through.)
    """
    multi_stream = make_multi_tool_call_stream()

    # Handler will be called with list[ToolCall] (raw protos from last response)
    handler: ToolHandler = MagicMock(
        return_value=[
            ToolResult(content_for_model="a", content="a", tool_call_id="call_a"),
            ToolResult(content_for_model="b", content="b", tool_call_id="call_b"),
        ]
    )

    # Provide a second stream so the non-image tool path's "continue" has a terminating text response
    finisher = make_text_only_stream("Done with tools.")
    iface = ChatInterface(
        mock_context,
        completer=create_responder([multi_stream, finisher]),
        tool_handler=handler,
    )
    runtime = _make_runtime()

    _ = list(iface.chat("Use two tools", "Question", runtime=runtime))

    handler.assert_called_once()  # type: ignore[attr-defined]
    # Handler now receives (active_tools, tool_calls).
    call_arg = handler.call_args[0][1]  # type: ignore[attr-defined]
    assert len(call_arg) == 2
    assert call_arg[0].id == "call_a"
    assert call_arg[1].function.arguments == '{"x":1}'


def test_chat_tool_result_without_image_mode_does_not_set_image(
    mock_context: MagicMock,
) -> None:
    """When a ToolResult (even image type) is returned in non-image mode, image is not set (the 'and is_image_mode' guard)."""
    tool_stream = make_tool_call_stream()
    weird_result = ToolResult(
        content_for_model="should not become image",
        content=b"not real png",
        tool_call_id="c",
        content_type="image",
    )
    handler: ToolHandler = MagicMock(return_value=[weird_result])

    # Second stream terminates the while-loop after the tool "continue" (non-image mode)
    finisher = make_text_only_stream("Acknowledged tool result.")
    iface = ChatInterface(
        mock_context,
        completer=create_responder([tool_stream, finisher]),
        tool_handler=handler,
    )
    runtime = _make_runtime()

    yields = list(iface.chat("Tool but not image mode", "Question", runtime=runtime))

    # Guard prevented image assignment
    final_runtime = yields[-1]["runtime"]
    assert final_runtime["current_image_path"] is None


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
    assert (
        iface._persist_message(
            session_id=None,
            role="user",
            content="x",
            access_token=None,
        )
        is None
    )

    model_context.persistence_client = None
    assert (
        iface._persist_message(
            session_id=1,
            role="user",
            content="x",
            access_token=None,
        )
        is None
    )

    model_context.persistence_client = SimpleNamespace(enabled=False)
    assert (
        iface._persist_message(
            session_id=1,
            role="user",
            content="x",
            access_token=None,
        )
        is None
    )

    # Success
    rpc_client = object()
    model_context.persistence_client = SimpleNamespace(
        enabled=True,
        rpc_client=rpc_client,
        rpc_client_for_role=lambda *_args, **_kwargs: rpc_client,
    )
    create_chat_message = MagicMock(return_value=123)
    monkeypatch.setattr("python_chat.chat.db.create_chat_message", create_chat_message)

    assert (
        iface._persist_message(
            session_id=1,
            role="assistant",
            content="ok",
            access_token="tok",
        )
        == 123
    )

    # Exception fallback
    create_chat_message.side_effect = RuntimeError("db fail")
    assert (
        iface._persist_message(
            session_id=1,
            role="assistant",
            content="ok",
            access_token="tok",
        )
        is None
    )


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
        access_token=None,
    )

    model_context.persistence_client = SimpleNamespace(
        enabled=True,
        rpc_client=object(),
        rpc_client_for_role=lambda *_args, **_kwargs: object(),
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
        access_token="tok",
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
    runtime = _make_runtime()
    spy_persist = MagicMock()
    iface._persist_tool_call = spy_persist  # type: ignore[method-assign]

    _ = list(iface.chat("x", "Question", runtime=runtime))

    assert spy_persist.called
    assert spy_persist.call_args.kwargs["input_args"] == {"raw": "{bad-json"}


def test_chat_image_mode_no_image_path_does_not_crash(
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
                content=None,
                content_type="image",
                tool_call_id="call_img",
            )
        ]
    )

    # test with no image path
    iface = ChatInterface(mock_context, completer=completer, tool_handler=handler)
    runtime = _make_runtime()

    turns = list(iface.chat("draw", "Generate Image", runtime=runtime))
    assert turns
    assert turns[-1]["runtime"]["current_image_path"] is None
    assert call_count == 2

    # test with previous image path
    call_count = 0
    iface = ChatInterface(mock_context, completer=completer, tool_handler=handler)
    runtime = _make_runtime()
    runtime["current_image_path"] = "previous_image.png"

    turns = list(iface.chat("draw", "Generate Image", runtime=runtime))
    assert turns
    assert turns[-1]["runtime"]["current_image_path"] == "previous_image.png"
    assert call_count == 2


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
