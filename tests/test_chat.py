"""Tests for chat.py module.

Covers the two pure choice functions and ChatInterface with dependency injection
for high coverage without real API calls or side effects.
"""

from __future__ import annotations

from collections.abc import Sequence
from io import BytesIO
from types import SimpleNamespace
from typing import Any, Generator
from unittest.mock import MagicMock

import pytest
from PIL import Image as PILImage
from xai_sdk.chat import user
from xai_sdk.proto import chat_pb2

from python_chat.api import Models
from python_chat.chat import (
    ChatCompleter,
    ChatInterface,
    ToolHandler,
    get_model_for_choice,
    get_system_message_for_choice,
    message_to_dict,
)
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
) -> list[tuple[Any, Any]]:
    """Stream that ends with a (client-side) tool call on the final response.

    Optionally includes preceding assistant text content. Tool calls are placed on
    *both* the response and chunk of the terminating pair because chat.py reads
    chunk.tool_calls (to decide client-side handling) and last_response.tool_calls
    (to attach to assistant history entry).
    """
    pairs: list[tuple[Any, Any]] = []
    tc = _make_tool_call("call_123", tool_name, arguments)
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
        stream = streams[min(call_idx, len(streams) - 1)]
        call_idx += 1
        for pair in stream:
            yield pair

    return completer


@pytest.fixture
def mock_context() -> MagicMock:
    """ModelContext double that only needs get_tools_for_model for chat()."""
    ctx = MagicMock()
    ctx.get_tools_for_model.return_value = []
    ctx.ensure_session.return_value = None
    ctx.persistence_client = None
    ctx.enqueue_log_event.return_value = None
    return ctx


@pytest.fixture
def sample_png_bytes() -> bytes:
    """Small valid PNG for image tool result tests."""
    buf = BytesIO()
    PILImage.new("RGB", (4, 4), color="red").save(buf, format="PNG")
    return buf.getvalue()


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


def test_chat_interface_init_sets_history_and_accepts_injections(
    mock_context: MagicMock,
) -> None:
    """Constructor initializes state and allows overriding completer/tool_handler for testability."""
    custom_completer: ChatCompleter = MagicMock()
    custom_handler: ToolHandler = MagicMock()

    iface = ChatInterface(
        mock_context, completer=custom_completer, tool_handler=custom_handler
    )

    assert iface.chat_history == []
    assert iface.image is None
    assert iface.modelContext is mock_context
    assert iface._completer is custom_completer
    assert iface._tool_handler is custom_handler


def test_chat_does_nothing_and_returns_history_when_message_empty(
    mock_context: MagicMock,
) -> None:
    """Early return path when no user message is provided."""
    iface = ChatInterface(mock_context)
    initial_history: list[dict[str, Any]] = [{"role": "user", "content": "prior"}]

    result = list(iface.chat("", initial_history, "Question"))

    # The early "return foo, bar" in a generator does not yield; list() gets [].
    # (The production caller in app.py guards the empty case before ever calling chat().)
    assert result[0][0] == initial_history
    assert result[0][1] is None
    assert iface.chat_history == []  # unchanged because early return before append


def test_chat_streams_single_simple_response_and_updates_history(
    mock_context: MagicMock,
) -> None:
    """Happy path: one user turn, model streams text tokens, history and yields are correct."""
    chunks = make_text_only_stream("Hello there, how can I help?")
    responder = create_responder([chunks])

    iface = ChatInterface(mock_context, completer=responder)

    yields = list(iface.chat("Hi", [], "Question"))

    # First yield: user appended + thinking placeholder assistant entry
    assert len(yields) >= 2
    first_history, _, _ = yields[0]
    assert first_history[-2]["role"] == "user"
    thinking = first_history[-1]
    assert thinking["role"] == "assistant"
    assert "Thinking" in thinking.get("metadata", {}).get("title", "")

    # Last yield should have the full assistant response
    final_history, final_img, _ = yields[-1]
    assert final_img is None
    assistant_msgs = [h for h in final_history if h["role"] == "assistant"]
    assert len(assistant_msgs) == 1
    assert "Hello there" in assistant_msgs[0]["content"]

    # Internal history should match (user + assistant, no tool entries)
    assert len(iface.chat_history) == 2
    assert message_to_dict(iface.chat_history[0])["role"] == "user"
    assert message_to_dict(iface.chat_history[1])["role"] == "assistant"


def test_chat_multiple_turns_accumulates_history_correctly(
    mock_context: MagicMock,
) -> None:
    """Second chat() call continues from previous history (multi-turn conversation)."""
    chunks1 = make_text_only_stream("First answer.")
    chunks2 = make_text_only_stream("Second answer following up.")

    iface = ChatInterface(mock_context, completer=create_responder([chunks1, chunks2]))

    _ = list(iface.chat("First question", [], "Question"))
    yields2 = list(iface.chat("Follow up?", [], "Question"))

    # After two turns we should have 4 entries in internal history
    assert len(iface.chat_history) == 4
    roles = [message_to_dict(e)["role"] for e in iface.chat_history]
    assert roles == ["user", "assistant", "user", "assistant"]

    # The last yield of second turn contains the latest assistant message
    last_hist, _, _ = yields2[-1]
    assert "Second answer" in last_hist[-1]["content"]


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

    yields = list(iface.chat("What day is it?", [], "Question"))

    handler.assert_called_once()  # type: ignore[attr-defined]
    # History: user, assistant(with tool_calls), tool(result), assistant(final)
    assert len(iface.chat_history) == 4
    assert message_to_dict(iface.chat_history[0])["role"] == "user"
    assert message_to_dict(iface.chat_history[1])["role"] == "assistant"
    assert len(getattr(iface.chat_history[1], "tool_calls", [])) == 1
    assert message_to_dict(iface.chat_history[2])["role"] == "tool"
    assert message_to_dict(iface.chat_history[2])["content"] == "2025-09-18"
    assert message_to_dict(iface.chat_history[3])["role"] == "assistant"
    assert "2025-09-18" in message_to_dict(iface.chat_history[3])["content"]

    # Tool role lives only in internal history (for model context); yields contain
    # user/assistant (+ metadata asst entries for Thinking/Calling). Verify a Calling
    # metadata entry was produced for the tool turn.
    calling_found = any(
        any("Calling" in (h.get("metadata", {}) or {}).get("title", "") for h in hist)
        for hist, _, _ in yields
    )
    assert calling_found


def test_chat_tool_call_to_generate_image_sets_image_and_stops(
    mock_context: MagicMock, sample_png_bytes: bytes
) -> None:
    """Image generation tool path: special result type sets self.image, yields it, and breaks without extra completion."""
    # Model "says" something then calls the (fake) image tool
    img_stream = make_tool_call_stream(
        tool_name="generate_image",
        preceding_text="Calling the image generation tool.",
    )

    tool_result = ToolResult(
        content_for_model="Image generated successfully",
        content=sample_png_bytes,
        tool_call_id="call_img",
        content_type="image",
    )
    handler: ToolHandler = MagicMock(return_value=[tool_result])

    iface = ChatInterface(
        mock_context,
        completer=create_responder([img_stream]),
        tool_handler=handler,
    )

    yields = list(iface.chat("Draw a red square", [], "Generate Image"))

    handler.assert_called_once()  # type: ignore[attr-defined]
    # Image must be populated
    assert iface.image is not None
    assert isinstance(iface.image, PILImage.Image)
    assert iface.image.size == (4, 4)

    # Final yield must carry the image
    final_hist, final_img, _ = yields[-1]
    assert final_img is iface.image
    # The preceding text from model should be present (check internal history after consumption
    # as it is mutated by appends that happen after the tool_result yield snapshot).
    assert any(
        "Calling the image generation tool" in (message_to_dict(h).get("content") or "")
        for h in iface.chat_history
        if message_to_dict(h).get("role") == "assistant"
    )
    # Tool entry present (in final internal state after append/extend that occur after the last yield)
    assert any(message_to_dict(h).get("role") == "tool" for h in iface.chat_history)


def test_chat_catches_exception_and_yields_generic_error(
    mock_context: MagicMock,
) -> None:
    """Exception path in the main chat loop produces a friendly error message."""

    def exploding_completer(
        model: str, messages: Sequence[chat_pb2.Message], tools: list[Any]
    ) -> Generator[tuple[Any, Any], None, None]:
        raise RuntimeError("boom from test")

    iface = ChatInterface(mock_context, completer=exploding_completer)

    yields = list(iface.chat("Trigger error", [], "Question"))

    assert len(iface.chat_history) == 2
    assert (
        "error processing your request"
        in message_to_dict(iface.chat_history[-1])["content"].lower()
    )

    # Last yield contains the error
    last_hist, _, _ = yields[-1]
    assert "error" in last_hist[-1]["content"].lower()


def test_clear_history_resets_chat_and_image_state(
    mock_context: MagicMock, sample_png_bytes: bytes
) -> None:
    """clear_history empties history and removes any generated image."""
    # Seed some state via a fake image-producing interaction (simplified)
    iface = ChatInterface(mock_context)
    iface.chat_history = [user("x")]
    iface.image = PILImage.open(BytesIO(sample_png_bytes))

    result = iface.clear_history()

    assert result == []
    assert iface.chat_history == []
    assert iface.image is None


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

    _ = list(iface.chat("Use two tools", [], "Question"))

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

    _ = list(iface.chat("Tool but not image mode", [], "Question"))

    # Guard prevented image assignment
    assert iface.image is None
