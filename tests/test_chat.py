"""Tests for chat.py module.

Covers the two pure choice functions and ChatInterface with dependency injection
for high coverage without real API calls or side effects.
"""

from __future__ import annotations

from collections.abc import Iterator
from io import BytesIO
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import MagicMock

import pytest
from PIL import Image as PILImage

from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from python_chat.api import Models
from python_chat.chat import (
    ChatCompleter,
    ChatInterface,
    ToolHandler,
    get_model_for_choice,
    get_system_message_for_choice,
)
from python_chat.tools import ToolResult

# --- Helpers for realistic fake streams ---


def _make_chunk(
    content: str | None = None,
    tool_calls: list[ChoiceDeltaToolCall] | None = None,
    finish_reason: str | None = None,
) -> ChatCompletionChunk:
    """Create a minimal valid ChatCompletionChunk for testing."""
    delta = ChoiceDelta(content=content, tool_calls=tool_calls)
    choice = Choice(index=0, delta=delta, finish_reason=finish_reason)  # type: ignore[arg-type]
    return ChatCompletionChunk(
        id="chunk-test",
        choices=[choice],
        created=1_700_000_000,
        model="test-model",
        object="chat.completion.chunk",
    )


def _make_tool_call_delta(
    index: int = 0,
    tool_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> ChoiceDeltaToolCall:
    """Create a tool call delta fragment (supports incremental streaming)."""
    func = None
    if name is not None or arguments is not None:
        func = ChoiceDeltaToolCallFunction(name=name, arguments=arguments)
    return ChoiceDeltaToolCall(index=index, id=tool_id, function=func)


def make_text_only_stream(text: str) -> list[ChatCompletionChunk]:
    """Simple single response stream (no tools)."""
    # Split into a couple tokens to exercise multiple yields
    tokens = [text[: len(text) // 2], text[len(text) // 2 :]]
    return [
        _make_chunk(content=tokens[0]),
        _make_chunk(content=tokens[1], finish_reason="stop"),
    ]


def make_tool_call_stream(
    tool_name: str = "today_date",
    arguments: str = "{}",
    preceding_text: str | None = None,
) -> list[ChatCompletionChunk]:
    """Stream that ends with a tool call (optionally with text first)."""
    chunks: list[ChatCompletionChunk] = []
    if preceding_text:
        chunks.append(_make_chunk(content=preceding_text))
    # Simulate split: id+name in first, arguments in second + finish
    chunks.append(
        _make_chunk(
            tool_calls=[
                _make_tool_call_delta(
                    index=0, tool_id="call_123", name=tool_name, arguments=""
                )
            ]
        )
    )
    chunks.append(
        _make_chunk(
            tool_calls=[_make_tool_call_delta(index=0, arguments=arguments)],
            finish_reason="tool_calls",
        )
    )
    return chunks


def make_multi_tool_call_stream() -> list[ChatCompletionChunk]:
    """Stream with two parallel tool calls to cover index handling."""
    return [
        _make_chunk(
            tool_calls=[
                _make_tool_call_delta(0, "call_a", "today_date", ""),
                _make_tool_call_delta(1, "call_b", "other", ""),
            ]
        ),
        _make_chunk(
            tool_calls=[
                _make_tool_call_delta(0, arguments="{}"),
                _make_tool_call_delta(1, arguments='{"x":1}'),
            ],
            finish_reason="tool_calls",
        ),
    ]


def make_error_stream() -> list[ChatCompletionChunk]:
    """A stream that will cause issues if iterated (for except path)."""
    # We trigger exception by raising inside the completer, not the stream itself.
    return []


def create_responder(
    streams: list[list[ChatCompletionChunk]],
) -> Callable[
    [str, list[dict[str, Any]], list[dict[str, Any]]], Iterator[ChatCompletionChunk]
]:
    """Stateful fake completer that returns successive streams on each call."""
    call_idx = 0

    def completer(
        model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Iterator[ChatCompletionChunk]:
        nonlocal call_idx
        stream = streams[min(call_idx, len(streams) - 1)]
        call_idx += 1
        return iter(stream)

    return completer


@pytest.fixture
def mock_context() -> MagicMock:
    """ModelContext double that only needs get_tools_for_model for chat()."""
    ctx = MagicMock()
    ctx.get_tools_for_model.return_value = []
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
    initial_history: list[dict[str, str]] = [{"role": "user", "content": "prior"}]

    result = list(iface.chat("", initial_history, "Question"))

    # The early "return foo, bar" in a generator does not yield; list() gets [].
    # (The production caller in app.py guards the empty case before ever calling chat().)
    assert result == [(initial_history, None)]
    assert iface.chat_history == []  # unchanged because early return before append


def test_chat_streams_single_simple_response_and_updates_history(
    mock_context: MagicMock,
) -> None:
    """Happy path: one user turn, model streams text tokens, history and yields are correct."""
    chunks = make_text_only_stream("Hello there, how can I help?")
    responder = create_responder([chunks])

    iface = ChatInterface(mock_context, completer=responder)  # type: ignore[arg-type]

    yields = list(iface.chat("Hi", [], "Question"))

    # First yield: user appended + thinking placeholder assistant entry
    assert len(yields) >= 2
    first_history, _ = yields[0]
    assert first_history[-2]["role"] == "user"
    thinking = first_history[-1]
    assert thinking["role"] == "assistant"
    assert "Thinking" in thinking.get("metadata", {}).get("title", "")

    # Last yield should have the full assistant response
    final_history, final_img = yields[-1]
    assert final_img is None
    assistant_msgs = [h for h in final_history if h["role"] == "assistant"]
    assert len(assistant_msgs) == 1
    assert "Hello there" in assistant_msgs[0]["content"]

    # Internal history should match (user + assistant, no tool entries)
    assert len(iface.chat_history) == 2
    assert iface.chat_history[0]["role"] == "user"
    assert iface.chat_history[1]["role"] == "assistant"


def test_chat_multiple_turns_accumulates_history_correctly(
    mock_context: MagicMock,
) -> None:
    """Second chat() call continues from previous history (multi-turn conversation)."""
    chunks1 = make_text_only_stream("First answer.")
    chunks2 = make_text_only_stream("Second answer following up.")

    iface = ChatInterface(mock_context, completer=create_responder([chunks1, chunks2]))  # type: ignore[arg-type]

    _ = list(iface.chat("First question", [], "Question"))
    yields2 = list(iface.chat("Follow up?", [], "Question"))

    # After two turns we should have 4 entries in internal history
    assert len(iface.chat_history) == 4
    roles = [e["role"] for e in iface.chat_history]
    assert roles == ["user", "assistant", "user", "assistant"]

    # The last yield of second turn contains the latest assistant message
    last_hist, _ = yields2[-1]
    assert "Second answer" in last_hist[-1]["content"]


def test_chat_handles_tool_call_and_continues_for_non_image_tool(
    mock_context: MagicMock,
) -> None:
    """Tool call path: _collect_stream yields tool_call/tool_result, handler invoked, loop continues for final answer."""
    # No preceding text so first collect produces only the tool items (full_response stays ""),
    # leading to history of exactly [user, tool, final_asst] after the continue + second turn.
    tool_stream = make_tool_call_stream()
    final_stream = make_text_only_stream("Today is 2025-09-18.")

    tool_responses = [
        {"role": "tool", "content": "2025-09-18", "tool_call_id": "call_123"}
    ]
    handler: ToolHandler = MagicMock(return_value=(tool_responses, [None]))

    iface = ChatInterface(
        mock_context,
        completer=create_responder([tool_stream, final_stream]),  # type: ignore[arg-type]
        tool_handler=handler,
    )

    yields = list(iface.chat("What day is it?", [], "Question"))

    handler.assert_called_once()  # type: ignore[attr-defined]
    # History: user, (tool role), assistant(final)
    assert len(iface.chat_history) == 3
    assert iface.chat_history[1]["role"] == "tool"
    assert "2025-09-18" in iface.chat_history[2]["content"]

    # At least one yield should contain the tool role entry (after first collect)
    tool_yield_found = any(
        any(h.get("role") == "tool" for h in hist) for hist, _ in yields
    )
    assert tool_yield_found


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
        content_type="image",
    )
    handler: ToolHandler = MagicMock(
        return_value=(
            [{"role": "tool", "content": "", "tool_call_id": "call_img"}],
            [tool_result],
        )
    )

    iface = ChatInterface(
        mock_context,
        completer=create_responder([img_stream]),  # type: ignore[arg-type]
        tool_handler=handler,
    )

    yields = list(iface.chat("Draw a red square", [], "Generate Image"))

    handler.assert_called_once()  # type: ignore[attr-defined]
    # Image must be populated
    assert iface.image is not None
    assert isinstance(iface.image, PILImage.Image)
    assert iface.image.size == (4, 4)

    # Final yield must carry the image
    final_hist, final_img = yields[-1]
    assert final_img is iface.image
    # The preceding text from model should be present (check internal history after consumption
    # as it is mutated by appends that happen after the tool_result yield snapshot).
    assert any(
        "Calling the image generation tool" in (h.get("content") or "")
        for h in iface.chat_history
        if h.get("role") == "assistant"
    )
    # Tool entry present (in final internal state after append/extend that occur after the last yield)
    assert any(h.get("role") == "tool" for h in iface.chat_history)


def test_chat_catches_exception_and_yields_generic_error(
    mock_context: MagicMock,
) -> None:
    """Exception path in the main chat loop produces a friendly error message."""

    def exploding_completer(
        model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Iterator[ChatCompletionChunk]:
        raise RuntimeError("boom from test")

    iface = ChatInterface(mock_context, completer=exploding_completer)  # type: ignore[arg-type]

    yields = list(iface.chat("Trigger error", [], "Question"))

    assert len(iface.chat_history) == 2
    assert "error processing your request" in iface.chat_history[-1]["content"].lower()

    # Last yield contains the error
    last_hist, _ = yields[-1]
    assert "error" in last_hist[-1]["content"].lower()


def test_clear_history_resets_chat_and_image_state(
    mock_context: MagicMock, sample_png_bytes: bytes
) -> None:
    """clear_history empties history and removes any generated image."""
    # Seed some state via a fake image-producing interaction (simplified)
    iface = ChatInterface(mock_context)
    iface.chat_history = [{"role": "user", "content": "x"}]
    iface.image = PILImage.open(BytesIO(sample_png_bytes))

    result = iface.clear_history()

    assert result == []
    assert iface.chat_history == []
    assert iface.image is None


def test_collect_stream_handles_multiple_tool_calls_and_argument_accumulation(
    mock_context: MagicMock,
) -> None:
    """_collect_stream correctly merges parallel tool calls and concatenates streamed arguments."""
    multi_stream = make_multi_tool_call_stream()

    # Handler will be called with reconstructed tool_calls (we just need it not to explode)
    handler: ToolHandler = MagicMock(
        return_value=(
            [
                {"role": "tool", "content": "a", "tool_call_id": "call_a"},
                {"role": "tool", "content": "b", "tool_call_id": "call_b"},
            ],
            [None, None],
        )
    )

    # Provide a second stream so the non-image tool path's "continue" has a terminating text response
    finisher = make_text_only_stream("Done with tools.")
    iface = ChatInterface(
        mock_context,
        completer=create_responder([multi_stream, finisher]),  # type: ignore[arg-type]
        tool_handler=handler,
    )

    _ = list(iface.chat("Use two tools", [], "Question"))

    handler.assert_called_once()  # type: ignore[attr-defined]
    # The reconstructed message passed to handler should have two tool_calls
    call_arg = handler.call_args[0][0]  # type: ignore[attr-defined]
    assert len(call_arg.tool_calls) == 2
    assert call_arg.tool_calls[0].id == "call_a"
    assert call_arg.tool_calls[1].function.arguments == '{"x":1}'


def test_chat_tool_result_without_image_mode_does_not_set_image(
    mock_context: MagicMock,
) -> None:
    """When a ToolResult (even image type) is returned in non-image mode, image is not set (the 'and is_image_mode' guard)."""
    tool_stream = make_tool_call_stream()
    weird_result = ToolResult(
        content_for_model="should not become image",
        content=b"not real png",
        content_type="image",
    )
    handler: ToolHandler = MagicMock(
        return_value=(
            [{"role": "tool", "content": "x", "tool_call_id": "c"}],
            [weird_result],
        )
    )

    # Second stream terminates the while-loop after the tool "continue" (non-image mode)
    finisher = make_text_only_stream("Acknowledged tool result.")
    iface = ChatInterface(
        mock_context,
        completer=create_responder([tool_stream, finisher]),  # type: ignore[arg-type]
        tool_handler=handler,
    )

    _ = list(iface.chat("Tool but not image mode", [], "Question"))

    # Guard prevented image assignment
    assert iface.image is None
