"""Tests for chat.py module.

Covers the two pure choice functions and ChatInterface with dependency injection
for high coverage without real API calls or side effects.
"""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from typing import Any, Callable, Generator
from unittest.mock import MagicMock

import pytest
from PIL import Image as PILImage

from python_chat.api import Models
from python_chat.chat import (
    ChatCompleter,
    ChatInterface,
    ToolHandler,
    get_model_for_choice,
    get_system_message_for_choice,
)
from python_chat.tools import ToolResult

# --- Helpers for realistic fake streams (xai-sdk style) ---
# The chat() loop now consumes Generator[tuple[response, chunk], ...] directly.
# Tool calls are provided on the *final* response object (no delta accumulation in our code).
# We use SimpleNamespace to simulate the minimal attrs accessed: chunk.content, response.tool_calls


def _make_chunk(content: str | None = None) -> Any:
    """Create a minimal chunk with .content (incremental text token or None)."""
    return SimpleNamespace(content=content)


def _make_response(tool_calls: list[Any] | None = None) -> Any:
    """Create a response object; tool_calls (if present) are inspected after streaming a turn."""
    return SimpleNamespace(tool_calls=tool_calls or [])


def make_text_only_stream(text: str) -> list[tuple[Any, Any]]:
    """Simple single response stream (no tools). Yields (resp, chunk) pairs."""
    # Split into a couple tokens to exercise multiple yields + final content
    tokens = [text[: len(text) // 2], text[len(text) // 2 :]]
    return [
        (_make_response(), _make_chunk(content=tokens[0])),
        (_make_response(), _make_chunk(content=tokens[1])),
    ]


def make_tool_call_stream(
    tool_name: str = "today_date",
    arguments: str = "{}",
    preceding_text: str | None = None,
) -> list[tuple[Any, Any]]:
    """Stream that ends with a (client-side) tool call on the final response.

    Optionally includes preceding assistant text content.
    """
    pairs: list[tuple[Any, Any]] = []
    if preceding_text:
        pairs.append((_make_response(), _make_chunk(content=preceding_text)))
    # The tool_calls live on the response of the last yielded pair for this turn.
    tc = SimpleNamespace(
        id="call_123",
        function=SimpleNamespace(name=tool_name, arguments=arguments),
    )
    pairs.append((_make_response(tool_calls=[tc]), _make_chunk(content=None)))
    return pairs


def make_multi_tool_call_stream() -> list[tuple[Any, Any]]:
    """Stream with two parallel client-side tool calls (on final response)."""
    tc0 = SimpleNamespace(
        id="call_a", function=SimpleNamespace(name="today_date", arguments="{}")
    )
    tc1 = SimpleNamespace(
        id="call_b", function=SimpleNamespace(name="other", arguments='{"x":1}')
    )
    return [(_make_response(tool_calls=[tc0, tc1]), _make_chunk(content=None))]


def create_responder(
    streams: list[list[tuple[Any, Any]]],
) -> Callable[
    [str, list[dict[str, Any]], list[Any]], Generator[tuple[Any, Any], None, None]
]:
    """Stateful fake completer that yields successive (response, chunk) streams.

    Matches the new ChatCompleter contract used by ChatInterface.chat.
    """
    call_idx = 0

    def completer(
        model: str, messages: list[dict[str, Any]], tools: list[Any]
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
    assert result == [(initial_history, None)]
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

    iface = ChatInterface(mock_context, completer=create_responder([chunks1, chunks2]))

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
    """Tool call path: stream yields content then final response carries tool_calls; handler invoked, loop continues for final answer."""
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
        completer=create_responder([tool_stream, final_stream]),
        tool_handler=handler,
    )

    yields = list(iface.chat("What day is it?", [], "Question"))

    handler.assert_called_once()  # type: ignore[attr-defined]
    # History: user, assistant(with tool_calls), tool(result), assistant(final)
    assert len(iface.chat_history) == 4
    assert iface.chat_history[0]["role"] == "user"
    assert iface.chat_history[1]["role"] == "assistant"
    assert "tool_calls" in iface.chat_history[1]
    assert iface.chat_history[2]["role"] == "tool"
    assert iface.chat_history[2]["content"] == "2025-09-18"
    assert iface.chat_history[3]["role"] == "assistant"
    assert "2025-09-18" in iface.chat_history[3]["content"]

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
        model: str, messages: list[dict[str, Any]], tools: list[Any]
    ) -> Generator[tuple[Any, Any], None, None]:
        raise RuntimeError("boom from test")

    iface = ChatInterface(mock_context, completer=exploding_completer)

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


def test_chat_handles_multiple_parallel_tool_calls(
    mock_context: MagicMock,
) -> None:
    """Chat loop correctly surfaces multiple parallel client-side tool calls to the handler.

    (Replaces prior _collect_stream delta accumulation test; xai-sdk provides complete
    tool_calls on the final Response, which chat() normalizes and passes through.)
    """
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
        completer=create_responder([multi_stream, finisher]),
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
        completer=create_responder([tool_stream, finisher]),
        tool_handler=handler,
    )

    _ = list(iface.chat("Tool but not image mode", [], "Question"))

    # Guard prevented image assignment
    assert iface.image is None
