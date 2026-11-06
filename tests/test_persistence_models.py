from __future__ import annotations

from datetime import timezone

import pytest
from pydantic import ValidationError

from python_chat.persistence.models import (
    ChatMessage,
    ChatSession,
    QueueEvent,
    QueueEventType,
    ToolCall,
)


def test_persistence_models_supply_utc_timestamps_and_independent_metadata() -> None:
    first = ChatSession(user_id="u-1", mode="Question")
    second = ChatSession(user_id="u-2", mode="Question")
    message = ChatMessage(session_id=1, role="user", content="hello")
    event = QueueEvent(payload={"kind": "test"})

    assert first.started_at.tzinfo == timezone.utc
    assert second.started_at.tzinfo == timezone.utc
    assert first.metadata == second.metadata == {}
    first.metadata["source"] = "test"
    assert second.metadata == {}
    assert message.started_at.tzinfo == timezone.utc
    assert event.occurred_at.tzinfo == timezone.utc
    assert event.event_type is QueueEventType.CHAT_LOG


def test_persistence_models_accept_optional_fields_and_reject_extras() -> None:
    session = ChatSession(
        user_id="u-1",
        mode="Question",
        title="Greetings",
        ended_at=None,
        metadata={"source": "unit-test"},
    )
    message = ChatMessage(
        session_id=3,
        role="assistant",
        content="hello",
        tool_calls=[{"id": "call-1"}],
        estimated_tokens=4,
        estimated_cost=0.01,
    )
    call = ToolCall(
        message_id=4,
        tool_name="echo",
        status="ok",
        input_args={"text": "hi"},
        output_result={"text": "hi"},
        latency_ms=10,
        estimated_cost=0.0,
    )

    assert session.title == "Greetings"
    assert message.estimated_tokens == 4
    assert call.output_result == {"text": "hi"}

    with pytest.raises(ValidationError):
        ChatSession(user_id="u-1", mode="Question", unexpected=True)  # type: ignore[call-arg]
