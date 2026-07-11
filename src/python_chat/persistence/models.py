from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class QueueEventType(str, Enum):
    CHAT_LOG = "chat_log"


class ChatSession(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str
    mode: str
    title: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: int
    role: str
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    estimated_tokens: int | None = None
    estimated_cost: float | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: int
    tool_name: str
    status: str
    input_args: dict[str, Any] | None = None
    output_result: dict[str, Any] | None = None
    error_message: str | None = None
    latency_ms: int | None = None
    estimated_cost: float | None = None


class QueueEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: QueueEventType = QueueEventType.CHAT_LOG
    session_id: int | None = None
    payload: dict[str, Any]
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
