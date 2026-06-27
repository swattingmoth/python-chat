from __future__ import annotations

import logging
import time
from typing import Any, Protocol, cast

from python_chat.persistence.models import ChatMessage, ChatSession, ToolCall

logger = logging.getLogger(__name__)


class RpcResult(Protocol):
    data: Any


class RpcClient(Protocol):
    def rpc(self, fn: str, params: dict[str, Any]) -> Any: ...


def _extract_scalar_id(raw_data: Any) -> int | None:
    if raw_data is None:
        return None
    if isinstance(raw_data, int):
        return raw_data
    if isinstance(raw_data, list) and raw_data:
        first = raw_data[0]
        if isinstance(first, int):
            return first
        if isinstance(first, dict):
            id_value = first.get("id")
            return int(id_value) if isinstance(id_value, (int, str)) else None
    if isinstance(raw_data, dict):
        id_value = raw_data.get("id")
        if isinstance(id_value, int):
            return id_value
    return None


def _rpc_with_retry(
    client: RpcClient,
    function_name: str,
    params: dict[str, Any],
    *,
    retries: int = 3,
    backoff_seconds: float = 0.2,
) -> Any:
    attempt = 0
    while True:
        try:
            response = client.rpc(function_name, params)
            executed = response.execute() if hasattr(response, "execute") else response
            result = cast(RpcResult, executed)
            return result.data
        except Exception:
            attempt += 1
            if attempt >= retries:
                raise
            sleep_for = backoff_seconds * (2 ** (attempt - 1))
            time.sleep(sleep_for)


def create_chat_session(client: RpcClient, session: ChatSession) -> int | None:
    data = _rpc_with_retry(
        client,
        "create_chat_session",
        {
            "p_user_id": session.user_id,
            "p_mode": session.mode,
            "p_title": session.title,
            "p_started_at": session.started_at.isoformat(),
            "p_metadata": session.metadata,
        },
    )
    return _extract_scalar_id(data)


def complete_chat_session(client: RpcClient, session_id: int, ended_at: str) -> None:
    _rpc_with_retry(
        client,
        "complete_chat_session",
        {
            "p_session_id": session_id,
            "p_ended_at": ended_at,
        },
    )


def create_chat_message(client: RpcClient, message: ChatMessage) -> int | None:
    data = _rpc_with_retry(
        client,
        "create_chat_message",
        {
            "p_session_id": message.session_id,
            "p_role": message.role,
            "p_content": message.content,
            "p_tool_calls": message.tool_calls,
            "p_estimated_tokens": message.estimated_tokens,
            "p_estimated_cost": message.estimated_cost,
            "p_started_at": message.started_at.isoformat(),
        },
    )
    return _extract_scalar_id(data)


def create_tool_call(client: RpcClient, call: ToolCall) -> int | None:
    data = _rpc_with_retry(
        client,
        "create_tool_call",
        {
            "p_message_id": call.message_id,
            "p_tool_name": call.tool_name,
            "p_status": call.status,
            "p_input_args": call.input_args,
            "p_output_result": call.output_result,
            "p_error_message": call.error_message,
            "p_latency_ms": call.latency_ms,
        },
    )
    return _extract_scalar_id(data)
