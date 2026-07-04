from __future__ import annotations

from typing import Any

import pytest

from python_chat import chatbot


class _FakePersistenceClient:
    def __init__(self, *, enabled: bool, rpc_client: Any = None) -> None:
        self.enabled = enabled
        self.rpc_client = rpc_client


def test_resolve_xai_api_key_raises_when_unset(monkeypatch: Any) -> None:
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    fake_persistence = _FakePersistenceClient(enabled=False)

    with pytest.raises(RuntimeError, match="XAI_API_KEY is not configured"):
        chatbot._resolve_xai_api_key(fake_persistence)  # type: ignore[arg-type]
