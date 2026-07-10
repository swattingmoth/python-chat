from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException, status


def test_extract_bearer_token_parses_expected_values(monkeypatch: Any) -> None:
    monkeypatch.setenv("PYTHON_CHAT_SKIP_SERVER_BOOTSTRAP", "1")

    from python_chat.server import _extract_bearer_token

    assert _extract_bearer_token(None) is None
    assert _extract_bearer_token("Basic abc") is None
    assert _extract_bearer_token("Bearer token-123") == "token-123"


def test_normalize_next_path_allows_only_safe_relative(monkeypatch: Any) -> None:
    monkeypatch.setenv("PYTHON_CHAT_SKIP_SERVER_BOOTSTRAP", "1")

    from python_chat.server import _normalize_next_path

    assert _normalize_next_path(None) == "/app"
    assert _normalize_next_path("https://evil.test") == "/app"
    assert _normalize_next_path("//evil.test") == "/app"
    assert _normalize_next_path("/app") == "/app"
    assert _normalize_next_path("/app/chat") == "/app/chat"


def test_extract_access_token_from_request_prefers_bearer(monkeypatch: Any) -> None:
    monkeypatch.setenv("PYTHON_CHAT_SKIP_SERVER_BOOTSTRAP", "1")

    from python_chat.server import _extract_access_token_from_request

    request: Any = SimpleNamespace(
        headers={"Authorization": "Bearer hdr-token"},
        cookies={"python_chat_access_token": "cookie-token"},
    )
    assert _extract_access_token_from_request(request) == "hdr-token"


def test_extract_access_token_from_request_uses_cookie(monkeypatch: Any) -> None:
    monkeypatch.setenv("PYTHON_CHAT_SKIP_SERVER_BOOTSTRAP", "1")

    from python_chat.server import _extract_access_token_from_request

    request: Any = SimpleNamespace(
        headers={},
        cookies={"python_chat_access_token": " abc "},
    )
    assert _extract_access_token_from_request(request) == "abc"


def test_cookie_secure_flag_respects_env(monkeypatch: Any) -> None:
    monkeypatch.setenv("PYTHON_CHAT_SKIP_SERVER_BOOTSTRAP", "1")

    from python_chat.server import _cookie_secure_flag

    monkeypatch.delenv("COOKIE_SECURE", raising=False)
    assert _cookie_secure_flag() is False

    monkeypatch.setenv("COOKIE_SECURE", "true")
    assert _cookie_secure_flag() is True


def test_resolve_identity_reads_request_state(monkeypatch: Any) -> None:
    monkeypatch.setenv("PYTHON_CHAT_SKIP_SERVER_BOOTSTRAP", "1")

    from python_chat.server import _resolve_identity

    request = SimpleNamespace(
        state=SimpleNamespace(user_id="u-1", access_token="t-1"),
    )
    wrapped_request = SimpleNamespace(request=request)

    assert _resolve_identity(wrapped_request) == ("u-1", "t-1")
    assert _resolve_identity(None) == (None, None)


def test_verify_supabase_user_returns_user_id(monkeypatch: Any) -> None:
    monkeypatch.setenv("PYTHON_CHAT_SKIP_SERVER_BOOTSTRAP", "1")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")

    from python_chat.server import _verify_supabase_user

    class _FakeAuth:
        def get_user(self, access_token: str) -> Any:
            assert access_token == "valid-token"
            return SimpleNamespace(user=SimpleNamespace(id="user-123"))

    class _FakeClient:
        auth = _FakeAuth()

    def _create_client(url: str, key: str) -> Any:
        assert url == "https://example.supabase.co"
        assert key == "anon-key"
        return _FakeClient()

    monkeypatch.setitem(
        sys.modules,
        "supabase",
        SimpleNamespace(create_client=_create_client),
    )

    assert _verify_supabase_user("valid-token") == "user-123"


def test_verify_supabase_user_raises_unauthorized_on_client_error(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("PYTHON_CHAT_SKIP_SERVER_BOOTSTRAP", "1")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")

    from python_chat.server import _verify_supabase_user

    class _FakeAuth:
        def get_user(self, access_token: str) -> Any:
            raise RuntimeError("token invalid")

    class _FakeClient:
        auth = _FakeAuth()

    monkeypatch.setitem(
        sys.modules,
        "supabase",
        SimpleNamespace(create_client=lambda _url, _key: _FakeClient()),
    )

    with pytest.raises(HTTPException) as exc_info:
        _verify_supabase_user("invalid-token")

    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert exc_info.value.detail == "Invalid Supabase access token."


def test_verify_supabase_user_raises_when_user_id_missing(monkeypatch: Any) -> None:
    monkeypatch.setenv("PYTHON_CHAT_SKIP_SERVER_BOOTSTRAP", "1")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")

    from python_chat.server import _verify_supabase_user

    class _FakeAuth:
        def get_user(self, access_token: str) -> Any:
            return SimpleNamespace(user=SimpleNamespace(id=""))

    class _FakeClient:
        auth = _FakeAuth()

    monkeypatch.setitem(
        sys.modules,
        "supabase",
        SimpleNamespace(create_client=lambda _url, _key: _FakeClient()),
    )

    with pytest.raises(HTTPException) as exc_info:
        _verify_supabase_user("valid-token")

    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert exc_info.value.detail == "Supabase token does not contain a valid user id."
