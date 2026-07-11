from __future__ import annotations

from types import SimpleNamespace
from typing import Any


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
    monkeypatch.delenv("CHATBOT_ENV", raising=False)
    assert _cookie_secure_flag() is True

    monkeypatch.setenv("CHATBOT_ENV", "production")
    assert _cookie_secure_flag() is True

    monkeypatch.setenv("CHATBOT_ENV", "development")
    assert _cookie_secure_flag() is False

    monkeypatch.setenv("COOKIE_SECURE", "true")
    assert _cookie_secure_flag() is True


def test_login_submit_propagates_server_errors(monkeypatch: Any) -> None:
    monkeypatch.setenv("PYTHON_CHAT_SKIP_SERVER_BOOTSTRAP", "1")

    from fastapi import HTTPException, status
    from fastapi.testclient import TestClient

    from python_chat import server as server_module

    monkeypatch.setattr(server_module, "initialize_runtime", lambda: None)
    monkeypatch.setattr(server_module, "launch_app", lambda **_: object())
    monkeypatch.setattr(server_module.gr, "mount_gradio_app", lambda *_, **__: None)  # type: ignore

    app = server_module.create_server_app()
    client = TestClient(app)

    def raise_server_error(email: str, password: str) -> tuple[str, str, int | None]:
        del email
        del password
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Supabase auth verification is not configured.",
        )

    monkeypatch.setattr(
        server_module,
        "_authenticate_supabase_password",
        raise_server_error,
    )

    response = client.post(
        "/login",
        data={"email": "user@example.com", "password": "secret"},
        follow_redirects=False,
    )

    assert response.status_code == 500


def test_resolve_identity_reads_request_state(monkeypatch: Any) -> None:
    monkeypatch.setenv("PYTHON_CHAT_SKIP_SERVER_BOOTSTRAP", "1")

    from python_chat.server import _resolve_identity

    request = SimpleNamespace(
        state=SimpleNamespace(user_id="u-1", access_token="t-1"),
    )

    assert _resolve_identity(request) == ("u-1", "t-1")  # type: ignore
    assert _resolve_identity(None) == (None, None)
