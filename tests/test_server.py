from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import Request


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


@pytest.fixture(autouse=True)
def _skip_server_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHON_CHAT_SKIP_SERVER_BOOTSTRAP", "1")


def _create_test_app(monkeypatch: Any, server_module: Any) -> Any:
    monkeypatch.setattr(server_module, "initialize_runtime", lambda: None)
    monkeypatch.setattr(server_module, "launch_app", lambda **_: object())
    monkeypatch.setattr(server_module.gr, "mount_gradio_app", lambda *_, **__: None)
    return server_module.create_server_app()


def test_authenticate_supabase_password_returns_session_details(
    monkeypatch: Any,
) -> None:
    import supabase
    from python_chat import server as server_module

    auth = SimpleNamespace(
        sign_in_with_password=lambda credentials: SimpleNamespace(
            session=SimpleNamespace(access_token="access", expires_in=600),
            user=SimpleNamespace(id="user-1"),
        )
    )
    monkeypatch.setenv("SUPABASE_URL", "https://supabase.example")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon")
    monkeypatch.setattr(
        supabase, "create_client", lambda *_: SimpleNamespace(auth=auth)
    )

    assert server_module._authenticate_supabase_password(" a@b.test ", "pw") == (
        "access",
        "user-1",
        600,
    )


def test_authenticate_supabase_password_rejects_missing_config(
    monkeypatch: Any,
) -> None:
    from fastapi import HTTPException

    from python_chat import server as server_module

    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
    with pytest.raises(HTTPException) as exc_info:
        server_module._authenticate_supabase_password("user@example.com", "pw")
    assert exc_info.value.status_code == 500


def test_authenticate_supabase_password_maps_auth_failure_to_unauthorized(
    monkeypatch: Any,
) -> None:
    from fastapi import HTTPException

    import supabase
    from python_chat import server as server_module

    auth = SimpleNamespace(
        sign_in_with_password=lambda _: (_ for _ in ()).throw(RuntimeError("offline"))
    )
    monkeypatch.setenv("SUPABASE_URL", "https://supabase.example")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon")
    monkeypatch.setattr(
        supabase, "create_client", lambda *_: SimpleNamespace(auth=auth)
    )

    with pytest.raises(HTTPException) as exc_info:
        server_module._authenticate_supabase_password("user@example.com", "pw")
    assert exc_info.value.status_code == 401


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(session=None, user=SimpleNamespace(id="user-1")),
        SimpleNamespace(
            session=SimpleNamespace(access_token="access", expires_in="bad"),
            user=None,
        ),
    ],
)
def test_authenticate_supabase_password_rejects_incomplete_response(
    monkeypatch: Any, response: Any
) -> None:
    from fastapi import HTTPException

    import supabase
    from python_chat import server as server_module

    monkeypatch.setenv("SUPABASE_URL", "https://supabase.example")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon")
    monkeypatch.setattr(
        supabase,
        "create_client",
        lambda *_: SimpleNamespace(
            auth=SimpleNamespace(sign_in_with_password=lambda _: response)
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        server_module._authenticate_supabase_password("user@example.com", "pw")
    assert exc_info.value.status_code == 401


def test_verify_supabase_user_success_and_invalid_user(monkeypatch: Any) -> None:
    from fastapi import HTTPException

    import supabase
    from python_chat import server as server_module

    monkeypatch.setenv("SUPABASE_URL", "https://supabase.example")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon")
    monkeypatch.setattr(
        supabase,
        "create_client",
        lambda *_: SimpleNamespace(
            auth=SimpleNamespace(
                get_user=lambda _: SimpleNamespace(user=SimpleNamespace(id="u-1"))
            )
        ),
    )
    assert server_module._verify_supabase_user("token") == "u-1"

    monkeypatch.setattr(
        supabase,
        "create_client",
        lambda *_: SimpleNamespace(
            auth=SimpleNamespace(get_user=lambda _: SimpleNamespace(user=None))
        ),
    )
    with pytest.raises(HTTPException) as exc_info:
        server_module._verify_supabase_user("token")
    assert exc_info.value.status_code == 401


def test_server_login_routes_and_root_redirect(monkeypatch: Any) -> None:
    monkeypatch.setenv("PYTHON_CHAT_SKIP_SERVER_BOOTSTRAP", "1")
    from fastapi import HTTPException
    from fastapi.testclient import TestClient

    from python_chat import server as server_module

    app = _create_test_app(monkeypatch, server_module)
    client = TestClient(app)

    assert client.get("/", follow_redirects=False).headers["location"] == "/login"
    page = client.get("/login?next=https%3A%2F%2Fevil.test&error=%3Cb%3Ebad%3C%2Fb%3E")
    assert "value='/app'" in page.text
    assert "&lt;b&gt;bad&lt;/b&gt;" in page.text

    missing = client.post(
        "/login", data={"email": "", "password": ""}, follow_redirects=False
    )
    assert missing.status_code == 303
    assert "Email+and+password+are+required" in missing.headers["location"]

    monkeypatch.setattr(
        server_module,
        "_authenticate_supabase_password",
        lambda *_: (_ for _ in ()).throw(
            HTTPException(status_code=401, detail="invalid")
        ),
    )
    rejected = client.post(
        "/login",
        data={"email": "u@example.com", "password": "bad"},
        follow_redirects=False,
    )
    assert rejected.status_code == 303
    assert "Invalid+email+or+password" in rejected.headers["location"]

    monkeypatch.setattr(
        server_module,
        "_authenticate_supabase_password",
        lambda *_: ("access-token", "user-1", 120),
    )
    monkeypatch.setenv("COOKIE_SECURE", "yes")
    accepted = client.post(
        "/login",
        data={"email": "u@example.com", "password": "pw", "next": "/app"},
        follow_redirects=False,
    )
    assert accepted.status_code == 303
    assert accepted.headers["location"] == "/app"
    assert "httponly" in accepted.headers["set-cookie"].lower()
    assert "secure" in accepted.headers["set-cookie"].lower()
    assert "max-age=120" in accepted.headers["set-cookie"].lower()

    logout = client.get("/logout", follow_redirects=False)
    assert logout.headers["location"] == "/login"
    assert "python_chat_access_token" in logout.headers.get("set-cookie", "")


def test_server_app_middleware_handles_missing_invalid_and_valid_tokens(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("PYTHON_CHAT_SKIP_SERVER_BOOTSTRAP", "1")
    from fastapi.testclient import TestClient

    from python_chat import server as server_module

    app = _create_test_app(monkeypatch, server_module)

    def probe(request: Request) -> dict[str, Any]:
        return {"identity": (request.state.user_id, request.state.access_token)}

    app.get("/app/probe")(probe)
    client = TestClient(app)

    missing = client.get("/app/probe", follow_redirects=False)
    assert missing.status_code == 302
    assert "next=%2Fapp%2Fprobe" in missing.headers["location"]

    from fastapi import HTTPException

    monkeypatch.setattr(
        server_module,
        "_verify_supabase_user",
        lambda _: (_ for _ in ()).throw(
            HTTPException(status_code=401, detail="bad token")
        ),
    )
    invalid = client.get(
        "/app/probe", headers={"Authorization": "Bearer bad"}, follow_redirects=False
    )
    assert invalid.status_code == 302
    assert "bad+token" in invalid.headers["location"]

    monkeypatch.setattr(server_module, "_verify_supabase_user", lambda _: "user-9")
    valid = client.get("/app/probe", headers={"Authorization": "Bearer good"})
    assert valid.json() == {"identity": ["user-9", "good"]}


def test_server_health_ready_and_global_error_handler(monkeypatch: Any) -> None:
    monkeypatch.setenv("PYTHON_CHAT_SKIP_SERVER_BOOTSTRAP", "1")
    from fastapi.testclient import TestClient

    from python_chat import server as server_module

    app = _create_test_app(monkeypatch, server_module)
    app.get("/explode")(lambda: (_ for _ in ()).throw(RuntimeError("private detail")))
    from python_chat.context import ModelContext

    monkeypatch.setattr(ModelContext, "current", lambda: object())
    client = TestClient(app, raise_server_exceptions=False)

    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"status": "ready"}
    error = client.get("/explode")
    assert error.status_code == 500
    assert error.json() == {"message": "An internal server error occurred."}
