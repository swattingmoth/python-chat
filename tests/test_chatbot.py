from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from python_chat import chatbot
from python_chat.context import ModelContext
from python_chat.persistence import db


class _FakePersistenceClient:
    def __init__(self, *, enabled: bool, rpc_client: Any = None) -> None:
        self.enabled = enabled
        self.rpc_client = rpc_client


def test_resolve_xai_api_key_raises_when_unset(monkeypatch: Any) -> None:
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    fake_persistence = _FakePersistenceClient(enabled=False)

    with pytest.raises(RuntimeError, match="XAI_API_KEY is not configured"):
        chatbot._resolve_xai_api_key(fake_persistence)  # type: ignore[arg-type]


def test_resolve_xai_api_key_reads_vault_secret(monkeypatch: Any) -> None:
    rpc_client = object()
    fake_persistence = _FakePersistenceClient(enabled=True, rpc_client=rpc_client)
    monkeypatch.setattr(db, "get_secret_from_vault", lambda client, name: "vault-key")

    assert chatbot._resolve_xai_api_key(fake_persistence) == "vault-key"  # type: ignore[arg-type]


def test_resolve_xai_api_key_rejects_missing_rpc_client() -> None:
    fake_persistence = _FakePersistenceClient(enabled=True)

    with pytest.raises(RuntimeError, match="XAI_API_KEY is not configured"):
        chatbot._resolve_xai_api_key(fake_persistence)  # type: ignore[arg-type]


def test_initialize_runtime_requires_image_folder_in_development(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("CHATBOT_ENV", "development")
    monkeypatch.delenv("IMAGE_FOLDER", raising=False)

    with pytest.raises(Exception, match="IMAGE_FOLDER environment variable is not set"):
        chatbot.initialize_runtime()


def test_initialize_runtime_builds_persistent_runtime(monkeypatch: Any) -> None:
    persistence = _FakePersistenceClient(enabled=True, rpc_client=object())
    log_queue = object()
    monkeypatch.setenv("CHATBOT_ENV", "production")
    monkeypatch.setenv("IMAGE_FOLDER", "images")
    monkeypatch.setattr(chatbot, "SupabaseClient", lambda: persistence)
    monkeypatch.setattr(chatbot, "AsyncLogQueue", lambda client: log_queue)
    monkeypatch.setattr(chatbot, "_resolve_xai_api_key", lambda client: "secret")

    def api_factory(key: str, url: str) -> tuple[str, str]:
        return key, url

    monkeypatch.setattr(chatbot, "init_api", api_factory)
    create_context = MagicMock()
    monkeypatch.setattr(ModelContext, "create", create_context)

    result = chatbot.initialize_runtime()

    assert result is create_context.return_value
    create_context.assert_called_once_with(
        ("secret", "https://api.x.ai/v1"),
        "images",
        persistence_client=persistence,
        log_queue=log_queue,
    )


def test_initialize_runtime_without_persistence_uses_no_queue(
    monkeypatch: Any,
) -> None:
    persistence = _FakePersistenceClient(enabled=False)
    monkeypatch.setenv("CHATBOT_ENV", "production")
    monkeypatch.delenv("IMAGE_FOLDER", raising=False)
    monkeypatch.setattr(chatbot, "SupabaseClient", lambda: persistence)
    monkeypatch.setattr(chatbot, "_resolve_xai_api_key", lambda client: "secret")
    monkeypatch.setattr(chatbot, "init_api", lambda key, url: "api-client")
    create_context = MagicMock()
    monkeypatch.setattr(ModelContext, "create", create_context)

    assert chatbot.initialize_runtime() is create_context.return_value
    create_context.assert_called_once_with(
        "api-client", "", persistence_client=None, log_queue=None
    )


def test_module_entry_point_initializes_and_launches_with_mocked_services(
    monkeypatch: Any,
) -> None:
    from python_chat import api, app, persistence

    fake_persistence = _FakePersistenceClient(enabled=True, rpc_client=object())
    fake_queue = object()
    fake_app = MagicMock()
    monkeypatch.setenv("CHATBOT_ENV", "production")
    monkeypatch.setenv("IMAGE_FOLDER", "images")
    monkeypatch.setattr(persistence, "SupabaseClient", lambda: fake_persistence)
    monkeypatch.setattr(persistence, "AsyncLogQueue", lambda _: fake_queue)
    monkeypatch.setattr(db, "get_secret_from_vault", lambda *_: "vault-key")
    monkeypatch.setattr(api, "init_api", lambda *_: "api-client")
    monkeypatch.setattr(ModelContext, "create", MagicMock(return_value=object()))
    configure_logging = MagicMock()
    launch_app = MagicMock(return_value=fake_app)
    monkeypatch.setattr(app, "configure_logging", configure_logging)
    monkeypatch.setattr(app, "launch_app", launch_app)

    runpy.run_path(str(Path(chatbot.__file__)), run_name="__main__")

    configure_logging.assert_called_once_with(log_to_file=True)
    launch_app.assert_called_once_with()
    fake_app.launch.assert_called_once_with()


def test_module_entry_point_logs_startup_failure(monkeypatch: Any, caplog: Any) -> None:
    from python_chat import app

    monkeypatch.setenv("CHATBOT_ENV", "development")
    monkeypatch.delenv("IMAGE_FOLDER", raising=False)
    monkeypatch.setattr(app, "configure_logging", MagicMock())
    launch_app = MagicMock()
    monkeypatch.setattr(app, "launch_app", launch_app)

    runpy.run_path(str(Path(chatbot.__file__)), run_name="__main__")

    launch_app.assert_not_called()
    assert "An error occurred while launching the app" in caplog.text
