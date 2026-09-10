"""Tests for persistence package (queue batching + RPC helpers)."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from python_chat.persistence import AsyncLogQueue, build_log_event, db
from python_chat.persistence.models import ChatMessage, ChatSession, ToolCall


class _FakeRpcCall:
    def __init__(self, data: Any) -> None:
        self._data = data

    def execute(self) -> Any:
        return SimpleNamespace(data=self._data)


class _FakeRpcClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def rpc(self, fn: str, params: dict[str, Any]) -> _FakeRpcCall:
        self.calls.append((fn, params))
        if fn == "create_chat_session":
            return _FakeRpcCall(101)
        if fn == "create_chat_message":
            return _FakeRpcCall(202)
        if fn == "create_tool_call":
            return _FakeRpcCall(303)
        return _FakeRpcCall(None)


class _FakeStorageClient:
    def __init__(self) -> None:
        self.uploads: list[dict[str, Any]] = []

    def upload_bytes(
        self,
        *,
        bucket: str,
        path: str,
        data: bytes,
        content_type: str,
        upsert: bool = True,
    ) -> None:
        self.uploads.append(
            {
                "bucket": bucket,
                "path": path,
                "data": data,
                "content_type": content_type,
                "upsert": upsert,
            }
        )


def test_db_helpers_call_expected_rpc_functions() -> None:
    client = _FakeRpcClient()

    session_id = db.create_chat_session(
        client,
        ChatSession(user_id="u1", mode="Question"),
    )
    message_id = db.create_chat_message(
        client,
        ChatMessage(session_id=101, role="user", content="hello"),
    )
    tool_id = db.create_tool_call(
        client,
        ToolCall(message_id=202, tool_name="today_date", status="ok"),
    )

    assert session_id == 101
    assert message_id == 202
    assert tool_id == 303
    assert [name for name, _ in client.calls] == [
        "create_chat_session",
        "create_chat_message",
        "create_tool_call",
    ]


def test_async_log_queue_batches_and_uploads_jsonl() -> None:
    fake_storage = _FakeStorageClient()

    queue = AsyncLogQueue(
        fake_storage,  # type: ignore[arg-type]
        flush_interval_seconds=0.01,
        max_batch_size=2,
    )
    queue.start()
    queue.enqueue(build_log_event(10, {"event": "a"}))
    queue.enqueue(build_log_event(10, {"event": "b"}))
    queue.stop()

    assert len(fake_storage.uploads) == 1
    upload = fake_storage.uploads[0]
    assert upload["bucket"] == "chat-logs"
    assert upload["content_type"] == "application/x-ndjson"
    assert b'"event": "a"' in upload["data"]
    assert b'"event": "b"' in upload["data"]


def test_async_log_queue_defaults_to_sixty_second_flush_window() -> None:
    fake_storage = _FakeStorageClient()

    queue = AsyncLogQueue(fake_storage, max_batch_size=2)  # type: ignore[arg-type]

    assert queue._flush_interval_seconds == 60.0


def test_async_log_queue_flushes_when_oldest_event_exceeds_timeout() -> None:
    fake_storage = _FakeStorageClient()
    queue = AsyncLogQueue(
        fake_storage,  # type: ignore[arg-type]
        flush_interval_seconds=60.0,
        max_batch_size=2,
    )

    now = datetime.now(timezone.utc)
    pending = [
        build_log_event(10, {"event": "old"}).model_copy(
            update={"occurred_at": now - timedelta(seconds=61)}
        )
    ]

    assert queue._should_flush(pending, now) is True


def test_async_log_queue_start_without_running_event_loop() -> None:
    fake_storage = _FakeStorageClient()
    queue = AsyncLogQueue(
        fake_storage,  # type: ignore[arg-type]
        flush_interval_seconds=0.01,
        max_batch_size=1,
    )

    queue.start()
    queue.enqueue(build_log_event(12, {"event": "startup"}))
    queue.stop()

    assert len(fake_storage.uploads) == 1
    assert b'"event": "startup"' in fake_storage.uploads[0]["data"]


def test_async_log_queue_stop_returns_quickly_with_default_timeout() -> None:
    fake_storage = _FakeStorageClient()
    queue = AsyncLogQueue(fake_storage)  # type: ignore[arg-type]

    queue.start()
    start = time.monotonic()
    queue.stop()
    elapsed = time.monotonic() - start

    assert elapsed < 1.0


class _FakeStorageBucket:
    def __init__(self) -> None:
        self.upload_calls: list[dict[str, Any]] = []

    def upload(self, *, path: str, file: bytes, file_options: dict[str, str]) -> None:
        self.upload_calls.append(
            {"path": path, "file": file, "file_options": file_options}
        )

    def get_public_url(self, _: str) -> Any:
        return "https://example.test/public.png"


class _FakeStorage:
    def __init__(self) -> None:
        self.bucket = _FakeStorageBucket()

    def from_(self, _: str) -> _FakeStorageBucket:
        return self.bucket


class _FakeSupabaseRuntimeClient:
    def __init__(self) -> None:
        self.storage = _FakeStorage()


def test_supabase_client_upload_and_public_url_variants() -> None:
    from python_chat.persistence import SupabaseClient

    runtime_client = _FakeSupabaseRuntimeClient()
    client = SupabaseClient(client=runtime_client)

    client.upload_bytes(
        bucket="images",
        path="x/y.png",
        data=b"img",
        content_type="image/png",
    )
    assert runtime_client.storage.bucket.upload_calls

    assert (
        client.get_public_url(bucket="images", path="x/y.png")
        == "https://example.test/public.png"
    )

    runtime_client.storage.bucket.get_public_url = lambda _: {  # type: ignore[method-assign]
        "publicUrl": "https://example.test/from-dict.png"
    }
    assert (
        client.get_public_url(bucket="images", path="x/y.png")
        == "https://example.test/from-dict.png"
    )

    runtime_client.storage.bucket.get_public_url = lambda _: {"publicUrl": 123}  # type: ignore[method-assign]
    assert client.get_public_url(bucket="images", path="x/y.png") is None


def test_supabase_client_public_url_maps_to_local_mount_path(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    from python_chat.persistence import SupabaseClient

    mapped_file = tmp_path / "folder" / "x y.png"
    mapped_file.parent.mkdir(parents=True)
    mapped_file.write_bytes(b"png-bytes")

    runtime_client = _FakeSupabaseRuntimeClient()
    runtime_client.storage.bucket.get_public_url = lambda _: (  # type: ignore[method-assign]
        "http://host.docker.internal:54321/storage/v1/object/public/images/folder/x%20y.png"
    )

    monkeypatch.setenv("SUPABASE_PUBLIC_IMAGE_BUCKET_MOUNT_PATH", str(tmp_path))
    monkeypatch.delenv("SUPABASE_PUBLIC_URL", raising=False)
    client = SupabaseClient(client=runtime_client)

    assert client.get_public_url(bucket="images", path="x/y.png") == str(mapped_file)


def test_supabase_client_public_url_falls_back_when_mount_path_missing_file(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """If the mounted path is invalid on this host (no matching file), fall back to the public URL."""
    from python_chat.persistence import SupabaseClient

    runtime_client = _FakeSupabaseRuntimeClient()
    public_url = "http://host.docker.internal:54321/storage/v1/object/public/images/folder/x%20y.png"
    runtime_client.storage.bucket.get_public_url = lambda _: public_url  # type: ignore[method-assign]

    monkeypatch.setenv("SUPABASE_PUBLIC_IMAGE_BUCKET_MOUNT_PATH", str(tmp_path))
    monkeypatch.delenv("SUPABASE_PUBLIC_URL", raising=False)
    client = SupabaseClient(client=runtime_client)

    assert client.get_public_url(bucket="images", path="x/y.png") == public_url


def test_supabase_client_public_url_maps_directory_object_to_nested_file(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    from python_chat.persistence import SupabaseClient

    runtime_client = _FakeSupabaseRuntimeClient()
    runtime_client.storage.bucket.get_public_url = lambda _: (  # type: ignore[method-assign]
        "http://host.docker.internal:54321/storage/v1/object/public/images/2026/08/01/file.png"
    )

    object_dir = tmp_path / "2026" / "08" / "01" / "file.png"
    object_dir.mkdir(parents=True)
    nested_file = object_dir / "content"
    nested_file.write_bytes(b"png-bytes")

    monkeypatch.setenv("SUPABASE_PUBLIC_IMAGE_BUCKET_MOUNT_PATH", str(tmp_path))
    client = SupabaseClient(client=runtime_client)

    assert client.get_public_url(bucket="images", path="x/y.png") == str(nested_file)


def test_supabase_client_public_url_mount_path_blocks_path_traversal(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A crafted object key with '../' segments must not escape the mount root."""
    from python_chat.persistence import SupabaseClient

    mount_root = tmp_path / "mount"
    mount_root.mkdir()

    secret_file = tmp_path / "secret.txt"
    secret_file.write_text("top-secret-local-file-contents")

    runtime_client = _FakeSupabaseRuntimeClient()
    runtime_client.storage.bucket.get_public_url = lambda _: (  # type: ignore[method-assign]
        "http://host.docker.internal:54321/storage/v1/object/public/images/"
        "..%2f..%2fsecret.txt"
    )

    monkeypatch.setenv("SUPABASE_PUBLIC_IMAGE_BUCKET_MOUNT_PATH", str(mount_root))
    client = SupabaseClient(client=runtime_client)

    resolved = client.get_public_url(bucket="images", path="x/y.png")

    assert resolved is not None
    resolved_path = Path(resolved)
    if resolved_path.exists():
        assert (
            resolved_path.resolve() != secret_file.resolve()
        ), f"path traversal exposed local file outside mount root: {resolved_path}"


def test_supabase_client_safe_when_disabled_and_errors() -> None:
    from python_chat.persistence import SupabaseClient

    monkeypatch_env = {
        "SUPABASE_URL": None,
        "SUPABASE_SERVICE_ROLE_KEY": None,
    }
    for key, value in monkeypatch_env.items():
        if value is None:
            import os

            os.environ.pop(key, None)
    client = SupabaseClient(client=None)
    assert client.enabled is False
    client.upload_bytes(bucket="b", path="p", data=b"x", content_type="text/plain")
    assert client.get_public_url(bucket="b", path="p") is None

    broken_storage = SimpleNamespace(
        from_=lambda _: SimpleNamespace(
            get_public_url=MagicMock(side_effect=RuntimeError("boom"))
        )
    )
    broken = SupabaseClient(client=SimpleNamespace(storage=broken_storage))
    assert broken.get_public_url(bucket="b", path="p") is None


def test_supabase_client_auto_init_with_fake_supabase_module(monkeypatch: Any) -> None:
    import sys
    from types import ModuleType

    from python_chat.persistence import SupabaseClient

    fake_module = ModuleType("supabase")

    class _Client:
        pass

    def _create_client(url: str, key: str) -> _Client:
        del url, key
        return _Client()

    fake_module.Client = _Client  # type: ignore[attr-defined]
    fake_module.create_client = _create_client  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "supabase", fake_module)
    monkeypatch.setenv("SUPABASE_URL", "https://example.test")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")

    client = SupabaseClient()
    assert client.enabled is True


def test_supabase_client_init_warns_on_unexpected_type_and_handles_import_error(
    monkeypatch: Any,
) -> None:
    import sys
    from types import ModuleType

    from python_chat.persistence import SupabaseClient

    fake_bad_module = ModuleType("supabase")

    class _Client:
        pass

    def _create_client_bad(_: str, __: str) -> object:
        return object()

    fake_bad_module.Client = _Client  # type: ignore[attr-defined]
    fake_bad_module.create_client = _create_client_bad  # type: ignore[attr-defined]

    logger_warning = MagicMock()
    monkeypatch.setattr("python_chat.persistence.logger.warning", logger_warning)
    monkeypatch.setitem(sys.modules, "supabase", fake_bad_module)
    monkeypatch.setenv("SUPABASE_URL", "https://example.test")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")

    bad_type_client = SupabaseClient()
    assert bad_type_client.enabled is True
    assert logger_warning.called

    original_import = __import__

    def _raise_import(
        _: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ) -> Any:
        if _ == "supabase":
            raise ImportError("missing")
        return original_import(_, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", _raise_import)
    import_fail_client = SupabaseClient(url="https://example.test", key="service-key")
    assert import_fail_client.enabled is True


def test_start_worker_and_local_image_copy(tmp_path: Path) -> None:
    from pathlib import Path

    from python_chat.persistence import ensure_local_image_copy, start_log_worker

    queue = MagicMock()
    queue.start = MagicMock()

    returned = start_log_worker(queue)
    assert returned is queue
    queue.start.assert_called_once()

    out = ensure_local_image_copy(str(tmp_path), "img.png", b"abc")
    assert Path(out).exists()
    assert Path(out).read_bytes() == b"abc"


def test_supabase_rpc_client_property_exposes_client() -> None:
    from python_chat.persistence import SupabaseClient

    class _RuntimeRpcClient(_FakeSupabaseRuntimeClient):
        def rpc(self, fn: str, params: dict[str, Any]) -> _FakeRpcCall:
            del fn, params
            return _FakeRpcCall(None)

    runtime_client = _RuntimeRpcClient()
    client = SupabaseClient(client=runtime_client)
    assert client.rpc_client is runtime_client


class _FlakyRpcClient:
    def __init__(self) -> None:
        self.calls = 0

    def rpc(self, fn: str, params: dict[str, Any]) -> Any:
        del fn, params
        self.calls += 1
        if self.calls < 3:
            raise RuntimeError("temporary")
        return SimpleNamespace(data=55)


class _AlwaysFailRpcClient:
    def rpc(self, fn: str, params: dict[str, Any]) -> Any:
        del fn, params
        raise RuntimeError("always")


def test_db_extract_scalar_id_variants() -> None:
    assert db._extract_scalar_id(None) is None
    assert db._extract_scalar_id(5) == 5
    assert db._extract_scalar_id([7]) == 7
    assert db._extract_scalar_id([{"id": "8"}]) == 8
    assert db._extract_scalar_id({"id": 9}) == 9
    assert db._extract_scalar_id({"id": "bad"}) is None


def test_db_rpc_with_retry_success_and_exhaustion(monkeypatch: Any) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(db, "_extract_scalar_id", db._extract_scalar_id)
    monkeypatch.setattr(db, "time", SimpleNamespace(sleep=lambda s: sleeps.append(s)))

    result = db._rpc_with_retry(_FlakyRpcClient(), "fn", {}, retries=3)
    assert result == 55
    assert sleeps == [0.2, 0.4]

    with pytest.raises(RuntimeError, match="always"):
        db._rpc_with_retry(_AlwaysFailRpcClient(), "fn", {}, retries=2)


def test_db_complete_chat_session_invokes_rpc(monkeypatch: Any) -> None:
    called: list[tuple[str, dict[str, Any]]] = []

    def _rpc_with_retry(
        _: Any, function_name: str, params: dict[str, Any], **__: Any
    ) -> None:
        called.append((function_name, params))

    monkeypatch.setattr(db, "_rpc_with_retry", _rpc_with_retry)

    db.complete_chat_session(
        MagicMock(), session_id=12, ended_at="2026-01-01T00:00:00Z"
    )
    assert called[0][0] == "complete_chat_session"


def test_get_secret_from_vault_handles_none_and_whitespace(monkeypatch: Any) -> None:
    monkeypatch.setattr(db, "_rpc_with_retry", lambda *_args, **_kwargs: None)
    assert db.get_secret_from_vault(MagicMock(), "XAI_API_KEY") is None

    monkeypatch.setattr(db, "_rpc_with_retry", lambda *_args, **_kwargs: "  ")
    assert db.get_secret_from_vault(MagicMock(), "XAI_API_KEY") is None


def test_async_log_queue_covers_timeout_empty_flush_and_upload_failure() -> None:
    class _UploadFailClient:
        def upload_bytes(self, **_: Any) -> None:
            raise RuntimeError("upload-fail")

    queue = AsyncLogQueue(_UploadFailClient(), flush_interval_seconds=0.01, max_batch_size=1)  # type: ignore[arg-type]

    queue._flush_batch([])
    queue._flush_batch([build_log_event(None, {"a": 1})])
    queue.enqueue(build_log_event(1, {"x": 1}))
    queue.start()
    queue.stop()


def test_supabase_client_rpc_role_routing(monkeypatch: Any) -> None:
    from python_chat.persistence import SupabaseClient

    class _RuntimeRpcClient(_FakeSupabaseRuntimeClient):
        def rpc(self, fn: str, params: dict[str, Any]) -> _FakeRpcCall:
            del fn, params
            return _FakeRpcCall(None)

    runtime_client = _RuntimeRpcClient()
    client = SupabaseClient(client=runtime_client)

    assert client.rpc_client_for_role("service") is runtime_client
    assert client.rpc_client_for_role("user", access_token=None) is None

    monkeypatch.setenv("SUPABASE_URL", "https://example.test")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")

    with_user = SupabaseClient(client=runtime_client)
    user_rpc_one = with_user.rpc_client_for_role("user", access_token="token-1")
    user_rpc_two = with_user.rpc_client_for_role("user", access_token="token-1")

    assert user_rpc_one is not None
    assert user_rpc_two is user_rpc_one
