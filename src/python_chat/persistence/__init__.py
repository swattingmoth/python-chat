from __future__ import annotations

import json
import logging
import os
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal
from urllib import error as urllib_error
from urllib import request as urllib_request
from uuid import uuid4

from python_chat.persistence import db
from python_chat.persistence.models import QueueEvent

logger = logging.getLogger(__name__)

PersistenceRole = Literal["service", "user"]


def _normalize_config_value(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = value.strip()
    return normalized if normalized else None


class HttpRpcClient:
    """Simple PostgREST RPC client for explicit role-scoped execution."""

    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        access_token: str | None = None,
    ) -> None:
        self._url = url.rstrip("/")
        self._api_key = api_key
        self._access_token = access_token

    def rpc(self, fn: str, params: dict[str, Any]) -> "HttpRpcCall":
        return HttpRpcCall(
            url=self._url,
            fn=fn,
            params=params,
            api_key=self._api_key,
            access_token=self._access_token,
        )


class HttpRpcCall:
    def __init__(
        self,
        *,
        url: str,
        fn: str,
        params: dict[str, Any],
        api_key: str,
        access_token: str | None,
    ) -> None:
        self._url = url
        self._fn = fn
        self._params = params
        self._api_key = api_key
        self._access_token = access_token

    def execute(self) -> Any:
        endpoint = f"{self._url}/rest/v1/rpc/{self._fn}"
        body = json.dumps(self._params).encode("utf-8")
        bearer_token = self._access_token or self._api_key
        headers = {
            "Content-Type": "application/json",
            "apikey": self._api_key,
            "Authorization": f"Bearer {bearer_token}",
        }
        request = urllib_request.Request(
            endpoint,
            data=body,
            headers=headers,
            method="POST",
        )

        try:
            with urllib_request.urlopen(request, timeout=10) as response:
                payload = response.read().decode("utf-8")
                data = json.loads(payload) if payload else None
                return SimpleNamespace(data=data)
        except urllib_error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"RPC '{self._fn}' failed with status {exc.code}: {details}"
            ) from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(
                f"RPC '{self._fn}' request failed: {exc.reason}"
            ) from exc


class SupabaseClient:
    """Thin wrapper around supabase-py with graceful no-op fallbacks."""

    def __init__(
        self,
        *,
        url: str | None = None,
        key: str | None = None,
        anon_key: str | None = None,
        user_id: str | None = None,
        client: Any = None,
    ) -> None:
        self._url = _normalize_config_value(url or os.getenv("SUPABASE_URL"))
        logger.info(f"Initializing SupabaseClient with URL: {self._url}")
        self._key = _normalize_config_value(
            key or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        )
        if self._key and self._key.startswith("sb_secret_"):
            logger.info("Retrieved service role key for Supabase client.")
        self._anon_key = _normalize_config_value(
            anon_key or os.getenv("SUPABASE_ANON_KEY")
        )
        if self._anon_key and self._anon_key.startswith("sb_publishable"):
            logger.info("Retrieved anon key for Supabase client.")

        self.user_id = _normalize_config_value(
            user_id or os.getenv("SUPABASE_AUTH_USER_ID")
        )
        self._client = client
        self._user_rpc_clients: dict[str, HttpRpcClient] = {}
        self._service_rpc_client: Any = None

        if self._client is None and self._url and self._key:
            try:
                from supabase import Client, create_client

                self._client = create_client(self._url, self._key)
                if not isinstance(self._client, Client):
                    logger.warning(
                        "Unexpected Supabase client type: %s", type(self._client)
                    )
                logger.info("Supabase client initialized successfully.")
            except Exception as exc:
                logger.warning("Failed to initialize Supabase client: %s", exc)
                self._client = None

        if self._client is not None and hasattr(self._client, "rpc"):
            self._service_rpc_client = self._client
        elif self._url and self._key:
            self._service_rpc_client = HttpRpcClient(url=self._url, api_key=self._key)
            logger.info("Initialized service RPC client using HttpRpcClient.")

    @property
    def enabled(self) -> bool:
        return self._service_rpc_client is not None

    @property
    def rpc_client(self) -> Any:
        return self.rpc_client_for_role("service")

    def rpc_client_for_role(
        self,
        role: PersistenceRole,
        *,
        access_token: str | None = None,
    ) -> Any:
        if role == "service":
            return self._service_rpc_client

        normalized_token = _normalize_config_value(access_token)

        if not self._url or not self._anon_key or not normalized_token:
            return None

        cached = self._user_rpc_clients.get(normalized_token)
        if cached is not None:
            return cached

        scoped_client = HttpRpcClient(
            url=self._url,
            api_key=self._anon_key,
            access_token=normalized_token,
        )
        self._user_rpc_clients[normalized_token] = scoped_client
        return scoped_client

    def upload_bytes(
        self,
        *,
        bucket: str,
        path: str,
        data: bytes,
        content_type: str,
        upsert: bool = True,
    ) -> None:
        if not self._client:
            return

        options = {"content-type": content_type, "upsert": str(upsert).lower()}
        self._client.storage.from_(bucket).upload(
            path=path, file=data, file_options=options  # pyright: ignore
        )

    def get_public_url(self, *, bucket: str, path: str) -> str | None:
        if not self._client:
            return None

        try:
            response = self._client.storage.from_(bucket).get_public_url(path)
            if isinstance(response, str):
                return response
            if isinstance(response, dict):
                maybe_url = response.get("publicUrl")
                if isinstance(maybe_url, str):
                    return maybe_url
                return None
            return None
        except Exception as exc:
            logger.warning("Failed to get public URL for %s/%s: %s", bucket, path, exc)
            return None


class AsyncLogQueue:
    """Batches detailed JSONL logs and uploads them to Supabase storage."""

    _stop_signal = object()

    def __init__(
        self,
        supabase_client: SupabaseClient,
        *,
        bucket: str = "chat-logs",
        flush_interval_seconds: float = 60.0,
        max_batch_size: int = 100,
    ) -> None:
        self._supabase_client = supabase_client
        self._bucket = bucket
        self._flush_interval_seconds = flush_interval_seconds
        self._max_batch_size = max_batch_size
        self._queue: queue.Queue[Any] = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

    def enqueue(self, event: QueueEvent) -> None:
        self._queue.put_nowait(event)

    def start(self) -> None:
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._worker,
                name="chat-log-worker",
                daemon=True,
            )
            self._worker_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._queue.put_nowait(self._stop_signal)
        if self._worker_thread is not None:
            worker_thread = self._worker_thread
            self._worker_thread = None
            worker_thread.join()

    def _worker(self) -> None:
        pending: list[QueueEvent] = []

        while not self._stop_event.is_set() or not self._queue.empty() or pending:
            if not pending:
                try:
                    item = self._queue.get(timeout=self._flush_interval_seconds)
                    if item is self._stop_signal:
                        break
                    pending.append(item)
                except queue.Empty:
                    continue

            while len(pending) < self._max_batch_size and not self._queue.empty():
                try:
                    pending.append(self._queue.get_nowait())
                except queue.Empty:
                    break

            now = datetime.now(timezone.utc)
            if pending and self._should_flush(pending, now):
                self._flush_batch(pending)
                pending = []
                continue

            if pending and not self._stop_event.is_set():
                remaining_seconds = self._remaining_flush_seconds(pending, now)
                try:
                    item = self._queue.get(timeout=remaining_seconds)
                    if item is self._stop_signal:
                        break
                    pending.append(item)
                except queue.Empty:
                    pass

        if pending:
            self._flush_batch(pending)

    def _oldest_pending_age_seconds(
        self, batch: list[QueueEvent], now: datetime
    ) -> float:
        oldest = batch[0].occurred_at
        return max(0.0, (now - oldest).total_seconds())

    def _remaining_flush_seconds(self, batch: list[QueueEvent], now: datetime) -> float:
        age_seconds = self._oldest_pending_age_seconds(batch, now)
        return max(0.0, self._flush_interval_seconds - age_seconds)

    def _should_flush(self, batch: list[QueueEvent], now: datetime) -> bool:
        return (
            len(batch) >= self._max_batch_size
            or self._oldest_pending_age_seconds(batch, now)
            >= self._flush_interval_seconds
            or self._stop_event.is_set()
        )

    def _flush_batch(self, batch: list[QueueEvent]) -> None:
        if not batch:
            return

        serialized = "\n".join(
            json.dumps(item.model_dump(mode="json")) for item in batch
        )
        serialized = f"{serialized}\n"

        first = batch[0]
        day_path = datetime.now(timezone.utc).strftime("%Y/%m/%d")
        session_part = (
            str(first.session_id) if first.session_id is not None else "unknown"
        )
        object_name = f"{day_path}/{session_part}/{uuid4()}.jsonl"

        try:
            self._supabase_client.upload_bytes(
                bucket=self._bucket,
                path=object_name,
                data=serialized.encode("utf-8"),
                content_type="application/x-ndjson",
            )
        except Exception as exc:
            logger.warning("Failed to upload chat log batch: %s", exc)


def start_log_worker(queue: AsyncLogQueue) -> AsyncLogQueue:
    queue.start()
    return queue


def build_log_event(session_id: int | None, payload: dict[str, Any]) -> QueueEvent:
    return QueueEvent(session_id=session_id, payload=payload)


def ensure_local_image_copy(directory: str, filename: str, content: bytes) -> str:
    """Write a local image copy for immediate UI display and debugging."""
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / filename
    file_path.write_bytes(content)
    return str(file_path)


__all__ = [
    "AsyncLogQueue",
    "SupabaseClient",
    "build_log_event",
    "db",
    "ensure_local_image_copy",
    "start_log_worker",
]
