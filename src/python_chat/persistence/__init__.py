from __future__ import annotations

import queue
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import threading
from typing import Any
from uuid import uuid4

from python_chat.persistence import db
from python_chat.persistence.models import QueueEvent

logger = logging.getLogger(__name__)


class SupabaseClient:
    """Thin wrapper around supabase-py with graceful no-op fallbacks."""

    def __init__(
        self,
        *,
        url: str | None = None,
        key: str | None = None,
        user_id: str | None = None,
        client: Any = None,
    ) -> None:
        self._url = url or os.getenv("SUPABASE_URL")
        self._key = key or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self.user_id = user_id or os.getenv("SUPABASE_AUTH_USER_ID")
        self._client = client

        if self._client is None and self._url and self._key:
            try:
                from supabase import Client, create_client

                self._client = create_client(self._url, self._key)
                if not isinstance(self._client, Client):
                    logger.warning(
                        "Unexpected Supabase client type: %s", type(self._client)
                    )
            except Exception as exc:
                logger.warning("Failed to initialize Supabase client: %s", exc)
                self._client = None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @property
    def rpc_client(self) -> Any:
        return self._client

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
                return maybe_url if isinstance(maybe_url, str) else None
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
