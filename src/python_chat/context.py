import code
from datetime import datetime, timezone
from contextlib import contextmanager
import logging
from types import SimpleNamespace
from typing import Any, Callable, Generator, Optional

from xai_sdk import Client
from xai_sdk.tools import web_search, code_execution
from xai_sdk.proto import chat_pb2

from python_chat.api import Models
from python_chat.persistence import AsyncLogQueue, SupabaseClient, build_log_event, db
from python_chat.persistence.models import ChatSession
from python_chat.tools import Tools, ToolResult

logger = logging.getLogger(__name__)


class ModelContext:
    _instance: Optional["ModelContext"] = None

    def __init__(
        self,
        client: Client,
        image_path: str,
        *,
        persistence_client: SupabaseClient | None = None,
        log_queue: AsyncLogQueue | None = None,
        user_id: str | None = None,
    ):
        self.model_name = Models.QUESTIONS
        self._client = client
        self._image_path = image_path
        self._tools = Tools()
        self._additional_tools: list[Any] = []
        self._persistence_client = persistence_client
        self._log_queue = log_queue
        self._user_id = user_id
        self._session_id: int | None = None
        self._session_mode: str | None = None

    @classmethod
    def create(
        cls,
        client: Client,
        image_path: str,
        *,
        persistence_client: SupabaseClient | None = None,
        log_queue: AsyncLogQueue | None = None,
        user_id: str | None = None,
    ) -> "ModelContext":
        """Initialize or return the singleton ModelContext instance.

        Args:
            client (Client): The xAI SDK client instance used for API calls.
            image_path (str): Directory path where generated images will be saved.

        Returns:
            ModelContext: The singleton ModelContext instance.
        """
        if cls._instance is None:
            cls._instance = cls(
                client,
                image_path,
                persistence_client=persistence_client,
                log_queue=log_queue,
                user_id=user_id,
            )
        return cls._instance

    @classmethod
    def current(cls) -> "ModelContext":
        """Return the currently initialized ModelContext instance.

        Raises:
            Exception: If the ModelContext has not been initialized.

        Returns:
            ModelContext: The current ModelContext instance.
        """
        if cls._instance:
            return cls._instance

        raise Exception(
            "ModelContext has not been initialized. Call ModelContext.create(client, image_path) first."
        )

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance.

        Intended primarily for tests to allow fresh initialization between cases
        without affecting production singleton semantics.
        """
        cls._instance = None

    @property
    def model_name(self) -> str:
        """Get the name of the current model in use."""
        return self._model_name

    @model_name.setter
    def model_name(self, value: str) -> None:
        """Set the current model name."""
        self._validate_model(value)
        self._model_name = value
        if self.model_name == Models.COMPLEX_QUESTIONS:
            # Use the native server-side web_search tool for complex questions.
            # The tool is executed inside the model's request (cost_usd covers it).
            self._additional_tools = [web_search(), code_execution()]
        else:
            self._additional_tools = []

    def _validate_model(self, model_name: str) -> None:
        if model_name not in [
            Models.IMAGES,
            Models.QUESTIONS,
            Models.COMPLEX_QUESTIONS,
        ]:
            raise ValueError(f"Invalid model name: {model_name}")

    @property
    def client(self) -> Client:
        """Get the xAI SDK client used by this context."""
        return self._client

    @property
    def image_path(self) -> str:
        """Get the directory path where generated images are stored."""
        return self._image_path

    @property
    def tools(self) -> "Tools":
        """Get the registered tool manager."""
        return self._tools

    @property
    def persistence_client(self) -> SupabaseClient | None:
        return self._persistence_client

    @property
    def log_queue(self) -> AsyncLogQueue | None:
        return self._log_queue

    @property
    def session_id(self) -> int | None:
        return self._session_id

    @property
    def user_id(self) -> str | None:
        return self._user_id

    def set_log_queue(self, queue: AsyncLogQueue | None) -> None:
        self._log_queue = queue

    def set_user_id(self, user_id: str | None) -> None:
        self._user_id = user_id

    def ensure_session(self, mode: str) -> int | None:
        """Ensure a chat session exists in the database for the current user and mode."""
        if self._session_id is not None and self._session_mode == mode:
            return self._session_id

        if not self._persistence_client or not self._persistence_client.enabled:
            return None

        if not self._user_id:
            return None

        try:
            created_session = ChatSession(user_id=self._user_id, mode=mode)
            new_session_id = db.create_chat_session(
                self._persistence_client.rpc_client,
                created_session,
            )
            self._session_id = new_session_id
            self._session_mode = mode
            return new_session_id
        except Exception as exc:
            logger.warning("Failed to create session: %s", exc)
            return None

    def complete_session(self) -> None:
        if not self._persistence_client or not self._persistence_client.enabled:
            return
        if self._session_id is None:
            return

        try:
            db.complete_chat_session(
                self._persistence_client.rpc_client,
                self._session_id,
                datetime.now(timezone.utc).isoformat(),
            )
        except Exception as exc:
            logger.warning("Failed to complete session %s: %s", self._session_id, exc)

    def enqueue_log_event(self, payload: dict[str, Any]) -> None:
        if not self._log_queue:
            return
        self._log_queue.enqueue(build_log_event(self._session_id, payload))

    def register_tool(
        self,
        func: Callable[..., Any],
        description: Optional[str] = None,
        name: Optional[str] = None,
        max_turns: Optional[int] = None,
    ) -> None:
        """Register a callable as a tool for the LLM to use.

        Args:
            func: The function to register.
            description: Optional description (falls back to docstring).
            name: Optional name override (defaults to func.__name__).
            max_turns: Optional maximum number of times the tool can be called.
        """
        self._tools.register_tool(
            func, description=description, name=name, max_turns=max_turns
        )

    def remove_tool(self, func: Callable[..., Any]) -> None:
        """Remove a registered tool from the tool registry."""
        self._tools.remove_tool(func)

    def handle_tool_calls(
        self, tool_calls: list[chat_pb2.ToolCall]
    ) -> list[ToolResult]:
        """Execute tool calls requested by the model and return their responses."""
        return self._tools.handle_tool_calls(tool_calls)

    def get_tools_for_model(self) -> list[Any]:
        """Return the tool objects (xai chat tool protos) formatted for model use."""
        return self._tools.get_tools_for_model(self._additional_tools)

    @contextmanager
    def use_model(self, model_name: str) -> Generator[None, None, None]:
        """Temporarily switch the current model within a context block."""
        self._validate_model(model_name)
        old_model = self.model_name
        self.model_name = model_name
        try:
            yield
        finally:
            self.model_name = old_model
