import code
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Callable, Generator, Optional

from xai_sdk import Client
from xai_sdk.tools import web_search, code_execution
from xai_sdk.proto import chat_pb2

from python_chat.api import Models
from python_chat.tools import Tools, ToolResult


class ModelContext:
    _instance: Optional["ModelContext"] = None

    def __init__(self, client: Client, image_path: str):
        self.model_name = Models.QUESTIONS
        self._client = client
        self._image_path = image_path
        self._tools = Tools()
        self._additional_tools: list[Any] = []

    @classmethod
    def create(cls, client: Client, image_path: str) -> "ModelContext":
        """Initialize or return the singleton ModelContext instance.

        Args:
            client (Client): The xAI SDK client instance used for API calls.
            image_path (str): Directory path where generated images will be saved.

        Returns:
            ModelContext: The singleton ModelContext instance.
        """
        if cls._instance is None:
            cls._instance = cls(client, image_path)
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

    def register_tool(
        self,
        func: Callable[..., Any],
        description: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """Register a callable as a tool for the LLM to use.

        Args:
            func: The function to register.
            description: Optional description (falls back to docstring).
            name: Optional name override (defaults to func.__name__).
        """
        self._tools.register_tool(func, description=description, name=name)

    def remove_tool(self, func: Callable[..., Any]) -> None:
        """Remove a registered tool from the tool registry."""
        self._tools.remove_tool(func)

    def handle_tool_calls(
        self, tool_calls: list[chat_pb2.ToolCall]
    ) -> tuple[list[dict[str, Any]], list[ToolResult | None]]:
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
