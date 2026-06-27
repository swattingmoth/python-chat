from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from logging import Logger
import logging
from typing import Any, Callable, Generator, Optional, Sequence
import os
from functools import wraps

# Disable OpenTelemetry tracing early to prevent context token issues
os.environ.setdefault("OTEL_TRACES_EXPORTER", "none")
os.environ.setdefault("OTEL_METRICS_EXPORTER", "none")

from google.protobuf import json_format
from PIL import Image

from xai_sdk.chat import Chunk, Response, assistant, system, tool_result, user
from xai_sdk.tools import get_tool_call_type

from python_chat.api import Models
from python_chat.context import ModelContext
from python_chat.persistence import db
from python_chat.persistence.models import ChatMessage, ToolCall
from python_chat.tools import ToolResult
from xai_sdk.proto import chat_pb2

logger = logging.getLogger(__name__)

# Suppress OpenTelemetry context detach errors that occur during stream cleanup
# This is a known issue when streaming crosses async/thread boundaries
logging.getLogger("opentelemetry.context").setLevel(logging.CRITICAL)

# Monkey-patch OpenTelemetry's context detach to silently ignore cross-context errors
try:
    from opentelemetry.context import contextvars_context

    _original_detach = contextvars_context.ContextVarsRuntimeContext.detach

    @wraps(_original_detach)
    def _patched_detach(self: Any, token: Any) -> None:
        """Patched detach that silently ignores cross-context token errors."""
        try:
            _original_detach(self, token)
        except ValueError as e:
            if "was created in a different Context" in str(e):
                # This error occurs when stream cleanup happens in a different context
                # It's safe to ignore as the context is being abandoned anyway
                logger.debug(f"Ignoring cross-context detach error: {e}")
            else:
                raise

    # Bypass type checker by using setattr via getattr to avoid assignment to method
    setattr(contextvars_context.ContextVarsRuntimeContext, "detach", _patched_detach)
except ImportError:
    # OpenTelemetry not available, no need to patch
    pass
    # The exact Response/Chunk types are runtime objects from the xai stream();
    # we use Any at runtime to avoid importing heavy protobuf wrappers in type position.
    pass

SYSTEM_MESSAGE: str = (
    "You are a helpful assistant. Only answer factually, and if you do not know the answer don't make anything up."
)

ChatCompleter = Callable[
    [str, Sequence[chat_pb2.Message], list[Any]],
    Generator[tuple[Response, Chunk], None, None],
]
"""Callable that performs a chat completion and yields (response, chunk) pairs.

The first item in each yielded tuple is the (partial or final) Response object;
the second is the incremental Chunk. Tool calls appear on the final Response.
"""

ToolHandler = Callable[[list[chat_pb2.ToolCall]], list[ToolResult]]


def get_model_for_choice(choice: str) -> str:
    """Map dropdown choice to appropriate model."""
    if choice == "Question":
        return Models.QUESTIONS
    elif choice == "Complex Question":
        return Models.COMPLEX_QUESTIONS
    elif choice == "Generate Image":
        return Models.QUESTIONS
    return Models.QUESTIONS


def get_system_message_for_choice(choice: str) -> str:
    """Get system message based on choice."""
    if choice == "Generate Image":
        return """You are an assistant that helps generate images based on user requests.
             The user will provide a description of the image they want and you will help them refine that description if needed, and then call the image generation tool with the final prompt.
             Call the image generation tool at most once per user request. If the request violates the content guidelines, do not call the tool at all; instead refuse the request and explain which part violates the guidelines.
             Do not issue multiple image generation tool calls for the same request.
             Always follow the content guidelines for image generation: the user must not ask for an image that contains nudity, suggestive content, or graphic violence. However, acts of affection (e.g. hugging, kissing), display of weapons (e.g. swords, guns, knives), preparation for warfare (e.g. building fortifications, assembling troops) are acceptable.
             If the user asks for an image but does not provide enough details, ask them for more information and suggest additional details.
             Only tell the user that you are calling the image generation tool when you are actually making a tool call. If the prompt that you pass to the tool is different from the user's original prompt, include the final prompt in your message to the user.

             Examples:
                User: I want a picture of a dog.
                Assistant: Can you provide more details about the dog picture you want? For example, what breed of dog, what setting or background, any specific colors or actions you want the dog to be doing?

                User: I would like a picture of a man with dark hair and skin wearing armor and holding a sword in an outstretched hand. The man is standing on a desolate battlefield with smoke in the background.
                Assistant: Calling the image generation tool.
                Tool Call: Generated Image

                User: I want a picture of a man with his head cut off.
                Assistant: I'm sorry, but I cannot generate that image because it violates the content guidelines regarding graphic violence. Specifically, the request for a picture of a man with his head cut off is not something I can assist with. Please let me know if you have another image request that follows the guidelines.

                User: I want a picture of a woman in a bikini.
                Assistant: I'm sorry, but I cannot generate that image because it violates the content guidelines regarding suggestive content. Specifically, the request for a picture of a woman in a bikini is not something I can assist with. Please let me know if you have another image request that follows the guidelines.
             """
    return SYSTEM_MESSAGE


ROLE_MAP: dict[chat_pb2.MessageRole, str] = {
    chat_pb2.MessageRole.ROLE_ASSISTANT: "assistant",
    chat_pb2.MessageRole.ROLE_USER: "user",
    chat_pb2.MessageRole.ROLE_SYSTEM: "system",
    chat_pb2.MessageRole.ROLE_TOOL: "tool",
}


def message_to_dict(msg: chat_pb2.Message) -> dict[str, Any]:
    """Convert a chat_pb2.Message into a dict for UI consumption."""
    role = ROLE_MAP.get(msg.role, "unknown")
    content = "\n".join([c.text or "" for c in msg.content or ""])  # type: ignore

    if msg.tool_calls:
        return {
            "role": role,
            "content": content,
            "tool_calls": [json_format.MessageToDict(tc) for tc in msg.tool_calls],
        }

    return {"role": role, "content": content}


class ChatInterface:
    SYSTEM_MESSAGE = SYSTEM_MESSAGE

    def __init__(
        self,
        modelContext: ModelContext,
        *,
        completer: ChatCompleter | None = None,
        tool_handler: ToolHandler | None = None,
    ) -> None:
        """Create a chat interface that streams responses and manages tool usage.

        The optional completer and tool_handler parameters are provided to support
        unit testing by injecting fake streaming responses and tool results,
        avoiding real API calls and side-effecting tool execution.

        Args:
            modelContext (ModelContext): The shared model context used for API calls and tools
                (still required for default behavior and tool registry access in production).
            completer (ChatCompleter | None): Test-only override for the chat completion
                streaming function.
            tool_handler (ToolHandler | None): Test-only override for executing tool calls
                and returning responses/results.
        """
        self.chat_history: list[chat_pb2.Message] = []
        self.modelContext = modelContext
        self.image: Optional[Image.Image] = None
        self._completer: ChatCompleter = completer or self._default_completer
        self._tool_handler: ToolHandler = tool_handler or self._default_tool_handler

    def _persist_message(
        self,
        *,
        session_id: int | None,
        role: str,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
        estimated_cost: float | None = None,
        estimated_tokens: int | None = None,
    ) -> int | None:
        if not isinstance(session_id, int):
            return None
        if not self.modelContext.persistence_client:
            return None
        if not self.modelContext.persistence_client.enabled:
            return None

        try:
            created = db.create_chat_message(
                self.modelContext.persistence_client.rpc_client,
                ChatMessage(
                    session_id=session_id,
                    role=role,
                    content=content,
                    tool_calls=tool_calls,
                    estimated_tokens=estimated_tokens,
                    estimated_cost=estimated_cost,
                    started_at=datetime.now(timezone.utc),
                ),
            )
            return created
        except Exception as exc:
            logger.warning("Failed to persist message: %s", exc)
            return None

    def _persist_tool_call(
        self,
        *,
        message_id: int | None,
        tool_name: str,
        status: str,
        input_args: dict[str, Any] | None,
        output_result: dict[str, Any] | None,
        error_message: str | None,
        latency_ms: int | None,
    ) -> None:
        if not isinstance(message_id, int):
            return
        if not self.modelContext.persistence_client:
            return
        if not self.modelContext.persistence_client.enabled:
            return

        try:
            db.create_tool_call(
                self.modelContext.persistence_client.rpc_client,
                ToolCall(
                    message_id=message_id,
                    tool_name=tool_name,
                    status=status,
                    input_args=input_args,
                    output_result=output_result,
                    error_message=error_message,
                    latency_ms=latency_ms,
                ),
            )
        except Exception as exc:
            logger.warning("Failed to persist tool call: %s", exc)

    def _default_completer(
        self, model: str, messages: Sequence[chat_pb2.Message], tools: list[Any]
    ) -> Generator[tuple[Response, Chunk], None, None]:
        """Default chat completion using the xAI SDK client.

        Builds xai message protos from our dict history, calls client.chat.create,
        then yields (response, chunk) pairs from the stream. Tool calls (if any)
        are present on the final Response object after iteration completes.
        """
        chat = self.modelContext.client.chat.create(
            model=model,
            messages=messages,
            tools=tools,
        )
        for response, chunk in chat.stream():
            yield response, chunk

    def _default_tool_handler(
        self, message: list[chat_pb2.ToolCall]
    ) -> list[ToolResult]:
        """Default tool execution using the model context's registered tools."""
        return self.modelContext.handle_tool_calls(message)

    def append_to_history(self, message: chat_pb2.Message) -> None:
        """Append a message to the chat history."""
        self.chat_history.append(message)
        logger.info(f"Appended message to history: {message_to_dict(message)}")

    def chat(
        self, message: str, chat_history: list[dict[str, Any]], choice: str
    ) -> Generator[Any, Any, Any]:
        """Stream chat responses token by token and handle tool calls.

        Args:
            message (str): The user's input message.
            chat_history (list[dict[str, Any]]): Current chat history from the UI.
            choice (str): Selected chat mode, such as question or image generation.

        Yields:
            tuple: Intermediate chat history and optional image data while streaming.
        """
        if not message:
            yield chat_history, self.image
            return

        new_model = get_model_for_choice(choice)
        system_message = get_system_message_for_choice(choice)
        is_image_mode = choice == "Generate Image"

        if self.modelContext.model_name != new_model:
            self.modelContext.model_name = get_model_for_choice(choice)
            logger.info(
                f"Set model to {self.modelContext.model_name}, is_image_mode={is_image_mode}"
            )
            logger.info(f"System message:\n{system_message}")

        try:
            session_id = self.modelContext.ensure_session(choice)

            # Add user message to history
            self.append_to_history(user(message))
            self._persist_message(
                session_id=session_id,
                role="user",
                content=message,
            )
            self.modelContext.enqueue_log_event(
                {
                    "event": "user_message",
                    "session_id": session_id,
                    "choice": choice,
                    "content": message,
                }
            )
            local_chat_history = [
                message_to_dict(c) for c in self.chat_history
            ]  # Create a local copy for this interaction
            yield local_chat_history + [
                {
                    "role": "assistant",
                    "content": "",
                    "metadata": {"title": "Thinking...", "status": "pending"},
                }
            ], self.image  # Yield initial state with user message added

            while True:
                # Build messages with system context
                messages = [system(system_message)]
                messages.extend(self.chat_history)

                # prevent the model from calling the image generation tool multiple times for a single response.
                tools = (
                    self.modelContext.get_tools_for_model()
                    if self.modelContext is not None
                    else []
                )
                stream = self._completer(self.modelContext.model_name, messages, tools)

                # Stream tokens; after completion, inspect final response for tool_calls
                client_tool_calls: list[chat_pb2.ToolCall] = []
                last_response: Optional[Response] = None
                for response, chunk in stream:
                    last_response = response
                    token = getattr(chunk, "content", None)
                    if token:
                        yield local_chat_history + [
                            message_to_dict(assistant(response.content))
                        ], self.image

                    for tool_call in chunk.tool_calls:
                        if get_tool_call_type(tool_call) == "client_side_tool":
                            client_tool_calls.append(tool_call)
                        yield local_chat_history + [
                            {
                                "role": "assistant",
                                "content": "",
                                "metadata": {
                                    "title": f"Calling {tool_call.function.name}...",
                                    "status": "pending",
                                },
                            }
                        ], self.image

                # Append this turn's assistant text (if any) to history
                assistant_message_id: int | None = None
                if last_response:
                    assistant_message = assistant(last_response.content)
                    assistant_message.tool_calls.extend(last_response.tool_calls)
                    self.append_to_history(assistant_message)

                    tool_call_dicts = [
                        json_format.MessageToDict(tc) for tc in last_response.tool_calls
                    ]
                    assistant_message_id = self._persist_message(
                        session_id=session_id,
                        role="assistant",
                        content=last_response.content,
                        tool_calls=tool_call_dicts if tool_call_dicts else None,
                        estimated_cost=last_response.cost_usd,
                        estimated_tokens=last_response.usage.total_tokens,
                    )
                    self.modelContext.enqueue_log_event(
                        {
                            "event": "assistant_message",
                            "session_id": session_id,
                            "content": last_response.content,
                            "tool_calls": tool_call_dicts,
                            "estimated_cost": last_response.cost_usd,
                        }
                    )

                    local_chat_history.append(message_to_dict(assistant_message))

                if client_tool_calls:
                    client_tool_calls_dict = {c.id: c for c in client_tool_calls}
                    tool_results = self._tool_handler(client_tool_calls)

                    for tr in tool_results:
                        tool_call_name = "unknown"
                        tool_call_args: dict[str, Any] | None = None
                        call = client_tool_calls_dict.get(tr.tool_call_id)
                        if call:
                            tool_call_name = call.function.name
                            if call.function.arguments:
                                try:
                                    parsed_args = json.loads(call.function.arguments)
                                    if isinstance(parsed_args, dict):
                                        tool_call_args = parsed_args
                                except json.JSONDecodeError:
                                    tool_call_args = {"raw": call.function.arguments}

                        output_payload = {
                            "content_type": tr.content_type,
                            "content_for_model": tr.content_for_model,
                            "metadata": tr.metadata,
                        }
                        self._persist_tool_call(
                            message_id=assistant_message_id,
                            tool_name=tool_call_name,
                            status="ok",
                            input_args=tool_call_args,
                            output_result=output_payload,
                            error_message=None,
                            latency_ms=None,
                        )
                        self.modelContext.enqueue_log_event(
                            {
                                "event": "tool_call",
                                "session_id": session_id,
                                "message_id": assistant_message_id,
                                "tool_name": tool_call_name,
                                "status": "ok",
                                "tool_call_id": tr.tool_call_id,
                                "output": output_payload,
                            }
                        )

                        if is_image_mode:
                            # Side-effect: capture image ToolResult for "Generate Image" mode
                            if tr and tr.content_type == "image" and tr.content:
                                try:
                                    self.image = Image.open(io.BytesIO(tr.content))
                                except Exception:
                                    pass
                            # Yield so the UI can display the image promptly
                            yield local_chat_history + [
                                message_to_dict(
                                    assistant(
                                        last_response.content if last_response else ""
                                    )
                                )
                            ], self.image

                        self.append_to_history(
                            tool_result(
                                tr.content_for_model, tool_call_id=tr.tool_call_id
                            )
                        )

                    if is_image_mode and self.image:
                        break
                else:
                    break

        except Exception as e:
            Logger(f"Error in chat: {e}")
            # Show generic error without details
            error_msg = (
                "I encountered an error processing your request. Please try again."
            )
            self.chat_history.append(assistant(error_msg))
            yield [message_to_dict(m) for m in self.chat_history], self.image

    def clear_history(self) -> list[dict[str, Any]]:
        """Clear stored chat history and reset the image output."""
        self.chat_history = []
        self.image = None
        logger.info("Cleared chat history and reset image.")
        return []
