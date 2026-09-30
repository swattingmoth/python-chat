from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Generator, Optional, Sequence, TypedDict

# Disable OpenTelemetry tracing early to prevent context token issues
os.environ.setdefault("OTEL_TRACES_EXPORTER", "none")
os.environ.setdefault("OTEL_METRICS_EXPORTER", "none")

from google.protobuf import json_format
from xai_sdk.chat import Chunk, Response, assistant, system, tool_result, user
from xai_sdk.proto import chat_pb2
from xai_sdk.tools import code_execution, get_tool_call_type, web_search

from python_chat.api import Models
from python_chat.context import ModelContext
from python_chat.persistence import db
from python_chat.persistence.models import ChatMessage, ToolCall
from python_chat.tools import RegisteredTool, ToolResult, Tools

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

ToolHandler = Callable[
    [list[RegisteredTool], list[chat_pb2.ToolCall]], list[ToolResult]
]


class SessionRuntime(TypedDict):
    user_id: str | None
    access_token: str | None
    session_id: int | None
    session_mode: str | None
    selected_choice: str
    model_name: str
    request_counter: int
    active_tools: list[RegisteredTool]
    chat_history: list[chat_pb2.Message]
    current_image_path: str | None


class ChatResult(TypedDict):
    history: list[dict[str, Any]]
    image_path: str | None
    runtime: SessionRuntime


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
    content = "\n".join([c.text or "" for c in msg.content])

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
        self.modelContext = modelContext
        self._completer: ChatCompleter = completer or self._default_completer
        self._tool_handler: ToolHandler = (
            tool_handler if tool_handler is not None else self._default_tool_handler
        )

    def _ensure_runtime(
        self, choice: str, runtime: SessionRuntime | None
    ) -> SessionRuntime:
        if runtime is None:
            return {
                "user_id": self.modelContext.user_id,
                "access_token": None,
                "session_id": None,
                "session_mode": None,
                "selected_choice": choice,
                "model_name": get_model_for_choice(choice),
                "request_counter": 0,
                "active_tools": [],
                "chat_history": [],
                "current_image_path": None,
            }

        resolved = dict(runtime)
        resolved.setdefault("user_id", self.modelContext.user_id)
        resolved.setdefault("access_token", None)
        resolved.setdefault("session_id", None)
        resolved.setdefault("session_mode", None)
        resolved.setdefault("selected_choice", choice)
        resolved.setdefault("model_name", get_model_for_choice(choice))
        resolved.setdefault("active_tools", [])
        resolved.setdefault("chat_history", [])
        resolved.setdefault("current_image_path", None)
        request_counter = resolved.get("request_counter", 0)
        resolved["request_counter"] = (
            request_counter + 1 if isinstance(request_counter, int) else 1
        )
        return resolved  # type: ignore[return-value]

    def _to_proto_history(
        self, history: list[dict[str, Any]]
    ) -> list[chat_pb2.Message]:
        messages: list[chat_pb2.Message] = []
        for item in history:
            role = item.get("role")
            content = item.get("content", "")
            if not isinstance(content, str):
                continue

            if role == "user":
                messages.append(user(content))
            elif role == "assistant":
                messages.append(assistant(content))
            elif role == "tool":
                tool_call_id = item.get("tool_call_id")
                if isinstance(tool_call_id, str) and tool_call_id:
                    messages.append(tool_result(content, tool_call_id=tool_call_id))
        return messages

    def _persist_message(
        self,
        *,
        session_id: int | None,
        role: str,
        content: str,
        access_token: str | None,
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

        user_rpc_client = self.modelContext.persistence_client.rpc_client_for_role(
            "user",
            access_token=access_token,
        )
        if user_rpc_client is None:
            logger.warning(
                "No user-scoped RPC client is available for message persistence"
            )
            return None

        try:
            created = db.create_chat_message(
                user_rpc_client,
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
        estimated_cost: float | None = None,
        access_token: str | None,
    ) -> None:
        if not isinstance(message_id, int):
            return
        if not self.modelContext.persistence_client:
            return
        if not self.modelContext.persistence_client.enabled:
            return

        user_rpc_client = self.modelContext.persistence_client.rpc_client_for_role(
            "user",
            access_token=access_token,
        )
        if user_rpc_client is None:
            logger.warning(
                "No user-scoped RPC client is available for tool persistence"
            )
            return

        try:
            db.create_tool_call(
                user_rpc_client,
                ToolCall(
                    message_id=message_id,
                    tool_name=tool_name,
                    status=status,
                    input_args=input_args,
                    output_result=output_result,
                    error_message=error_message,
                    latency_ms=latency_ms,
                    estimated_cost=estimated_cost,
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
        self, active_tools: list[RegisteredTool], message: list[chat_pb2.ToolCall]
    ) -> list[ToolResult]:
        """Default tool execution using the model context's registered tools."""
        return Tools.handle_tool_calls(active_tools, message)

    def _update_history(
        self,
        session_runtime: SessionRuntime,
        history: list[chat_pb2.Message],
        message: chat_pb2.Message,
    ) -> None:
        """Append a message to the chat history."""
        history.append(message)
        session_runtime["chat_history"] = history
        logger.debug(f"Appended message to history: {message_to_dict(message)}")

    def get_history(
        self,
        history: list[chat_pb2.Message],
        thinking: bool = False,
        message: chat_pb2.Message | None = None,
        calling_function: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve the chat history as a list of dictionaries."""
        history_to_return = [
            message_to_dict(msg)
            for msg in history
            if not msg.role == chat_pb2.MessageRole.ROLE_TOOL
        ]

        if message is not None:
            history_to_return.append(message_to_dict(message))

        if calling_function is not None:
            history_to_return.append(
                {
                    "role": "assistant",
                    "content": "",
                    "metadata": {
                        "title": f"Calling {calling_function}...",
                        "status": "pending",
                    },
                }
            )

        if thinking:
            history_to_return.append(
                {
                    "role": "assistant",
                    "content": "",
                    "metadata": {"title": "Thinking"},
                }
            )
        return history_to_return

    def chat(
        self,
        message: str,
        choice: str,
        *,
        runtime: SessionRuntime | None = None,
    ) -> Generator[ChatResult, None, None]:
        """Stream chat responses token by token and handle tool calls.

        Args:
            message (str): The user's input message.
            runtime (SessionRuntime | None): The current session runtime containing chat history and other session-specific data.
            choice (str): Selected chat mode, such as question or image generation.

        Yields:
            ChatResult: Intermediate chat history, optional image data, and updated session runtime.
        """
        session_runtime = self._ensure_runtime(choice, runtime)
        current_image_path = session_runtime.get("current_image_path")
        local_history = list(session_runtime.get("chat_history", []))
        if not message:
            yield {
                "history": self.get_history(local_history),
                "image_path": current_image_path,
                "runtime": session_runtime,
            }
            return

        new_model = get_model_for_choice(choice)
        system_message = get_system_message_for_choice(choice)
        is_image_mode = choice == "Generate Image"
        session_runtime["selected_choice"] = choice
        session_runtime["model_name"] = new_model

        logger.info(f"Set model to {new_model}, is_image_mode={is_image_mode}")
        logger.debug(f"System message:\n{system_message}")

        try:
            session_id, session_mode = self.modelContext.ensure_session_for(
                mode=choice,
                user_id=session_runtime.get("user_id"),
                current_session_id=session_runtime.get("session_id"),
                current_session_mode=session_runtime.get("session_mode"),
                access_token=session_runtime.get("access_token"),
            )
            session_runtime["session_id"] = session_id
            session_runtime["session_mode"] = session_mode

            # Add user message to history
            self._update_history(session_runtime, local_history, user(message))
            self._persist_message(
                session_id=session_id,
                role="user",
                content=message,
                access_token=session_runtime.get("access_token"),
            )
            self.modelContext.enqueue_log_event_for(
                session_id,
                {
                    "event": "user_message",
                    "session_id": session_id,
                    "choice": choice,
                    "content": message,
                },
            )

            yield {
                "history": self.get_history(local_history, thinking=True),
                "image_path": current_image_path,
                "runtime": session_runtime,
            }

            while True:
                # Build messages with system context
                messages = [system(system_message)]
                messages.extend(local_history)  # pyright: ignore

                # Gather tools: combine per-session active tools with model-based additional tools
                per_session_tools: list[RegisteredTool] = session_runtime.get(
                    "active_tools", []
                )
                additional_tools = [web_search(), code_execution()]
                if new_model != Models.COMPLEX_QUESTIONS:
                    additional_tools = []

                tools = Tools.get_tools_for_model(per_session_tools, additional_tools)
                stream = self._completer(new_model, messages, tools)

                # Stream tokens; after completion, inspect final response for tool_calls
                client_tool_calls: list[chat_pb2.ToolCall] = []
                last_response: Optional[Response] = None
                for response, chunk in stream:
                    last_response = response
                    token = getattr(chunk, "content", None)
                    if token:
                        yield {
                            "history": self.get_history(
                                local_history,  # pyright: ignore
                                message=assistant(response.content),
                            ),
                            "image_path": current_image_path,
                            "runtime": session_runtime,
                        }

                    for tool_call in chunk.tool_calls:
                        if get_tool_call_type(tool_call) == "client_side_tool":
                            client_tool_calls.append(tool_call)
                        yield {
                            "history": self.get_history(
                                local_history,  # pyright: ignore
                                calling_function=tool_call.function.name,
                            ),
                            "image_path": current_image_path,
                            "runtime": session_runtime,
                        }

                # Append this turn's assistant text (if any) to history
                assistant_message_id: int | None = None
                if last_response:
                    assistant_message = assistant(last_response.content)
                    assistant_message.tool_calls.extend(last_response.tool_calls)
                    self._update_history(
                        session_runtime,
                        local_history,
                        assistant_message,  # pyright: ignore
                    )

                    tool_call_dicts = [
                        json_format.MessageToDict(tc) for tc in last_response.tool_calls
                    ]
                    assistant_message_id = self._persist_message(
                        session_id=session_id,
                        role="assistant",
                        content=last_response.content,
                        access_token=session_runtime.get("access_token"),
                        tool_calls=tool_call_dicts if tool_call_dicts else None,
                        estimated_cost=last_response.cost_usd,
                        estimated_tokens=last_response.usage.total_tokens,
                    )

                    self.modelContext.enqueue_log_event_for(
                        session_id,
                        {
                            "event": "assistant_message",
                            "session_id": session_id,
                            "content": last_response.content,
                            "tool_calls": tool_call_dicts,
                            "estimated_cost": last_response.cost_usd,
                        },
                    )

                if client_tool_calls:
                    image_generated = False

                    client_tool_calls_dict = {c.id: c for c in client_tool_calls}
                    tool_results = self._tool_handler(
                        per_session_tools, client_tool_calls
                    )

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
                            estimated_cost=tr.cost,
                            access_token=session_runtime.get("access_token"),
                        )
                        self.modelContext.enqueue_log_event_for(
                            session_id,
                            {
                                "event": "tool_call",
                                "session_id": session_id,
                                "message_id": assistant_message_id,
                                "tool_name": tool_call_name,
                                "status": "ok",
                                "tool_call_id": tr.tool_call_id,
                                "output": output_payload,
                            },
                        )

                        if is_image_mode:
                            # Side-effect: capture image ToolResult for "Generate Image" mode
                            if (
                                tr
                                and tr.content_type == "image"
                                and tr.content
                                and isinstance(tr.content, str)
                            ):
                                current_image_path = tr.content
                                image_generated = True
                                session_runtime["current_image_path"] = (
                                    current_image_path
                                )

                        self._update_history(
                            session_runtime,
                            local_history,
                            tool_result(
                                tr.content_for_model, tool_call_id=tr.tool_call_id
                            ),
                        )

                    if is_image_mode:
                        yield {
                            "history": self.get_history(local_history),
                            "image_path": current_image_path,
                            "runtime": session_runtime,
                        }

                        if current_image_path and image_generated:
                            break
                else:
                    break

        except Exception as e:
            logger.exception(e)
            # Show generic error without details
            error_msg = (
                "I encountered an error processing your request. Please try again."
            )
            error_message = assistant(error_msg)
            self._update_history(session_runtime, local_history, error_message)

        yield {
            "history": self.get_history(local_history),
            "image_path": current_image_path,
            "runtime": session_runtime,
        }
