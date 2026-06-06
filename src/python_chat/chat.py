from __future__ import annotations

from http import client
import io
from json import tool
from types import SimpleNamespace
from typing import Any, Callable, Generator, Optional, TYPE_CHECKING, Sequence

from google.protobuf import json_format
from PIL import Image

import xai_sdk
from xai_sdk.chat import Chunk, Response, assistant, system, tool_result, user
import xai_sdk.chat
from xai_sdk.tools import get_tool_call_type
from xai_sdk.types import Content

from python_chat.api import Models
from python_chat.context import ModelContext
from python_chat.tools import ToolResult
from xai_sdk.proto import chat_pb2

if TYPE_CHECKING:
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


# def _to_xai_message(msg: dict[str, Any]) -> Any:
#     """Convert internal chat history dict into an xai_sdk chat message.

#     Handles system/user/tool_result and assistant messages.
#     For assistant messages carrying prior tool_calls (to continue a tool loop),
#     we construct a low-level chat_pb2.Message so that tool_calls are attached.
#     """
#     role = msg.get("role")
#     content: str = msg.get("content") or ""

#     if role == "system":
#         return system(content)
#     if role == "user":
#         return user(content)
#     if role == "tool":
#         tool_call_id: str = msg.get("tool_call_id", "")
#         return tool_result(tool_call_id=tool_call_id, result=content or "")
#     if role == "assistant":
#         assistant_msg = assistant(content)
#         tool_calls = msg.get("tool_calls")
#         if tool_calls:
#             for tc in tool_calls:
#                 if isinstance(tc, dict):
#                     fid = tc.get("id", "")
#                     fn = tc.get("function", {}) or {}
#                     fname = fn.get("name", "") if isinstance(fn, dict) else ""
#                     fargs = fn.get("arguments", "") if isinstance(fn, dict) else ""
#                 else:
#                     fid = getattr(tc, "id", "")
#                     fn = getattr(tc, "function", None)
#                     fname = getattr(fn, "name", "") if fn else ""
#                     fargs = getattr(fn, "arguments", "") if fn else ""
#                 assistant_msg.tool_calls.append(
#                     chat_pb2.ToolCall(
#                         id=fid,
#                         function=chat_pb2.FunctionCall(name=fname, arguments=fargs),
#                     )
#                 )
#     # Fallback
#     return user(str(content))


# def _normalize_tool_call(tc: Any) -> SimpleNamespace:
#     """Normalize an xai tool call (proto, dict, or ns) to the SimpleNamespace shape
#     expected by ToolHandler ( {id, function: {name, arguments}} ).
#     """
#     if isinstance(tc, SimpleNamespace):
#         return tc
#     if isinstance(tc, dict):
#         fid = tc.get("id", "")
#         fn = tc.get("function", {}) or {}
#         fname = fn.get("name", "") if isinstance(fn, dict) else getattr(fn, "name", "")
#         fargs = (
#             fn.get("arguments", "")
#             if isinstance(fn, dict)
#             else getattr(fn, "arguments", "")
#         )
#         return SimpleNamespace(
#             id=fid,
#             function=SimpleNamespace(name=fname, arguments=fargs),
#         )
#     # assume protobuf / object with attributes
#     fid = getattr(tc, "id", "") or ""
#     fn = getattr(tc, "function", None)
#     fname = getattr(fn, "name", "") if fn is not None else ""
#     fargs = getattr(fn, "arguments", "") if fn is not None else ""
#     return SimpleNamespace(
#         id=fid,
#         function=SimpleNamespace(name=fname, arguments=fargs),
#     )

ROLE_MAP: dict[chat_pb2.MessageRole, str] = {
    chat_pb2.MessageRole.ROLE_ASSISTANT: "assistant",
    chat_pb2.MessageRole.ROLE_USER: "user",
    chat_pb2.MessageRole.ROLE_SYSTEM: "system",
    chat_pb2.MessageRole.ROLE_TOOL: "tool",
}


def message_to_dict(msg: chat_pb2.Message) -> dict[str, Any]:
    """Convert a chat_pb2.Message into a dict for UI consumption."""
    role = ROLE_MAP.get(msg.role, "unknown")
    content = msg.content or ""

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

        # Add user message to history
        self.chat_history.append(user(message))
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

        self.modelContext.model_name = get_model_for_choice(choice)
        is_image_mode = choice == "Generate Image"

        try:
            # Process in a loop to handle (client-side) tool calls and follow-ups.
            # Server-side tools (web_search etc.) are executed by xAI inside create();
            # only client_side_tool calls require us to run local handlers and loop.
            while True:
                # Build messages with system context
                messages = [system(get_system_message_for_choice(choice))]
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
                            {"role": "assistant", "content": response.content}
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
                if last_response:
                    assistant_message = assistant(last_response.content)
                    assistant_message.tool_calls.extend(last_response.tool_calls)
                    self.chat_history.append(assistant_message)

                    local_chat_history.append(
                        {
                            "role": "assistant",
                            "content": last_response.content or "",
                        }
                    )

                if client_tool_calls:
                    tool_results = self._tool_handler(client_tool_calls)

                    for tr in tool_results:
                        if is_image_mode:
                            # Side-effect: capture image ToolResult for "Generate Image" mode
                            if tr and tr.content_type == "image" and tr.content:
                                try:
                                    self.image = Image.open(io.BytesIO(tr.content))
                                except Exception:
                                    pass
                            # Yield so the UI can display the image promptly
                            yield local_chat_history + [
                                {
                                    "role": "assistant",
                                    "content": (
                                        last_response.content if last_response else ""
                                    ),
                                }
                            ], self.image

                        self.chat_history.append(
                            tool_result(
                                tr.content_for_model, tool_call_id=tr.tool_call_id
                            )
                        )

                    if is_image_mode and self.image:
                        # ensure we stop after one image generation per user turn
                        self._print_chat()
                        break
                else:
                    self._print_chat()
                    break

        except Exception as e:
            print(f"Error in chat: {e}")
            # Show generic error without details
            error_msg = (
                "I encountered an error processing your request. Please try again."
            )
            self.chat_history.append(assistant(error_msg))
            yield [message_to_dict(m) for m in self.chat_history], self.image

    def _print_chat(self) -> None:
        print("*********** Finished a chat iteration ***********")
        for entry in self.chat_history:
            print(entry)
        print("**********************************************")

    def clear_history(self) -> list[dict[str, Any]]:
        """Clear stored chat history and reset the image output."""
        self.chat_history = []
        self.image = None
        return []
