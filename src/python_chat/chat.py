from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any, Callable, Generator, Optional, cast

from PIL import Image

from openai import Stream
from openai.types.chat import ChatCompletionChunk

from python_chat.api import Models
from python_chat.context import ModelContext
from python_chat.tools import ToolResult

SYSTEM_MESSAGE: str = (
    "You are a helpful assistant. Only answer factually, and if you do not know the answer don't make anything up."
)

ChatCompleter = Callable[
    [str, list[dict[str, Any]], list[dict[str, Any]]], Stream[ChatCompletionChunk]
]
ToolHandler = Callable[
    [SimpleNamespace], tuple[list[dict[str, Any]], list[ToolResult | None]]
]


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
        self.chat_history: list[dict[str, str]] = []
        self.modelContext = modelContext
        self.image: Optional[Image.Image] = None
        self._completer: ChatCompleter = completer or self._default_completer
        self._tool_handler: ToolHandler = tool_handler or self._default_tool_handler

    def _default_completer(
        self, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Stream[ChatCompletionChunk]:
        """Default chat completion using the injected model context's OpenAI client."""
        return cast(
            Stream[ChatCompletionChunk],
            self.modelContext.client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                tools=tools,  # type: ignore[arg-type]
                stream=True,
            ),
        )

    def _default_tool_handler(
        self, message: SimpleNamespace
    ) -> tuple[list[dict[str, Any]], list[ToolResult | None]]:
        """Default tool execution using the model context's registered tools."""
        return self.modelContext.handle_tool_calls(message)

    def _collect_stream(
        self, stream: Stream[ChatCompletionChunk]
    ) -> Generator[tuple[str, Any], None, None]:
        """Collect all chunks from a stream, yielding text content and accumulating tool calls."""
        full_content = ""
        tool_calls_by_index: dict[int, dict[str, Any]] = {}

        for chunk in stream:
            choice = chunk.choices[0]
            delta = choice.delta

            if delta.content:
                token = delta.content
                full_content += token
                yield ("text", token)

            if delta.tool_calls:
                print(f"Received tool call chunk: {delta.tool_calls}")
                for tc in delta.tool_calls:
                    idx = getattr(tc, "index", 0) or 0
                    if idx not in tool_calls_by_index:
                        tool_calls_by_index[idx] = {
                            "id": "",
                            "function": {"name": "", "arguments": ""},
                        }
                    acc = tool_calls_by_index[idx]
                    if getattr(tc, "id", None) is not None:
                        acc["id"] = tc.id
                    if getattr(tc, "function", None) is not None:
                        if getattr(tc.function, "name", None) is not None:
                            acc["function"]["name"] = tc.function.name  # type: ignore[union-attr]
                        if getattr(tc.function, "arguments", None) is not None:
                            acc["function"]["arguments"] += tc.function.arguments or ""  # type: ignore[union-attr]

            if (
                getattr(choice, "finish_reason", None) == "tool_calls"
                and tool_calls_by_index
            ):
                tool_message = SimpleNamespace(tool_calls=[])
                tool_calls_list: list[dict[str, Any]] = []
                for tool_call_info in tool_calls_by_index.values():
                    function = SimpleNamespace(
                        name=tool_call_info["function"]["name"],
                        arguments=tool_call_info["function"]["arguments"],
                    )
                    tool_message.tool_calls.append(
                        SimpleNamespace(
                            id=tool_call_info["id"],
                            function=function,
                        )
                    )
                    tool_calls_list.append(
                        {
                            "id": tool_call_info["id"],
                            "type": "function",
                            "function": {
                                "name": tool_call_info["function"]["name"],
                                "arguments": tool_call_info["function"]["arguments"],
                            },
                        }
                    )
                tool_calls_by_index.clear()

                tool_responses, tool_results = self._tool_handler(tool_message)
                yield ("assistant_tool_call", tool_calls_list)
                yield ("tool_call", tool_responses)
                yield ("tool_result", tool_results)

    def chat(
        self, message: str, chat_history: list[dict[str, str]], choice: str
    ) -> Generator[Any, Any, Any]:
        """Stream chat responses token by token and handle tool calls.

        Args:
            message (str): The user's input message.
            chat_history (list[dict[str, str]]): Current chat history from the UI.
            choice (str): Selected chat mode, such as question or image generation.

        Yields:
            tuple: Intermediate chat history and optional image data while streaming.
        """
        if not message:
            yield chat_history, self.image
            return

        # Add user message to history
        self.chat_history.append({"role": "user", "content": message})
        local_chat_history = [
            c for c in self.chat_history
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
            # Process in a loop to handle tool calls and follow-ups
            while True:
                # Build messages with system context
                messages = [
                    {
                        "role": "system",
                        "content": get_system_message_for_choice(choice),
                    }
                ]
                messages.extend(self.chat_history)

                # Create streaming response
                # prevent the model from calling the image generation tool multiple times for a single response.
                tools = (
                    self.modelContext.get_tools_for_model()
                    if self.modelContext is not None
                    else []
                )
                stream = self._completer(self.modelContext.model_name, messages, tools)

                # Collect and stream all chunks
                full_response = ""
                tool_calls = []
                tool_calls_for_history = None
                for item_type, tool_results in self._collect_stream(stream):
                    if item_type == "text":
                        full_response += tool_results
                        yield local_chat_history + [
                            {"role": "assistant", "content": full_response or ""}
                        ], self.image
                    elif item_type == "assistant_tool_call":
                        tool_calls_for_history = tool_results
                    elif item_type == "tool_call":
                        tool_calls.extend(tool_results)
                    elif item_type == "tool_result" and is_image_mode:
                        for image_data in [
                            i.content
                            for i in tool_results
                            if i and i.content_type == "image"
                        ]:
                            self.image = Image.open(io.BytesIO(image_data))
                            yield local_chat_history + [
                                {"role": "assistant", "content": full_response or ""}
                            ], self.image

                # Add assistant response to history
                if full_response:
                    history_entry = {
                        "role": "assistant",
                        "content": full_response or "",
                    }
                    self.chat_history.append(history_entry)
                    local_chat_history.append(history_entry)
                if tool_calls_for_history:
                    # Add the assistant message with tool_calls before adding tool results
                    assistant_message = {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": tool_calls_for_history,
                    }
                    self.chat_history.append(assistant_message)
                    local_chat_history.append(assistant_message)
                if tool_calls:
                    self.chat_history.extend(tool_calls)
                    local_chat_history.extend(tool_calls)
                    if is_image_mode and self.image:
                        # make sure no additonal images are generated for the same prompt.
                        self._print_chat()
                        break

                    continue

                self._print_chat()
                break

        except Exception as e:
            print(f"Error in chat: {e}")
            # Show generic error without details
            error_msg = (
                "I encountered an error processing your request. Please try again."
            )
            self.chat_history.append({"role": "assistant", "content": error_msg})
            yield self.chat_history, self.image

    def _print_chat(self) -> None:
        print("*********** Finished a chat iteration ***********")
        for entry in self.chat_history:
            print(entry)
        print("**********************************************")

    def clear_history(self) -> list[dict[str, str]]:
        """Clear stored chat history and reset the image output."""
        self.chat_history = []
        self.image = None
        return []
