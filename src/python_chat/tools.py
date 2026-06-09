from collections import defaultdict
from email.policy import default
import inspect
import json
from datetime import datetime
from json import tool
from types import SimpleNamespace
from typing import Any, Callable, Optional

from attr import dataclass
from xai_sdk.chat import tool as xai_tool

from xai_sdk.proto import chat_pb2


def today_date() -> str:
    """Tool function to return today's date"""
    return datetime.now().strftime("%Y-%m-%d")


@dataclass
class ToolResult:
    content_for_model: str
    content: Any
    tool_call_id: str
    content_type: str = "text"

    def __str__(self) -> str:
        # Convert content to string and slice first 100 characters
        content_str = str(self.content)
        truncated = content_str[:100] + "..." if len(content_str) > 100 else content_str

        return (
            f"ToolResult:\n"
            f"  Model Content: {self.content_for_model}\n"
            f"  Tool Call ID: {self.tool_call_id}\n"
            f"  Type: {self.content_type}\n"
            f"  Content Preview: {truncated}"
        )

    def __repr__(self) -> str:
        # Convert content to string, slice first 100 characters, and keep it safe for a single line
        content_str = str(self.content)
        truncated = content_str[:100] + "..." if len(content_str) > 100 else content_str

        return (
            f"ToolResult(content_for_model={self.content_for_model!r}, "
            f"tool_call_id={self.tool_call_id!r}, "
            f"content_type={self.content_type!r}, "
            f"content_preview={truncated!r})"
        )


class Tools:
    def __init__(self) -> None:
        """Initialize the tool registry."""
        self.tools: dict[str, dict[str, Any]] = {}

    def register_tool(
        self,
        func: Callable[..., Any],
        description: Optional[str] = None,
        name: Optional[str] = None,
        max_turns: Optional[int] = None,
    ) -> None:
        """Register a function as a tool that the model can call.

        Parameter types are derived from function annotations, and descriptions are derived from the function docstrings.

        Args:
            func (Callable): The function to register as a tool.
            description (Optional[str]): A brief description of what the tool does.
            name (Optional[str]): Optional override for the tool name (defaults to func.__name__).
            max_turns (Optional[int]): The maximum number of turns the tool can be used in a single handle_tool_calls session
        """
        tool_name = name or func.__name__
        tool_desc = description or (func.__doc__ or "")

        parameters: dict[str, Any] = {}
        accept_tool_call_id = False
        sig = inspect.signature(func)
        for param_name, param in sig.parameters.items():
            if param_name == "tool_call_id":
                accept_tool_call_id = True
            else:
                parameters[param_name] = {
                    "type": get_property_type(param.annotation),
                    "description": get_description(func.__doc__, param_name),
                }

        param_schema: dict[str, Any] = {
            "type": "object",
            "properties": parameters,
            "required": [
                pname
                for pname, p in sig.parameters.items()
                if p.default is inspect.Parameter.empty
                and p.kind
                not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            ],
            "additionalProperties": False,
        }

        self.tools[tool_name] = {
            "function": func,
            "accept_tool_call_id": accept_tool_call_id,
            "max_turns": max_turns,
            "tool": xai_tool(
                name=tool_name,
                description=tool_desc,
                parameters=param_schema,
            ),
        }

    def remove_tool(self, func: Callable[..., Any]) -> None:
        """Removes a tool from the registry."""
        if func.__name__ in self.tools:
            del self.tools[func.__name__]

    def handle_tool_calls(
        self, tool_calls: list[chat_pb2.ToolCall]
    ) -> list[ToolResult]:
        """Handles tool calls from the model by executing the corresponding functions and returning their results."""
        tool_results: list[ToolResult] = []
        tool_call_counts: dict[str, int] = defaultdict(int)
        for tool_call in tool_calls:
            print(f"Got tool call {tool_call}")
            # check if tool call count for this tool exceeds max_turns

            func = self.tools.get(tool_call.function.name, {})
            if func:
                max_turns = func.get("max_turns")
                if max_turns and tool_call_counts[tool_call.function.name] >= max_turns:
                    print(
                        f"Skipping tool call {tool_call}. Already called {tool_call_counts[tool_call.function.name]} time(s)."
                    )
                    continue
                tool_call_counts[tool_call.function.name] += 1

                try:
                    arguments = json.loads(tool_call.function.arguments)
                    print(
                        f"Calling function {func['function']} with arguments {arguments}"
                    )
                    if func["accept_tool_call_id"]:
                        arguments["tool_call_id"] = tool_call.id
                    result = func["function"](**arguments)
                    print(f"Got result {result} from tool call")

                    if isinstance(result, ToolResult):
                        tool_results.append(result)
                    else:
                        tool_results.append(
                            ToolResult(str(result), result, tool_call.id, "text")
                        )

                except Exception as e:
                    print(f"Error executing tool {tool_call.function.name}: {e}")
                    arguments = {}
                    tool_results.append(
                        ToolResult(
                            f"Error executing tool {tool_call.function.name}",
                            None,
                            tool_call.id,
                            "text",
                        )
                    )

        return tool_results

    def get_tools_for_model(
        self, additional_tools: Optional[list[Any]] = None
    ) -> list[Any]:
        """Return the tool objects (xai chat tool protos) formatted for the model's tool interface.

        Client-side tools are converted to xai_sdk.chat.tool(...) protos.
        Server-side tools (e.g. web_search()) may be passed in via additional_tools.
        """
        tools: list[Any] = [self.tools[t]["tool"] for t in self.tools]

        if additional_tools:
            tools.extend(additional_tools)
        return tools


def get_property_type(annotation: Any) -> str:
    """Return the JSON Schema type name for a function annotation.

    Maps Python types to JSON Schema primitive types:
    str -> "string", int -> "integer", float -> "number",
    bool -> "boolean", list[...] -> "array", dict[...] -> "object".
    Supports both concrete types and string annotations (from `from __future__ import annotations`).
    """
    if isinstance(annotation, str):
        mapping = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object",
            "object": "object",
        }
        return mapping.get(annotation, annotation or "string")

    # concrete types or empty
    if annotation in (str, inspect.Parameter.empty):
        return "string"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    if annotation is bool:
        return "boolean"
    if annotation in (
        list,
        list[str],
        list[int],
        list[float],
        list[bool],
        list[dict[Any, Any]],
        list[Any],
    ):
        return "array"
    if annotation in (dict, dict[Any, Any], dict[str, Any], dict[str, str]):
        return "object"

    # fallback to __name__ for custom/annotated types
    name = getattr(annotation, "__name__", None)
    if name == "str":
        return "string"
    if name == "int":
        return "integer"
    if name == "float":
        return "number"
    if name == "bool":
        return "boolean"
    return name or "string"


def get_description(docstring: Optional[str], param_name: str) -> str:
    """Extract a parameter description from a function docstring.

    Args:
        docstring (str): The function docstring to parse.
        param_name (str): The parameter name to locate.

    Returns:
        str: The description text for the parameter, or an empty string if none found.
    """
    if not docstring:
        return ""
    lines = docstring.splitlines()
    for line in lines:
        line = line.strip()
        if param_name in line and ":" in line:
            return line.partition(":")[2].strip()
    return ""
