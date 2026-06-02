import inspect
import json
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Callable, Optional
from attr import dataclass


def today_date() -> str:
    """Tool function to return today's date"""
    return datetime.now().strftime("%Y-%m-%d")


@dataclass
class ToolResult:
    content_for_model: str
    content: Any
    content_type: str = "text"

    def __str__(self) -> str:
        # Convert content to string and slice first 100 characters
        content_str = str(self.content)
        truncated = content_str[:100] + "..." if len(content_str) > 100 else content_str

        return (
            f"ToolResult:\n"
            f"  Model Content: {self.content_for_model}\n"
            f"  Type: {self.content_type}\n"
            f"  Content Preview: {truncated}"
        )

    def __repr__(self) -> str:
        # Convert content to string, slice first 100 characters, and keep it safe for a single line
        content_str = str(self.content)
        truncated = content_str[:100] + "..." if len(content_str) > 100 else content_str

        return (
            f"ToolResult(content_for_model={self.content_for_model!r}, "
            f"content_type={self.content_type!r}, "
            f"content_preview={truncated!r})"
        )


class Tools:
    def __init__(self) -> None:
        """Initialize the tool registry."""
        self.tools: dict[str, dict[str, Any]] = {}

    def register_tool(self, func: Callable[..., Any], description: str) -> None:
        """Register a function as a tool that the model can call.

        Parameter types are derived from function annotations, and descriptions are derived from the function docstrings.

        Args:
            func (Callable): The function to register as a tool.
            description (str): A brief description of what the tool does.
        """
        parameters = {}
        sig = inspect.signature(func)
        for param_name, param in sig.parameters.items():
            parameters[param_name] = {
                "type": get_property_type(param.annotation, True),
                "description": get_description(func.__doc__, param_name),
            }
        self.tools[func.__name__] = {
            "function": func,
            "json": {
                "name": func.__name__,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                },
                "required": [
                    name
                    for name, param in sig.parameters.items()
                    if param.default is inspect.Parameter.empty
                    and param.kind
                    not in (
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    )
                ],
                "additional_properties": False,
            },
        }

    def remove_tool(self, func: Callable[..., Any]) -> None:
        """Removes a tool from the registry."""
        if func.__name__ in self.tools:
            del self.tools[func.__name__]

    def handle_tool_calls(
        self, message: SimpleNamespace
    ) -> tuple[list[dict[str, Any]], list[ToolResult | None]]:
        """Handles tool calls from the model by executing the corresponding functions and returning their results."""
        responses = []
        tool_results: list[ToolResult | None] = []
        for tool_call in message.tool_calls:
            print(f"Got tool call {tool_call}")
            func = self.tools.get(tool_call.function.name, {})
            if func:
                try:
                    arguments = json.loads(tool_call.function.arguments)
                    print(
                        f"Calling function {func['function']} with arguments {arguments}"
                    )
                    result = func["function"](**arguments)
                    print(f"Got result {result} from tool call")

                    result_for_model = result
                    if isinstance(result, ToolResult):
                        result_for_model = result.content_for_model
                        tool_results.append(result)
                    else:
                        tool_results.append(None)

                    responses.append(
                        {
                            "role": "tool",
                            "content": result_for_model or "",
                            "tool_call_id": tool_call.id,
                        }
                    )
                except Exception as e:
                    print(f"Error executing tool {tool_call.function.name}: {e}")
                    arguments = {}
                    responses.append(
                        {
                            "role": "tool",
                            "content": f"Error executing tool {tool_call.function.name}",
                            "tool_call_id": tool_call.id,
                        }
                    )

        return responses, tool_results

    def get_tools_for_model(
        self, additional_tools: Optional[list[dict[str, Any]]] = None
    ) -> list[dict[str, Any]]:
        """Return the tool metadata formatted for the model's tool interface."""
        tools = [
            {"type": "function", "function": self.tools[t]["json"]} for t in self.tools
        ]

        if additional_tools:
            tools.extend(additional_tools)
        return tools


def get_property_type(annotation: Any, for_xai: bool) -> str:
    """Return the JSON property type name for a function annotation.

    Supports both concrete types and string annotations (from `from __future__ import annotations`).
    """
    if isinstance(annotation, str):
        # Handle postponed annotations (string form)
        if for_xai and annotation in {"str", "string"}:
            return "string"
        if annotation in {"int", "float", "bool", "list", "dict", "object"}:
            return annotation
        return annotation if annotation else "str"

    if for_xai:
        if annotation == str or annotation == inspect.Parameter.empty:
            return "string"
    return annotation.__name__ if annotation != inspect.Parameter.empty else "str"


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
