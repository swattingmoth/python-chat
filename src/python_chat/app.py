import atexit
import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Generator, Optional, cast

import gradio as gr
from PIL import Image

from python_chat.chat import ChatInterface, SessionRuntime, get_model_for_choice
from python_chat.context import ModelContext
from python_chat.images import generate_image_tool
from python_chat.persistence import start_log_worker
from python_chat.tools import RegisteredTool, Tools, today_date
from python_chat.utils import get_environment

logger = logging.getLogger(__name__)

IdentityResolver = Callable[[gr.Request | None], tuple[str | None, str | None]]


class _JsonFormatter(logging.Formatter):
    """Format log records as compact JSON for telemetry pipelines."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.datetime.fromtimestamp(
                record.created, tz=datetime.timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def _resolve_daily_log_path(logfile_path: Optional[str] = None) -> str:
    if logfile_path:
        return logfile_path

    tdy = datetime.date.today().strftime("%Y-%m-%d")
    log_dir = Path(os.getenv("CHATBOT_LOG_DIR", "c:\\temp"))
    return str(log_dir / f"chatbot_{tdy}.log")


def configure_logging(
    logfile_path: Optional[str] = None,
    log_to_console: bool = True,
    log_to_file: bool = False,
) -> None:

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if get_environment() == "production":
        json_stdout_handler = logging.StreamHandler(stream=sys.stdout)
        json_stdout_handler.setFormatter(_JsonFormatter())
        json_stdout_handler.setLevel(logging.INFO)
        root_logger.addHandler(json_stdout_handler)
        return

    if log_to_file:
        resolved_path = _resolve_daily_log_path(logfile_path)
        json_file_handler = logging.FileHandler(resolved_path, encoding="utf-8")
        json_file_handler.setFormatter(_JsonFormatter())
        root_logger.addHandler(json_file_handler)

    if log_to_console:
        detailed_console_handler = logging.StreamHandler()
        detailed_console_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(detailed_console_handler)


def get_tools_for_choice(choice: str) -> list[RegisteredTool]:
    """Compute the active tools for a given choice (per-session).

    Args:
        choice: The selected choice from the dropdown.

    Returns:
        A list of tool callables that should be active for this choice.
    """
    tools: list[RegisteredTool] = []
    tools = Tools.register_tool(
        tools,
        today_date,
        description="Get today's date in YYYY-MM-DD format",
    )
    if choice == "Generate Image":
        tools = Tools.register_tool(
            tools,
            generate_image_tool,
            description=(
                "Generate an image from a user prompt. "
                "Returns image bytes plus metadata for model follow-up."
            ),
            name="generate_image",
            max_turns=1,
        )
    return tools


def launch_app(
    *,
    identity_resolver: IdentityResolver | None = None,
) -> gr.Blocks:
    """Launch the Gradio Blocks interface."""
    model_context = ModelContext.current()
    chat_interface = ChatInterface(model_context)

    def resolve_identity(
        state: SessionRuntime | dict[str, Any] | None = None,
        request: gr.Request | None = None,
    ) -> tuple[str | None, str | None]:
        # Priority 1: Session state
        if state is not None and ("user_id" in state or "access_token" in state):
            user_id = state.get("user_id")
            access_token = state.get("access_token")
            if (isinstance(user_id, str) or user_id is None) and (
                isinstance(access_token, str) or access_token is None
            ):
                normalized_user_id = (
                    user_id.strip() if isinstance(user_id, str) else user_id
                )
                normalized_access_token = (
                    access_token.strip()
                    if isinstance(access_token, str)
                    else access_token
                )
                if normalized_user_id == "":
                    normalized_user_id = None
                if normalized_access_token == "":
                    normalized_access_token = None
                if (
                    normalized_user_id is not None
                    or normalized_access_token is not None
                ):
                    return normalized_user_id, normalized_access_token

        # Priority 2: request-scoped resolver (FastAPI mount)
        if identity_resolver is not None:
            resolved_user_id, resolved_access_token = identity_resolver(request)
            return resolved_user_id, resolved_access_token

        # Development fallback only.
        if get_environment() == "development":
            user = os.getenv("SUPABASE_AUTH_USER_ID")
            return user, None

        return None, None

    def build_default_runtime(
        choice: str = "Question",
        request: gr.Request | None = None,
    ) -> SessionRuntime:
        user_id, access_token = resolve_identity(request=request)
        return {
            "user_id": user_id,
            "access_token": access_token,
            "session_id": None,
            "session_mode": None,
            "selected_choice": choice,
            "model_name": get_model_for_choice(choice),
            "request_counter": 0,
            "active_tools": get_tools_for_choice(choice),
        }

    def ensure_runtime(
        state: SessionRuntime,
        choice: str,
        request: gr.Request | None = None,
    ) -> SessionRuntime:
        merged = build_default_runtime(choice, request=request)
        user_id, access_token = resolve_identity(state=state, request=request)
        merged["user_id"] = user_id
        merged["access_token"] = access_token

        session_id = state.get("session_id")
        if isinstance(session_id, int) or session_id is None:
            merged["session_id"] = session_id

        session_mode = state.get("session_mode")
        if isinstance(session_mode, str) or session_mode is None:
            merged["session_mode"] = session_mode

        request_counter = state.get("request_counter")
        if isinstance(request_counter, int):
            merged["request_counter"] = request_counter

        merged["selected_choice"] = choice
        merged["model_name"] = get_model_for_choice(choice)
        merged["active_tools"] = get_tools_for_choice(choice)
        merged["request_counter"] = int(merged.get("request_counter", 0)) + 1
        return merged

    if model_context.log_queue:
        start_log_worker(model_context.log_queue)

        def _shutdown_persistence() -> None:
            model_context.complete_session()
            if not model_context.log_queue:
                return
            try:
                model_context.log_queue.stop()
            except RuntimeError:
                pass

        atexit.register(_shutdown_persistence)

    with gr.Blocks(title="AI Assistant") as demo:
        gr.Markdown("# AI Assistant")

        with gr.Row():
            choice = gr.Dropdown(
                choices=["Question", "Complex Question", "Generate Image"],
                value="Question",
                label="Select Mode",
                interactive=True,
            )

        chatbot = gr.Chatbot(label="Chat", height=400)
        session_state = gr.State(value=build_default_runtime())

        image_output = gr.Image(label="Generated Image", visible=False, type="pil")

        with gr.Row():
            message_input = gr.Textbox(
                label="Message",
                placeholder="Type your message here...",
            )

        with gr.Row():
            submit_btn = gr.Button("Submit", variant="primary")
            clear_btn = gr.Button("Clear")

        def on_choice_change(
            selected_choice: str,
            state: SessionRuntime,
            request: gr.Request | None = None,
        ) -> tuple[dict[str, Any], SessionRuntime]:
            """Update image visibility based on choice (no longer mutates global tool registry)."""
            runtime = ensure_runtime(state, selected_choice, request=request)
            return gr.update(visible=(selected_choice == "Generate Image")), runtime

        choice.change(
            fn=on_choice_change,
            inputs=[choice, session_state],
            outputs=[image_output, session_state],
        )

        def handle_submit(
            message: str,
            chat_history: list[dict[str, Any]],
            selected_choice: str,
            state: SessionRuntime,
            request: gr.Request | None = None,
        ) -> Generator[
            tuple[list[dict[str, Any]], str, Optional[Image.Image], SessionRuntime],
            None,
            None,
        ]:
            """Handle message submission."""
            runtime = ensure_runtime(state, selected_choice, request=request)
            if not message:
                yield chat_history, "", None, runtime
                return

            for updated_history, image_data, updated_runtime in chat_interface.chat(
                message,
                chat_history,
                selected_choice,
                runtime=runtime,
            ):
                yield updated_history, "", image_data, updated_runtime

        def handle_clear(
            state: SessionRuntime,
            request: gr.Request | None = None,
        ) -> tuple[list[dict[str, Any]], str, Optional[Image.Image], SessionRuntime]:
            """Handle clear button."""
            runtime = ensure_runtime(
                state,
                state.get("selected_choice", "Question"),
                request=request,
            )
            if runtime.get("session_id") is not None:
                model_context.complete_session_for(
                    runtime.get("session_id"),
                    access_token=runtime.get("access_token"),
                )

            cleared_history = chat_interface.clear_history()
            runtime["session_id"] = None
            runtime["session_mode"] = None
            return cleared_history, "", None, runtime

        submit_event = {
            "fn": handle_submit,
            "inputs": [message_input, chatbot, choice, session_state],
            "outputs": [chatbot, message_input, image_output, session_state],
        }

        submit_btn.click(**submit_event)  # type: ignore[arg-type]
        message_input.submit(**submit_event)  # type: ignore[arg-type]

        clear_btn.click(
            fn=handle_clear,
            inputs=[session_state],
            outputs=[chatbot, message_input, image_output, session_state],
        )

    return cast(gr.Blocks, demo)
