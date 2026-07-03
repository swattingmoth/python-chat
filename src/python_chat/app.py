import atexit
import datetime
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Generator, Optional, cast

import gradio as gr
from PIL import Image

from python_chat.chat import ChatInterface, SessionRuntime, get_model_for_choice
from python_chat.context import ModelContext
from python_chat.images import generate_image_tool
from python_chat.persistence import start_log_worker
from python_chat.tools import RegisteredTool, Tools, today_date
from python_chat.utils import get_environment

logger = logging.getLogger(__name__)


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
    logfile_path: Optional[str] = None, log_to_console: bool = True
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


def launch_app() -> gr.Blocks:
    """Launch the Gradio Blocks interface."""
    model_context = ModelContext.current()
    chat_interface = ChatInterface(model_context)

    def resolve_user_id(
        state: SessionRuntime | dict[str, Any] | None = None,
    ) -> str | None:
        # Priority 1: Session state
        if state is not None:
            user_id = state.get("user_id")
            if isinstance(user_id, str) or user_id is None:
                return user_id

        # Priority 2: Supabase Python library (authenticated user from SDK)
        persistence_client = model_context.persistence_client
        if persistence_client and persistence_client.rpc_client:
            try:
                user = persistence_client.rpc_client.auth.get_user()
                if user and hasattr(user, "id") and isinstance(user.id, str):
                    return user.id
            except Exception as exc:
                logger.debug("Failed to get user from Supabase SDK: %s", exc)

        # Priority 3: Environment variable fallback
        env_user_id = os.getenv("SUPABASE_AUTH_USER_ID")
        return env_user_id if env_user_id else None

    def build_default_runtime(choice: str = "Question") -> SessionRuntime:
        return {
            "user_id": resolve_user_id(),
            "session_id": None,
            "session_mode": None,
            "selected_choice": choice,
            "model_name": get_model_for_choice(choice),
            "request_counter": 0,
            "active_tools": get_tools_for_choice(choice),
        }

    def ensure_runtime(state: SessionRuntime, choice: str) -> SessionRuntime:
        merged = build_default_runtime(choice)
        merged["user_id"] = resolve_user_id(state)

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
            selected_choice: str, state: SessionRuntime
        ) -> tuple[dict[str, Any], SessionRuntime]:
            """Update image visibility based on choice (no longer mutates global tool registry)."""
            runtime = ensure_runtime(state, selected_choice)
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
        ) -> Generator[
            tuple[list[dict[str, Any]], str, Optional[Image.Image], SessionRuntime],
            None,
            None,
        ]:
            """Handle message submission."""
            runtime = ensure_runtime(state, selected_choice)
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
        ) -> tuple[list[dict[str, Any]], str, Optional[Image.Image], SessionRuntime]:
            """Handle clear button."""
            runtime = ensure_runtime(state, state.get("selected_choice", "Question"))
            if runtime.get("session_id") is not None:
                model_context.complete_session_for(runtime.get("session_id"))

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
