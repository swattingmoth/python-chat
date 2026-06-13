import datetime
import logging
from math import log
from typing import Any, Generator, Optional, cast

import gradio as gr

from python_chat.chat import ChatInterface
from python_chat.context import ModelContext
from python_chat.images import generate_image_tool


def configure_logging(
    logfile_path: Optional[str] = None, log_to_console: bool = True
) -> None:

    if not logfile_path:
        tdy = datetime.date.today().strftime("%Y-%m-%d")
        logfile_path = rf"c:\temp\chatbot_logs\chatbot_{tdy}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=(
            [logging.FileHandler(logfile_path), logging.StreamHandler()]
            if log_to_console
            else [logging.FileHandler(logfile_path)]
        ),
    )


def launch_app() -> gr.Blocks:
    """Launch the Gradio Blocks interface."""
    chat_interface = ChatInterface(ModelContext.current())

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

        image_output = gr.Image(label="Generated Image", visible=False, type="pil")

        with gr.Row():
            message_input = gr.Textbox(
                label="Message",
                placeholder="Type your message here...",
            )

        with gr.Row():
            submit_btn = gr.Button("Submit", variant="primary")
            clear_btn = gr.Button("Clear")

        def on_choice_change(selected_choice: str) -> dict[str, Any]:
            """Update image visibility based on choice."""
            if selected_choice == "Generate Image":
                ModelContext.current().register_tool(
                    generate_image_tool,
                    "Generate an image based on a text prompt",
                    max_turns=1,
                )
            else:
                ModelContext.current().remove_tool(generate_image_tool)

            return gr.update(visible=(selected_choice == "Generate Image"))

        choice.change(fn=on_choice_change, inputs=choice, outputs=image_output)

        def handle_submit(
            message: str, chat_history: list[dict[str, Any]], selected_choice: str
        ) -> Generator[tuple[list[dict[str, Any]], str, Optional[bytes]], None, None]:
            """Handle message submission."""
            if not message:
                yield chat_history, "", None
                return

            for updated_history, image_data in chat_interface.chat(
                message, chat_history, selected_choice
            ):
                yield updated_history, "", image_data

        def handle_clear() -> tuple[list[dict[str, Any]], str, Optional[bytes]]:
            """Handle clear button."""
            cleared_history = chat_interface.clear_history()
            return cleared_history, "", None

        submit_event = {
            "fn": handle_submit,
            "inputs": [message_input, chatbot, choice],
            "outputs": [chatbot, message_input, image_output],
        }

        # Submit on button click
        submit_btn.click(**submit_event)  # type: ignore[arg-type]

        # Submit on Enter key (textbox submission)
        message_input.submit(**submit_event)  # type: ignore[arg-type]

        # Clear button
        clear_btn.click(fn=handle_clear, outputs=[chatbot, message_input, image_output])

    return cast(gr.Blocks, demo)
