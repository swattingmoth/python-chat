"""A chatbot interface the supports question answering and tool calls with streaming. It also supports image generation as a tool call.
This assumes use of the XAI API, but can be modified to work with other APIs.

All generated images are written to disk. The default path is c:\temp, but can be changed by modifying the call to ModelContext.create in the __main__ block.
"""

import os

os.environ["XAI_SDK_DISABLE_TRACING"] = "true"

from dotenv import load_dotenv

from python_chat.app import configure_logging, launch_app
from python_chat.api import init_api
from python_chat.context import ModelContext
from python_chat.tools import today_date

if __name__ == "__main__":
    load_dotenv()
    configure_logging()

    xai_api_key = os.getenv("XAI_API_KEY")
    image_folder = os.getenv("IMAGE_FOLDER")
    if not image_folder:
        raise Exception("IMAGE_FOLDER environment variable is not set.")
    if not xai_api_key:
        raise Exception("XAI_API_KEY environment variable is not set.")

    client = init_api("XAI_API_KEY", "https://api.x.ai/v1")
    ModelContext.create(client, image_folder)
    ModelContext.current().register_tool(
        today_date, "Get today's date in YYYY-MM-DD format"
    )
    app = launch_app()
    app.launch()
