"""A chatbot interface the supports question answering and tool calls with streaming. It also supports image generation as a tool call.
This assumes use of the XAI API, but can be modified to work with other APIs.

All generated images are written to disk. The default path is c:\temp, but can be changed by modifying the call to ModelContext.create in the __main__ block.
"""

import logging
import os

os.environ["XAI_SDK_DISABLE_TRACING"] = "true"

from dotenv import load_dotenv

from python_chat.app import configure_logging, launch_app
from python_chat.utils import get_environment
from python_chat.api import init_api
from python_chat.context import ModelContext
from python_chat.persistence import AsyncLogQueue, SupabaseClient, db

if __name__ == "__main__":
    load_dotenv()
    configure_logging()
    logger = logging.getLogger()

    try:

        image_folder = os.getenv("IMAGE_FOLDER") or ""
        if not image_folder and get_environment() == "development":
            raise Exception("IMAGE_FOLDER environment variable is not set.")

        logger.info(f"Starting in environment: {get_environment()}")

        client = init_api("XAI_API_KEY", "https://api.x.ai/v1")
        persistence_client = SupabaseClient()
        log_queue = AsyncLogQueue(persistence_client)

        xai_api_key = db.get_secret_from_vault(
            persistence_client.rpc_client, "XAI_API_KEY"
        )
        if not xai_api_key:
            raise Exception("XAI_API_KEY secret is not set in the vault.")
        ModelContext.create(
            client,
            image_folder,
            persistence_client=persistence_client,
            log_queue=log_queue,
        )
        app = launch_app()
        app.launch()
    except Exception as e:
        logger.error("An error occurred while launching the app")
        logger.exception(e)
