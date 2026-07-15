# type: ignore
# ignore mypy errors in this file due to an issue with dotenv in github workwflows
from __future__ import annotations

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


def load_env_file() -> bool:
    """Best-effort dotenv loader that never crashes process startup.

    Returns:
        bool: True when environment variables were loaded from a dotenv file.
    """
    loader: Callable[..., bool] | None = None
    try:
        from dotenv import load_dotenv as imported_loader

        loader = imported_loader
    except Exception as exc:
        logger.warning("python-dotenv is unavailable; skipping .env loading: %s", exc)
        return False

    try:
        return bool(loader())
    except Exception as exc:
        logger.warning("Failed to load .env file: %s", exc)
        return False
