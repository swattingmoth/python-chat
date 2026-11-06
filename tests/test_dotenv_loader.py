from __future__ import annotations

import builtins
import sys
from types import ModuleType
from typing import Any

from python_chat.dotenv_loader import load_env_file


def test_load_env_file_returns_loader_result(monkeypatch: Any) -> None:
    module = ModuleType("dotenv")
    module.load_dotenv = lambda: True  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dotenv", module)

    assert load_env_file() is True


def test_load_env_file_returns_false_when_loader_raises(monkeypatch: Any) -> None:
    module = ModuleType("dotenv")

    def fail() -> bool:
        raise RuntimeError("bad dotenv file")

    module.load_dotenv = fail  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dotenv", module)

    assert load_env_file() is False


def test_load_env_file_returns_false_when_dotenv_cannot_be_imported(
    monkeypatch: Any,
) -> None:
    original_import = builtins.__import__

    def blocked_import(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ) -> Any:
        if name == "dotenv":
            raise ImportError("dotenv unavailable")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    assert load_env_file() is False
