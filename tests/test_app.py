from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Generator, Literal
from unittest.mock import MagicMock

from python_chat import app as app_module


class _FakeCtx:
    def __enter__(self) -> "_FakeCtx":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Literal[False]:
        return False


class _FakeBlocks(_FakeCtx):
    def __init__(self, title: str):
        self.title = title


class _FakeRow(_FakeCtx):
    pass


class _FakeComponent:
    def __init__(
        self, registry: dict[str, list[dict[str, Any]]], **kwargs: Any
    ) -> None:
        self._registry = registry
        self.kwargs = kwargs

    def change(
        self, fn: Callable[..., Any], inputs: Any = None, outputs: Any = None
    ) -> None:
        self._registry["change"].append(
            {"fn": fn, "inputs": inputs, "outputs": outputs}
        )

    def click(self, **kwargs: Any) -> None:
        self._registry["click"].append(kwargs)

    def submit(self, **kwargs: Any) -> None:
        self._registry["submit"].append(kwargs)


class _FakeGr:
    def __init__(self) -> None:
        self.registry: dict[str, list[dict[str, Any]]] = {
            "change": [],
            "click": [],
            "submit": [],
        }

    def Blocks(self, title: str) -> _FakeBlocks:  # noqa: N802
        return _FakeBlocks(title=title)

    def Row(self) -> _FakeRow:  # noqa: N802
        return _FakeRow()

    def Markdown(self, _: str) -> None:  # noqa: N802
        return None

    def Dropdown(self, **kwargs: Any) -> _FakeComponent:  # noqa: N802
        return _FakeComponent(self.registry, **kwargs)

    def Chatbot(self, **kwargs: Any) -> _FakeComponent:  # noqa: N802
        return _FakeComponent(self.registry, **kwargs)

    def State(self, value: Any) -> _FakeComponent:  # noqa: N802
        return _FakeComponent(self.registry, value=value)

    def Image(self, **kwargs: Any) -> _FakeComponent:  # noqa: N802
        return _FakeComponent(self.registry, **kwargs)

    def Textbox(self, **kwargs: Any) -> _FakeComponent:  # noqa: N802
        return _FakeComponent(self.registry, **kwargs)

    def Button(self, *_: Any, **kwargs: Any) -> _FakeComponent:  # noqa: N802
        return _FakeComponent(self.registry, **kwargs)

    def update(self, **kwargs: Any) -> dict[str, Any]:
        return kwargs


class _FakeChatInterface:
    def __init__(self, model_context: Any) -> None:
        self.model_context = model_context

    def chat(
        self, message: str, chat_history: list[dict[str, Any]], choice: str
    ) -> Generator[tuple[list[dict[str, Any]], bytes | None], None, None]:
        del choice
        yield chat_history + [{"role": "assistant", "content": f"echo:{message}"}], None

    def clear_history(self) -> list[dict[str, Any]]:
        return []


def test_configure_logging_supports_defaults_and_file_only(monkeypatch: Any) -> None:
    basic_config = MagicMock()
    file_handler = MagicMock(return_value="fh")
    stream_handler = MagicMock(return_value="sh")

    monkeypatch.setattr(app_module.logging, "basicConfig", basic_config)  # type: ignore
    monkeypatch.setattr(app_module.logging, "FileHandler", file_handler)  # type: ignore
    monkeypatch.setattr(app_module.logging, "StreamHandler", stream_handler)  # type: ignore

    app_module.configure_logging(logfile_path=None, log_to_console=True)
    app_module.configure_logging(logfile_path="c:/tmp/chat.log", log_to_console=False)

    assert basic_config.call_count == 2
    assert file_handler.call_count == 2


def test_launch_app_registers_handlers_and_executes_callbacks(monkeypatch: Any) -> None:
    fake_gr = _FakeGr()
    model_context = SimpleNamespace(
        session_id=77,
        log_queue=None,
        register_tool=MagicMock(),
        remove_tool=MagicMock(),
        complete_session=MagicMock(),
    )

    monkeypatch.setattr(app_module, "gr", fake_gr)
    monkeypatch.setattr(app_module.ModelContext, "current", lambda: model_context)  # type: ignore
    monkeypatch.setattr(app_module, "ChatInterface", _FakeChatInterface)

    demo = app_module.launch_app()

    assert isinstance(demo, _FakeBlocks)

    on_choice_change = fake_gr.registry["change"][0]["fn"]
    show = on_choice_change("Generate Image")
    hide = on_choice_change("Question")
    assert show == {"visible": True}
    assert hide == {"visible": False}
    model_context.register_tool.assert_called_once()
    model_context.remove_tool.assert_called_once()

    submit_fn = fake_gr.registry["click"][0]["fn"]
    empty_turns = list(
        submit_fn("", [{"role": "user", "content": "old"}], "Question", {})
    )
    assert empty_turns == [([{"role": "user", "content": "old"}], "", None, {})]

    msg_turns = list(submit_fn("hello", [], "Question", {}))
    assert msg_turns[-1][0][-1]["content"] == "echo:hello"
    assert msg_turns[-1][3] == {"session_id": 77}

    clear_fn = fake_gr.registry["click"][1]["fn"]
    assert clear_fn() == ([], "", None, {"session_id": 77})


def test_launch_app_registers_shutdown_when_log_queue_exists(monkeypatch: Any) -> None:
    fake_gr = _FakeGr()

    class _FakeQueue:
        def stop(self) -> None:
            return None

    model_context = SimpleNamespace(
        session_id=42,
        log_queue=_FakeQueue(),
        register_tool=MagicMock(),
        remove_tool=MagicMock(),
        complete_session=MagicMock(),
    )
    registered: list[Callable[[], None]] = []

    monkeypatch.setattr(app_module, "gr", fake_gr)
    monkeypatch.setattr(app_module.ModelContext, "current", lambda: model_context)  # type: ignore
    monkeypatch.setattr(app_module, "ChatInterface", _FakeChatInterface)
    monkeypatch.setattr(app_module, "start_log_worker", MagicMock())
    monkeypatch.setattr(app_module.atexit, "register", lambda fn: registered.append(fn))  # type: ignore

    app_module.launch_app()

    assert len(registered) == 1
    registered[0]()
    model_context.complete_session.assert_called_once()


def test_launch_app_shutdown_handles_runtime_error(monkeypatch: Any) -> None:
    fake_gr = _FakeGr()

    class _FakeQueue:
        def stop(self) -> None:
            raise RuntimeError("stop failed")

    model_context = SimpleNamespace(
        session_id=1,
        log_queue=_FakeQueue(),
        register_tool=MagicMock(),
        remove_tool=MagicMock(),
        complete_session=MagicMock(),
    )
    registered: list[Callable[[], None]] = []

    monkeypatch.setattr(app_module, "gr", fake_gr)
    monkeypatch.setattr(app_module.ModelContext, "current", lambda: model_context)  # type: ignore
    monkeypatch.setattr(app_module, "ChatInterface", _FakeChatInterface)
    monkeypatch.setattr(app_module, "start_log_worker", MagicMock())
    monkeypatch.setattr(app_module.atexit, "register", lambda fn: registered.append(fn))  # type: ignore

    app_module.launch_app()
    registered[0]()

    model_context.complete_session.assert_called_once()
