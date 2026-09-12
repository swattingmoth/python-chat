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
    class Request:
        pass

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

    def HTML(self, **kwargs: Any) -> _FakeComponent:  # noqa: N802
        return _FakeComponent(self.registry, **kwargs)

    def Textbox(self, **kwargs: Any) -> _FakeComponent:  # noqa: N802
        return _FakeComponent(self.registry, **kwargs)

    def Button(self, *_: Any, **kwargs: Any) -> _FakeComponent:  # noqa: N802
        return _FakeComponent(self.registry, **kwargs)

    def update(self, **kwargs: Any) -> dict[str, Any]:
        return kwargs

    def skip(self, **kwargs: Any) -> dict[Any, Any]:
        return {}


class _FakeChatInterface:
    def __init__(self, model_context: Any) -> None:
        self.model_context = model_context

    def chat(
        self,
        message: str,
        chat_history: list[dict[str, Any]],
        choice: str,
        *,
        runtime: dict[str, Any] | None = None,
    ) -> Generator[
        tuple[list[dict[str, Any]], bytes | None, dict[str, Any]], None, None
    ]:
        del choice
        resolved_runtime = runtime or {}
        yield (
            chat_history + [{"role": "assistant", "content": f"echo:{message}"}],
            None,
            resolved_runtime,
        )

    def clear_history(self) -> list[dict[str, Any]]:
        return []


def test_configure_logging_uses_root_handlers_for_local_and_server(
    monkeypatch: Any,
) -> None:
    class _FakePath:
        def __init__(self) -> None:
            self.parent = self

        def __truediv__(self, _: str) -> "_FakePath":
            return self

        def mkdir(self, *, parents: bool, exist_ok: bool) -> None:
            del parents
            del exist_ok

    file_handler = MagicMock(return_value=MagicMock())
    stream_handler = MagicMock(return_value=MagicMock())
    root_logger = MagicMock(handlers=[])

    monkeypatch.setattr(app_module, "Path", lambda _: _FakePath())
    monkeypatch.setattr(app_module.logging, "FileHandler", file_handler)  # type: ignore
    monkeypatch.setattr(app_module.logging, "StreamHandler", stream_handler)  # type: ignore
    monkeypatch.setattr(app_module.logging, "getLogger", lambda *_: root_logger)  # type: ignore

    monkeypatch.setenv("CHATBOT_ENV", "development")
    app_module.configure_logging(
        logfile_path=None, log_to_console=True, log_to_file=True
    )
    monkeypatch.setenv("CHATBOT_ENV", "production")
    app_module.configure_logging(logfile_path="c:/tmp/chat.log", log_to_console=False)

    assert file_handler.call_count == 1
    assert stream_handler.call_count == 2
    assert root_logger.addHandler.call_count == 3


def test_build_image_display_html_escapes_url_and_renders_copy_button() -> None:
    image_url = "https://example.com/image?name='demo'&size=large"

    rendered = app_module.build_image_display_html(image_url)

    assert "<img" in rendered
    assert "Copy image URL" in rendered
    assert "navigator.clipboard.writeText(" in rendered
    assert "src='https://example.com/image?name=&#x27;demo&#x27;&amp;size=large'" in rendered
    assert "navigator.clipboard.writeText(\"https://example.com/image?name='demo'&size=large\")" in rendered


def test_launch_app_registers_handlers_and_executes_callbacks(monkeypatch: Any) -> None:
    fake_gr = _FakeGr()
    model_context = SimpleNamespace(
        user_id="user-1",
        persistence_client=None,
        log_queue=None,
        register_tool=MagicMock(),
        remove_tool=MagicMock(),
        complete_session_for=MagicMock(),
        complete_session=MagicMock(),
    )

    monkeypatch.setattr(app_module, "gr", fake_gr)
    monkeypatch.setattr(app_module.ModelContext, "current", lambda: model_context)  # type: ignore
    monkeypatch.setattr(app_module, "ChatInterface", _FakeChatInterface)

    demo = app_module.launch_app()

    assert isinstance(demo, _FakeBlocks)

    on_choice_change = fake_gr.registry["change"][0]["fn"]
    _, _, show, show_state = on_choice_change("Generate Image", {})
    _, _, hide, hide_state = on_choice_change("Question", {})
    assert show == {"value": "", "visible": True}
    assert hide == {"value": "", "visible": False}
    assert show_state["selected_choice"] == "Generate Image"
    assert hide_state["selected_choice"] == "Question"

    # Verify per-session tool isolation: active_tools should be in state, not global registry
    assert "active_tools" in show_state
    assert "active_tools" in hide_state
    assert (
        len(show_state["active_tools"]) > 1
    )  # Generate Image includes base + image tools
    assert len(hide_state["active_tools"]) == 1  # Question includes base tools only

    # Global registry should NOT have been called (tools are now per-session)
    model_context.register_tool.assert_not_called()
    model_context.remove_tool.assert_not_called()

    submit_fn = fake_gr.registry["click"][0]["fn"]
    empty_turns = list(
        submit_fn("", [{"role": "user", "content": "old"}], "Question", {})
    )
    assert empty_turns[0][0] == [{"role": "user", "content": "old"}]
    assert empty_turns[0][1] == ""
    assert isinstance(empty_turns[0][2], dict)
    assert empty_turns[0][3]["selected_choice"] == "Question"

    msg_turns = list(submit_fn("hello", [], "Question", {}))
    assert msg_turns[-1][0][-1]["content"] == "echo:hello"
    assert msg_turns[-1][3]["selected_choice"] == "Question"

    clear_fn = fake_gr.registry["click"][1]["fn"]
    assert clear_fn({"selected_choice": "Question"})[0] == []


def test_launch_app_registers_shutdown_when_log_queue_exists(monkeypatch: Any) -> None:
    fake_gr = _FakeGr()

    class _FakeQueue:
        def stop(self) -> None:
            return None

    model_context = SimpleNamespace(
        user_id="user-1",
        persistence_client=None,
        log_queue=_FakeQueue(),
        register_tool=MagicMock(),
        remove_tool=MagicMock(),
        complete_session_for=MagicMock(),
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
        user_id="user-1",
        persistence_client=None,
        log_queue=_FakeQueue(),
        register_tool=MagicMock(),
        remove_tool=MagicMock(),
        complete_session_for=MagicMock(),
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


def test_launch_app_uses_identity_resolver_for_runtime(monkeypatch: Any) -> None:
    fake_gr = _FakeGr()
    model_context = SimpleNamespace(
        user_id=None,
        persistence_client=None,
        log_queue=None,
        register_tool=MagicMock(),
        remove_tool=MagicMock(),
        complete_session_for=MagicMock(),
        complete_session=MagicMock(),
    )

    monkeypatch.setattr(app_module, "gr", fake_gr)
    monkeypatch.setattr(app_module.ModelContext, "current", lambda: model_context)  # type: ignore
    monkeypatch.setattr(app_module, "ChatInterface", _FakeChatInterface)

    app_module.launch_app(identity_resolver=lambda _request: ("user-abc", "tok-xyz"))

    on_choice_change = fake_gr.registry["change"][0]["fn"]
    _, _, _, state = on_choice_change("Question", {})
    assert state["user_id"] == "user-abc"
    assert state["access_token"] == "tok-xyz"


def test_launch_app_ignores_blank_state_identity_values(monkeypatch: Any) -> None:
    fake_gr = _FakeGr()
    model_context = SimpleNamespace(
        user_id=None,
        persistence_client=None,
        log_queue=None,
        register_tool=MagicMock(),
        remove_tool=MagicMock(),
        complete_session_for=MagicMock(),
        complete_session=MagicMock(),
    )

    monkeypatch.setattr(app_module, "gr", fake_gr)
    monkeypatch.setattr(app_module.ModelContext, "current", lambda: model_context)  # type: ignore
    monkeypatch.setattr(app_module, "ChatInterface", _FakeChatInterface)

    app_module.launch_app(identity_resolver=lambda _request: ("user-abc", "tok-xyz"))

    on_choice_change = fake_gr.registry["change"][0]["fn"]
    _, _, _, state = on_choice_change("Question", {"user_id": "", "access_token": "  "})
    assert state["user_id"] == "user-abc"
    assert state["access_token"] == "tok-xyz"
