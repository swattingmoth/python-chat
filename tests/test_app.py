from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Generator, Literal, cast
from unittest.mock import MagicMock

from xai_sdk.chat import user

from python_chat import app as app_module
from python_chat.chat import ChatInterface
from python_chat.context import ModelContext


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
        choice: str,
        *,
        runtime: dict[str, Any] | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        del choice
        resolved_runtime = runtime or {}
        history = list(resolved_runtime.get("chat_history", []))
        history.append({"role": "assistant", "content": f"echo:{message}"})
        resolved_runtime["chat_history"] = history
        yield {"history": history, "image_path": None, "runtime": resolved_runtime}

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
    assert (
        "src='https://example.com/image?name=&#x27;demo&#x27;&amp;size=large'"
        in rendered
    )
    assert (
        "navigator.clipboard.writeText(&quot;https://example.com/image?name=&#x27;demo&#x27;&amp;size=large&quot;)"
        in rendered
    )


def test_resolve_identity_prefers_and_normalizes_session_state(
    monkeypatch: Any,
) -> None:
    resolver = MagicMock(return_value=("request-user", "request-token"))

    resolved = app_module.resolve_identity(
        state={"user_id": "  session-user ", "access_token": " session-token "},
        request=object(),  # type: ignore[arg-type]
        identity_resolver=resolver,
    )

    assert resolved == ("session-user", "session-token")
    resolver.assert_not_called()

    resolved = app_module.resolve_identity(
        state={"user_id": "", "access_token": ""},
        request=object(),  # type: ignore[arg-type]
        identity_resolver=resolver,
    )

    assert resolved == ("request-user", "request-token")
    resolver.assert_called_once()


def test_build_default_runtime_uses_choice_and_resolved_identity(
    monkeypatch: Any,
) -> None:
    tools = [MagicMock()]
    monkeypatch.setattr(app_module, "get_tools_for_choice", lambda _: tools)

    runtime = app_module.build_default_runtime(
        "Complex Question",
        identity_resolver=lambda _: ("user-1", "token-1"),
    )

    assert runtime["user_id"] == "user-1"
    assert runtime["access_token"] == "token-1"
    assert runtime["selected_choice"] == "Complex Question"
    assert runtime["request_counter"] == 0
    assert runtime["active_tools"] is tools
    assert runtime["chat_history"] == []
    assert runtime["current_image_path"] is None


def test_ensure_runtime_merges_existing_session_state(
    monkeypatch: Any,
) -> None:
    tools = [MagicMock()]
    monkeypatch.setattr(app_module, "get_tools_for_choice", lambda _: tools)
    state = {
        "user_id": "state-user",
        "access_token": "state-token",
        "session_id": 42,
        "session_mode": "chat",
        "request_counter": 3,
        "chat_history": [{"role": "user", "content": "Hello"}],
        "current_image_path": "image.png",
    }

    runtime = app_module.ensure_runtime(
        state,
        "Generate Image",
        identity_resolver=lambda _: ("request-user", "request-token"),
    )

    assert runtime["user_id"] == "state-user"
    assert runtime["access_token"] == "state-token"
    assert runtime["session_id"] == 42
    assert runtime["session_mode"] == "chat"
    assert runtime["selected_choice"] == "Generate Image"
    assert runtime["request_counter"] == 4
    assert runtime["active_tools"] is tools
    assert runtime["chat_history"] == state["chat_history"]
    assert runtime["current_image_path"] == "image.png"


def test_handle_clear_completes_session_and_resets_runtime() -> None:
    model_context = SimpleNamespace(complete_session_for=MagicMock())
    state = app_module.build_default_runtime()
    state["session_id"] = 42
    state["access_token"] = "token-1"
    state["chat_history"] = [user("Hello")]
    state["current_image_path"] = "image.png"

    history, message, image, runtime = app_module.handle_clear(
        state, cast(ModelContext, model_context)
    )

    assert history == []
    assert message == ""
    assert image == ""
    assert runtime["session_id"] is None
    assert runtime["chat_history"] == []
    assert runtime["current_image_path"] is None
    model_context.complete_session_for.assert_called_once_with(
        42, access_token="token-1"
    )


def test_handle_submit_delegates_to_chat_interface() -> None:
    state = app_module.build_default_runtime()
    chat_interface = _FakeChatInterface(SimpleNamespace())
    history = [{"role": "user", "content": "Earlier"}]

    turns = list(
        app_module.handle_submit(
            "Hello",
            history,
            "Question",
            state,
            cast(ChatInterface, chat_interface),
        )
    )

    assert turns[-1][0][-1] == {"role": "assistant", "content": "echo:Hello"}
    assert turns[-1][1] == ""
    assert isinstance(turns[-1][2], dict)
    assert turns[-1][2].get("__type__") == "update"
    assert turns[-1][3]["chat_history"][-1]["content"] == "echo:Hello"  # pyright:ignore


def test_on_choice_change_clears_chat_and_updates_image_visibility(
    monkeypatch: Any,
) -> None:
    fake_gr = _FakeGr()
    monkeypatch.setattr(app_module, "gr", fake_gr)
    model_context = SimpleNamespace(complete_session_for=MagicMock())

    history, message, image, runtime = app_module.on_choice_change(
        "Generate Image",
        app_module.build_default_runtime(),
        cast(ModelContext, model_context),
    )

    assert history == []
    assert message == ""
    assert image == {"value": "", "visible": True}
    assert runtime["selected_choice"] == "Generate Image"


def test_shutdown_persistence_completes_and_stops_worker() -> None:
    queue = SimpleNamespace(stop=MagicMock())
    model_context = SimpleNamespace(
        log_queue=queue,
        complete_session=MagicMock(),
    )

    app_module.shutdown_persistence(cast(ModelContext, model_context))

    model_context.complete_session.assert_called_once()
    queue.stop.assert_called_once()


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
