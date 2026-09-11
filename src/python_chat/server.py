from __future__ import annotations

import html
import logging
import os
from typing import Any
from urllib import parse as urllib_parse

import gradio as gr
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from python_chat.app import configure_logging, launch_app
from python_chat.chatbot import initialize_runtime
from python_chat.context import ModelContext
from python_chat.dotenv_loader import load_env_file
from python_chat.utils import get_environment, get_project_version

logger = logging.getLogger(__name__)

ACCESS_TOKEN_COOKIE_NAME = "python_chat_access_token"


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return None

    normalized_token = token.strip()
    return normalized_token or None


def _normalize_next_path(candidate: str | None) -> str:
    if not candidate:
        return "/app"
    if not candidate.startswith("/"):
        return "/app"
    if candidate.startswith("//"):
        return "/app"
    return candidate


def _cookie_secure_flag() -> bool:
    configured = os.getenv("COOKIE_SECURE")
    if configured is None:
        if get_environment() == "production":
            return True
        return False

    return configured.strip().lower() in {"1", "true", "yes", "on"}


def _build_login_url(next_path: str, error: str | None = None) -> str:
    params: dict[str, str] = {"next": _normalize_next_path(next_path)}
    if error:
        params["error"] = error
    return f"/login?{urllib_parse.urlencode(params)}"


def _extract_access_token_from_request(request: Request) -> str | None:
    bearer = _extract_bearer_token(request.headers.get("Authorization"))
    if bearer:
        return bearer

    cookie_token = request.cookies.get(ACCESS_TOKEN_COOKIE_NAME)
    if not cookie_token:
        return None

    normalized = cookie_token.strip()
    return normalized or None


def _authenticate_supabase_password(
    email: str,
    password: str,
) -> tuple[str, str, int | None]:
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_anon_key = os.getenv("SUPABASE_ANON_KEY")

    logger.info("Authenticating user with Supabase auth service")
    if not supabase_url or not supabase_anon_key:
        logger.error("Supabase auth verification is not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Supabase auth verification is not configured.",
        )

    try:
        from supabase import create_client

        client = create_client(supabase_url, supabase_anon_key)
        auth_response = client.auth.sign_in_with_password(
            {
                "email": email,
                "password": password,
            }
        )
    except Exception as exc:
        logger.error("Supabase auth request failed")
        logger.exception(exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        ) from exc

    session = auth_response.session
    user = auth_response.user

    access_token = session.access_token if session else None
    user_id = user.id if user else None
    expires_in = session.expires_in if session else None

    if not access_token or not user_id:
        logger.error("Supabase auth response was missing required fields")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Supabase auth response was missing required fields.",
        )

    max_age: int | None = expires_in if isinstance(expires_in, int) else None
    return access_token, user_id, max_age


def _login_page_html(next_path: str, error: str | None = None) -> str:
    escaped_next = html.escape(next_path, quote=True)
    error_block = ""
    if error:
        escaped_error = html.escape(error, quote=True)
        error_block = (
            f"<p style='color:#b00020;margin-bottom:12px;'>{escaped_error}</p>"
        )

    return f"""
<!doctype html>
<html>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>python-chat Login</title>
</head>
<body style='font-family:Segoe UI,Tahoma,sans-serif;background:#f7f7f8;margin:0;'>
  <main style='max-width:420px;margin:48px auto;padding:28px;background:#fff;border:1px solid #e5e7eb;border-radius:12px;'>
    <h1 style='margin:0 0 8px 0;font-size:24px;'>Sign in</h1>
    <p style='margin:0 0 20px 0;color:#374151;'>Use your Supabase credentials to continue.</p>
    {error_block}
    <form method='post' action='/login'>
      <input type='hidden' name='next' value='{escaped_next}' />
      <label for='email' style='display:block;margin-bottom:6px;'>Email</label>
      <input id='email' name='email' type='email' required style='width:100%;padding:10px;margin-bottom:12px;border:1px solid #d1d5db;border-radius:8px;' />
      <label for='password' style='display:block;margin-bottom:6px;'>Password</label>
      <input id='password' name='password' type='password' required style='width:100%;padding:10px;margin-bottom:16px;border:1px solid #d1d5db;border-radius:8px;' />
      <button type='submit' style='width:100%;padding:10px;border:0;border-radius:8px;background:#0f766e;color:white;font-weight:600;'>Sign in</button>
    </form>
  </main>
</body>
</html>
"""


def _verify_supabase_user(access_token: str) -> str:
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_anon_key = os.getenv("SUPABASE_ANON_KEY")

    if not supabase_url or not supabase_anon_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Supabase auth verification is not configured.",
        )

    try:
        from supabase import create_client

        client = create_client(supabase_url, supabase_anon_key)
        auth_response = client.auth.get_user(access_token)
    except Exception as exc:
        logger.error("Supabase auth verification request failed")
        logger.exception(exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Supabase access token.",
        ) from exc

    user = getattr(auth_response, "user", None)
    user_id = getattr(user, "id", None)
    if not isinstance(user_id, str) or not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Supabase token does not contain a valid user id.",
        )
    return user_id


def _resolve_identity(request: gr.Request | None) -> tuple[str | None, str | None]:
    if request is None:
        return None, None

    state = request.state
    if state is None:
        return None, None

    user_id = state.user_id if state else None
    access_token = state.access_token if state else None

    resolved_user_id = user_id if isinstance(user_id, str) else None
    resolved_access_token = access_token if isinstance(access_token, str) else None
    return resolved_user_id, resolved_access_token


def create_server_app() -> FastAPI:
    load_env_file()
    configure_logging()
    logger.info("Starting python-chat version %s", get_project_version())
    initialize_runtime()

    app = FastAPI(title="python-chat API")
    blocks = launch_app(identity_resolver=_resolve_identity)

    @app.get("/")
    def root() -> RedirectResponse:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    @app.get("/login")
    def login_page(request: Request) -> HTMLResponse:
        next_path = _normalize_next_path(request.query_params.get("next"))
        error = request.query_params.get("error")
        return HTMLResponse(content=_login_page_html(next_path, error=error))

    @app.post("/login")
    async def login_submit(request: Request) -> RedirectResponse:
        body_text = (await request.body()).decode("utf-8")
        form_data = urllib_parse.parse_qs(body_text, keep_blank_values=True)

        email = (form_data.get("email", [""])[0]).strip()
        password = form_data.get("password", [""])[0]
        next_path = _normalize_next_path(form_data.get("next", ["/app"])[0])

        if not email or not password:
            return RedirectResponse(
                url=_build_login_url(
                    next_path, error="Email and password are required."
                ),
                status_code=status.HTTP_303_SEE_OTHER,
            )

        try:
            access_token, _, max_age = _authenticate_supabase_password(email, password)
        except HTTPException as exc:
            if exc.status_code >= 500:
                raise
            return RedirectResponse(
                url=_build_login_url(next_path, error="Invalid email or password."),
                status_code=status.HTTP_303_SEE_OTHER,
            )

        response = RedirectResponse(
            url=next_path,
            status_code=status.HTTP_303_SEE_OTHER,
        )
        response.set_cookie(
            key=ACCESS_TOKEN_COOKIE_NAME,
            value=access_token,
            httponly=True,
            secure=_cookie_secure_flag(),
            samesite="lax",
            path="/",
            max_age=max_age,
        )
        return response

    @app.get("/logout")
    def logout() -> RedirectResponse:
        response = RedirectResponse(
            url="/login",
            status_code=status.HTTP_303_SEE_OTHER,
        )
        response.delete_cookie(key=ACCESS_TOKEN_COOKIE_NAME, path="/")
        return response

    @app.middleware("http")
    async def require_supabase_auth_for_app(
        request: Request,
        call_next: Any,
    ) -> Any:
        if request.url.path.startswith("/app"):
            access_token = _extract_access_token_from_request(request)
            if access_token is None:
                return RedirectResponse(
                    url=_build_login_url(request.url.path),
                    status_code=status.HTTP_302_FOUND,
                )

            try:
                user_id = _verify_supabase_user(access_token)
            except HTTPException as exc:
                response = RedirectResponse(
                    url=_build_login_url(request.url.path, error=str(exc.detail)),
                    status_code=status.HTTP_302_FOUND,
                )
                response.delete_cookie(key=ACCESS_TOKEN_COOKIE_NAME, path="/")
                return response

            request.state.user_id = user_id
            request.state.access_token = access_token

        return await call_next(request)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict[str, str]:
        ModelContext.current()
        return {"status": "ready"}

    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.error("Global exception caught in request lifecycle")
        logger.exception(exc)
        return JSONResponse(
            status_code=500,
            content={"message": "An internal server error occurred."},
        )

    gr.mount_gradio_app(app, blocks, path="/app")
    return app


def main() -> None:
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if os.getenv("PYTHON_CHAT_SKIP_SERVER_BOOTSTRAP") == "1" and os.getenv(
    "PYTEST_CURRENT_TEST"
):
    app = FastAPI(title="python-chat API (bootstrap skipped)")
else:
    app = create_server_app()


if __name__ == "__main__":
    main()
