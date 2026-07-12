"""FastAPI application with bounded, authenticated native model analysis."""

from __future__ import annotations

import asyncio
import hmac
import logging
import re
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from startrain.contracts import (
    ACTION_LAYOUT_SCHEMA_ID,
    EXTERNAL_FEATURE_SCHEMA_ID,
    FEATURE_SCHEMA_HASH,
    FEATURE_SCHEMA_VERSION,
    RULES_HASH_WIRE,
    RULES_SCHEMA_ID,
    RULES_VERSION,
)
from startrain.model import MODEL_SCHEMA_VERSION

from .config import SERVER_CONFIG_SCHEMA_VERSION, ServerConfig, load_server_config
from .runtime import AnalysisError, NativeAnalysisService
from .schemas import AnalyzeRequest, AnalyzeResponse

SERVICE_VERSION = "2.0.0"
_LOGGER = logging.getLogger("starserve")
_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class AnalysisServiceProtocol(Protocol):
    def startup(self) -> None: ...

    def health(self) -> dict[str, object]: ...

    def analyze(
        self, request: AnalyzeRequest, cancellation: threading.Event
    ) -> dict[str, object]: ...


class RequestSizeLimitMiddleware:
    def __init__(self, app: Any, *, maximum_bytes: int) -> None:
        self.app = app
        self.maximum_bytes = maximum_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        request_id = _request_id_from_headers(headers)
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                declared = -1
            if declared < 0:
                await _send_error(
                    send,
                    status_code=400,
                    code="invalid_content_length",
                    message="Content-Length must be a non-negative integer",
                    request_id=request_id,
                )
                return
            if declared > self.maximum_bytes:
                await _send_error(
                    send,
                    status_code=413,
                    code="request_too_large",
                    message="request body exceeds the configured size limit",
                    request_id=request_id,
                )
                return
        received = 0
        buffered: list[dict[str, Any]] = []
        while True:
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.maximum_bytes:
                    await _send_error(
                        send,
                        status_code=413,
                        code="request_too_large",
                        message="request body exceeds the configured size limit",
                        request_id=request_id,
                    )
                    return
                buffered.append(message)
                if not message.get("more_body", False):
                    break
            elif message["type"] == "http.disconnect":
                return

        async def replay_receive() -> dict[str, Any]:
            if buffered:
                return buffered.pop(0)
            return await receive()

        await self.app(scope, replay_receive, send)


async def _send_error(
    send: Any,
    *,
    status_code: int,
    code: str,
    message: str,
    request_id: str,
) -> None:
    response = JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "details": None,
                "request_id": request_id,
            }
        },
    )
    response.headers["X-Request-ID"] = request_id
    response.headers["Cache-Control"] = "no-store"
    await response({"type": "http"}, _empty_receive, send)


async def _empty_receive() -> dict[str, Any]:
    return {"type": "http.request", "body": b"", "more_body": False}


def _request_id_from_headers(headers: dict[bytes, bytes]) -> str:
    raw = headers.get(b"x-request-id")
    if raw is not None and 0 < len(raw) <= 128:
        try:
            supplied = raw.decode("ascii")
        except UnicodeDecodeError:
            supplied = ""
        else:
            if _REQUEST_ID.fullmatch(supplied):
                return supplied
    return uuid.uuid4().hex


def create_app(
    config: ServerConfig | str | Path,
    *,
    service: AnalysisServiceProtocol | None = None,
) -> FastAPI:
    settings = (
        config if isinstance(config, ServerConfig) else load_server_config(config)
    )
    analysis_service = service or NativeAnalysisService(settings)
    bearer_token = settings.security.bearer_token()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        startup = getattr(analysis_service, "startup", None)
        if callable(startup):
            await asyncio.to_thread(startup)
        yield

    app = FastAPI(
        title="Double *Star model service",
        version=SERVICE_VERSION,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url="/v2/openapi.json",
    )
    app.state.config = settings
    app.state.analysis_service = analysis_service
    app.state.analysis_slots = asyncio.Semaphore(settings.limits.max_concurrency)
    app.add_middleware(
        RequestSizeLimitMiddleware,
        maximum_bytes=settings.limits.max_request_bytes,
    )
    if settings.security.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.security.cors_allow_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
            expose_headers=["X-Request-ID"],
            max_age=600,
        )

    def error_response(
        request: Request,
        *,
        status_code: int,
        code: str,
        message: str,
        details: object | None = None,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "code": code,
                    "message": message,
                    "details": details,
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )

    @app.middleware("http")
    async def request_context(
        request: Request,
        call_next: Callable[[Request], Awaitable[Any]],
    ) -> Any:
        supplied = request.headers.get("x-request-id")
        request_id = (
            supplied
            if supplied is not None and _REQUEST_ID.fullmatch(supplied)
            else uuid.uuid4().hex
        )
        request.state.request_id = request_id
        if (
            bearer_token is not None
            and request.method == "POST"
            and request.url.path in ("/v2/analyze", "/v2/move")
        ):
            authorization = request.headers.get("authorization")
            candidate = (
                authorization.removeprefix("Bearer ")
                if authorization is not None and authorization.startswith("Bearer ")
                else ""
            )
            if not candidate or not hmac.compare_digest(candidate, bearer_token):
                response = error_response(
                    request,
                    status_code=401,
                    code="unauthorized",
                    message="a valid bearer token is required",
                )
                response.headers["X-Request-ID"] = request_id
                response.headers["Cache-Control"] = "no-store"
                response.headers["WWW-Authenticate"] = "Bearer"
                return response
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(AnalysisError)
    async def analysis_error_handler(
        request: Request, error: AnalysisError
    ) -> JSONResponse:
        return error_response(
            request,
            status_code=error.status_code,
            code=error.code,
            message=error.message,
            details=error.details,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        details = [
            {
                "location": [str(part) for part in item["loc"]],
                "message": item["msg"],
                "type": item["type"],
            }
            for item in error.errors()
        ]
        return error_response(
            request,
            status_code=422,
            code="invalid_request",
            message="request does not match the v2 analysis schema",
            details=details,
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(
        request: Request, error: Exception
    ) -> JSONResponse:
        _LOGGER.error(
            "unhandled analysis service error",
            exc_info=(type(error), error, error.__traceback__),
        )
        return error_response(
            request,
            status_code=500,
            code="internal_error",
            message="the analysis service encountered an internal error",
        )

    @app.get("/healthz")
    @app.get("/v2/health")
    async def health() -> JSONResponse:
        model_health = analysis_service.health()
        ready = bool(model_health.get("ready", True))
        degraded = bool(model_health.get("last_reload_error"))
        payload = {
            "status": "degraded" if degraded else ("ok" if ready else "starting"),
            "service_version": SERVICE_VERSION,
            "api_schema_version": 2,
            "server_config_schema_version": SERVER_CONFIG_SCHEMA_VERSION,
            "model_schema_version": MODEL_SCHEMA_VERSION,
            "device": settings.device,
            "model": model_health,
            "search": {
                "defaults": {
                    "simulations": settings.search.default_simulations,
                    "max_considered": settings.search.default_max_considered,
                },
                "maximums": {
                    "simulations": settings.search.maximum_simulations,
                    "max_considered": settings.search.maximum_max_considered,
                },
                "presets": settings.search.named_presets(),
            },
            "rules": {
                "schema_id": RULES_SCHEMA_ID,
                "version": RULES_VERSION,
                "hash": RULES_HASH_WIRE,
            },
            "features": {
                "schema_id": EXTERNAL_FEATURE_SCHEMA_ID,
                "version": FEATURE_SCHEMA_VERSION,
                "hash": f"{FEATURE_SCHEMA_HASH:016x}",
            },
            "actions": {
                "schema_id": ACTION_LAYOUT_SCHEMA_ID,
                "types": ["place"],
            },
            "outcomes": {
                "classes": ["loss", "win"],
                "value": "P(win)-P(loss)",
            },
        }
        return JSONResponse(payload, status_code=200 if ready else 503)

    async def analyze(
        payload: AnalyzeRequest,
        request: Request,
    ) -> AnalyzeResponse:
        if payload.search.simulations > settings.search.maximum_simulations:
            raise AnalysisError(
                "search_budget_exceeded",
                "simulations exceed the configured maximum",
                status_code=422,
                details={"maximum_simulations": settings.search.maximum_simulations},
            )
        if payload.search.max_considered > settings.search.maximum_max_considered:
            raise AnalysisError(
                "search_budget_exceeded",
                "max_considered exceeds the configured maximum",
                status_code=422,
                details={
                    "maximum_max_considered": (settings.search.maximum_max_considered)
                },
            )
        result, queue_ms = await _run_bounded(
            app,
            analysis_service,
            payload,
            request,
        )
        result = dict(result)
        result["request_id"] = request.state.request_id
        raw_timing = result.get("timing_ms")
        if not isinstance(raw_timing, dict):
            raise AnalysisError(
                "native_search_error",
                "analysis service returned malformed timing metrics",
            )
        timing: dict[str, float] = {}
        for name, value in raw_timing.items():
            if not isinstance(name, str) or not isinstance(value, (int, float)):
                raise AnalysisError(
                    "native_search_error",
                    "analysis service returned malformed timing metrics",
                )
            timing[name] = float(value)
        if "total" not in timing:
            raise AnalysisError(
                "native_search_error",
                "analysis service omitted total timing",
            )
        timing["queue"] = queue_ms
        timing["total"] += queue_ms
        result["timing_ms"] = timing
        return AnalyzeResponse.model_validate(result)

    app.post(
        "/v2/analyze",
        response_model=AnalyzeResponse,
        response_model_exclude_none=False,
    )(analyze)
    app.post(
        "/v2/move",
        response_model=AnalyzeResponse,
        response_model_exclude_none=False,
        include_in_schema=False,
    )(analyze)
    return app


async def _run_bounded(
    app: FastAPI,
    service: AnalysisServiceProtocol,
    payload: AnalyzeRequest,
    request: Request,
) -> tuple[dict[str, object], float]:
    semaphore: asyncio.Semaphore = app.state.analysis_slots
    settings: ServerConfig = app.state.config
    queued = time.perf_counter()
    try:
        await asyncio.wait_for(
            semaphore.acquire(),
            timeout=settings.limits.queue_timeout_seconds,
        )
    except TimeoutError as exc:
        raise AnalysisError(
            "service_busy",
            "analysis concurrency limit is saturated",
            status_code=503,
        ) from exc
    queue_ms = (time.perf_counter() - queued) * 1_000.0
    cancellation = threading.Event()
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(None, service.analyze, payload, cancellation)
    disconnect = asyncio.create_task(_wait_for_disconnect(request))
    deferred_release = False
    try:
        remaining = max(
            0.001,
            settings.limits.request_timeout_seconds - queue_ms / 1_000.0,
        )
        done, _ = await asyncio.wait(
            {future, disconnect},
            timeout=remaining,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if future in done:
            return future.result(), queue_ms
        cancellation.set()
        deferred_release = True
        _release_when_done(future, semaphore, loop)
        if disconnect in done and disconnect.result():
            raise AnalysisError(
                "client_disconnected",
                "client disconnected during analysis",
                status_code=499,
            )
        raise AnalysisError(
            "analysis_timeout",
            "analysis exceeded the configured request timeout",
            status_code=504,
        )
    except asyncio.CancelledError:
        cancellation.set()
        deferred_release = True
        _release_when_done(future, semaphore, loop)
        raise
    finally:
        disconnect.cancel()
        if not deferred_release:
            semaphore.release()


async def _wait_for_disconnect(request: Request) -> bool:
    while True:
        if await request.is_disconnected():
            return True
        await asyncio.sleep(0.05)


def _release_when_done(
    future: asyncio.Future[Any],
    semaphore: asyncio.Semaphore,
    loop: asyncio.AbstractEventLoop,
) -> None:
    def release(completed: asyncio.Future[Any]) -> None:
        if not completed.cancelled():
            completed.exception()
        loop.call_soon_threadsafe(semaphore.release)

    future.add_done_callback(release)
