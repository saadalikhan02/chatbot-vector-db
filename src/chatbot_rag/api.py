"""FastAPI serving layer wrapping pipeline.answer() - see README's
"Serving" section for how to run it.

Loads the model once at startup (not per-request). LLM_BACKEND selects
which of three generate_fn implementations to use - "plain" (default, the
local PyTorch/transformers backend, ~1.6-2.9 tok/s), "gguf" (the local
llama.cpp backend, ~12.8 tok/s - see README's "Speed"; also selected
implicitly when GGUF_MODEL_PATH is set and LLM_BACKEND is unset, for
backward compatibility), or "api" (an OpenAI-compatible cloud endpoint -
OpenAI, DeepSeek, Groq, etc. - see api_llm.py). Every reliability guarantee
(guardrails.py, fact_check.py) applies identically across all three - see
pipeline.py's module docstring for why.

Auth: a single shared API key via the X-API-Key header, checked against
the API_KEY environment variable. Deliberately fails closed - if API_KEY
isn't configured, every request is rejected (503) rather than silently
allowing unauthenticated access. This is not a full auth system (no
per-user identity, no key rotation, no scopes) - adequate for a small
deployment behind one trusted key, not for multi-tenant access control.

Rate limiting: see ratelimit.py's docstring for what this does and does
not cover (single-process only).

Error handling: per SECURITY.md's "Error handling / information
disclosure" finding - every exception is logged server-side with full
detail (logger.exception, not print) and only a generic message is ever
returned in a response body. Never a raw stack trace, internal path, or
dependency version.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from .api_llm import LLMProviderError
from .guardrails import MAX_USER_INPUT_CHARS
from .pipeline import answer
from .prompting import SYSTEM_PROMPT
from .ratelimit import RateLimiter
from .retrieval import open_index
from .sessions import ConversationStore, append_conversation_log
from .utils import get_hf_token, load_dotenv_if_present

logger = logging.getLogger("chatbot_rag.api")

# A separate logger channel from the normal request log above - every
# guardrail refusal and every fact-check-blocked response is logged here
# too, so a human reviewer can watch (or route to a separate file/Slack
# channel/etc.) just the things this project's reliability layer actually
# caught, without wading through all traffic. The review PROCESS itself
# (someone actually looking at this) is not something code can do for
# you - this only sets up the plumbing.
flagged_logger = logging.getLogger("chatbot_rag.flagged")

REQUESTS_TOTAL = Counter(
    "chatbot_rag_requests_total",
    "Total /v1/chat requests by outcome",
    ["outcome"],  # generated | guardrail_refused | fact_check_blocked | error
)
REQUEST_LATENCY_SECONDS = Histogram(
    "chatbot_rag_request_latency_seconds",
    "Latency of /v1/chat requests",
)
AUTH_FAILURES_TOTAL = Counter("chatbot_rag_auth_failures_total", "Requests rejected for missing/invalid API key")
RATE_LIMIT_REJECTIONS_TOTAL = Counter(
    "chatbot_rag_rate_limit_rejections_total", "Requests rejected by the rate limiter"
)


MAX_HISTORY_MESSAGES = 40
DEFAULT_QUEUE_TIMEOUT_SECONDS = 60.0


def cors_origins(value: str | None = None) -> list[str]:
    """Parse an explicit comma-separated CORS allowlist.

    Empty is deliberately not ``*``: a public website deployment must list
    ``https://technyxsystems.com`` (and any approved preview origin) before a
    browser can call this API.
    """
    raw = os.environ.get("CORS_ORIGINS", "") if value is None else value
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=MAX_USER_INPUT_CHARS)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_USER_INPUT_CHARS)
    # 40 turns fit alongside the answer in GGUF's measured n_ctx=2048.
    # Reject, rather than silently truncating, so callers know context was lost.
    history: list[ChatMessage] | None = Field(default=None, max_length=MAX_HISTORY_MESSAGES)
    conversation_id: str | None = Field(default=None, max_length=128)


class ChatResponse(BaseModel):
    response: str
    retrieved_fact_count: int
    generated: bool  # False if a guardrail answered without calling the model
    latency_ms: float
    conversation_id: str


def _to_pipeline_history(history: list[ChatMessage] | None) -> list[dict[str, Any]] | None:
    """Convert the API's plain {role, content} messages into this
    project's internal list-of-parts content form (prompting.py,
    retrieval.build_retrieval_query) - kept as an API-layer detail so the
    rest of the codebase doesn't need to know about the wire format."""
    if not history:
        return None
    return [{"role": m.role, "content": [{"type": "text", "text": m.content}]} for m in history]


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv_if_present()

    app.state.retrieval_index = open_index()

    gguf_path = os.environ.get("GGUF_MODEL_PATH")
    # Explicit switch; falls back to the old implicit "GGUF_MODEL_PATH set
    # => gguf" inference when unset, so existing .env files keep working.
    llm_backend = os.environ.get("LLM_BACKEND", "").strip().lower() or ("gguf" if gguf_path else "plain")

    if llm_backend == "api":
        from .api_llm import make_generate_fn as make_api_generate_fn

        missing = [var for var in ("LLM_API_BASE_URL", "LLM_API_KEY", "LLM_API_MODEL") if not os.environ.get(var)]
        if missing:
            raise RuntimeError(f"LLM_BACKEND=api requires {', '.join(missing)} - see .env.example.")
        base_url = os.environ["LLM_API_BASE_URL"]
        api_model = os.environ["LLM_API_MODEL"]
        logger.info("Loading cloud API backend (%s, model=%s)", base_url, api_model)
        app.state.generate_fn = make_api_generate_fn(
            base_url=base_url, api_key=os.environ["LLM_API_KEY"], model=api_model
        )
        app.state.backend = "api"
    elif llm_backend == "gguf":
        if not gguf_path:
            raise RuntimeError("LLM_BACKEND=gguf requires GGUF_MODEL_PATH - see .env.example.")
        from .gguf_llm import load_gguf_model
        from .gguf_llm import make_generate_fn as make_gguf_generate_fn

        logger.info("Loading GGUF backend from %s", gguf_path)
        llm = load_gguf_model(gguf_path)
        app.state.generate_fn = make_gguf_generate_fn(llm)
        app.state.backend = "gguf"
    else:
        from .llm import load_model, load_tokenizer, make_generate_fn

        model_name = os.environ.get("MODEL_NAME", "google/gemma-3-1b-it")
        logger.info(
            "Loading plain PyTorch backend (%s) - set LLM_BACKEND=gguf or LLM_BACKEND=api for alternatives",
            model_name,
        )
        hf_token = get_hf_token()
        tokenizer = load_tokenizer(model_name, hf_token=hf_token)
        model = load_model(model_name, hf_token=hf_token)
        app.state.generate_fn = make_generate_fn(model, tokenizer)
        app.state.backend = "plain"

    max_requests = int(os.environ.get("RATE_LIMIT_MAX_REQUESTS", "20"))
    window_seconds = float(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))
    app.state.rate_limiter = RateLimiter(max_requests, window_seconds)
    visitor_max_requests = int(os.environ.get("VISITOR_RATE_LIMIT_MAX_REQUESTS", "10"))
    visitor_window_seconds = float(os.environ.get("VISITOR_RATE_LIMIT_WINDOW_SECONDS", "60"))
    app.state.visitor_rate_limiter = RateLimiter(visitor_max_requests, visitor_window_seconds)
    app.state.sessions = ConversationStore(
        max_sessions=int(os.environ.get("SESSION_MAX_CONVERSATIONS", "1000")),
        ttl_seconds=float(os.environ.get("SESSION_TTL_SECONDS", "3600")),
        max_messages=MAX_HISTORY_MESSAGES,
    )
    app.state.conversation_log_path = os.environ.get("CONVERSATION_LOG_PATH", "logs/conversations.jsonl")

    # There is exactly one shared model instance (loaded once above), and
    # it is NOT thread-safe for concurrent generation calls - load-tested
    # directly: two concurrent /v1/chat requests dispatched to threads
    # without this lock corrupted the GGUF backend's internal KV-cache
    # state ("GGML_ASSERT(i1 >= 0 && i1 < ne1) failed" in llama.cpp) and
    # hung the whole process, taking every in-flight request down with
    # it, not just the concurrent ones. This lock serializes access to
    # the model while asyncio.to_thread (see the /v1/chat handler) still
    # keeps the event loop itself unblocked - a health check or a
    # rate-limited request's immediate 429 don't wait on this lock, only
    # actual model access does.
    app.state.answer_lock = asyncio.Lock()

    logger.info("Ready (backend=%s, rate limit=%d req/%.0fs)", app.state.backend, max_requests, window_seconds)
    yield
    app.state.retrieval_index.close()


load_dotenv_if_present()
app = FastAPI(title="Technyx RAG Chatbot API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=False,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)
STATIC_DIR = Path(__file__).resolve().parents[2] / "static"


def require_api_key(x_api_key: str | None = Header(default=None)) -> str:
    expected = os.environ.get("API_KEY")
    if not expected:
        # Fail closed: an unconfigured API_KEY must never mean "accept
        # everything" - see this module's docstring.
        raise HTTPException(status_code=503, detail="Service is not configured for authentication.")
    if x_api_key != expected:
        AUTH_FAILURES_TOTAL.inc()
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
    return x_api_key


def enforce_rate_limit(request: Request, api_key: str = Depends(require_api_key)) -> None:
    limiter: RateLimiter = request.app.state.rate_limiter
    if not limiter.allow(api_key):
        RATE_LIMIT_REJECTIONS_TOTAL.inc()
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please slow down.")


def visitor_rate_limit_key(request: Request) -> str:
    """Return a visitor IP key. X-Forwarded-For is trusted only behind an
    explicitly configured reverse proxy; otherwise a client can forge it."""
    if os.environ.get("TRUST_PROXY") == "1":
        forwarded = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        if forwarded:
            return forwarded
    return request.client.host if request.client else "unknown"


def enforce_visitor_rate_limit(request: Request) -> None:
    limiter: RateLimiter = request.app.state.visitor_rate_limiter
    if not limiter.allow(visitor_rate_limit_key(request)):
        RATE_LIMIT_REJECTIONS_TOTAL.inc()
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please slow down.")


def request_history(req: ChatRequest, request: Request) -> tuple[str, list[dict[str, Any]] | None]:
    """Prefer explicit history for backward compatibility, otherwise reuse a
    bounded server session.  New callers always receive an opaque ID."""
    conversation_id = req.conversation_id or str(uuid.uuid4())
    history = (
        _to_pipeline_history(req.history)
        if req.history is not None
        else request.app.state.sessions.get(conversation_id)
    )
    return conversation_id, history


def save_conversation(request: Request, conversation_id: str, user_text: str, response: str, outcome: str) -> None:
    request.app.state.sessions.append(conversation_id, user_text, response)
    append_conversation_log(
        request.app.state.conversation_log_path,
        {
            "timestamp": time.time(),
            "conversation_id": conversation_id,
            "user": user_text,
            "assistant": response,
            "outcome": outcome,
        },
    )


def answer_outcome(result: Any) -> str:
    if not result.generated:
        return "guardrail_refused"
    if result.unsupported_claims:
        return "fact_check_blocked"
    return "generated"


def sse(event: str, data: dict[str, Any]) -> str:
    """Encode one SSE event without allowing user/model newlines to break it."""
    import json

    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def acquire_answer_lock(request: Request) -> None:
    """Wait only a bounded time for the single safe generation slot.

    GGUF's shared KV cache was load-tested as unsafe under simultaneous
    generation, so removing the lock is not an option.  A timeout prevents a
    traffic burst from retaining requests indefinitely; 60 seconds is long
    enough for the measured fast GGUF answer path but callers receive a
    retryable generic 503 when the service is saturated.
    """
    timeout = float(os.environ.get("QUEUE_TIMEOUT_SECONDS", str(DEFAULT_QUEUE_TIMEOUT_SECONDS)))
    try:
        await asyncio.wait_for(request.app.state.answer_lock.acquire(), timeout=timeout)
    except TimeoutError:
        raise HTTPException(status_code=503, detail="Chat service is busy. Please try again shortly.") from None


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception processing %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    return {
        "status": "ok",
        "backend": request.app.state.backend,
        "facts_indexed": request.app.state.retrieval_index.count(),
    }


@app.get("/widget.js", include_in_schema=False)
async def widget() -> FileResponse:
    """Serve the dependency-free browser widget from this API host."""
    return FileResponse(STATIC_DIR / "widget.js", media_type="application/javascript")


@app.get("/metrics")
async def metrics(x_metrics_key: str | None = Header(default=None)) -> Response:
    """Prometheus metrics gated by ``X-Metrics-Key`` / ``METRICS_KEY``.

    Metrics can expose traffic volume and timing. Keeping the port public is
    acceptable only when the independently configured scrape key is handled
    as a secret; network isolation remains useful defense in depth.
    """
    expected = os.environ.get("METRICS_KEY")
    if not expected:
        raise HTTPException(status_code=503, detail="Metrics are not configured.")
    if x_metrics_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing metrics key.")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/chat", response_model=ChatResponse, dependencies=[Depends(enforce_rate_limit)])
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    t0 = time.monotonic()
    conversation_id, history = request_history(req, request)
    try:
        # answer() is a blocking, CPU-bound call (generation takes seconds
        # to tens of seconds). Dispatched to a thread explicitly rather
        # than awaited directly, so this doesn't block the single asyncio
        # event loop for the full generation duration - otherwise a
        # second request (even a /health check) couldn't be handled at
        # all until the first one's generation finished. Load-tested:
        # without this, correctness would depend on whichever generation
        # backend happens to release the GIL during its own compute
        # (llama-cpp-python appears to; the plain-transformers backend
        # was not separately verified to) - explicit dispatch makes the
        # non-blocking behavior a guarantee, not a backend-specific
        # accident. The lock (see lifespan()'s comment) serializes actual
        # model access across concurrent requests - required, not
        # optional, for correctness with a single shared model instance.
        await acquire_answer_lock(request)
        try:
            result = await asyncio.to_thread(
                answer,
                req.message,
                request.app.state.generate_fn,
                request.app.state.retrieval_index,
                system_prompt=SYSTEM_PROMPT,
                history=history,
            )
        finally:
            request.app.state.answer_lock.release()
    except LLMProviderError as exc:
        REQUESTS_TOTAL.labels(outcome="error").inc()
        logger.warning("LLM provider error: provider_status=%s http_status=%s", exc.provider_status, exc.http_status)
        headers = {"Retry-After": exc.retry_after} if exc.retry_after else None
        raise HTTPException(status_code=exc.http_status, detail=exc.public_detail, headers=headers) from None
    except HTTPException:
        raise
    except Exception:
        # Caught explicitly (in addition to the global handler above) so
        # this specific, unexpected failure mode is logged with request
        # context, not just "some exception happened somewhere."
        REQUESTS_TOTAL.labels(outcome="error").inc()
        logger.exception("Error answering chat request")
        raise HTTPException(status_code=500, detail="Internal error processing your request.") from None

    latency_seconds = time.monotonic() - t0
    REQUEST_LATENCY_SECONDS.observe(latency_seconds)

    outcome = answer_outcome(result)
    REQUESTS_TOTAL.labels(outcome=outcome).inc()

    logger.info(
        "chat request handled: outcome=%s retrieved=%d latency_ms=%.0f",
        outcome,
        len(result.retrieved),
        latency_seconds * 1000,
    )
    if outcome != "generated":
        # See flagged_logger's definition above - a separate channel so a
        # human reviewer can watch just this, not all traffic.
        flagged_logger.info(
            "outcome=%s question=%r response=%r unsupported_claims=%r",
            outcome,
            req.message,
            result.response,
            result.unsupported_claims,
        )

    save_conversation(request, conversation_id, req.message, result.response, outcome)

    return ChatResponse(
        response=result.response,
        retrieved_fact_count=len(result.retrieved),
        generated=result.generated,
        latency_ms=latency_seconds * 1000,
        conversation_id=conversation_id,
    )


@app.post("/v1/chat/stream", dependencies=[Depends(enforce_visitor_rate_limit)])
async def chat_stream(req: ChatRequest, request: Request) -> StreamingResponse:
    """Public SSE chat endpoint for the embeddable widget.

    The existing generation backends return a complete completion, and the
    fact checker necessarily needs that complete text. Consequently this
    endpoint emits token-sized chunks *after* fact checking, rather than
    exposing a fast but potentially unsupported claim and trying to retract
    it later. This is lower-latency than a non-streaming browser UI but is
    intentionally not a claim of first-token model streaming.
    """
    t0 = time.monotonic()
    conversation_id, history = request_history(req, request)
    try:
        await acquire_answer_lock(request)
        try:
            result = await asyncio.to_thread(
                answer,
                req.message,
                request.app.state.generate_fn,
                request.app.state.retrieval_index,
                system_prompt=SYSTEM_PROMPT,
                history=history,
            )
        finally:
            request.app.state.answer_lock.release()
    except LLMProviderError as exc:
        REQUESTS_TOTAL.labels(outcome="error").inc()
        logger.warning(
            "LLM provider error on streaming request: provider_status=%s http_status=%s",
            exc.provider_status,
            exc.http_status,
        )
        headers = {"Retry-After": exc.retry_after} if exc.retry_after else None
        raise HTTPException(status_code=exc.http_status, detail=exc.public_detail, headers=headers) from None
    except HTTPException:
        raise
    except Exception:
        REQUESTS_TOTAL.labels(outcome="error").inc()
        logger.exception("Error answering streaming chat request")
        raise HTTPException(status_code=500, detail="Internal error processing your request.") from None

    latency_seconds = time.monotonic() - t0
    outcome = answer_outcome(result)
    REQUESTS_TOTAL.labels(outcome=outcome).inc()
    REQUEST_LATENCY_SECONDS.observe(latency_seconds)
    save_conversation(request, conversation_id, req.message, result.response, outcome)

    async def events():
        # Whitespace-preserving chunks let a tiny widget append immediately
        # without needing a Markdown/tokenizer dependency in the browser.
        for chunk in result.response.splitlines(keepends=True):
            yield sse("token", {"text": chunk})
        yield sse(
            "done",
            {
                "conversation_id": conversation_id,
                "retrieved_fact_count": len(result.retrieved),
                "generated": result.generated,
                "latency_ms": latency_seconds * 1000,
            },
        )

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
