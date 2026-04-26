import hashlib
import json
import os
import time
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel, Field

load_dotenv()

from agent.agent_builder import sales_agent  # noqa: E402
from tools.rag import RAGPipeline  # noqa: E402
from tools.rag_tool import set_rag_instance  # noqa: E402
from tools.request_context import current_user_id  # noqa: E402

# Env: GROQ, OPENAI, TAVILY, WEAVIATE, MEM0, REDIS (optional MEM0/REDIS)
REDIS_URL = os.getenv("REDIS_URL")
REDIS_TTL_SECONDS = int(os.getenv("REDIS_TTL_SECONDS", "900"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "86400"))
# Max *messages* in Redis (user+assistant), e.g. 24 = 12 full turns
SESSION_MAX_MESSAGES = int(os.getenv("SESSION_MAX_MESSAGES", "24"))
MEM0_API_KEY = os.getenv("MEM0_API_KEY")
# Bump when cache key shape changes
CACHE_VERSION = "3"
SESSION_HISTORY_VERSION = "1"

_redis = None
if REDIS_URL:
    try:
        import redis as redis_lib

        _redis = redis_lib.from_url(REDIS_URL, decode_responses=True)
    except Exception:
        _redis = None

_mem0 = None
if MEM0_API_KEY:
    try:
        from mem0 import MemoryClient

        _mem0 = MemoryClient(api_key=MEM0_API_KEY)
    except Exception:
        _mem0 = None

# Same RAG instance for /upload and agent tool rag_search (Weaviate v4 client)
rag = RAGPipeline()
set_rag_instance(rag)
agent = sales_agent()


@asynccontextmanager
async def _lifespan(_: FastAPI):
    yield
    try:
        rag.client.close()
    except Exception:
        pass


app = FastAPI(
    title="Foodnests sales agent API",
    version="1.0.0",
    lifespan=_lifespan,
    description=(
        "Chat with a sales agent that can answer from **uploaded .txt documents** (RAG) or "
        "the **open web** (Tavily). "
        "Use **Swagger UI** at `/docs`: upload a file with **POST /upload**, then ask questions "
        "with **POST /chat**. Alternatively use **/redoc**."
    ),
    openapi_tags=[
        {
            "name": "docs",
            "description": "API discovery and OpenAPI (Swagger) links.",
        },
        {
            "name": "chat",
            "description": "Conversational agent. Routes internally: document questions use RAG, "
            "web-style questions use search, short greetings are answered without tools when appropriate.",
        },
        {
            "name": "documents",
            "description": "Upload plain-text files into the Weaviate-backed knowledge base for rag_search.",
        },
    ],
)


def _norm_session_id(session_id: str | None) -> str:
    s = (session_id or "").strip()
    return s if s else "default"


def _cache_key(user_id: str, session_id: str | None, message: str) -> str:
    sid = _norm_session_id(session_id)
    h = hashlib.sha256(
        f"{CACHE_VERSION}|{user_id}|{sid}|{message}".encode("utf-8")
    ).hexdigest()
    return f"chat:{h}"


def _session_history_key(user_id: str, session_id: str) -> str:
    return f"session:hist:{SESSION_HISTORY_VERSION}:{user_id}:{session_id}"


def _session_history_get(user_id: str, session_id: str | None) -> list[dict[str, str]]:
    if not _redis or not session_id or not (session_id or "").strip():
        return []
    try:
        raw = _redis.get(_session_history_key(user_id, session_id.strip()))
        if not raw:
            return []
        data = json.loads(raw)
        if not isinstance(data, list):
            return []
        return [x for x in data if isinstance(x, dict) and x.get("role") in ("user", "assistant") and "content" in x]
    except Exception:
        return []


def _session_history_save(
    user_id: str, session_id: str, user_msg: str, assistant_msg: str
) -> None:
    if not _redis or not session_id or not session_id.strip():
        return
    try:
        sid = session_id.strip()
        hist = _session_history_get(user_id, sid)
        hist.append({"role": "user", "content": user_msg})
        hist.append({"role": "assistant", "content": assistant_msg})
        if len(hist) > SESSION_MAX_MESSAGES:
            hist = hist[-SESSION_MAX_MESSAGES:]
        _redis.setex(
            _session_history_key(user_id, sid),
            SESSION_TTL_SECONDS,
            json.dumps(hist),
        )
    except Exception:
        return


def _cache_get(key: str):
    """Return (text, response_source) or (None, None) on miss. Legacy values are plain strings -> source *cached*."""
    if not _redis:
        return None, None
    try:
        raw = _redis.get(key)
        if raw is None:
            return None, None
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "response" in obj:
                return obj["response"], str(
                    obj.get("response_source", "unknown")
                )
        except (json.JSONDecodeError, TypeError):
            pass
        return raw, "cached"
    except Exception:
        return None, None


def _cache_set(key: str, value: str, response_source: str) -> None:
    if not _redis:
        return
    try:
        payload = json.dumps(
            {"response": value, "response_source": response_source}
        )
        _redis.setex(key, REDIS_TTL_SECONDS, payload)
    except Exception:
        pass


def _infer_response_source(agent_messages) -> str:
    """Classify the last turn: which tools ran (rag_search / tavily_search)."""
    saw_rag = False
    saw_tav = False
    for m in agent_messages:
        if getattr(m, "type", None) == "tool" or type(m).__name__ == "ToolMessage":
            n = getattr(m, "name", None)
            if n == "rag_search":
                saw_rag = True
            elif n == "tavily_search":
                saw_tav = True
        for tc in getattr(m, "tool_calls", None) or ():
            n = None
            if isinstance(tc, dict):
                n = tc.get("name")
            else:
                n = getattr(tc, "name", None)
            if n == "rag_search":
                saw_rag = True
            elif n == "tavily_search":
                saw_tav = True
    if saw_rag and saw_tav:
        return "rag_and_web"
    if saw_rag:
        return "rag"
    if saw_tav:
        return "web_search"
    return "direct"


def _mem0_context(user_id: str, query: str) -> str:
    if not _mem0 or not (query and query.strip()):
        return ""
    try:
        # Mem0 V3 search API (/v3/memories/search/); V1 can return 400 for current payloads
        res = _mem0.search(
            query,
            version="v3",
            filters={"user_id": user_id},
        )
        if not isinstance(res, dict):
            return ""
        results = res.get("results", [])
        parts = [
            str(x.get("memory", "")).strip()
            for x in results
            if isinstance(x, dict) and x.get("memory")
        ]
        if not parts:
            return ""
        return "Relevant long-term user facts: " + "; ".join(parts[:12])
    except Exception:
        return ""


def _mem0_add(user_id: str, user_text: str, assistant_text: str) -> None:
    if not _mem0:
        return
    try:
        _mem0.add(
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ],
            user_id=user_id,
        )
    except Exception:
        return


# =========================
# REQUEST SCHEMA
# =========================
class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        description="User message. Can be a greeting, a question about an uploaded document, or a request that needs a web search.",
        examples=["What does the uploaded file say about pricing?"],
    )
    user_id: str = Field(
        "default",
        description="Identifier for the end user. Used for RAG (uploads), Mem0 long-term memory, and (with session_id) response cache keys.",
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "Optional conversation / thread id. When set, the server (with Redis) keeps a **rolling in-session message history** "
            "and scopes **response cache** to this id so the same `message` in another session is not a cache hit. "
            "Use a fresh UUID per browser tab or chat thread. Omitted = single-turn behavior (no server-side history)."
        ),
    )


class ChatResponse(BaseModel):
    response: str
    processing_time: float
    response_source: str = Field(
        ...,
        description=(
            "How the answer was produced: **direct** (no tools), **rag** (rag_search only), "
            "**web_search** (tavily_search only), **rag_and_web** (both), **cached** (Redis hit; "
            "no model run for this request)."
        ),
    )


# =========================
# DOCS / SWAGGER DISCOVERY
# =========================
@app.get("/", tags=["docs"])
async def root():
    return {
        "swagger_ui": "/docs",
        "openapi_json": "/openapi.json",
        "redoc": "/redoc",
        "endpoints": {"upload": "POST /upload", "chat": "POST /chat"},
    }


# =========================
# CHAT ENDPOINT
# =========================
@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["chat"],
    summary="Send a message to the sales agent",
    response_description="Assistant text, which tools informed the answer, and server time in seconds.",
)
async def chat(req: ChatRequest):
    start_time = time.time()
    sid = (req.session_id or "").strip() or None
    ck = _cache_key(req.user_id, req.session_id, req.message)
    cached_text, _ = _cache_get(ck)
    if cached_text is not None:
        elapsed = time.time() - start_time
        return {
            "response": cached_text,
            "processing_time": round(elapsed, 2),
            "response_source": "cached",
        }

    token = current_user_id.set(req.user_id)
    try:
        mem_ctx = _mem0_context(req.user_id, req.message)
        session_hist = _session_history_get(req.user_id, req.session_id) if sid else []

        messages = []
        if mem_ctx:
            messages.append({"role": "system", "content": mem_ctx})
        for turn in session_hist:
            messages.append(turn)
        messages.append({"role": "user", "content": req.message})

        result = agent.invoke({"messages": messages})
        out_msgs = result["messages"]
        text = out_msgs[-1].content
        response_source = _infer_response_source(out_msgs)

        _mem0_add(req.user_id, req.message, text)
        _cache_set(ck, text, response_source)
        if sid:
            _session_history_save(req.user_id, sid, req.message, text)

        elapsed_time = time.time() - start_time
        return {
            "response": text,
            "processing_time": round(elapsed_time, 2),
            "response_source": response_source,
        }
    finally:
        current_user_id.reset(token)


# =========================
# FILE UPLOAD (RAG)
# =========================
@app.post(
    "/upload",
    tags=["documents"],
    summary="Upload a .txt file for RAG (document Q&A)",
    response_description="Success message or validation error. Indexing is synchronous. Scope uploads with user_id.",
)
async def upload_file(
    file: UploadFile = File(
        ...,
        description="Plain-text .txt only; UTF-8. Content is chunked and embedded for rag_search for this user only.",
    ),
    user_id: str = Form(
        "default",
        description="Binds the document to this user; rag_search only returns chunks for the same user_id as in /chat.",
    ),
):
    if not (file.filename or "").endswith(".txt"):
        return {"error": "Only .txt files supported"}

    content = await file.read()
    text = content.decode("utf-8")

    rag.add_document(text, file.filename, user_id=user_id)

    return {
        "message": f"{file.filename} uploaded successfully for user {user_id!r}"
    }
