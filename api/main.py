"""
api/main.py — FastAPI wrapper around Ollama or llama.cpp HTTP server.

Memory recall behaviour
-----------------------
Warm (recently-accessed) facts are searched first.  If warm results are
insufficient (fewer than top_k facts above min_score), the cold store is
searched as well.  Cold recall incurs a configurable async delay
(cold_recall_delay_seconds) to simulate the slower recognition humans
experience when recalling something they haven't thought about in months.
Cold memories that are successfully recalled are automatically restored to
warm storage.
"""

import asyncio
import os
from typing import AsyncGenerator

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.authenticator import authenticate
from api.memory import MemoryStore

CONFIG_PATH = "config.yaml"

with open(CONFIG_PATH) as f:
    _config = yaml.safe_load(f)

_api_cfg = _config.get("api", {})
_mem_cfg = _config.get("memory", {})

BACKEND = _api_cfg.get("backend", "ollama")
OLLAMA_URL = _api_cfg.get("ollama_url", "http://localhost:11434")
LLAMACPP_URL = _api_cfg.get("llamacpp_url", "http://localhost:8080")
MODEL_NAME = _api_cfg.get("model_name", "my-custom-model")
API_KEY = _api_cfg.get("api_key", "") or os.getenv("OPENSENSE_API_KEY", "")

MEMORY_ENABLED = _mem_cfg.get("enabled", True)
MEMORY_DIR = _mem_cfg.get("persist_dir", "memory/db")
MEMORY_TOP_K = int(_mem_cfg.get("top_k", 5))
MEMORY_MIN_SCORE = float(_mem_cfg.get("min_score", 0.45))
MEMORY_MODEL_CHECK = _mem_cfg.get("model_check", True)
MEMORY_MIN_CONFIDENCE = float(_mem_cfg.get("min_confidence", 0.6))

# Cold-tier config
DUMP_AFTER_DAYS = int(_mem_cfg.get("dump_after_days", 180))
RECENCY_HALF_LIFE_DAYS = int(_mem_cfg.get("recency_half_life_days", 30))
COLD_SCORE_PENALTY = float(_mem_cfg.get("cold_score_penalty", 0.6))
COLD_RECALL_DELAY = float(_mem_cfg.get("cold_recall_delay_seconds", 3.0))

app = FastAPI(title="opensense API", version="0.1.0")

_memory: MemoryStore | None = None


@app.on_event("startup")
async def _startup():
    global _memory
    if MEMORY_ENABLED:
        _memory = MemoryStore(
            persist_dir=MEMORY_DIR,
            dump_after_days=DUMP_AFTER_DAYS,
            recency_half_life_days=RECENCY_HALF_LIFE_DAYS,
            cold_score_penalty=COLD_SCORE_PENALTY,
        )


def _check_auth(request: Request):
    if not API_KEY:
        return
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    memory_hits: list[str] = []        # warm facts injected (for transparency)
    cold_memory_hits: list[str] = []   # cold facts recalled after delay (for transparency)
    cold_recall_delay_applied: bool = False


class LearnRequest(BaseModel):
    information: str
    source: str = "user"


class LearnResponse(BaseModel):
    stored: bool
    fact_id: str | None = None
    verdict: str
    confidence: float
    reason: str


class MemoryListResponse(BaseModel):
    count: int
    facts: list[dict]


class MemoryDumpResponse(BaseModel):
    dumped: int          # facts moved to cold this run
    warm_remaining: int  # facts still in warm storage
    cold_total: int      # total facts now in cold storage


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _build_prompt(message: str, memory_facts: list[str]) -> str:
    """Prepend remembered facts to the user message as context."""
    if not memory_facts:
        return message
    facts_block = "\n".join(f"- {f}" for f in memory_facts)
    return (
        f"[Remembered context]\n{facts_block}\n\n"
        f"[User message]\n{message}"
    )


async def _retrieve_memory_facts(
    query: str,
) -> tuple[list[str], list[str], bool]:
    """
    Two-tier memory retrieval.

    1. Search warm (fast, recency-weighted).
    2. If warm results are fewer than top_k, search cold after applying a
       delay (cold_recall_delay_seconds) to simulate slow recall.
    3. Cold memories that are recalled are automatically restored to warm.

    Returns
    -------
    warm_texts : list[str]
        Texts from warm memory that passed the score threshold.
    cold_texts : list[str]
        Texts recalled from cold memory (after delay) that passed threshold.
    cold_delay_applied : bool
        True if the cold store was consulted and the delay was incurred.
    """
    if _memory is None:
        return [], [], False

    # --- Warm tier ---
    warm_hits = _memory.search(query, top_k=MEMORY_TOP_K)
    good_warm = [h for h in warm_hits if h["score"] >= MEMORY_MIN_SCORE]

    # --- Cold tier (only if warm results are insufficient) ---
    cold_texts: list[str] = []
    cold_delay_applied = False
    cold_needed = MEMORY_TOP_K - len(good_warm)

    if cold_needed > 0 and _memory.count_cold() > 0:
        # Simulate slow recall — this is the "slower than main memory" penalty
        await asyncio.sleep(COLD_RECALL_DELAY)
        cold_delay_applied = True

        cold_hits = _memory.search_cold(query, top_k=cold_needed)
        for hit in cold_hits:
            if hit["score"] >= MEMORY_MIN_SCORE:
                cold_texts.append(hit["text"])
                # Re-learning through recognition — restore to warm
                _memory.restore_from_cold(hit["id"])

    warm_texts = [h["text"] for h in good_warm]
    return warm_texts, cold_texts, cold_delay_applied


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.get("/health")
def health():
    warm_count = _memory.count() if _memory else 0
    cold_count = _memory.count_cold() if _memory else 0
    return {
        "status": "ok",
        "backend": BACKEND,
        "model": MODEL_NAME,
        "memory_enabled": MEMORY_ENABLED,
        "memory_facts_warm": warm_count,
        "memory_facts_cold": cold_count,
        "dump_after_days": DUMP_AFTER_DAYS,
        "recency_half_life_days": RECENCY_HALF_LIFE_DAYS,
        "cold_recall_delay_seconds": COLD_RECALL_DELAY,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request):
    _check_auth(request)

    warm_facts, cold_facts, cold_delay = await _retrieve_memory_facts(body.message)
    all_facts = warm_facts + cold_facts
    prompt = _build_prompt(body.message, all_facts)

    async with httpx.AsyncClient(timeout=120) as client:
        if BACKEND == "ollama":
            r = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
            )
            r.raise_for_status()
            return ChatResponse(
                response=r.json()["response"],
                memory_hits=warm_facts,
                cold_memory_hits=cold_facts,
                cold_recall_delay_applied=cold_delay,
            )
        else:
            r = await client.post(
                f"{LLAMACPP_URL}/completion",
                json={"prompt": prompt, "n_predict": 512, "stream": False},
            )
            r.raise_for_status()
            return ChatResponse(
                response=r.json()["content"],
                memory_hits=warm_facts,
                cold_memory_hits=cold_facts,
                cold_recall_delay_applied=cold_delay,
            )


@app.post("/learn", response_model=LearnResponse)
async def learn(body: LearnRequest, request: Request):
    """
    Submit new information for the model to learn.

    The information passes a two-layer authenticity gate:
      1. Heuristic checks (length, gibberish, injection patterns)
      2. Model-assisted plausibility check (optional, controlled by config)

    If it passes, the fact is embedded and stored in the persistent vector
    memory so future /chat calls can retrieve and use it.
    """
    _check_auth(request)

    if not MEMORY_ENABLED or _memory is None:
        raise HTTPException(status_code=503, detail="Memory is disabled in config.")

    result = await authenticate(
        body.information,
        use_model=MEMORY_MODEL_CHECK,
        backend=BACKEND,
        ollama_url=OLLAMA_URL,
        llamacpp_url=LLAMACPP_URL,
        model_name=MODEL_NAME,
        min_confidence=MEMORY_MIN_CONFIDENCE,
    )

    if result.verdict == "rejected":
        return LearnResponse(
            stored=False,
            verdict=result.verdict,
            confidence=result.confidence,
            reason=result.reason,
        )

    fact_id = _memory.add(
        body.information,
        source=body.source,
        extra={"verdict": result.verdict, "confidence": result.confidence},
    )

    return LearnResponse(
        stored=True,
        fact_id=fact_id,
        verdict=result.verdict,
        confidence=result.confidence,
        reason=result.reason,
    )


@app.get("/memory", response_model=MemoryListResponse)
def memory_list(request: Request):
    """List all facts currently stored in warm memory."""
    _check_auth(request)
    if _memory is None:
        return MemoryListResponse(count=0, facts=[])
    facts = _memory.list_all()
    return MemoryListResponse(count=len(facts), facts=facts)


@app.get("/memory/cold", response_model=MemoryListResponse)
def memory_cold_list(request: Request):
    """List all facts currently in cold (dumped) storage."""
    _check_auth(request)
    if _memory is None:
        return MemoryListResponse(count=0, facts=[])
    facts = _memory.list_cold()
    return MemoryListResponse(count=len(facts), facts=facts)


@app.post("/memory/dump", response_model=MemoryDumpResponse)
def memory_dump(request: Request):
    """
    Trigger a memory dump: move all warm memories not accessed for
    ``dump_after_days`` to cold storage.

    Run this periodically (e.g. via a cron job or scheduled task).
    """
    _check_auth(request)
    if _memory is None:
        raise HTTPException(status_code=503, detail="Memory is disabled in config.")
    dumped = _memory.dump_stale()
    return MemoryDumpResponse(
        dumped=dumped,
        warm_remaining=_memory.count(),
        cold_total=_memory.count_cold(),
    )


@app.delete("/memory/{fact_id}")
def memory_delete(fact_id: str, request: Request):
    """Remove a specific fact from memory by its ID."""
    _check_auth(request)
    if _memory is None:
        raise HTTPException(status_code=503, detail="Memory is disabled in config.")
    removed = _memory.delete(fact_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Fact not found.")
    return {"deleted": fact_id}


@app.post("/chat/stream")
async def chat_stream(body: ChatRequest, request: Request):
    _check_auth(request)

    warm_facts, cold_facts, _cold_delay = await _retrieve_memory_facts(body.message)
    prompt = _build_prompt(body.message, warm_facts + cold_facts)

    async def _stream_ollama() -> AsyncGenerator[str, None]:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_URL}/api/generate",
                json={"model": MODEL_NAME, "prompt": prompt, "stream": True},
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if line:
                        import json
                        chunk = json.loads(line)
                        yield f"data: {chunk.get('response', '')}\n\n"

    async def _stream_llamacpp() -> AsyncGenerator[str, None]:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{LLAMACPP_URL}/completion",
                json={"prompt": prompt, "n_predict": 512, "stream": True},
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if line.startswith("data:"):
                        import json
                        chunk = json.loads(line[5:].strip())
                        yield f"data: {chunk.get('content', '')}\n\n"

    generator = _stream_ollama() if BACKEND == "ollama" else _stream_llamacpp()
    return StreamingResponse(generator, media_type="text/event-stream")
