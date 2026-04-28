"""
api/main.py — FastAPI wrapper around Ollama or llama.cpp HTTP server.
"""

import os
from typing import AsyncGenerator

import httpx
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

CONFIG_PATH = "config.yaml"

with open(CONFIG_PATH) as f:
    _config = yaml.safe_load(f)

_api_cfg = _config.get("api", {})
BACKEND = _api_cfg.get("backend", "ollama")
OLLAMA_URL = _api_cfg.get("ollama_url", "http://localhost:11434")
LLAMACPP_URL = _api_cfg.get("llamacpp_url", "http://localhost:8080")
MODEL_NAME = _api_cfg.get("model_name", "my-custom-model")
API_KEY = _api_cfg.get("api_key", "") or os.getenv("OPENSENSE_API_KEY", "")

app = FastAPI(title="opensense API", version="0.1.0")


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


@app.get("/health")
def health():
    return {"status": "ok", "backend": BACKEND, "model": MODEL_NAME}


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request):
    _check_auth(request)
    async with httpx.AsyncClient(timeout=120) as client:
        if BACKEND == "ollama":
            r = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": MODEL_NAME, "prompt": body.message, "stream": False},
            )
            r.raise_for_status()
            return ChatResponse(response=r.json()["response"])
        else:
            r = await client.post(
                f"{LLAMACPP_URL}/completion",
                json={"prompt": body.message, "n_predict": 512, "stream": False},
            )
            r.raise_for_status()
            return ChatResponse(response=r.json()["content"])


@app.post("/chat/stream")
async def chat_stream(body: ChatRequest, request: Request):
    _check_auth(request)

    async def _stream_ollama() -> AsyncGenerator[str, None]:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_URL}/api/generate",
                json={"model": MODEL_NAME, "prompt": body.message, "stream": True},
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
                json={"prompt": body.message, "n_predict": 512, "stream": True},
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if line.startswith("data:"):
                        import json
                        chunk = json.loads(line[5:].strip())
                        yield f"data: {chunk.get('content', '')}\n\n"

    generator = _stream_ollama() if BACKEND == "ollama" else _stream_llamacpp()
    return StreamingResponse(generator, media_type="text/event-stream")
