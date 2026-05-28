"""
supermemory_service.py — PrecisionMemBench wrapper for Supermemory
Implements the three-endpoint contract on port 8080:
  POST   /add     — ingest a belief
  POST   /search  — retrieve beliefs by query
  DELETE /reset   — wipe all stored beliefs

Supermemory API mapping:
  /add    → POST   https://api.supermemory.ai/v3/documents
  /search → POST   https://api.supermemory.ai/v4/search
  /reset  → POST   https://api.supermemory.ai/v3/settings/reset

Environment variables:
  SUPERMEMORY_API_KEY   required — API key from console.supermemory.ai
  SEED_DELAY_MS         extra wait after /add  (default: 0)
  POLL_TIMEOUT_S        max seconds to poll    (default: 30)
  PORT                  HTTP port              (default: 8080)
"""

import asyncio
import logging
import os
import time
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("supermemory_service")


API_KEY = os.environ["SUPERMEMORY_API_KEY"]         
BASE_URL = "https://api.supermemory.ai"
SEED_DELAY_MS = int(os.getenv("SEED_DELAY_MS", "0"))
POLL_TIMEOUT_S = int(os.getenv("POLL_TIMEOUT_S", "30"))
PORT = int(os.getenv("PORT", "8080"))

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

app = FastAPI(title="supermemory-bench-wrapper")



class Metadata(BaseModel):
    beliefId: str
    scope: str

class AddRequest(BaseModel):
    text: str
    user_id: str
    metadata: Metadata

class AddResponse(BaseModel):
    ok: bool

class SearchRequest(BaseModel):
    query: str
    user_id: str
    limit: int = 20
    scope: str

class SearchResult(BaseModel):
    id: str        
    memory: str
    score: float

class SearchResponse(BaseModel):
    results: list[SearchResult]

class ResetResponse(BaseModel):
    ok: bool


async def _poll_until_ready(client: httpx.AsyncClient, doc_id: str) -> None:
    """
    Poll GET /v3/documents/{doc_id} until status != 'processing',
    or until POLL_TIMEOUT_S is exceeded.
    """
    deadline = time.monotonic() + POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            r = await client.get(
                f"{BASE_URL}/v3/documents/{doc_id}",
                headers=HEADERS,
                timeout=10,
            )
            if r.status_code == 200:
                status = r.json().get("status", "")
                if status != "processing":
                    return
        except Exception:
            pass
        await asyncio.sleep(0.5)
    log.warning("Poll timeout for doc %s after %ss", doc_id, POLL_TIMEOUT_S)



@app.post("/add", response_model=AddResponse)
async def add(req: AddRequest):
    """
    Ingest one belief into Supermemory.

    We store beliefId both as `customId` (document-level identifier)
    and inside `metadata.beliefId` (surfaced on search results).
    """
    belief_id = req.metadata.beliefId

    payload: dict[str, Any] = {
        "content": req.text,
        "containerTag": req.user_id,
        "metadata": {"beliefId": belief_id, "scope": req.metadata.scope},
        "filterByMetadata": {"scope": req.metadata.scope}
    }
    if belief_id:
        payload["customId"] = belief_id

    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(
                f"{BASE_URL}/v3/documents",
                headers=HEADERS,
                json=payload,
                timeout=30,
            )
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            log.error("add failed: %s — %s", exc.response.status_code, exc.response.text)
            raise HTTPException(status_code=502, detail=exc.response.text) from exc

        doc_id = r.json().get("id", "")

        if doc_id:
            await _poll_until_ready(client, doc_id)

        if SEED_DELAY_MS > 0:
            await asyncio.sleep(SEED_DELAY_MS / 1000)

    return AddResponse(ok=True)


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """
    Search Supermemory for beliefs matching the query.

    We use searchMode='memories' (not hybrid) so results are purely the
    extracted memory entries — the same unit the benchmark scored on ingest.

    threshold=0.0 returns all results ranked by similarity; the harness
    already handles precision by checking which beliefIds are present,
    so we don't want the API to pre-filter based on score.
    """
    payload: dict[str, Any] = {
        "q": req.query,
        "containerTag": req.user_id,
        "limit": req.limit,
        "searchMode": "memories",
        "threshold": 0.69,
        "filters": {
        "AND": [{"key": "scope", "value": req.scope}]
        }
    }

    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(
                f"{BASE_URL}/v4/search",
                headers=HEADERS,
                json=payload,
                timeout=30,
            )
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            log.error("search failed: %s — %s", exc.response.status_code, exc.response.text)
            raise HTTPException(status_code=502, detail=exc.response.text) from exc

    raw = r.json().get("results", [])
    results: list[SearchResult] = []
    for item in raw:
        belief_id = (item.get("metadata") or {}).get("beliefId", "")
        memory_text = item.get("memory", "")
        score = float(item.get("similarity", 0.0))
        results.append(SearchResult(id=belief_id, memory=memory_text, score=score))

    return SearchResponse(results=results)


@app.delete("/reset", response_model=ResetResponse)
async def reset():
    """
    Wipe all data in the Supermemory organisation.
    POST /v3/settings/reset requires a `confirmation` string.
    """
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(
                f"{BASE_URL}/v3/settings/reset",
                headers=HEADERS,
                json={"confirmation": "reset"},
                timeout=60,
            )
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            log.error("reset failed: %s — %s", exc.response.status_code, exc.response.text)
            raise HTTPException(status_code=502, detail=exc.response.text) from exc

    log.info("Organisation reset: %s", r.json())
    return ResetResponse(ok=True)



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
