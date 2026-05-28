from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import uvicorn
import os

ATOMICMEMORY_URL = os.getenv("ATOMICMEMORY_URL", "http://localhost:17350")
API_KEY = os.getenv("ATOMICMEMORY_API_KEY", "local-dev-key")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

TIMEOUT = httpx.Timeout(120.0)

app = FastAPI()
id_map: dict[str, str] = {}

class Metadata(BaseModel):
    beliefId: str
    scope: str

class AddRequest(BaseModel):
    text: str
    user_id: str
    metadata: Metadata

class SearchRequest(BaseModel):
    query: str
    user_id: str
    limit: int = 20
    scope: str

@app.post("/add")
def add(req: AddRequest):
    belief_id = req.metadata.beliefId
    payload = {
        "user_id": req.user_id,
        "conversation": f"user: {req.text}",
        "source_site": "precisionmembench",
        "agent_scope": req.metadata.scope
    }
    response = httpx.post(
        f"{ATOMICMEMORY_URL}/v1/memories/ingest", json=payload, headers=HEADERS, timeout=TIMEOUT,
    )
    data = response.json()
    if belief_id:
        for internal_id in data.get("stored_memory_ids", []):
            id_map[internal_id] = belief_id
        for internal_id in data.get("updated_memory_ids", []):
            id_map[internal_id] = belief_id
    return {"ok": True}

@app.post("/search")
def search(req: SearchRequest):
    payload = {
        "user_id": req.user_id,
        "query": req.query,
        "limit": req.limit,
        "agent_scope": req.scope
    }
    response = httpx.post(
        f"{ATOMICMEMORY_URL}/v1/memories/search", json=payload, headers=HEADERS, timeout=TIMEOUT,
    )
    data = response.json()
    memories = data.get("memories", [])

    normalized = []
    seen = set()
    for r in memories:
        internal_id = r.get("id")
        belief_id = id_map.get(internal_id)
        if not belief_id or belief_id in seen:
            continue
        seen.add(belief_id)
        normalized.append(
            {
                "id": belief_id,
                "memory": r.get("content", ""),
                "score": r.get("score", r.get("similarity", 1.0)),
            }
        )

    return {"results": normalized}


@app.delete("/reset")
def reset():
    response = httpx.post(
        f"{ATOMICMEMORY_URL}/v1/memories/search",
        json={"query": "*", "limit": 1000},
        headers=HEADERS,
    )
    memories = response.json().get("memories", [])
    for m in memories:
        memory_id = m.get("id")
        if memory_id:
            httpx.delete(
                f"{ATOMICMEMORY_URL}/v1/memories/{memory_id}", headers=HEADERS
            )
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)