import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE = "http://iii-engine:3111"

app = FastAPI()
_belief_map: dict[str, str] = {}


class Metadata(BaseModel):
    beliefId: str
    scope: str


class AddRequest(BaseModel):
    text: str
    user_id: str
    metadata: Metadata
    aliases: list[str] = []


class SearchRequest(BaseModel):
    query: str
    user_id: str
    limit: int = 20
    scope: str


class UpdateRequest(BaseModel):
    beliefId: str
    text: str
    user_id: str
    metadata: dict = {}


@app.post("/add")
def add(req: AddRequest):
    r = httpx.post(
        f"{BASE}/agentmemory/remember",
        json={
            "content": req.text,
            "project": req.metadata.scope,
            "userId": req.user_id,
            "title": req.metadata.beliefId,
            "metadata": {"beliefId": req.metadata.beliefId},
            "concepts": req.aliases,
        },
        timeout=30,
    )
    if not r.text:
        return {"ok": True}
    data = r.json()
    mem_id = (
        (data.get("memory") or {}).get("id") or data.get("id") or data.get("memoryId")
    )
    if mem_id and req.metadata.beliefId:
        _belief_map[str(mem_id)] = req.metadata.beliefId
    return {"ok": True}


@app.put("/update")
def update(req: UpdateRequest):
    mem_id = next((k for k, v in _belief_map.items() if v == req.beliefId), None)
    if mem_id is None:
        raise HTTPException(
            status_code=404, detail=f"beliefId {req.beliefId} not in belief map"
        )

    httpx.post(
        f"{BASE}/agentmemory/forget",
        json={
            "memoryId": mem_id,
        },
        timeout=30,
    )
    _belief_map.pop(mem_id, None)

    r = httpx.post(
        f"{BASE}/agentmemory/remember",
        json={
            "content": req.text,
            "project": req.metadata.get("scope", ""),
            "userId": req.user_id,
            "metadata": {"beliefId": req.beliefId},
        },
        timeout=30,
    )
    data = r.json()
    new_id = (
        (data.get("memory") or {}).get("id") or data.get("id") or data.get("memoryId")
    )
    if new_id:
        _belief_map[str(new_id)] = req.beliefId

    return {"ok": True}


@app.post("/search")
def search(req: SearchRequest):
    r = httpx.post(
        f"{BASE}/agentmemory/smart-search",
        json={
            "query": req.query,
            "project": req.scope,
            "userId": req.user_id,
            "limit": req.limit,
        },
        timeout=30,
    )

    memories = r.json().get("results") or r.json().get("memories", [])
    results = []
    seen = set()
    for m in memories:
        obs = m.get("observation", m)
        meta_bid = (obs.get("metadata") or {}).get("beliefId")
        bid = meta_bid or _belief_map.get(str(m.get("obsId") or m.get("id", "")))
        if not bid or bid in seen:
            continue
        seen.add(bid)
        results.append(
            {
                "id": bid,
                "memory": obs.get("content")
                or obs.get("narrative")
                or obs.get("memory", ""),
                "score": m.get("score", 1.0),
                "metadata": {"beliefId": bid},
            }
        )
    return {"results": results}


@app.delete("/reset")
def reset():
    _belief_map.clear()
    httpx.post(
        f"{BASE}/agentmemory/governance/bulk-delete",
        json={"all": True},
        timeout=30,
    )

    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8084)
