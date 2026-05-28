

import httpx
from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel

BASE = "http://yourmemory:8005"  

TOP_K = 6


app = FastAPI()
_belief_map: dict[int, str] = {}

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

class UpdateRequest(BaseModel):
    beliefId: str
    text: str
    user_id: str
    metadata: dict = {}


@app.post("/add")
def add(req: AddRequest):
    r = httpx.post(f"{BASE}/memories", json={
        "userId": req.user_id,
        "content": req.text,
        "importance": 0.7,
        "contextPaths": [req.metadata.scope]
    })
    data = r.json()
    if (ym_id := data.get("id")) and (bid := req.metadata.beliefId):
        _belief_map[ym_id] = bid
    return {"ok": True}

@app.put("/update")
def update(req: UpdateRequest):
    ym_id = next((k for k, v in _belief_map.items() if v == req.beliefId), None)
    if ym_id is None:
        raise HTTPException(status_code=404, detail=f"beliefId {req.beliefId} not in belief map")

    r = httpx.put(f"{BASE}/memories/{ym_id}", json={
        "content": req.text,
        "importance": 0.7,
    })
    return r.json()


@app.post("/search")
def search(req: SearchRequest):
    r = httpx.post(f"{BASE}/retrieve", json={
        "userId": req.user_id,
        "query": req.query,
        "topK": TOP_K,
        "scoreThreshold": 0.55,
        "currentPath": req.scope
    })
    memories = r.json().get("memories", [])
    results = []
    seen = set()
    for m in memories:
        bid = _belief_map.get(m["id"])
        if not bid or bid in seen:
            continue
        seen.add(bid)
        results.append({
            "id": bid,
            "memory": m["content"],
            "score": m.get("score", 1.0),
            "metadata": {"beliefId": bid},
        })
    return {"results": results}


@app.delete("/reset")
def reset():
    _belief_map.clear()
    r = httpx.get(f"{BASE}/memories", params={"userId": "test-user", "limit": 40})
    for m in r.json().get("memories", []):
        httpx.delete(f"{BASE}/memories/{m['id']}")
    r = httpx.get(f"{BASE}/memories", params={"userId": "other-user", "limit": 5})
    for m in r.json().get("memories", []):
        httpx.delete(f"{BASE}/memories/{m['id']}")
    r = httpx.get(f"{BASE}/memories", params={"userId": "brand-new-user", "limit": 5})
    for m in r.json().get("memories", []):
        httpx.delete(f"{BASE}/memories/{m['id']}")
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)