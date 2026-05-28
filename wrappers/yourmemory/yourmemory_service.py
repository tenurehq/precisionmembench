import httpx
from fastapi import FastAPI
from pydantic import BaseModel

BASE = "http://yourmemory:8005"  
USER = "test-user"

app = FastAPI()
_belief_map: dict[int, str] = {}  


class AddRequest(BaseModel):
    text: str
    user_id: str
    metadata: dict = {}


class SearchRequest(BaseModel):
    query: str
    user_id: str
    limit: int = 20


@app.post("/add")
def add(req: AddRequest):
    r = httpx.post(f"{BASE}/memories", json={
        "userId": req.user_id,
        "content": req.text,
        "importance": 0.7,
    })
    data = r.json()
    if (ym_id := data.get("id")) and (bid := req.metadata.get("beliefId")):
        _belief_map[ym_id] = bid
    return {"ok": True}


@app.post("/search")
def search(req: SearchRequest):
    r = httpx.post(f"{BASE}/retrieve", json={
        "userId": req.user_id,
        "query": req.query,
        "topK": req.limit,
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
    r = httpx.get(f"{BASE}/memories", params={"userId": USER, "limit": 500})
    for m in r.json().get("memories", []):
        httpx.delete(f"{BASE}/memories/{m['id']}")
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)