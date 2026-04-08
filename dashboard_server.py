import asyncio
from collections import deque
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

app = FastAPI(title="Smart Factory RL Dashboard API")
_connections: list[WebSocket] = []
_history: deque[dict[str, Any]] = deque(maxlen=2000)
_latest: dict[str, Any] = {}


class StepPayload(BaseModel):
    step: int
    jobs_completed: int
    queue_length: int
    machines: list[dict[str, Any]] | list[str]
    reward: float
    completion_rate: float


@app.get("/health")
def health():
    return {"status": "ok", "clients": len(_connections), "events": len(_history)}


@app.get("/state")
def state():
    return _latest


@app.get("/history")
def history():
    return list(_history)


@app.post("/ingest")
async def ingest(payload: StepPayload):
    global _latest
    data = payload.model_dump()
    _latest = data
    _history.append(data)

    dead = []
    for ws in _connections:
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in _connections:
            _connections.remove(ws)
    return {"ok": True}


@app.websocket("/ws")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    _connections.append(websocket)
    try:
        if _latest:
            await websocket.send_json(_latest)
        while True:
            # Keep connection alive; dashboard is mostly server-push
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _connections:
            _connections.remove(websocket)
