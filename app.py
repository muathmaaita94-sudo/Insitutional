"""Run:  python server.py   then open http://127.0.0.1:8000 (on Render: uses $PORT)"""
import asyncio
import json
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from engine import STEP_SECONDS, Sim, load_cfg

HERE = os.path.dirname(os.path.abspath(__file__))
state = {"sim": None}


async def loop():
    while True:
        try:
            state["sim"].step()
        except Exception as e:  # keep the loop alive, show the error in the terminal panel
            state["sim"].log("WARN", f"engine error: {type(e).__name__}: {e}")
        await asyncio.sleep(STEP_SECONDS)


@asynccontextmanager
async def lifespan(app):
    state["sim"] = Sim(load_cfg())
    task = asyncio.create_task(loop())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()

    async def reader():
        while True:
            msg = json.loads(await sock.receive_text())
            cmd = msg.get("cmd")
            if cmd == "reset":
                state["sim"] = Sim(load_cfg())
            elif cmd in ("pause", "resume", "kill"):
                state["sim"].command(cmd)

    task = asyncio.create_task(reader())
    try:
        while True:
            await sock.send_text(json.dumps(state["sim"].snapshot()))
            await asyncio.sleep(1)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        task.cancel()


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", 8000)), log_level="warning")