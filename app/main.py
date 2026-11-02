from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import get_settings
from app.binance_client import BinanceFuturesClient
from app.engine import TradingEngine

settings = get_settings()
client = BinanceFuturesClient(
    api_key=settings.api_key, api_secret=settings.api_secret,
    rest_base=settings.rest_base, ws_base=settings.ws_base
)
engine = TradingEngine(settings, client)

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.validate_live()
    yield
    await engine.stop()
    await client.close()

app = FastAPI(title="Scalper Pro", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/dashboard.html", "r") as f:
        return f.read()

@app.get("/healthz")
async def health():
    return {"status": "ok", "mode": settings.APP_MODE}

@app.get("/api/status")
async def get_status():
    return {
        "mode": settings.APP_MODE,
        "is_running": engine.is_running,
        "state": engine.state,
        "balance": engine.risk.current_balance,
        "positions": engine.open_positions,
        "logs": engine.logs[-20:],
        "prices": engine.latest_prices
    }

@app.get("/api/stream")
async def stream():
    async def event_generator():
        while True:
            data = {
                "balance": engine.risk.current_balance,
                "is_running": engine.is_running,
                "positions": len(engine.open_positions),
                "logs": engine.logs[-5:]
            }
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(2)
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/start")
async def start_bot():
    await engine.start()
    return {"status": "started"}

@app.post("/api/stop")
async def stop_bot():
    await engine.stop()
    return {"status": "stopped"}

@app.post("/api/panic")
async def panic():
    await engine.panic()
    return {"status": "panic_triggered"}
