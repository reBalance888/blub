"""
main.py — FastAPI server with WebSocket for the BLUB Ocean simulation.
"""
from __future__ import annotations

import asyncio
import os
import json
from pathlib import Path
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .ocean import Ocean


# Load config
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

# Global ocean instance
ocean: Ocean | None = None

# WebSocket connections
viewer_connections: list[WebSocket] = []
agent_connections: dict[str, WebSocket] = {}  # agent_id -> ws

# Tick loop task
tick_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ocean, tick_task
    ocean = Ocean(CONFIG)
    tick_task = asyncio.create_task(tick_loop())
    print(f"BLUB Ocean started | Size: {ocean.size}x{ocean.size} | Active zone: {ocean.active_zone_size} | Epoch length: {CONFIG['economy']['epoch_length_ticks']} ticks")
    yield
    tick_task.cancel()
    try:
        await tick_task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="BLUB Ocean", lifespan=lifespan)


# ----- Tick loop -----

async def tick_loop():
    """Main simulation loop."""
    while True:
        await asyncio.sleep(ocean.tick_interval)
        try:
            ocean.process_tick()
        except Exception as e:
            import traceback
            print(f"[TICK_LOOP ERROR] tick={ocean.tick}: {e}")
            traceback.print_exc()
            continue

        # Broadcast viewer state
        viewer_state = ocean.get_viewer_state()
        dead = []
        for ws in viewer_connections:
            try:
                await ws.send_json(viewer_state)
            except Exception:
                dead.append(ws)
        for ws in dead:
            viewer_connections.remove(ws)

        # Broadcast agent states
        dead_agents = []
        for agent_id, ws in agent_connections.items():
            state = ocean.get_agent_state(agent_id)
            if state:
                try:
                    await ws.send_json(state)
                except Exception:
                    dead_agents.append(agent_id)
        for aid in dead_agents:
            agent_connections.pop(aid, None)

        # Clear speaking after broadcast
        for lob in ocean.lobsters.values():
            lob.speaking = []


# ----- HTTP Endpoints -----

@app.post("/connect")
async def connect_agent(body: dict):
    name = body.get("name", "unnamed")
    lob = ocean.add_lobster(name)
    zone_min, zone_max = ocean._active_zone_bounds()
    return {
        "agent_id": lob.id,
        "starting_balance": lob.balance,
        "position": [lob.x, lob.y],
        "ocean_size": ocean.size,
        "active_zone": {
            "size": ocean.active_zone_size,
            "min": zone_min,
            "max": zone_max,
        },
    }


@app.post("/disconnect")
async def disconnect_agent(body: dict):
    agent_id = body.get("agent_id")
    if agent_id and agent_id in ocean.lobsters:
        ocean.remove_lobster(agent_id)
        return {"ok": True}
    return JSONResponse({"error": "unknown agent"}, status_code=400)


@app.post("/action")
async def agent_action(body: dict):
    agent_id = body.get("agent_id")
    actions = body.get("actions", {})

    if not agent_id or agent_id not in ocean.lobsters:
        return JSONResponse({"error": "unknown agent"}, status_code=400)

    ocean.submit_action(agent_id, actions)
    lob = ocean.lobsters[agent_id]
    return {
        "ok": True,
        "credits_earned": round(lob.credits_earned, 2),
        "credits_spent": round(lob.credits_spent, 2),
        "net_credits": round(lob.credits_earned - lob.credits_spent, 2),
    }


@app.get("/state/{agent_id}")
async def get_state(agent_id: str):
    state = ocean.get_agent_state(agent_id)
    if not state:
        return JSONResponse({"error": "unknown agent"}, status_code=404)
    return state


@app.get("/stats")
async def get_stats():
    top = sorted(
        ocean.lobsters.values(),
        key=lambda l: l.credits_earned - l.credits_spent,
        reverse=True,
    )[:10]

    return {
        "tick": ocean.tick,
        "epoch": ocean.epoch_mgr.epoch,
        "epoch_ticks_remaining": ocean.epoch_mgr.ticks_remaining(),
        "total_lobsters": len(ocean.lobsters),
        "alive_lobsters": sum(1 for l in ocean.lobsters.values() if l.alive),
        "active_rifts": len(ocean.rift_mgr.active_rifts),
        "active_predators": len(ocean.predator_mgr.active_predators),
        "epoch_pool": CONFIG["economy"]["simulated_epoch_pool"],
        "leaderboard": [
            {
                "id": l.id,
                "name": l.name,
                "net_credits": round(l.credits_earned - l.credits_spent, 2),
                "tier": ocean.get_tier(l),
            }
            for l in top
        ],
    }


@app.post("/knowledge/deposit")
async def knowledge_deposit(body: dict):
    agent_id = body.get("agent_id")
    if not agent_id or agent_id not in ocean.lobsters:
        return JSONResponse({"error": "unknown agent"}, status_code=400)
    lob = ocean.lobsters[agent_id]
    min_age = ocean.config.get("cultural_cache", {}).get("min_agent_age_to_contribute", 300)
    if lob.age < min_age:
        return JSONResponse({"error": f"agent too young (age={lob.age}, need {min_age})"}, status_code=400)
    data = body.get("data", {})
    ocean.cultural_cache.contribute(data, agent_id=agent_id)
    return {"ok": True, "message": "deposit OK"}


@app.get("/knowledge/bootstrap")
async def knowledge_bootstrap():
    return ocean.cultural_cache.bootstrap()


@app.get("/knowledge/mentor")
async def knowledge_mentor(agent_id: str = ""):
    """Return knowledge snapshot from nearest experienced social agent (mentor).
    Colony-restricted: prefer mentors from same colony, fallback to any nearby."""
    lob = ocean.lobsters.get(agent_id)
    if not lob:
        return {"production": {}, "comprehension": {}}

    # Check if this lobster belongs to a colony
    my_colony = ocean.colony_mgr.get_colony_for(agent_id)
    colony_members = set(my_colony.members) if my_colony else set()

    # Find nearest mentor — prioritize same-colony members
    best_colony_dist = float("inf")
    best_colony_snapshot = None
    best_any_dist = float("inf")
    best_any_snapshot = None

    for other_id, other_lob in ocean.lobsters.items():
        if other_id == agent_id:
            continue
        if other_lob.agent_type != "social" or other_lob.age < 500 or not other_lob.alive:
            continue
        if other_id not in ocean.cultural_cache.agent_snapshots:
            continue
        dist = abs(lob.x - other_lob.x) + abs(lob.y - other_lob.y)
        if dist > 15:
            continue
        snapshot = ocean.cultural_cache.agent_snapshots[other_id]
        # Same colony mentor (preferred)
        if other_id in colony_members and dist < best_colony_dist:
            best_colony_dist = dist
            best_colony_snapshot = snapshot
        # Any nearby mentor (fallback)
        if dist < best_any_dist:
            best_any_dist = dist
            best_any_snapshot = snapshot

    return best_colony_snapshot or best_any_snapshot or {"production": {}, "comprehension": {}}


@app.post("/reset")
async def reset_world():
    ocean.reset()
    return {"ok": True, "message": "World reset"}


# ----- WebSocket Endpoints -----
# NOTE: /ws/viewer MUST be registered before /ws/{agent_id} to avoid
# "viewer" being captured as an agent_id parameter.

@app.websocket("/ws/viewer")
async def viewer_websocket(ws: WebSocket):
    await ws.accept()
    viewer_connections.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        if ws in viewer_connections:
            viewer_connections.remove(ws)
    except Exception:
        if ws in viewer_connections:
            viewer_connections.remove(ws)


@app.websocket("/ws/{agent_id}")
async def agent_websocket(ws: WebSocket, agent_id: str):
    await ws.accept()

    if agent_id not in ocean.lobsters:
        await ws.send_json({"error": "unknown agent, call /connect first"})
        await ws.close()
        return

    agent_connections[agent_id] = ws
    try:
        while True:
            data = await ws.receive_json()
            if "actions" in data:
                ocean.submit_action(agent_id, data["actions"])
    except WebSocketDisconnect:
        agent_connections.pop(agent_id, None)
        ocean.remove_lobster(agent_id)
    except Exception:
        agent_connections.pop(agent_id, None)
        ocean.remove_lobster(agent_id)


# ----- Serve viewer -----

VIEWER_PATH = Path(__file__).parent.parent / "viewer"

@app.get("/viewer")
async def serve_viewer():
    index = VIEWER_PATH / "index.html"
    if index.exists():
        return FileResponse(index)
    return HTMLResponse("<h1>Viewer not built yet</h1>")


# ----- Entry point -----

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
