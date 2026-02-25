"""
base_agent.py — Base class for all BLUB Ocean agents.
"""
from __future__ import annotations

import asyncio
import aiohttp

SOUNDS = [
    "blub", "glorp", "skree", "klak", "mrrp",
    "woosh", "pop", "zzzub", "frrr", "tink",
    "bloop", "squee", "drrrn", "gulp", "hiss",
    "bonk", "splat", "chirr", "wub", "clonk",
    "fizz", "grumble", "ping", "splish", "croak",
    "zzzt", "plop", "whirr", "snap", "burble",
]


class BlubAgent:
    """Base class for creating $BLUB Ocean agents (lobsters)."""

    def __init__(self, name: str, server_url: str = "http://localhost:8000"):
        self.name = name
        self.server_url = server_url
        self.agent_id: str | None = None
        self.state: dict | None = None
        self.memory: dict = {}
        self.sound_model: dict = {}

    async def connect(self) -> dict:
        """Connect to the ocean server."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.server_url}/connect",
                json={"name": self.name},
            ) as resp:
                data = await resp.json()
                self.agent_id = data["agent_id"]
                return data

    async def get_state(self) -> dict:
        """Get current state from server."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.server_url}/state/{self.agent_id}"
            ) as resp:
                self.state = await resp.json()
                return self.state

    async def do_action(self, move: str = "stay", speak: list | None = None, act: str | None = None) -> dict:
        """Send an action to the server."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.server_url}/action",
                json={
                    "agent_id": self.agent_id,
                    "actions": {
                        "move": move,
                        "speak": speak or [],
                        "act": act,
                    },
                },
            ) as resp:
                return await resp.json()

    def think(self, state: dict) -> dict:
        """
        Override this method.
        Receives state, returns {"move": ..., "speak": [...], "act": ...}
        """
        return {"move": "stay", "speak": [], "act": None}

    def on_sounds_heard(self, sounds: list):
        """
        Override this method.
        Called when the agent hears sounds. Build/update language model here.
        """
        pass

    async def run(self):
        """Main agent loop."""
        await self.connect()
        print(f"[{self.name}] Connected as {self.agent_id}")

        while True:
            try:
                state = await self.get_state()

                if not state.get("alive", True):
                    await asyncio.sleep(1)
                    continue

                # Update language model
                if state.get("sounds_heard"):
                    self.on_sounds_heard(state["sounds_heard"])

                # Think and act
                action = self.think(state)
                await self.do_action(**action)

                # Telemetry every 50 ticks
                if state.get("tick", 0) % 50 == 0:
                    print(f"[{self.name}] tick={state.get('tick')} credits={state.get('my_net_credits', '?')} pos={state.get('my_position', '?')}")

            except aiohttp.ClientError as e:
                print(f"[{self.name}] Connection error: {e}")
            except Exception as e:
                print(f"[{self.name}] Error: {e}")

            await asyncio.sleep(1)
