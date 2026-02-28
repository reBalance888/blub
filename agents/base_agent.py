"""
base_agent.py — Base class for all BLUB Ocean agents.
"""
from __future__ import annotations

import asyncio
import aiohttp

SOUNDS = [
    "blub", "glorp", "skree", "klak", "mrrp",
    "woosh", "pop", "zzzub", "frrr", "tink",
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
        self.local_age: int = 0
        self._was_alive: bool = True  # track alive→dead transitions
        self._my_colony_id: str | None = None  # cached colony_id for reconnect

    async def connect(self, colony_id: str | None = None) -> dict:
        """Connect to the ocean server, then bootstrap from cultural cache."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.server_url}/connect",
                json={"name": self.name},
            ) as resp:
                data = await resp.json()
                self.agent_id = data["agent_id"]
                self.local_age = 0
        # Bootstrap cultural knowledge (oblique transmission)
        try:
            bootstrap_data = await self.bootstrap_knowledge(colony_id=colony_id)
            if bootstrap_data and bootstrap_data.get("production"):
                inherit_frac = bootstrap_data.pop("_inherit_frac", None)
                self.on_bootstrap(bootstrap_data, inherit_frac=inherit_frac)
        except Exception as e:
            print(f"[{self.name}] Bootstrap failed (empty cache?): {e}")
        # Mentor fallback (horizontal transmission)
        try:
            mentor_data = await self.get_mentor_knowledge()
            if mentor_data:
                self.on_mentor(mentor_data)
        except Exception as e:
            print(f"[{self.name}] Mentor fetch failed: {e}")
        self._was_alive = True
        return data

    async def get_state(self) -> dict:
        """Get current state from server."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.server_url}/state/{self.agent_id}"
            ) as resp:
                self.state = await resp.json()
                return self.state

    async def do_action(self, move: str = "stay", speak: list | None = None,
                        act: str | None = None, **kwargs) -> dict:
        """Send an action to the server."""
        actions = {
            "move": move,
            "speak": speak or [],
            "act": act,
        }
        # Pass through extra fields (e.g. role for Phase 2 specialization)
        actions.update(kwargs)
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.server_url}/action",
                json={
                    "agent_id": self.agent_id,
                    "actions": actions,
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

    async def deposit_knowledge(self, data: dict, colony_id: str | None = None) -> dict:
        """POST knowledge to the cultural cache on the server."""
        body = {"agent_id": self.agent_id, "data": data}
        if colony_id:
            body["colony_id"] = colony_id
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.server_url}/knowledge/deposit",
                json=body,
            ) as resp:
                return await resp.json()

    async def bootstrap_knowledge(self, colony_id: str | None = None) -> dict:
        """GET bootstrapped knowledge from the cultural cache."""
        params = {}
        if colony_id:
            params["colony_id"] = colony_id
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.server_url}/knowledge/bootstrap",
                params=params,
            ) as resp:
                return await resp.json()

    async def get_mentor_knowledge(self) -> dict | None:
        """GET knowledge from nearest experienced social agent (mentor system)."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.server_url}/knowledge/mentor",
                params={"agent_id": self.agent_id},
            ) as resp:
                data = await resp.json()
                if data.get("production"):
                    return data
                return None

    async def on_death(self):
        """Called when agent transitions from alive to dead.
        Override in subclasses to deposit knowledge before death timeout."""
        pass

    def on_bootstrap(self, data: dict, inherit_frac: float | None = None):
        """Called after connect with bootstrapped cultural knowledge.
        Override in subclasses to import knowledge."""
        pass

    def on_mentor(self, data: dict):
        """Called after bootstrap with mentor's knowledge (horizontal transmission).
        Override in subclasses to blend mentor knowledge at 15%."""
        pass

    async def on_pre_retire(self):
        """Called before on_retired and reconnect.
        Override in subclasses to deposit knowledge before death."""
        pass

    async def periodic_check(self, state: dict):
        """Called after every action. Override for periodic deposits etc."""
        pass

    def on_retired(self):
        """Called when the server flags this agent as retired.
        Override in subclasses to reset learned state (turnover)."""
        pass

    async def _reconnect(self):
        """Disconnect old lobster and reconnect as a naive agent (turnover)."""
        old_id = self.agent_id
        print(f"[{self.name}] Retiring {old_id}, reconnecting as naive...")
        # Remove old lobster from server to prevent ghost accumulation
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(
                    f"{self.server_url}/disconnect",
                    json={"agent_id": old_id},
                )
        except Exception:
            pass
        try:
            data = await self.connect(colony_id=self._my_colony_id)
            print(f"[{self.name}] Reborn as {self.agent_id}")
        except Exception as e:
            print(f"[{self.name}] Reconnect failed: {e}")

    async def run(self):
        """Main agent loop."""
        await self.connect()
        print(f"[{self.name}] Connected as {self.agent_id}")

        while True:
            try:
                state = await self.get_state()

                # Check retirement signal from server
                if state.get("retired", False):
                    await self.on_pre_retire()
                    self.on_retired()
                    await self._reconnect()
                    continue

                if not state.get("alive", True):
                    # Detect alive→dead transition: deposit knowledge
                    if self._was_alive:
                        self._was_alive = False
                        await self.on_death()
                    await asyncio.sleep(0.1)
                    self.local_age += 1
                    continue
                else:
                    self._was_alive = True

                # Update language model
                if state.get("sounds_heard"):
                    self.on_sounds_heard(state["sounds_heard"])

                # Think and act
                action = self.think(state)
                await self.do_action(**action)

                # Periodic check (deposits etc.)
                await self.periodic_check(state)
                self.local_age += 1

                # Telemetry every 50 ticks
                if state.get("tick", 0) % 50 == 0:
                    print(f"[{self.name}] tick={state.get('tick')} credits={state.get('my_net_credits', '?')} pos={state.get('my_position', '?')}")

            except aiohttp.ClientError as e:
                print(f"[{self.name}] Connection error: {e}")
            except Exception as e:
                print(f"[{self.name}] Error: {e}")

            await asyncio.sleep(0.1)
