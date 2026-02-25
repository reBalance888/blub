"""
random_agent.py — Baseline agent with random actions.
"""
from __future__ import annotations

import random
from base_agent import BlubAgent, SOUNDS


class RandomAgent(BlubAgent):
    """Moves randomly, occasionally speaks random sounds."""

    def think(self, state: dict) -> dict:
        move = random.choice(["north", "south", "east", "west", "stay"])
        speak = (
            random.sample(SOUNDS, random.randint(1, 2))
            if random.random() < 0.5
            else []
        )
        return {"move": move, "speak": speak, "act": None}


if __name__ == "__main__":
    import asyncio

    agent = RandomAgent("random_solo", "http://localhost:8000")
    asyncio.run(agent.run())
