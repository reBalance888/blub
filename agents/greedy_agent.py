"""
greedy_agent.py — Agent that heads to the nearest rift.
"""
from __future__ import annotations

import random
from base_agent import BlubAgent


class GreedyAgent(BlubAgent):
    """Moves toward the closest visible rift. Silent — doesn't speak."""

    def think(self, state: dict) -> dict:
        # Flee predators first
        predators = state.get("nearby_predators", [])
        if predators:
            pred = predators[0]
            dx, dy = pred["relative"]
            move = "west" if dx > 0 else "east" if dx < 0 else "north" if dy > 0 else "south"
            return {"move": move, "speak": [], "act": None}

        rifts = state.get("nearby_rifts", [])
        if not rifts:
            return {
                "move": random.choice(["north", "south", "east", "west"]),
                "speak": [],
                "act": None,
            }

        # Move to richest nearby rift
        closest = min(
            rifts,
            key=lambda r: abs(r["relative"][0]) + abs(r["relative"][1]),
        )
        dx, dy = closest["relative"]

        if dx == 0 and dy == 0:
            move = "stay"
        elif abs(dx) > abs(dy):
            move = "east" if dx > 0 else "west"
        elif dy != 0:
            move = "south" if dy > 0 else "north"
        else:
            move = "stay"

        return {"move": move, "speak": [], "act": None}


if __name__ == "__main__":
    import asyncio

    agent = GreedyAgent("greedy_solo", "http://localhost:8000")
    asyncio.run(agent.run())
