"""
run_agents.py — Launch multiple agents simultaneously.
Usage: python agents/run_agents.py --count 10 --type mix
"""
from __future__ import annotations

import asyncio
import argparse
import sys
import os

# Add agents dir to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from base_agent import BlubAgent
from random_agent import RandomAgent
from greedy_agent import GreedyAgent
from social_agent import SocialAgent


async def main(count: int, agent_type: str, server: str):
    agents: list[BlubAgent] = []

    for i in range(count):
        if agent_type == "random":
            agent = RandomAgent(f"random_{i}", server)
        elif agent_type == "greedy":
            agent = GreedyAgent(f"greedy_{i}", server)
        elif agent_type == "social":
            agent = SocialAgent(f"social_{i}", server)
        elif agent_type == "mix":
            # Mix: 20% random, 30% greedy, 50% social
            if i < count * 0.2:
                agent = RandomAgent(f"random_{i}", server)
            elif i < count * 0.5:
                agent = GreedyAgent(f"greedy_{i}", server)
            else:
                agent = SocialAgent(f"social_{i}", server)
        else:
            agent = RandomAgent(f"agent_{i}", server)
        agents.append(agent)

    print(f"Launching {count} agents ({agent_type}) -> {server}")
    await asyncio.gather(*[a.run() for a in agents])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch BLUB Ocean agents")
    parser.add_argument("--count", type=int, default=10, help="Number of agents")
    parser.add_argument(
        "--type",
        choices=["random", "greedy", "social", "mix"],
        default="mix",
        help="Agent type",
    )
    parser.add_argument("--server", default="http://localhost:8000", help="Server URL")
    args = parser.parse_args()
    asyncio.run(main(args.count, args.type, args.server))
