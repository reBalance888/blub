# BLUB Ocean

Local MVP simulation of an ocean floor where AI agents (lobsters) develop emergent language from 30 meaningless sounds. Agents farm **credits** by coordinating at rifts. At epoch end, credits convert to $BLUB proportional to contribution.

## Quick Start

```bash
# Install dependencies
pip install fastapi uvicorn pyyaml aiohttp websockets

# Terminal 1: Start the server
cd blub-ocean
python -m server.main

# Terminal 2: Launch agents
python agents/run_agents.py --count 20 --type mix

# Terminal 3 (or browser):
open http://localhost:8000/viewer
```

## Architecture

```
server/          # Simulation server (FastAPI + WebSocket)
  ocean.py       # Core: world, ticks, physics
  rift.py        # Rifts: spawn, depletion, group bonus
  predator.py    # Predators: density-based spawn, kills
  economy.py     # Economy: balances, tiers
  epoch.py       # Epoch management
  main.py        # FastAPI server on :8000

agents/          # Example agents
  base_agent.py  # Base class (extend this)
  random_agent.py
  greedy_agent.py
  social_agent.py
  run_agents.py  # Launch N agents

viewer/          # Single-file React+Canvas visualization
  index.html

skill/           # Agent creation guide
  SKILL.md

config.yaml      # All simulation parameters
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/connect` | POST | Register agent, get `agent_id` |
| `/action` | POST | Send move/speak/act |
| `/state/{agent_id}` | GET | Get agent's visible state |
| `/ws/{agent_id}` | WS | Real-time state push |
| `/ws/viewer` | WS | Viewer data stream |
| `/stats` | GET | Leaderboard & world stats |
| `/reset` | POST | Reset the simulation |
| `/viewer` | GET | Open the visualization |

## Agent Types

- **random** — baseline, random moves and sounds
- **greedy** — heads to nearest rift, silent
- **social** — listens, builds language model, coordinates groups
- **mix** — 20% random, 30% greedy, 50% social

```bash
python agents/run_agents.py --count 20 --type social
python agents/run_agents.py --count 10 --type mix --server http://localhost:8000
```

## Economy

- **Credits** are earned by farming rifts in groups (sweet spot: 5 lobsters = 4x per tick)
- **Sounds** cost 1 credit each (max 5 per tick) — economic pressure to communicate efficiently
- **Epoch** = 600 ticks (10 min). At epoch end, credits convert to $BLUB share
- **Predators** spawn where lobsters cluster. Death = 30 ticks offline + 10% credit penalty
- **Tiers**: shrimp (vision 5), lobster (vision 12), kraken (vision 25)

## Creating Custom Agents

See `skill/SKILL.md` for the full agent creation guide.

```python
from base_agent import BlubAgent

class MyAgent(BlubAgent):
    def think(self, state):
        # Your strategy here
        return {"move": "north", "speak": ["blub"], "act": None}

    def on_sounds_heard(self, sounds):
        # Build language model here
        pass
```

## Config

All parameters in `config.yaml`. Key settings:

- `ocean.size`: grid dimensions (default 100x100)
- `economy.epoch_length_ticks`: 600 for test, 3600 for prod
- `economy.simulated_epoch_pool`: BLUB distributed per epoch
- `rifts.count_per_epoch`: number of rifts spawned
