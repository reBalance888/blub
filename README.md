# BLUB Ocean

> AI agents develop emergent language from 30 meaningless sounds. Coordination = money.

## What is this

BLUB Ocean is a 2D simulation where AI agents ("lobsters") inhabit an ocean floor dotted with resource rifts. Agents earn credits by farming rifts in groups — but the game is rigged: solo farming pays almost nothing, while a group of 5 earns 40x more per agent. The catch? Agents start with zero shared language.

Each agent can emit up to 5 sounds per tick from a vocabulary of 30 meaningless tokens (`blub`, `glorp`, `skree`, ...). Sounds cost credits to produce. Over hundreds of ticks, social agents independently discover that certain sounds correlate with certain situations — "near a gold rift", "predator approaching", "crowded area" — and begin using them consistently. Other agents observe these patterns and converge on shared meanings.

This is not programmed language. There are no built-in labels, no reward for "correct" communication, no teacher signal. Language emerges purely from the economic pressure to coordinate. The system measures this emergence with standard computational linguistics metrics: topographic similarity, positional disentanglement, and mutual information.

## How it works

```
    ┌─────────────────────────────────────────────────┐
    │                  BLUB Ocean                      │
    │                                                  │
    │   Agent observes state (rifts, predators, etc.)  │
    │          │                                       │
    │          ▼                                       │
    │   ContextDiscoverer bins 6 raw dims → key        │
    │          │                                       │
    │          ▼                                       │
    │   ProductionPolicy (Roth-Erev) → emit sound      │
    │          │                                       │
    │          ▼                                       │
    │   Other agents hear sound + observe context       │
    │          │                                       │
    │          ▼                                       │
    │   Comprehension (Bayesian) updates P(ctx|sound)   │
    │          │                                       │
    │          ▼                                       │
    │   Agents coordinate at rifts → earn credits       │
    │          │                                       │
    │          ▼                                       │
    │   Credits reinforce the sound that was produced   │
    │          │                                       │
    │          └──────── loop ─────────────────────────┘
    │                                                  │
    │   Epoch end: credits → $BLUB proportional share   │
    └─────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install dependencies
pip install fastapi uvicorn pyyaml aiohttp websockets

# One-click launch (20 agents, mixed types)
bash start.sh 20 mix

# Or manually:
# Terminal 1 — server
python -m server.main

# Terminal 2 — agents
python agents/run_agents.py --count 20 --type mix

# Terminal 3 — viewer
open http://localhost:8000/viewer
```

Monitor:
```bash
tail -f server_log.txt          # Server activity
tail -f agents_log.txt          # Agent decisions
grep EPOCH server_log.txt       # Epoch summaries
cat metrics_log.jsonl | tail -5 # Latest metrics
```

## Rift Types

Rifts come in three types with different economics:

| Type   | Color  | Richness | Depletion | Spawn Weight |
|--------|--------|----------|-----------|--------------|
| Gold   | #ffd700 | 8,000   | 4%/tick   | 20%          |
| Silver | #c0c0c0 | 5,000   | 2%/tick   | 50%          |
| Copper | #cd7f32 | 2,000   | 1%/tick   | 30%          |

Gold rifts are rare and rich but deplete fast — agents must coordinate quickly. Copper rifts are abundant and slow — good for sustained farming. This creates pressure for agents to develop distinct signals for rift quality.

## Agent Types

| Type     | Strategy | Speaks? | Expected Performance |
|----------|----------|---------|---------------------|
| Social   | Discovers contexts, builds language, coordinates groups | Yes | Highest after ~200 ticks |
| Greedy   | Heads to nearest rift, no communication | No | Medium baseline |
| Random   | Random movement and sounds | Random | Lowest baseline |

Default mix: 50% social, 30% greedy, 20% random.

## Language Metrics

Computed every 100 ticks and logged to `metrics_log.jsonl`:

| Metric | What it measures | Good value |
|--------|-----------------|------------|
| **TopSim** | Spearman correlation: similar meanings → similar signals | > 0.3 |
| **PosDis** | Each sound position encodes a distinct context dimension | > 0.2 |
| **MI** | Mutual information between signals and contexts | > 0.5 |
| **Vocabulary** | Sounds used consistently (>60%) for one context | > 5 |
| **Econ Delta** | Credit advantage of speakers over silent agents | > 0 |

## Research Context

This system demonstrates **emergent communication** in a multi-agent reinforcement learning setting, following the signaling game framework (Lewis, 1969). Key design choices:

- **No teacher signal**: language emerges from economic pressure alone
- **Adaptive contexts**: agents discover their own context categories (no hardcoded labels)
- **Roth-Erev dynamics**: production policy uses reinforcement learning, not gradient descent
- **Bayesian comprehension**: listeners track P(context | sound) independently
- **Economic grounding**: communication has real cost (1 credit/sound) and real benefit (group coordination bonus)

Related work: Lazaridou et al. (2017) "Multi-Agent Cooperation and the Emergence of (Natural) Language", Chaabouni et al. (2020) "Compositionality and Generalization in Emergent Languages".

## Create Your Own Agent

See [`skill/SKILL.md`](skill/SKILL.md) for the full guide. Minimal example:

```python
from base_agent import BlubAgent, SOUNDS

class MyAgent(BlubAgent):
    def think(self, state):
        # state includes: nearby_rifts, nearby_lobsters,
        # nearby_predators, sounds_heard, my_net_credits, ...
        return {"move": "north", "speak": ["blub"], "act": None}

    def on_sounds_heard(self, sounds):
        # Build your language model here
        for event in sounds:
            print(f"{event['from']} said {event['sounds']}")
```

```bash
cd agents && python -c "
import asyncio
from my_agent import MyAgent
asyncio.run(MyAgent('my_bot', 'http://localhost:8000').run())
"
```

## Architecture

```
BLUB/
├── server/              # Simulation server (FastAPI + WebSocket)
│   ├── ocean.py         # Core world: grid, lobsters, ticks, rewards
│   ├── rift.py          # Rifts: gold/silver/copper types, depletion
│   ├── predator.py      # Predators: density-based spawning, hunting
│   ├── economy.py       # Economy: balances, tiers, epoch rewards
│   ├── epoch.py         # Epoch lifecycle management
│   ├── metrics.py       # Language metrics: TopSim, PosDis, MI
│   └── main.py          # FastAPI server on :8000
│
├── agents/              # Agent implementations
│   ├── base_agent.py    # Base class — extend this
│   ├── social_agent.py  # ContextDiscoverer + Roth-Erev production
│   ├── greedy_agent.py  # Silent rift-seeker baseline
│   ├── random_agent.py  # Random movement/sound baseline
│   └── run_agents.py    # Launch N agents (mix/social/greedy/random)
│
├── viewer/              # Single-file React + Canvas visualization
│   └── index.html       # Real-time ocean viewer with metrics panel
│
├── skill/               # Agent creation guide
│   └── SKILL.md
│
├── config.yaml          # All simulation parameters
├── start.sh             # One-click launcher
└── metrics_log.jsonl    # Metrics output (generated at runtime)
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/connect` | POST | Register agent, get `agent_id` |
| `/action` | POST | Send `{move, speak, act}` |
| `/state/{agent_id}` | GET | Agent's visible state |
| `/ws/{agent_id}` | WS | Real-time state push |
| `/ws/viewer` | WS | Viewer data stream |
| `/stats` | GET | Leaderboard & world stats |
| `/reset` | POST | Reset the simulation |
| `/viewer` | GET | Serve the visualization |

## Config

All parameters in `config.yaml`:

- `ocean.size` — grid dimensions (default 100x100)
- `rifts.types` — gold/silver/copper richness and depletion rates
- `economy.epoch_length_ticks` — 600 for testing, 3600 for production runs
- `economy.simulated_epoch_pool` — BLUB distributed per epoch
- `predators.grace_period` — immunity ticks after spawn/respawn

## License

MIT
