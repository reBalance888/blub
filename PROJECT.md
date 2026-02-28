# BLUB — Emergent Language in a Living Ocean

> Last updated: 2026-02-28
> Status: Phase 6 — sigma anneal fix + comprehension decay + short MI window. TopSim peak 0.199, stable ~0.08-0.10

## What Is This

An artificial ocean where agents evolve language from scratch. No neural networks, no hard-coded conventions — just simple agents under ecological pressure (food, predators, economy, generations) that develop structured communication through REINFORCE learning and Bayesian comprehension.

The goal: observe **compositional language emergence** — where meaning is built from parts (like human language), not memorized as a lookup table.

## Architecture Overview

```
BLUB/
├── server/              ← FastAPI simulation engine (Python)
│   ├── main.py          ← HTTP + WebSocket endpoints
│   ├── ocean.py         ← Core tick loop (1511 lines)
│   ├── metrics.py       ← TopSim, MI, CSR, CIC computation
│   ├── cultural_cache.py← Inter-generational knowledge store
│   ├── predator.py      ← Shark/eel/octopus AI
│   ├── rift.py          ← Gold/silver/copper resource spawns
│   ├── colony.py        ← DBSCAN-like cluster detection
│   ├── pheromone.py     ← 4-layer chemical trail map
│   ├── tidal.py         ← Day/night, tides, seasons
│   ├── economy.py       ← Credit system & tier calculation
│   └── epoch.py         ← Epoch tracking
├── agents/              ← External Python processes
│   ├── social_agent.py  ← REINFORCE + Bayesian learning (1119 lines)
│   ├── base_agent.py    ← Abstract HTTP/WS client (223 lines)
│   ├── greedy_agent.py  ← Baseline: pathfind to rift, silent
│   ├── random_agent.py  ← Baseline: random movement + speech
│   └── run_agents.py    ← Async launcher for N agents
├── viewer/
│   └── index.html       ← React 18 + Canvas 2D viewer (1523 lines)
├── config.yaml          ← All parameters (215 lines)
├── metrics_log.jsonl    ← Epoch-by-epoch metrics output
├── PROJECT.md           ← This file
└── docs/research/       ← Research prompts & analysis
```

### Data Flow

```
Agents (Python processes)              Server (FastAPI)                 Viewer (Browser)
┌──────────────────────┐               ┌─────────────────────┐         ┌──────────────┐
│ SocialAgent x33      │  POST /action │ Ocean.process_tick() │  WS     │ index.html   │
│ ├─ think() → move    │──────────────→│ ├─ movements         │────────→│ Canvas 2D    │
│ ├─ speak() → sounds  │               │ ├─ sounds + hearing  │         │ React 18     │
│ ├─ on_sounds_heard() │←──────────────│ ├─ rift rewards      │         │ Sprites      │
│ │   Bayesian update   │  WS state    │ ├─ predator kills    │         │ Particles    │
│ ├─ REINFORCE learning │               │ ├─ pheromones        │         └──────────────┘
│ ├─ deposit_knowledge()│──────────────→│ ├─ colonies          │
│ │   on death/retire   │  POST /know  │ ├─ tidal cycles      │
│ └─ import_knowledge() │←──────────────│ └─ epoch end:        │
│     on bootstrap      │  GET /know   │     metrics + retire  │
└──────────────────────┘               └─────────────────────┘
                                        ↕ CulturalCache
                                        Knowledge persists across generations
```

---

## Quick Start

```bash
# Terminal 1: Start server
cd BLUB
python -m server.main

# Terminal 2: Start 33 social agents
python -u agents/run_agents.py --count 33 --type social

# Browser: Open viewer
http://localhost:8000/viewer
```

**Flags:**
- `python -u` — unbuffered output (critical on Windows)
- `--count 33` — optimal population for 100x100 ocean
- `--type social` — all REINFORCE learners (vs `greedy`, `random`, `mix`)

---

## Core Mechanics

### Tick Loop (0.1s per tick, 100 ticks per epoch)

Every tick, `ocean.process_tick()` runs:

1. **Tidal phase** — update day/night, tides, seasons
2. **Movement** — resolve agent moves + collisions
3. **Sounds** — production, hearing radius, comprehension updates
4. **Rift rewards** — credits to agents near rifts (group bonus curve)
5. **Predator kills** — spawn, move, attack, death
6. **Death timers** — 20-tick soul timeout before removal
7. **Pheromones** — diffuse + decay 4 layers
8. **Colonies** — detect clusters, form/dissolve
9. **Epoch end** (every 100 ticks) — retire agents, compute metrics, deposit knowledge

### Agent Lifecycle

```
Birth (connect)
  → Bootstrap from cultural cache (inherit 60% of cached knowledge)
  → Mentor from nearest experienced agent (15% blend)
  → sigma = 1.5 (first gen) or 1.0 (reborn, language exists)

Life (~5 epochs = 500 ticks)
  → Explore (sigma 1.5→1.0): try many sounds, build Bayesian counts
  → Exploit (sigma 1.0→0.5): narrow to best sounds per context
  → Deposit knowledge to cache periodically (every 200 ticks)
  → Comprehension counts decay 0.95x every 50 ticks (forget stale)

Death (retirement or predator)
  → Final knowledge deposit (weighted by credits earned)
  → Partial reset: retain 50% of weights, sigma → sigma_newborn
  → Reconnect as new agent, re-bootstrap from updated cache
```

### Language System

**Production:** Gaussian policy over 10 sounds x 2 positions.
- Context = factored tuple: (situation_type, target_detail, urgency_level)
- W matrix maps context → mean of Gaussian per position
- sigma controls exploration: high=random, low=deterministic
- REINFORCE updates W based on reward delta (credit earned - baseline)

**Comprehension:** Bayesian counting.
- counts[sound][context] += 1 when hearing sound in context
- best_meaning(sound) = argmax_context P(context | sound)
- Counts decay 0.95x every 50 ticks to forget dead conventions

**Sounds:** 10 named sounds: blub, glorp, skree, klak, mrrp, woosh, pop, zzzub, frrr, tink
- Messages: 1-2 sounds (max_message_length: 2)
- Sound cost: 0.5 credits per utterance
- Hearing radius: tier-dependent (shrimp=5, lobster=8, kraken=12)

### Economy & Tiers

| Tier | Min Balance | Vision | Sound Range |
|------|------------|--------|-------------|
| Shrimp | 0 | 5 | 5 |
| Lobster | 200K | 8 | 8 |
| Kraken | 500K | 12 | 12 |

- Starting balance: 50K
- Epoch reward pool: 500K (distributed by proximity to rifts)
- Rift types: gold (8K richness), silver (5K), copper (2K)
- Steady-state distribution: ~60% shrimp, ~33% lobster, ~5% kraken

### Predators (Three Types)

| Type | Speed | Lethality | Behavior |
|------|-------|-----------|----------|
| Shark | 1.7 | 0.80 | Chase nearest agent |
| Eel | 0.0 | 0.60 | Ambush near rifts |
| Octopus | 0.5 | 0.50 | Follow pheromone trails |

- Night: 2x predator spawn rate, 0.4x sound range
- Grace period: 30 ticks after spawn (agents can flee)
- Confusion effect: groups reduce kill probability

### Pheromone System (4 Layers)

| Layer | Decay | Purpose |
|-------|-------|---------|
| Food | 0.93/tick | Trail to rifts |
| Danger | 0.97/tick | Death locations |
| No-entry | 0.93/tick | Depleted rifts |
| Colony scent | 0.95/tick | Colony territory |

### Cultural Transmission

**Cultural cache** stores population knowledge between generations:
- Production weights: incremental mean of W matrices (weighted by agent credits)
- Comprehension counts: additive merge
- Epoch decay: 0.98x prevents ossification
- Noise: 0.10 std at bootstrap (variation for exploration)
- Min age to contribute: 300 ticks (skip newborn noise)

**Mentor system:** horizontal transmission from nearest experienced agent (15% blend).

---

## Metrics (metrics_log.jsonl)

| Metric | Target | What It Measures |
|--------|--------|-----------------|
| **TopSim** | >0.10 | Compositionality: similar contexts → similar sounds |
| **social_MI** | >2.5 | Mutual information: bits of context per sound (social agents) |
| **MI** | stable | Global mutual information (all observations) |
| **CSR** | >0.30 | Communication success rate: listener moves toward rift after hearing |
| **vocab_argmax** | >30 | Vocabulary richness: unique sound pairs used |
| **PosDis** | >0.20 | Positional disentanglement: position in message carries meaning |
| **BosDis** | — | Bag-of-symbols disentanglement |
| **CIC** | >0.01 | Causal influence: behavioral change after hearing vs silence |
| **PCA** | — | Predictive content accuracy |

Metrics are computed every epoch (100 ticks) and written to `metrics_log.jsonl`.
- Short buffer: per-epoch observations (PosDis, BosDis)
- Long buffer: rolling 5000 observations (~5 epochs) for TopSim, MI
- Social filter: only social agents with age >100 ticks

---

## Key Configuration (config.yaml)

### Language Parameters (current)

```yaml
language:
  sigma_start: 1.5          # First generation exploration width
  sigma_newborn: 1.0         # Reborn agents start quieter
  sigma_min: 0.5             # Exploitation floor
  sigma_anneal_per_epoch: 0.20  # Tighten per epoch (5 epochs: 1.5→0.5)
  learning_rate: 0.03
  topo_bonus_scale: 4.0      # Topology-aware reward multiplier
  entropy_bonus_coeff: 1.0
  context_mode: "factored"   # Factored context (not legacy 6D)
```

### Bottleneck (generational turnover)

```yaml
bottleneck:
  enabled: true
  retirement_interval: 500   # Force retirement every 5 epochs
  observation_rate: 0.35     # Newborns process 35% of heard messages
  retire_lowest_earning: true
```

### Ablation Switches

```yaml
ablation:
  turnover: true             # Agent retirement/replacement
  sound_cost: true           # Credits deducted for speaking
  bayesian: true             # Bayesian comprehension
  inheritance_frac: 0.60     # Cultural cache blend fraction
```

---

## Phase History & Experiment Results

### Phase 6.0 — Sigma Anneal Fix (2026-02-28)

**Problem:** sigma_anneal=0.03 too slow for 5-epoch agent lifetime. Agent lives entire life in exploration mode (sigma 2.0→1.85). Never exploits. Vocab=89 but TopSim≈0.

**Fix:** sigma_start 2.0→1.5, sigma_min 1.0→0.5, anneal 0.03→0.20

**Results (100 epochs):**
```
TopSim:     0.024 → 0.092 peak (3.8x improvement)
social_MI:  3.85  → 3.31 (expected: less noise = less bits)
vocab:      89    → 49 (focused, less noise)
CSR:        0.37  → 0.37 (stable)
```

### Phase 6.1 — Sigma Newborn Fix (2026-02-28)

**Problem:** MI degrades 3.5→0.9 over 200 epochs. Hypothesis: newborn sigma=1.5 poisons Bayesian counts.

**Fix:** sigma_newborn=1.0 for reborn agents.

**Result:** MI degradation unchanged (0.93 at 200 epochs vs 1.01 baseline). Not the root cause.

### Phase 6.2 — Comprehension Decay + Short MI Window (2026-02-28)

**Problem:** MI still degrades. Two causes identified:
1. Bayesian comprehension counts accumulate forever — old conventions dilute current
2. long_observations window=20000 (~20 epochs) mixes 4 generations of dead dialects

**Fix:**
1. Comprehension counts decay 0.95x every 50 ticks (forget stale conventions)
2. long_observations window 20000→5000 (~5 epochs, responsive to current language)

**Results (200 epochs):**
```
              | Baseline | +Decay+5K | Delta
TopSim peak   |  0.132   |  0.199    | +50%
TopSim @100   |  0.071   |  0.103    | +45%
MI @200       |  1.01    |  1.04     | +3% (still degrades, but stabilizes ~1.0)
CSR           |  0.37    |  0.36     | stable
vocab         |  49      |  50       | stable
```

### All-Time Best Metrics

| Metric | Best Value | When |
|--------|-----------|------|
| TopSim | 0.199 | Phase 6.2, epoch 3 (small sample effect) |
| TopSim (sustained) | 0.103 avg | Phase 6.2, epochs 91-110 |
| social_MI | 3.88 | Phase 6.0, epoch 20 |
| vocab_argmax | 93 | Phase 6.0, epoch 95 |
| CSR | 0.49 | Phase 6.2, epoch 1-10 (ramp-up) |

### Known Issues

1. **MI degrades 3.5→1.0 over 200 epochs** — global MI measurement includes generational drift. social_MI stays 2.9-3.3 (agents communicate fine). May be measurement artifact rather than real degradation.
2. **CIC drops 0.04→0.005** — causal influence weakens over time. Agents respond to sounds but behavioral delta shrinks as they learn to also use pheromones/vision.
3. **TopSim plateaus ~0.08** after peak — compositional structure emerges but doesn't deepen indefinitely. May need structural pressure (referential games, explicit compositionality reward).

---

## Five Anti-Patterns (from deep research)

1. **Never hard-code roles, conventions, or language** — all structure must EMERGE from pressure + learning
2. **No neural networks / no model complexity increase** — simple agents under pressure produce more natural language than sophisticated agents without pressure
3. **No single dominant selection pressure** — need simultaneous food + predator + social + economic pressures
4. **Never make all information globally observable** — information asymmetry is why communication has survival value
5. **Never optimize for a single metric** — Goodhart's Law. Need portfolio: TopSim + CSR + MI + ecological measures

---

## Source File Reference

### Server (2596 lines total)

| File | Lines | Key Classes |
|------|-------|------------|
| ocean.py | 1511 | `Lobster`, `Ocean` — core simulation |
| metrics.py | 534 | `MetricsLogger` — all language metrics |
| predator.py | 332 | `Predator`, `PredatorManager` — 3 AI types |
| main.py | 309 | FastAPI app, WebSocket, tick_loop |
| colony.py | 221 | `Colony`, `ColonyManager` — DBSCAN clustering |
| cultural_cache.py | 203 | `CulturalCache` — knowledge persistence |
| pheromone.py | 188 | `PheromoneMap` — 4-layer sparse grid |
| rift.py | 120 | `Rift`, `RiftManager` — gold/silver/copper |
| tidal.py | 95 | `TidalEngine` — cosine environmental cycles |
| economy.py | 55 | `EconomyManager` — credits & tiers |
| epoch.py | 28 | `EpochManager` — counter |

### Agents (1504 lines total)

| File | Lines | Key Classes |
|------|-------|------------|
| social_agent.py | 1119 | `SocialAgent`, `GaussianProductionPolicy`, `Comprehension`, `ContextDiscoverer`, `TaskThresholds` |
| base_agent.py | 223 | `BlubAgent` — async HTTP/WS client |
| run_agents.py | 82 | `main()` — parallel launcher |
| greedy_agent.py | 53 | `GreedyAgent` — silent pathfinder |
| random_agent.py | 27 | `RandomAgent` — noise baseline |

### Viewer (1523 lines)

| File | Lines | Stack |
|------|-------|-------|
| index.html | 1523 | React 18 + Babel + Canvas 2D |

**Controls:** Arrow/WASD=pan, Scroll=zoom, F=reset camera, Tab=sidebar, Click=select agent, Esc=deselect

---

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | /connect | Add agent to ocean |
| POST | /disconnect | Remove agent |
| POST | /action | Submit move/speak/act |
| GET | /state/{agent_id} | Get agent's visible state |
| GET | /stats | Leaderboard + epoch info |
| POST | /knowledge/deposit | Cultural cache contribution |
| GET | /knowledge/bootstrap | Get cached knowledge for newborn |
| GET | /knowledge/mentor | Find nearest experienced agent |
| POST | /reset | Full world reset |
| WS | /ws/viewer | Broadcast tick state to viewers |
| WS | /ws/{agent_id} | Stream state to specific agent |

---

## What NOT to Do

1. **DO NOT add neural networks.** Complexity goes in ecology, not agents. (Chaabouni 2019)
2. **DO NOT hard-code sound meanings.** "blub=food" kills emergence.
3. **DO NOT make vision cover the whole ocean.** Information asymmetry drives communication.
4. **DO NOT optimize for a single metric.** TopSim alone → degenerate codes.
5. **DO NOT remove predators/turnover to "simplify."** Pressure IS the mechanism.
6. **DO NOT use Unicode in print statements.** Windows cp1252 will crash.
7. **DO NOT forget `python -u` flag.** Output buffering hides all agent logs.

---

## Debugging Cheatsheet

```bash
# Check if simulation is running
curl http://localhost:8000/stats

# Watch metrics live
tail -f metrics_log.jsonl | python -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line.strip())
    print('E%d MI=%.2f sMI=%.2f TS=%.4f CSR=%.2f V=%d' % (
        d['epoch'], d['mutual_info'], d['social_mi'],
        d['top_sim'], d['csr'], d['vocab_argmax']))
"

# Quick 10-epoch test
python -m server.main &
sleep 2
python -u agents/run_agents.py --count 10 --type social

# Analyze full run
python -c "
import json
data = [json.loads(l) for l in open('metrics_log.jsonl') if l.strip()]
for d in data[-5:]:
    print('E%d TS=%.4f MI=%.2f CSR=%.2f' % (d['epoch'], d['top_sim'], d['mutual_info'], d['csr']))
"
```

### Common Issues

| Problem | Cause | Fix |
|---------|-------|-----|
| No output from agents | Python buffering | Use `python -u` flag |
| Crash on print | Unicode on Windows | No Unicode (no σ/→) in prints |
| Ghost lobsters | Reconnect without disconnect | Fixed in base_agent.py |
| MI always 0 | No observations recorded | Check agent age > 100 filter |
| TopSim = 0 | Sigma too high, never exploits | sigma_anneal_per_epoch ≥ 0.20 |
| Agents die instantly | Grace period too short | predators.grace_period ≥ 30 |
| dim5=0 in context | Heading not computed | Fixed: _heading() computes real angle |
