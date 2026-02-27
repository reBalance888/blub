# Changelog

All notable changes to BLUB Ocean will be documented here.

This is a living document — we're building in public. Expect honest numbers, including bad ones.

## [Unreleased] — Phase 2.5: Bootstrap Language

### Critical Bugfix
- **Fix position 1 context distribution** — Position 1 was receiving `(0, 0, 0)` as context, meaning `sigmoid(bias) ≈ 0.73` → sound ~7 always. Half of every message was frozen. This alone explains PosDis stagnation and vocab collapse.

### Added
- Causal influence intrinsic reward — speakers get rewarded when their messages change listener behavior
- Message entropy regularization — gentle pressure against message collapse
- Population heterogeneity — variable learning rates per agent (0.7×–1.4× base)

### Expected Impact
- PosDis: 0.15 → >0.25 (position 1 now carries information)
- CIC: 0.003 → >0.01 (messages start mattering)
- TopSim: 0.03 → >0.06 (two orthogonal positions with structured mapping)

---

## [0.2.0] — Phase 2: Foundation

### Added
- Factored context system (situation × target × urgency = 60 values)
- 4 pheromone layers (food, danger, no-entry, colony scent)
- Colony formation via DBSCAN clustering
- Tidal engine with 3 nested cycles (day/night, tidal, seasonal)
- Bottleneck mechanism — lowest earner replaced every 200 ticks
- Cultural cache — dying agents pass 40% of knowledge to newborns
- Task specialization (forage/scout/guard/teach)
- 20% novelty holdout for zero-shot generalization testing

### Metrics at epoch 824
| Metric | Value | Note |
|--------|-------|------|
| TopSim | 0.03 | Random noise level — not yet structured |
| CIC | 0.003 | Communication is illusory |
| CSR | 0.19 | Weak context-sound correlation |
| PosDis | 0.15 | Position 1 frozen (bug) |
| Vocab | 0 | Complete collapse |
| Colonies | 3–4 | Avg size 4.8 |

---

## [0.1.0] — Phase 1: Baseline

### Added
- Core ocean simulation with FastAPI + WebSocket
- 33 social agents with GaussianProductionPolicy + Bayesian listener
- 10 sounds, 2-position messages (100 combinations)
- 3 predator types (shark, eel, octopus)
- 8 rifts per epoch (gold/silver/copper)
- Superlinear group rewards (solo 0.5 → group of 5: 4.0 per tick)
- React 18 + Canvas 2D bioluminescent viewer
- REINFORCE learning with topographic bonus
- Listener feedback (speaker gets 30% of listener reward)

### Metrics at epoch 142
| Metric | Value |
|--------|-------|
| TopSim | 0.036 |
| MI | 2.12 |
| CSR | 25.7% |
| PosDis | 0.238 |
| Vocab | 84 |
