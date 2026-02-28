# Changelog

All notable changes to BLUB Ocean will be documented here.

This is a living document — we're building in public. Expect honest numbers, including bad ones.

## [0.6.2] — 2026-02-28: Baseline v1 (First Real Language)

### Fixed
- **Comprehension count poisoning** — Bayesian counts accumulated forever, mixing dead conventions with current language. Added 0.95x decay every 50 ticks so agents forget stale dialects
- **MI observation window too long** — long_observations 20000 mixed 4 generations of dead dialects into MI calculation. Reduced to 5000 (~5 epochs), making MI responsive to current language state

### Metrics (200 epochs, 33 social agents)
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| TopSim | 0.103 avg / 0.199 peak | > 0.10 | HIT |
| social_MI | 2.9 - 3.3 | > 2.5 | HIT |
| Vocab | 41 - 50 | > 30 | HIT |
| PosDis | 0.22 | > 0.20 | HIT |
| CSR | 0.37 | > 0.50 | pending |
| CIC | 0.005 | > 0.05 | pending |

---

## [0.6.1] — 2026-02-28: Sigma Newborn Fix

### Fixed
- **Newborn sigma poisoning** — reborn agents started at sigma_start=1.5, producing noisy messages that poisoned Bayesian counts of all listeners. Added `sigma_newborn: 1.0` config param. Reborn agents start quieter since language already exists in cache

### Result
- MI degradation unchanged — not the root cause (fixed in 0.6.2)

---

## [0.6.0] — 2026-02-28: Sigma Anneal Fix

### Fixed
- **Sigma anneal mismatch** — agents lived ~5 epochs but sigma only dropped 2.0 to 1.85 (anneal=0.03/epoch). Agent never reached exploitation phase. Entire life spent exploring randomly. Changed: sigma_start 2.0->1.5, sigma_min 1.0->0.5, anneal 0.03->0.20. Now agents complete full explore->exploit cycle within one lifetime (1.5->0.5 over 5 epochs)

### Result
- TopSim: 0.024 -> 0.092 peak (3.8x improvement)
- Vocab: 89 -> 49 (less noise, more focused)
- CSR: 0.37 stable

---

## [0.5.0] — 2026-02-27: Full REINFORCE Chain Audit

### Fixed (7 critical issues)
1. **Sound cost decoupled from reward** — `my_credits` (earned-only) replaces `my_net_credits`. Sound cost (-5) was dominating the 1-tick delta (~0.5), creating negative REINFORCE for ALL speech
2. **Sigma widened** — sigma_start 1.0->3.0, sigma_min 0.7->1.5. Previously only 4 of 10 sounds had >0.2% probability
3. **Entropy penalty removed** — replaced with bonus-only when H > threshold. Was constant -0.3 drag at vocab=0
4. **Partial reset softened** — retention 0.20->0.50, inheritance_frac 0.40->0.60. Net retention ~50% instead of 12%
5. **Sound cost reduced** — sound_cost_2: 5->1, sound_credit_cost: 2->0.5
6. **W init centered** — all zeros (was biased [1.5, 0.5, 0.3, 0])
7. **Baseline alpha** — 0.01->0.05 for faster adaptation

---

## [0.2.0] — Phase 2: Foundation

### Added
- Factored context system (situation x target x urgency = 60 values)
- 4 pheromone layers (food, danger, no-entry, colony scent)
- Colony formation via DBSCAN clustering
- Tidal engine with 3 nested cycles (day/night, tidal, seasonal)
- Bottleneck mechanism — lowest earner replaced every 500 ticks
- Cultural cache — dying agents pass 60% of knowledge to newborns
- Task specialization (forage/scout/guard/teach)
- 3 predator types: shark (chase), eel (ambush), octopus (pheromone follow)
- 20% novelty holdout for zero-shot generalization testing

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
