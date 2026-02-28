"""
cultural_cache.py — Collective knowledge store that persists across agent lifetimes.
Agents deposit learned production weights and comprehension counts;
new agents bootstrap from the cache to inherit cultural knowledge.
"""
from __future__ import annotations

import random
from copy import deepcopy


class CulturalCache:
    """Stores aggregated language knowledge from all agents.

    Production weights are stored as incremental means (each contributor
    has equal weight).  Comprehension counts are additive.  A small
    Gaussian noise is added at bootstrap time so agents don't become
    identical clones.
    """

    def __init__(self, config: dict):
        cc = config.get("cultural_cache", {})
        self.inheritance_frac: float = cc.get("inheritance_frac", 0.40)
        self.decay_rate: float = cc.get("cache_decay_rate", 0.98)
        self.noise_std: float = cc.get("noise_std", 0.10)
        self.min_prod_total: float = 35.0  # skip near-uniform contexts (30*1.0=30)
        self.min_comp_confidence: int = 5   # minimum observations to contribute

        # {ctx_key_str: list[float]} — incremental mean of production weights (Roth-Erev)
        self.production_weights: dict[str, list[float]] = {}
        # Number of contributors per context (for incremental mean)
        self._prod_n: dict[str, int] = {}
        # Neural production cache
        self._neural_W: list = []
        self._neural_baseline: list[float] = []

        # {sound_idx: {ctx_key_str: count}} — additive comprehension counts
        self.comprehension_counts: dict[int, dict[str, int]] = {}

        # Per-agent knowledge snapshots for mentor system
        self.agent_snapshots: dict[str, dict] = {}

    def contribute(self, data: dict, agent_id: str | None = None, weight: float = 1.0):
        """Accept a knowledge deposit from an agent.

        Handles both Roth-Erev format {ctx_str: [weights]} and
        Neural format {"W": [...], "baseline": [...]}.
        Weight parameter: higher-earning agents have more influence on cache (Darwinian selection).
        """
        # Store snapshot for mentor system
        if agent_id:
            self.agent_snapshots[agent_id] = data

        prod = data.get("production", {})

        # Neural production format: {"W": [...], "baseline": [...]}
        if "W" in prod:
            self._contribute_neural(prod, weight=weight)
        else:
            # Roth-Erev format: {ctx_key_str: [weights...]}
            for ctx_str, weights in prod.items():
                # Quality filter: skip contexts where agent hasn't learned much
                if sum(weights) < self.min_prod_total:
                    continue
                if ctx_str not in self.production_weights:
                    self.production_weights[ctx_str] = list(weights)
                    self._prod_n[ctx_str] = 1
                else:
                    n = self._prod_n[ctx_str] + 1
                    self._prod_n[ctx_str] = n
                    existing = self.production_weights[ctx_str]
                    for i in range(min(len(existing), len(weights))):
                        existing[i] += (weights[i] - existing[i]) / n

        comp = data.get("comprehension", {})
        for sidx_str, ctx_counts in comp.items():
            sidx = int(sidx_str)
            if sidx not in self.comprehension_counts:
                self.comprehension_counts[sidx] = {}
            for ctx_str, count in ctx_counts.items():
                # Quality filter: skip low-confidence observations
                if count < self.min_comp_confidence:
                    continue
                self.comprehension_counts[sidx][ctx_str] = (
                    self.comprehension_counts[sidx].get(ctx_str, 0) + count
                )

    def _contribute_neural(self, prod: dict, weight: float = 1.0):
        """Weighted incremental mean of neural W matrices.
        Higher-earning agents contribute more to the cache (Darwinian cultural selection).
        Handles both flat (Gaussian: W[pos]=[floats]) and nested (Neural: W[pos]=[[floats]]) formats."""
        W = prod.get("W", [])
        baseline = prod.get("baseline", [])
        key = "__neural__"
        if key not in self._prod_n:
            self._neural_W = deepcopy(W)
            self._neural_baseline = list(baseline)
            self._prod_n[key] = weight
        else:
            total_w = self._prod_n[key] + weight
            alpha = weight / total_w  # higher earnings → more pull
            self._prod_n[key] = total_w
            for pos in range(min(len(W), len(self._neural_W))):
                pw = W[pos]
                sw = self._neural_W[pos]
                if isinstance(pw, list) and pw and isinstance(pw[0], (int, float)):
                    for j in range(min(len(pw), len(sw))):
                        sw[j] += (pw[j] - sw[j]) * alpha
                elif isinstance(pw, list) and pw and isinstance(pw[0], list):
                    for i in range(min(len(pw), len(sw))):
                        for j in range(min(len(pw[i]), len(sw[i]))):
                            sw[i][j] += (pw[i][j] - sw[i][j]) * alpha
            for i in range(min(len(baseline), len(self._neural_baseline))):
                self._neural_baseline[i] += (baseline[i] - self._neural_baseline[i]) * alpha

    def bootstrap(self) -> dict:
        """Return cached knowledge with Gaussian noise for a new agent.

        Handles both neural (W matrices) and Roth-Erev (per-context weights) formats.
        """
        # Comprehension: return a deep copy (same for both formats)
        comp_copy: dict[str, dict[str, int]] = {}
        for sidx, ctx_counts in self.comprehension_counts.items():
            comp_copy[str(sidx)] = dict(ctx_counts)

        # Neural/Gaussian format: return noisy W matrix
        if hasattr(self, "_neural_W") and self._neural_W:
            noisy_W = deepcopy(self._neural_W)
            for pos in range(len(noisy_W)):
                pw = noisy_W[pos]
                if isinstance(pw, list) and pw and isinstance(pw[0], (int, float)):
                    # Flat format
                    for j in range(len(pw)):
                        w = pw[j]
                        noise = random.gauss(0, self.noise_std * abs(w)) if abs(w) > 0.01 else 0
                        noisy_W[pos][j] = w + noise
                elif isinstance(pw, list) and pw and isinstance(pw[0], list):
                    # Nested format
                    for i in range(len(pw)):
                        for j in range(len(pw[i])):
                            w = pw[i][j]
                            noise = random.gauss(0, self.noise_std * abs(w)) if abs(w) > 0.01 else 0
                            noisy_W[pos][i][j] = w + noise
            return {
                "production": {"W": noisy_W, "baseline": list(getattr(self, "_neural_baseline", []))},
                "comprehension": comp_copy,
            }

        # Roth-Erev format
        noisy_prod: dict[str, list[float]] = {}
        for ctx_str, weights in self.production_weights.items():
            noisy = []
            for w in weights:
                noise = random.gauss(0, self.noise_std * abs(w)) if abs(w) > 0.01 else 0
                noisy.append(max(0.01, w + noise))
            noisy_prod[ctx_str] = noisy

        return {
            "production": noisy_prod,
            "comprehension": comp_copy,
        }

    def epoch_decay(self):
        """Apply decay to all stored weights."""
        # Neural/Gaussian weights: gentle decay toward zero
        if self._neural_W:
            for pos in range(len(self._neural_W)):
                pw = self._neural_W[pos]
                if isinstance(pw, list) and pw and isinstance(pw[0], (int, float)):
                    for j in range(len(pw)):
                        pw[j] *= self.decay_rate
                elif isinstance(pw, list) and pw and isinstance(pw[0], list):
                    for i in range(len(pw)):
                        for j in range(len(pw[i])):
                            pw[i][j] *= self.decay_rate

        # Roth-Erev weights: decay with floor
        prune_prod = []
        for ctx_str, weights in self.production_weights.items():
            for i in range(len(weights)):
                weights[i] = max(0.5, weights[i] * self.decay_rate)
            if max(weights) < 0.51:
                prune_prod.append(ctx_str)
        for ctx_str in prune_prod:
            del self.production_weights[ctx_str]
            self._prod_n.pop(ctx_str, None)

        # Decay comprehension counts (floor to int)
        prune_comp = []
        for sidx, ctx_counts in self.comprehension_counts.items():
            prune_ctx = []
            for ctx_str, count in ctx_counts.items():
                new_count = int(count * self.decay_rate)
                if new_count < 1:
                    prune_ctx.append(ctx_str)
                else:
                    ctx_counts[ctx_str] = new_count
            for ctx_str in prune_ctx:
                del ctx_counts[ctx_str]
            if not ctx_counts:
                prune_comp.append(sidx)
        for sidx in prune_comp:
            del self.comprehension_counts[sidx]
