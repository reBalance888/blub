"""
social_agent.py — Agent that discovers contexts, builds language via Roth-Erev reinforcement.
"""
from __future__ import annotations

import math
import random
from base_agent import BlubAgent, SOUNDS


class ContextDiscoverer:
    """Adaptive context binning: tracks running min/max per dimension,
    quantizes raw observation vectors into discrete context keys.
    Dynamically doubles bins for most-used dimensions (P3-A)."""

    MAX_BINS = 8
    REFINE_INTERVAL = 500  # ticks between refinement checks

    def __init__(self, dims: int = 6, bins: int = 4, warmup: int = 50):
        self.dims = dims
        self.bins_per_dim = [bins] * dims
        self.mins = [float('inf')] * dims
        self.maxs = [float('-inf')] * dims
        self.count = 0
        self.warmup = warmup
        # Track usage frequency per dimension (non-zero variance)
        self._dim_usage: list[int] = [0] * dims
        self._last_refine: int = 0

    def get_context(self, raw_vec: list[float]) -> tuple:
        self.count += 1
        for i in range(self.dims):
            if raw_vec[i] < self.mins[i]:
                self.mins[i] = raw_vec[i]
            if raw_vec[i] > self.maxs[i]:
                self.maxs[i] = raw_vec[i]
        if self.count < self.warmup:
            return (0,) * self.dims
        key = []
        for i in range(self.dims):
            r = self.maxs[i] - self.mins[i]
            b = self.bins_per_dim[i]
            if r > 0:
                idx = min(int((raw_vec[i] - self.mins[i]) / r * b), b - 1)
                self._dim_usage[i] += 1
            else:
                idx = 0
            key.append(idx)

        # Dynamic refinement every REFINE_INTERVAL ticks
        if self.count - self._last_refine >= self.REFINE_INTERVAL:
            self._refine_bins()
            self._last_refine = self.count

        return tuple(key)

    def _refine_bins(self):
        """Double bins for top-2 most-used dimensions (capped at MAX_BINS)."""
        # Rank dims by usage, pick top 2 that are below cap
        ranked = sorted(range(self.dims), key=lambda i: -self._dim_usage[i])
        refined = 0
        for i in ranked:
            if refined >= 2:
                break
            if self.bins_per_dim[i] < self.MAX_BINS:
                old = self.bins_per_dim[i]
                self.bins_per_dim[i] = min(old * 2, self.MAX_BINS)
                refined += 1
        # Reset usage counts
        self._dim_usage = [0] * self.dims


class ProductionPolicy:
    """Roth-Erev reinforcement learning for sound production per context."""

    LATERAL_INHIBITION = 0.05  # competing sounds lose 5% of reinforcement (Franke 2014)
    MUTATION_RATE = 0.05  # 5% chance of random sound instead of learned (anti-ossification)

    def __init__(self, n_sounds: int = 30, init_weight: float = 1.0, decay: float = 0.95):
        self.weights: dict[tuple, list[float]] = {}
        self.n_sounds = n_sounds
        self.init = init_weight
        self.decay = decay

    def _ensure(self, ctx_key: tuple):
        if ctx_key not in self.weights:
            self.weights[ctx_key] = [self.init] * self.n_sounds

    def produce(self, ctx_key: tuple) -> int:
        """Sample a sound index proportional to weights for this context.
        Mutation: small chance of random sound to prevent ossification."""
        # Anti-ossification: mutation produces random sound (Kirby 2015)
        if random.random() < self.MUTATION_RATE:
            return random.randint(0, self.n_sounds - 1)
        self._ensure(ctx_key)
        w = self.weights[ctx_key]
        total = sum(w)
        if total <= 0:
            return random.randint(0, self.n_sounds - 1)
        r = random.random() * total
        cumulative = 0.0
        for i, wi in enumerate(w):
            cumulative += wi
            if r <= cumulative:
                return i
        return self.n_sounds - 1

    def reinforce(self, ctx_key: tuple, sound_idx: int, reward: float):
        """Increase weight for this context-sound pair.
        Lateral inhibition: competing sounds are suppressed (Franke 2014).
        Spill-over: also reinforce Hamming-distance neighbors for generalization."""
        self._ensure(ctx_key)
        self.weights[ctx_key][sound_idx] += reward
        # Lateral inhibition: suppress competing sounds for this context
        if reward > 0:
            inhibition = reward * self.LATERAL_INHIBITION
            for i in range(self.n_sounds):
                if i != sound_idx:
                    self.weights[ctx_key][i] = max(
                        self.init * 0.1,  # floor to prevent negative/zero weights
                        self.weights[ctx_key][i] - inhibition,
                    )
            # Spill-over to neighboring context bins
            self._spillover(ctx_key, sound_idx, reward)

    def _spillover(self, ctx_key: tuple, sound_idx: int, reward: float):
        """Propagate reward to Hamming distance 1 (30%) and 2 (10%) neighbors."""
        dims = len(ctx_key)
        # Distance 1: flip one dimension by +/-1
        for i in range(dims):
            for delta in (-1, 1):
                neighbor = list(ctx_key)
                neighbor[i] = ctx_key[i] + delta
                if neighbor[i] < 0:
                    continue
                nb = tuple(neighbor)
                self._ensure(nb)
                self.weights[nb][sound_idx] += reward * 0.3
        # Distance 2: flip two dimensions by +/-1 each
        for i in range(dims):
            for j in range(i + 1, dims):
                for di in (-1, 1):
                    for dj in (-1, 1):
                        neighbor = list(ctx_key)
                        neighbor[i] = ctx_key[i] + di
                        neighbor[j] = ctx_key[j] + dj
                        if neighbor[i] < 0 or neighbor[j] < 0:
                            continue
                        nb = tuple(neighbor)
                        self._ensure(nb)
                        self.weights[nb][sound_idx] += reward * 0.1

    def decay_all(self):
        """Multiply all weights by decay factor."""
        for ctx_key in self.weights:
            self.weights[ctx_key] = [w * self.decay for w in self.weights[ctx_key]]

    def top_sounds(self, ctx_key: tuple, n: int = 3) -> list[tuple[int, float]]:
        """Return top-n (sound_idx, weight) for a context key."""
        self._ensure(ctx_key)
        w = self.weights[ctx_key]
        total = sum(w)
        if total <= 0:
            return []
        indexed = [(i, wi / total) for i, wi in enumerate(w)]
        indexed.sort(key=lambda x: -x[1])
        return indexed[:n]


class GaussianProductionPolicy:
    """Gaussian ordinal policy: each position outputs a scalar μ∈[0,1],
    sound sampled from discretized Gaussian centered at μ*(n_sounds-1).
    Topographic similarity is built into the architecture — nearby contexts
    automatically produce nearby sounds.

    Only 4 parameters per position (vs 40 in categorical policy).
    """

    MUTATION_RATE = 0.05
    MAX_NORM = 8.0

    def __init__(self, n_sounds: int = 10, n_dims: int = 3, max_len: int = 2,
                 lr: float = 0.03, language_cfg: dict | None = None):
        self.n_sounds = n_sounds
        self.n_dims = n_dims
        self.max_len = max_len
        self.lr = lr
        lang = language_cfg or {}
        self.sigma_start = lang.get("sigma_start", 1.0)
        self.sigma = self.sigma_start
        self.sigma_min = lang.get("sigma_min", 0.5)
        self.sigma_anneal = lang.get("sigma_anneal_per_epoch", 0.02)
        self.weight_decay = lang.get("weight_decay", 0.9995)
        # Structured init: W @ normalized_ctx → sigmoid → [0,1] → sound index.
        # Position 0: high spatial → HIGH sounds (positive gradient)
        # Position 1: high social → LOW sounds (flipped gradient)
        # This creates position-specific encoding → high PosDis + high TopSim.
        self.W: list[list[float]] = []
        for pos in range(max_len):
            if pos == 0:
                self.W.append([1.5, 0.5, 0.3, 0.0])
            else:
                # Flipped gradient: high values → low sounds, centered by bias=1.0
                self.W.append([-1.5, -0.5, -0.3, 1.0])
        self.baseline: list[float] = [0.0] * max_len

    def split_context(self, full_ctx: tuple) -> list[tuple]:
        return [
            (full_ctx[0], full_ctx[1], full_ctx[2]),
            (full_ctx[3], full_ctx[4], full_ctx[5]),
        ]

    def _normalize(self, sub_ctx: tuple) -> list[float]:
        return [v / self.MAX_NORM for v in sub_ctx] + [1.0]

    def _forward_mu(self, pos: int, sub_ctx: tuple) -> float:
        """Compute μ = sigmoid(W @ x) ∈ [0, 1]."""
        x = self._normalize(sub_ctx)
        z = sum(self.W[pos][j] * x[j] for j in range(len(x)))
        # Clip to prevent overflow
        z = max(-10.0, min(10.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def _sound_probs(self, mu: float) -> list[float]:
        """Discretized Gaussian: P(sound=i) ∝ exp(-(i - μ*(n-1))² / 2σ²)."""
        center = mu * (self.n_sounds - 1)
        sig2 = 2.0 * self.sigma * self.sigma
        probs = [math.exp(-(i - center) ** 2 / sig2) for i in range(self.n_sounds)]
        total = sum(probs)
        return [p / total for p in probs] if total > 0 else [1.0 / self.n_sounds] * self.n_sounds

    def produce(self, full_ctx: tuple) -> list[int]:
        sub_ctxs = self.split_context(full_ctx)
        message = []
        for pos in range(self.max_len):
            if random.random() < self.MUTATION_RATE:
                message.append(random.randint(0, self.n_sounds - 1))
                continue
            mu = self._forward_mu(pos, sub_ctxs[pos])
            probs = self._sound_probs(mu)
            r = random.random()
            cumulative = 0.0
            chosen = self.n_sounds - 1
            for i, p in enumerate(probs):
                cumulative += p
                if r <= cumulative:
                    chosen = i
                    break
            message.append(chosen)
        return message

    def reinforce(self, full_ctx: tuple, message: list[int], rewards: list[float]):
        """REINFORCE for Gaussian policy.
        Gradient: push μ toward chosen sound if advantage > 0."""
        sub_ctxs = self.split_context(full_ctx)
        for pos in range(min(len(message), self.max_len)):
            reward = rewards[pos] if pos < len(rewards) else rewards[-1]
            self.baseline[pos] += 0.01 * (reward - self.baseline[pos])
            advantage = reward - self.baseline[pos]
            if abs(advantage) < 0.01:
                continue

            sub_ctx = sub_ctxs[pos]
            x = self._normalize(sub_ctx)
            mu = self._forward_mu(pos, sub_ctx)
            action = message[pos]

            # Correct REINFORCE for discretized Gaussian:
            # d log P(k)/dW ≈ (k - center)/σ² × (n-1) × sigmoid'(z) × x
            center = mu * (self.n_sounds - 1)  # mu_scaled ∈ [0, n-1]
            # Gaussian score: (k - center) / σ²
            score = (action - center) / (self.sigma ** 2)
            # Chain rule: d(center)/d(mu_raw) = (n-1), d(mu_raw)/d(z) = mu*(1-mu)
            grad_mu = score * (self.n_sounds - 1) * mu * (1 - mu)

            for j in range(len(x)):
                self.W[pos][j] += self.lr * advantage * grad_mu * x[j]

    def anneal_sigma(self):
        """Reduce σ by fixed step each epoch (exploration → exploitation)."""
        old = self.sigma
        self.sigma = max(self.sigma_min, self.sigma - self.sigma_anneal)
        return old, self.sigma

    def decay_all(self):
        """Gentle weight decay."""
        for pos in range(self.max_len):
            for j in range(len(self.W[pos])):
                self.W[pos][j] *= self.weight_decay

    def top_sounds(self, ctx_key: tuple, n: int = 3) -> list[tuple[int, float]]:
        """Return top sounds for a context (for logging)."""
        sub_ctxs = self.split_context(ctx_key)
        mu = self._forward_mu(0, sub_ctxs[0])
        probs = self._sound_probs(mu)
        indexed = sorted(enumerate(probs), key=lambda x: -x[1])
        return indexed[:n]


# Keep old NeuralProductionPolicy for backward compat reference
NeuralProductionPolicy = GaussianProductionPolicy


class Comprehension:
    """Bayesian comprehension: tracks P(context | sound heard).
    Handles both individual sounds and multi-sound sequences."""

    def __init__(self):
        # {sound_idx: {ctx_key: count}} — individual sounds
        self.counts: dict[int, dict[tuple, int]] = {}
        # {sound_seq_tuple: {ctx_key: count}} — full sequences
        self.seq_counts: dict[tuple, dict[tuple, int]] = {}

    def update(self, sound_idx: int, ctx_key: tuple):
        if sound_idx not in self.counts:
            self.counts[sound_idx] = {}
        self.counts[sound_idx][ctx_key] = self.counts[sound_idx].get(ctx_key, 0) + 1

    def update_sequence(self, sound_seq: tuple[int, ...], ctx_key: tuple):
        """Update sequence-level comprehension (enables compositionality)."""
        if len(sound_seq) < 2:
            return  # single sounds handled by update()
        if sound_seq not in self.seq_counts:
            self.seq_counts[sound_seq] = {}
        self.seq_counts[sound_seq][ctx_key] = self.seq_counts[sound_seq].get(ctx_key, 0) + 1

    def best_meaning(self, sound_idx: int) -> tuple | None:
        """Return the most likely context key for a sound, or None."""
        if sound_idx not in self.counts:
            return None
        ctx_counts = self.counts[sound_idx]
        if not ctx_counts:
            return None
        total = sum(ctx_counts.values())
        best_ctx = max(ctx_counts, key=ctx_counts.get)
        confidence = ctx_counts[best_ctx] / total
        if confidence > 0.4 and total >= 3:
            return best_ctx
        return None

    def best_seq_meaning(self, sound_seq: tuple[int, ...]) -> tuple | None:
        """Return most likely context for a full sequence, or None."""
        if sound_seq not in self.seq_counts:
            return None
        ctx_counts = self.seq_counts[sound_seq]
        if not ctx_counts:
            return None
        total = sum(ctx_counts.values())
        best_ctx = max(ctx_counts, key=ctx_counts.get)
        confidence = ctx_counts[best_ctx] / total
        if confidence > 0.35 and total >= 3:
            return best_ctx
        return None

    def has_rift_meaning(self, sound_idx: int) -> bool:
        """Check if this sound is associated with a rift-related context
        (dim 0 = low distance to rift, i.e. bin 0)."""
        meaning = self.best_meaning(sound_idx)
        if meaning is None:
            return False
        # dim 0 is distance to nearest rift; bin 0 = very close
        return meaning[0] == 0

    def has_rift_meaning_seq(self, sound_seq: tuple[int, ...]) -> bool:
        """Check if a sound sequence is associated with rift proximity."""
        meaning = self.best_seq_meaning(sound_seq)
        if meaning is not None and meaning[0] == 0:
            return True
        # Fallback: check first sound individually
        if sound_seq:
            return self.has_rift_meaning(sound_seq[0])
        return False


class SocialAgent(BlubAgent):
    """
    Discovers contexts from raw state, produces sounds via Roth-Erev policy,
    comprehends heard sounds via Bayesian updating.

    6 context dimensions:
      0: distance to nearest rift (Manhattan)
      1: richness_pct of nearest rift
      2: rift_type of nearest rift (gold=3, silver=2, copper=1, none=0)
      3: nearby lobster count
      4: nearby predator count
      5: heading to nearest rift (0-7 compass, 8=none)
    """

    def __init__(self, *args, ablation: dict | None = None,
                 language_cfg: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ablation = ablation or {}
        self.language_cfg = language_cfg or {}
        self.ctx_discoverer = ContextDiscoverer(dims=6, bins=4, warmup=50)
        self.production = GaussianProductionPolicy(
            n_sounds=len(SOUNDS), n_dims=3, max_len=2, lr=0.03,
            language_cfg=self.language_cfg,
        )
        self.comprehension = Comprehension()
        self.last_ctx: tuple = (0,) * 6
        self.last_sound_idx: int | None = None
        self.last_sound_seq: list[int] = []
        self.last_credits: float = 0.0
        self.last_group_hits: int = 0
        self.last_deaths: int = 0
        self.decay_counter: int = 0
        self.last_epoch: int = -1  # for epoch-change detection (sigma annealing)
        self.topo_bonus_scale: float = self.language_cfg.get("topo_bonus_scale", 4.0)
        # Topographic tracking: recent (message_tuple, ctx_key) pairs
        self._msg_history: list[tuple[tuple, tuple]] = []

    @staticmethod
    def _heading(relative: list[int]) -> int:
        """Convert relative [dx, dy] to compass heading 0-7."""
        dx, dy = relative
        if dx == 0 and dy == 0:
            return 8  # at target
        angle = math.atan2(dy, dx)
        # 8 compass bins: E=0, SE=1, S=2, SW=3, W=4, NW=5, N=6, NE=7
        idx = int((angle + math.pi) / (2 * math.pi) * 8) % 8
        return idx

    def _extract_raw(self, state: dict) -> list[float]:
        """Extract 6 raw dimensions from agent state."""
        rifts = state.get("nearby_rifts", [])
        if rifts:
            closest = min(rifts, key=lambda r: abs(r["relative"][0]) + abs(r["relative"][1]))
            d0 = abs(closest["relative"][0]) + abs(closest["relative"][1])
            d1 = closest.get("richness_pct", 0.5)
            rt = closest.get("rift_type", "silver")
            d2 = {"gold": 3, "silver": 2, "copper": 1}.get(rt, 0)
            d5 = self._heading(closest["relative"])
        else:
            d0, d1, d2, d5 = 99, 0, 0, 8

        d3 = len(state.get("nearby_lobsters", []))
        d4 = len(state.get("nearby_predators", []))

        return [d0, d1, d2, d3, d4, d5]

    def export_knowledge(self) -> dict:
        """Serialize neural weights + comprehension counts for deposit."""
        prod = {
            "W": self.production.W,
            "baseline": self.production.baseline,
        }
        comp: dict[str, dict[str, int]] = {}
        for sidx, ctx_counts in self.comprehension.counts.items():
            comp[str(sidx)] = {str(k): v for k, v in ctx_counts.items()}
        return {"production": prod, "comprehension": comp}

    def import_knowledge(self, data: dict, frac: float = 0.40):
        """Import weights + comprehension, blending with current."""
        prod = data.get("production", {})
        imported_W = prod.get("W")
        if imported_W and len(imported_W) == self.production.max_len:
            for pos in range(self.production.max_len):
                w = self.production.W[pos]
                iw = imported_W[pos]
                # Handle both flat (Gaussian: [w0,w1,w2,bias]) and nested (Neural: [[...], ...]) formats
                if isinstance(w, list) and w and not isinstance(w[0], list):
                    # Flat format: Gaussian policy W[pos] = [w0, w1, w2, bias]
                    for j in range(min(len(w), len(iw) if isinstance(iw, list) else 0)):
                        try:
                            val = iw[j] if not isinstance(iw[j], list) else iw[j][0]
                            self.production.W[pos][j] = w[j] * (1 - frac) + val * frac
                        except (IndexError, TypeError):
                            pass
                else:
                    # Nested format: Neural policy W[pos] = [[...], ...]
                    for i in range(min(len(w), len(iw))):
                        for j in range(min(len(w[i]), len(iw[i]))):
                            try:
                                self.production.W[pos][i][j] = (
                                    w[i][j] * (1 - frac) + iw[i][j] * frac
                                )
                            except (IndexError, TypeError):
                                pass

        comp = data.get("comprehension", {})
        for sidx_str, ctx_counts in comp.items():
            try:
                sidx = int(sidx_str)
            except ValueError:
                continue
            if sidx not in self.comprehension.counts:
                self.comprehension.counts[sidx] = {}
            for ctx_str, count in ctx_counts.items():
                try:
                    ctx_key = tuple(int(x.strip()) for x in ctx_str.strip("()").split(",") if x.strip())
                except (ValueError, AttributeError):
                    continue
                old = self.comprehension.counts[sidx].get(ctx_key, 0)
                self.comprehension.counts[sidx][ctx_key] = old + int(count * frac)
        n_w = sum(len(pos_w) if isinstance(pos_w, list) else 1 for pos_w in (imported_W or []))
        n_comp = sum(len(v) if isinstance(v, dict) else 1 for v in comp.values())
        print(f"[{self.name}] Imported knowledge: {n_w} weights, {n_comp} comp entries (frac={frac})")

    def partial_reset(self, retention: float = 0.20):
        """Partial reset: shrink weights toward zero, keep direction."""
        for pos in range(self.production.max_len):
            w = self.production.W[pos]
            if isinstance(w, list) and w and not isinstance(w[0], list):
                # Flat format (Gaussian policy)
                for j in range(len(w)):
                    self.production.W[pos][j] *= retention
            else:
                # Nested format (Neural policy)
                for i in range(len(w)):
                    for j in range(len(w[i])):
                        self.production.W[pos][i][j] *= retention
            self.production.baseline[pos] *= retention
        # Keep comprehension counts scaled down
        for sidx in self.comprehension.counts:
            for ctx_key in self.comprehension.counts[sidx]:
                self.comprehension.counts[sidx][ctx_key] = max(
                    1, int(self.comprehension.counts[sidx][ctx_key] * retention)
                )
        self.ctx_discoverer.count = 0
        self.last_credits = 0.0
        self.decay_counter = 0
        # Reset sigma to allow exploration in new life
        old_sigma = self.production.sigma
        self.production.sigma = self.production.sigma_start
        print(f"[{self.name}] Partial reset (retention={retention}, sigma {old_sigma:.3f}->{self.production.sigma:.3f})")

    def on_bootstrap(self, data: dict):
        """Called by base_agent after connect — import cultural cache (oblique transmission)."""
        self.import_knowledge(data, frac=self.ablation.get("inheritance_frac", 0.40))

    def on_mentor(self, data: dict):
        """Horizontal transmission: blend 15% from nearest experienced social agent."""
        self.import_knowledge(data, frac=0.15)

    async def on_death(self):
        """Deposit knowledge to cultural cache when killed by predator."""
        try:
            knowledge = self.export_knowledge()
            result = await self.deposit_knowledge(knowledge)
            print(f"[{self.name}] Death deposit: {result.get('message', 'ok')}")
        except Exception as e:
            print(f"[{self.name}] Death deposit failed: {e}")

    async def on_pre_retire(self):
        """Deposit knowledge before retirement, then partial reset."""
        try:
            knowledge = self.export_knowledge()
            result = await self.deposit_knowledge(knowledge)
            print(f"[{self.name}] Pre-retire deposit: {result.get('message', 'ok')}")
        except Exception as e:
            print(f"[{self.name}] Pre-retire deposit failed: {e}")
        self.partial_reset(retention=0.20)

    async def periodic_check(self, state: dict):
        """Every contribution_interval ticks, deposit knowledge if old enough."""
        interval = 200  # from config cultural_cache.contribution_interval
        min_age = 300   # from config cultural_cache.min_agent_age_to_contribute
        if self.local_age > 0 and self.local_age % interval == 0 and self.local_age >= min_age:
            try:
                knowledge = self.export_knowledge()
                result = await self.deposit_knowledge(knowledge)
                print(f"[{self.name}] Periodic deposit at local_age={self.local_age}: {result.get('message', 'ok')}")
            except Exception as e:
                print(f"[{self.name}] Periodic deposit failed: {e}")

    def on_retired(self):
        """Turnover: partial reset (on_pre_retire already deposited)."""
        pass

    def on_sounds_heard(self, sounds_events: list):
        """Update comprehension when sounds are received (individual + sequences)."""
        if not self.ablation.get("bayesian", True):
            return  # ablation: skip Bayesian comprehension
        raw = self._extract_raw(self.state)
        ctx_key = self.ctx_discoverer.get_context(raw)

        for event in sounds_events:
            seq_indices = []
            for sound_name in event["sounds"]:
                if sound_name in SOUNDS:
                    sound_idx = SOUNDS.index(sound_name)
                    self.comprehension.update(sound_idx, ctx_key)
                    seq_indices.append(sound_idx)
            # Also track the full sequence for compositionality
            if len(seq_indices) >= 2:
                self.comprehension.update_sequence(tuple(seq_indices), ctx_key)

    def think(self, state: dict) -> dict:
        # Extract context
        raw = self._extract_raw(state)
        ctx_key = self.ctx_discoverer.get_context(raw)

        # Differential reinforcement: pos0=spatial (credit delta), pos1=social (group/survival)
        current_credits = state.get("my_net_credits", 0)
        delta = current_credits - self.last_credits

        current_group = state.get("group_hits", 0)
        current_deaths = state.get("deaths_this_epoch", 0)
        group_delta = current_group - self.last_group_hits
        survived_predator = len(state.get("nearby_predators", [])) > 0 and state.get("alive", True)

        if self.last_sound_seq:
            # pos0: spatial reward from rift credits
            pos0_reward = max(0, delta)
            # pos1: social reward from group coordination + predator survival
            pos1_reward = group_delta * 5.0  # group hits are valuable
            if survived_predator:
                pos1_reward += 3.0  # surviving near predator rewards social encoding
            if current_deaths > self.last_deaths:
                pos1_reward = 0  # died = no social reward
            if pos0_reward > 0 or pos1_reward > 0:
                self.production.reinforce(
                    self.last_ctx, self.last_sound_seq,
                    [pos0_reward, pos1_reward],
                )

        self.last_credits = current_credits
        self.last_group_hits = current_group
        self.last_deaths = current_deaths

        # Sigma annealing at epoch boundary
        current_epoch = state.get("epoch", 0)
        if current_epoch != self.last_epoch and self.last_epoch >= 0:
            old_s, new_s = self.production.anneal_sigma()
            print(f"[{self.name}] sigma anneal: {old_s:.3f} -> {new_s:.3f} (epoch {current_epoch})")
        self.last_epoch = current_epoch

        # Decay weights every 50 ticks
        self.decay_counter += 1
        if self.decay_counter >= 50:
            self.production.decay_all()
            self.decay_counter = 0

        # Log every 100 ticks
        if state["tick"] % 100 == 0:
            top = self.production.top_sounds(ctx_key, n=3)
            top_info = {SOUNDS[i]: f"{p:.0%}" for i, p in top}
            w_info = [f"{w:.2f}" for w in self.production.W[0]]
            print(f"[{self.name}] tick={state['tick']} sig={self.production.sigma:.2f} top={top_info} W0={w_info}")

        speak: list[str] = []

        # Produce sounds based on current context
        # Always speak near rifts/predators, 30% elsewhere
        near_rift = bool(state.get("nearby_rifts"))
        near_predator = bool(state.get("nearby_predators"))

        if near_rift or near_predator or random.random() < 0.3:
            # Compositional production: always 2-sound message
            sound_indices = self.production.produce(ctx_key)
            speak = [SOUNDS[i] for i in sound_indices]
            self.last_sound_idx = sound_indices[0]
            self.last_sound_seq = sound_indices
            self.last_ctx = ctx_key

            # Topographic bonus: reward when context→sound mapping is smooth
            # (close contexts → close sounds, distant contexts → distant sounds)
            msg_t = tuple(sound_indices)
            self._msg_history.append((msg_t, ctx_key))
            if len(self._msg_history) > 200:
                self._msg_history.pop(0)
            if len(self._msg_history) >= 15:
                topo = self._topographic_bonus(ctx_key, sound_indices)
                if abs(topo) > 0.05:
                    self.production.reinforce(ctx_key, sound_indices, [topo, topo])
        else:
            self.last_sound_idx = None
            self.last_sound_seq = []
            self.last_ctx = ctx_key

        # Movement priority: flee predators > heard rift sounds > visible rift >
        #                    follow food pheromone > avoid danger pheromone > random walk
        if near_predator:
            pred = state["nearby_predators"][0]
            dx, dy = pred["relative"]
            # Also factor in danger pheromone — stronger flee if danger trails nearby
            move = "west" if dx > 0 else "east" if dx < 0 else "north" if dy > 0 else "south"
            return {"move": move, "speak": speak, "act": None}

        # If heard sounds that mean "near rift" — go toward the speaker
        # Check sequences first (more specific), then individual sounds
        for event in state.get("sounds_heard", []):
            seq_indices = tuple(
                SOUNDS.index(s) for s in event["sounds"] if s in SOUNDS
            )
            is_rift = False
            if len(seq_indices) >= 2:
                is_rift = self.comprehension.has_rift_meaning_seq(seq_indices)
            if not is_rift and seq_indices:
                is_rift = self.comprehension.has_rift_meaning(seq_indices[0])
            if is_rift:
                for lob in state.get("nearby_lobsters", []):
                    if lob["id"] == event["from"]:
                        dx, dy = lob["relative"]
                        if dx > 0:
                            move = "east"
                        elif dx < 0:
                            move = "west"
                        elif dy > 0:
                            move = "south"
                        elif dy < 0:
                            move = "north"
                        else:
                            move = "stay"
                        return {"move": move, "speak": speak, "act": None}

        # Move toward nearest rift
        rifts = state.get("nearby_rifts", [])
        if rifts:
            closest = min(rifts, key=lambda r: abs(r["relative"][0]) + abs(r["relative"][1]))
            dx, dy = closest["relative"]
            if dx > 0:
                move = "east"
            elif dx < 0:
                move = "west"
            elif dy > 0:
                move = "south"
            elif dy < 0:
                move = "north"
            else:
                move = "stay"
        else:
            # Priority 4: follow food pheromone gradient
            food_trails = state.get("nearby_food_trails", [])
            if food_trails:
                best = max(food_trails, key=lambda t: t["intensity"])
                dx, dy = best["dx"], best["dy"]
                if abs(dx) >= abs(dy):
                    move = "east" if dx > 0 else "west"
                else:
                    move = "south" if dy > 0 else "north"
            else:
                # Priority 5: avoid danger pheromone zones
                danger = state.get("nearby_danger_trails", [])
                if danger:
                    worst = max(danger, key=lambda t: t["intensity"])
                    dx, dy = worst["dx"], worst["dy"]
                    # Move AWAY from strongest danger
                    if abs(dx) >= abs(dy):
                        move = "west" if dx > 0 else "east"
                    else:
                        move = "north" if dy > 0 else "south"
                else:
                    move = random.choice(["north", "south", "east", "west"])

        return {"move": move, "speak": speak, "act": None}

    def _topographic_bonus(self, ctx_key: tuple, sound_indices: list[int]) -> float:
        """Reward for topographic consistency: close contexts → close sounds.
        Directly incentivizes TopSim at the individual agent level."""
        bonus = 0.0
        n = 0
        for prev_msg, prev_ctx in self._msg_history[-50:]:
            if prev_ctx == ctx_key:
                continue  # skip self-comparisons
            # Meaning distance: normalized Hamming
            m_dist = sum(1 for a, b in zip(ctx_key, prev_ctx) if a != b) / len(ctx_key)
            # Signal distance: ordinal distance per position, normalized
            if len(prev_msg) != len(sound_indices):
                continue
            s_dist = sum(abs(a - b) for a, b in zip(sound_indices, prev_msg)) / (
                (self.production.n_sounds - 1) * len(sound_indices)
            )
            # Reward correlated distances (close→close, far→far)
            # Penalize anti-correlation (close→far, far→close)
            # Base magnitudes scaled by topo_bonus_scale (default 4x → ~15% of main reward)
            scale = self.topo_bonus_scale
            if m_dist < 0.35:  # close meanings
                if s_dist < 0.25:
                    bonus += 2.0 * scale  # close sounds → strong reward
                elif s_dist > 0.6:
                    bonus -= 1.5 * scale  # distant sounds → penalty
            elif m_dist > 0.65:  # distant meanings
                if s_dist > 0.6:
                    bonus += 1.0 * scale  # distant sounds → moderate reward
                elif s_dist < 0.25:
                    bonus -= 0.5 * scale  # close sounds for distant meanings → small penalty
            n += 1
        return bonus / max(n, 1)

    def reset_language(self):
        """Reset all learned language state (for turnover/rebirth as naive)."""
        self.ctx_discoverer = ContextDiscoverer(dims=6, bins=4, warmup=50)
        self.production = GaussianProductionPolicy(
            n_sounds=len(SOUNDS), n_dims=3, max_len=2, lr=0.03,
            language_cfg=self.language_cfg,
        )
        self.comprehension = Comprehension()
        self.last_ctx = (0,) * 6
        self.last_sound_idx = None
        self.last_sound_seq = []
        self.last_credits = 0.0
        self.last_group_hits = 0
        self.last_deaths = 0
        self.decay_counter = 0
        self.last_epoch = -1
        print(f"[{self.name}] Language state RESET (turnover rebirth)")


if __name__ == "__main__":
    import asyncio

    agent = SocialAgent("social_solo", "http://localhost:8000")
    asyncio.run(agent.run())
