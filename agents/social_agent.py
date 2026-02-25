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

    def __init__(self, n_sounds: int = 30, init_weight: float = 1.0, decay: float = 0.95):
        self.weights: dict[tuple, list[float]] = {}
        self.n_sounds = n_sounds
        self.init = init_weight
        self.decay = decay

    def _ensure(self, ctx_key: tuple):
        if ctx_key not in self.weights:
            self.weights[ctx_key] = [self.init] * self.n_sounds

    def produce(self, ctx_key: tuple) -> int:
        """Sample a sound index proportional to weights for this context."""
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
        Spill-over: also reinforce Hamming-distance neighbors for generalization."""
        self._ensure(ctx_key)
        self.weights[ctx_key][sound_idx] += reward
        # Spill-over to neighboring context bins
        if reward > 0:
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

    def __init__(self, *args, ablation: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ablation = ablation or {}
        self.ctx_discoverer = ContextDiscoverer(dims=6, bins=4, warmup=50)
        self.production = ProductionPolicy(n_sounds=len(SOUNDS), init_weight=1.0, decay=0.97)
        self.comprehension = Comprehension()
        self.last_ctx: tuple = (0,) * 6
        self.last_sound_idx: int | None = None
        self.last_sound_seq: tuple = ()
        self.last_credits: float = 0.0
        self.decay_counter: int = 0

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

    def _sequence_length(self, ctx_key: tuple) -> int:
        """Determine how many sounds to produce based on context complexity.
        Simple contexts (open water) → 1 sound.
        Rich contexts (near specific rift type + crowded + heading) → 2-3 sounds.
        This enables compositionality: position in sequence carries meaning."""
        complexity = 0
        # dim 0: near rift (bin 0 = very close)
        if ctx_key[0] == 0:
            complexity += 1
        # dim 2: rift type known (gold=3, silver=2, copper=1)
        if ctx_key[2] > 0:
            complexity += 1
        # dim 3: crowded (high bin)
        if ctx_key[3] >= 2:
            complexity += 1
        # dim 4: predator nearby
        if ctx_key[4] >= 1:
            complexity += 1
        # 0-1 complexity → 1 sound, 2 → 2 sounds, 3+ → 3 sounds
        if complexity <= 1:
            return 1
        if complexity == 2:
            return 2
        return 3

    def on_retired(self):
        """Turnover: reset all language state when retired by server."""
        self.reset_language()

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

        # Reinforce last action based on credit delta
        current_credits = state.get("my_net_credits", 0)
        delta = current_credits - self.last_credits
        if self.last_sound_idx is not None and delta > 0:
            self.production.reinforce(self.last_ctx, self.last_sound_idx, delta)
        self.last_credits = current_credits

        # Decay weights every 50 ticks
        self.decay_counter += 1
        if self.decay_counter >= 50:
            self.production.decay_all()
            self.decay_counter = 0

        # Log every 100 ticks
        if state["tick"] % 100 == 0:
            active_contexts = len(self.production.weights)
            top_ctx = sorted(
                self.production.weights.keys(),
                key=lambda k: max(self.production.weights[k]),
                reverse=True,
            )[:3] if self.production.weights else []
            top_info = {}
            for ck in top_ctx:
                top_s = self.production.top_sounds(ck, 1)
                if top_s:
                    top_info[str(ck)] = f"{SOUNDS[top_s[0][0]]}({top_s[0][1]:.0%})"
            print(f"[{self.name}] tick={state['tick']} contexts={active_contexts} top={top_info}")

        speak: list[str] = []

        # Produce sounds based on current context
        # Always speak near rifts/predators, 30% elsewhere
        near_rift = bool(state.get("nearby_rifts"))
        near_predator = bool(state.get("nearby_predators"))

        if near_rift or near_predator or random.random() < 0.3:
            # Multi-sound production: context complexity determines sequence length
            # Rich contexts (rift type, crowded, heading) → 2-3 sounds for compositionality
            seq_len = self._sequence_length(ctx_key)
            sound_indices = []
            for _ in range(seq_len):
                idx = self.production.produce(ctx_key)
                sound_indices.append(idx)
            speak = [SOUNDS[i] for i in sound_indices]
            self.last_sound_idx = sound_indices[0]  # primary sound for reinforcement
            self.last_sound_seq = tuple(sound_indices)
            self.last_ctx = ctx_key
        else:
            self.last_sound_idx = None
            self.last_sound_seq = ()
            self.last_ctx = ctx_key

        # Movement: flee predators > respond to heard rift sounds > go to rift > wander
        if near_predator:
            pred = state["nearby_predators"][0]
            dx, dy = pred["relative"]
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
            move = random.choice(["north", "south", "east", "west"])

        return {"move": move, "speak": speak, "act": None}

    def reset_language(self):
        """Reset all learned language state (for turnover/rebirth as naive)."""
        self.ctx_discoverer = ContextDiscoverer(dims=6, bins=4, warmup=50)
        self.production = ProductionPolicy(n_sounds=len(SOUNDS), init_weight=1.0, decay=0.97)
        self.comprehension = Comprehension()
        self.last_ctx = (0,) * 6
        self.last_sound_idx = None
        self.last_sound_seq = ()
        self.last_credits = 0.0
        self.decay_counter = 0
        print(f"[{self.name}] Language state RESET (turnover rebirth)")


if __name__ == "__main__":
    import asyncio

    agent = SocialAgent("social_solo", "http://localhost:8000")
    asyncio.run(agent.run())
