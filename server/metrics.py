"""
metrics.py — Language emergence metrics: TopSim, PosDis, MI, vocabulary size, economic delta.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path


class MetricsLogger:
    def __init__(self, log_file: str = "metrics_log.jsonl"):
        self.log_file = Path(log_file)
        # Each observation: (sound_seq tuple, context_key tuple, reward float)
        self.observations: list[tuple[tuple, tuple, float]] = []
        self.latest: dict = {}

    def record(self, sound_seq: list[str], context_key: tuple, reward: float):
        """Called each tick for each sound event with server-side context."""
        self.observations.append((tuple(sound_seq), context_key, reward))

    def compute(self, tick: int, epoch: int, lobsters,
                csr_heard: int = 0, csr_successes: int = 0) -> dict:
        """Compute all metrics. Called every 100 ticks."""
        csr = round(csr_successes / csr_heard, 4) if csr_heard > 0 else 0.0
        result = {
            "tick": tick,
            "epoch": epoch,
            "vocabulary_size": self._vocab_size(),
            "top_sim": self._topographic_similarity(),
            "pos_dis": self._positional_disentanglement(),
            "bos_dis": self._bag_of_symbols_disentanglement(),
            "mutual_info": self._mutual_information(),
            "economic_delta": self._economic_delta(lobsters),
            "csr": csr,
            "csr_heard": csr_heard,
            "csr_successes": csr_successes,
        }
        self.latest = result

        # Append to JSONL file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(result) + "\n")

        return result

    def _vocab_size(self) -> int:
        """Count distinct sound sequences that appear with >60% consistency for one context."""
        if not self.observations:
            return 0

        # sound_seq -> {context_key: count}
        seq_ctx: dict[tuple, dict[tuple, int]] = defaultdict(lambda: defaultdict(int))
        for seq, ctx, _ in self.observations:
            seq_ctx[seq][ctx] += 1

        vocab = 0
        for seq, ctx_counts in seq_ctx.items():
            total = sum(ctx_counts.values())
            if total < 3:
                continue
            best_count = max(ctx_counts.values())
            if best_count / total > 0.6:
                vocab += 1
        return vocab

    def _topographic_similarity(self) -> float:
        """Spearman correlation between meaning distances and signal distances.
        Sample up to 200 (meaning, signal) pairs."""
        if len(self.observations) < 10:
            return 0.0

        # Build (context_key, sound_seq) pairs — deduplicate by most common mapping
        ctx_to_seq: dict[tuple, dict[tuple, int]] = defaultdict(lambda: defaultdict(int))
        for seq, ctx, _ in self.observations:
            ctx_to_seq[ctx][seq] += 1

        pairs = []
        for ctx, seqs in ctx_to_seq.items():
            best_seq = max(seqs, key=seqs.get)
            pairs.append((ctx, best_seq))

        if len(pairs) < 4:
            return 0.0

        # Sample up to 200 pairs
        if len(pairs) > 200:
            pairs = random.sample(pairs, 200)

        # Compute all pairwise distances
        meaning_dists = []
        signal_dists = []
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                md = _hamming(pairs[i][0], pairs[j][0])
                sd = _levenshtein(pairs[i][1], pairs[j][1])
                meaning_dists.append(md)
                signal_dists.append(sd)

        if not meaning_dists:
            return 0.0

        return _spearman(meaning_dists, signal_dists)

    def _positional_disentanglement(self) -> float:
        """For multi-sound messages: how well each position encodes a single dimension."""
        if len(self.observations) < 20:
            return 0.0

        # Only use multi-sound sequences
        multi = [(seq, ctx) for seq, ctx, _ in self.observations if len(seq) > 1]
        if len(multi) < 10:
            return 0.0

        max_len = max(len(seq) for seq, _ in multi)
        if max_len < 2:
            return 0.0

        n_dims = len(multi[0][1]) if multi else 0
        if n_dims == 0:
            return 0.0

        scores = []
        for j in range(min(max_len, 5)):  # up to 5 positions
            # Collect (sound_at_j, context) pairs
            pos_data = [(seq[j], ctx) for seq, ctx in multi if len(seq) > j]
            if len(pos_data) < 5:
                continue

            # H(sound_j)
            h_sj = _entropy_list([s for s, _ in pos_data])
            if h_sj < 0.01:
                continue

            # MI(sound_j, dim_k) for each k
            mis = []
            for k in range(n_dims):
                mi = _mi_two_lists(
                    [s for s, _ in pos_data],
                    [c[k] for _, c in pos_data],
                )
                mis.append(mi)

            mis.sort(reverse=True)
            if len(mis) >= 2 and h_sj > 0:
                gap = (mis[0] - mis[1]) / h_sj
                scores.append(max(0.0, min(1.0, gap)))

        return sum(scores) / len(scores) if scores else 0.0

    def _bag_of_symbols_disentanglement(self) -> float:
        """BosDis (Chaabouni 2020): for each context dimension, how well is it
        predicted by the BAG (unordered set) of symbols in the message?
        Complementary to PosDis — measures order-independent encoding."""
        if len(self.observations) < 20:
            return 0.0

        multi = [(seq, ctx) for seq, ctx, _ in self.observations if len(seq) > 1]
        if len(multi) < 10:
            return 0.0

        n_dims = len(multi[0][1]) if multi else 0
        if n_dims == 0:
            return 0.0

        scores = []
        for k in range(n_dims):
            # Build bag-of-symbols representation for each message
            bags = [frozenset(seq) for seq, _ in multi]
            dims_vals = [ctx[k] for _, ctx in multi]

            if len(set(dims_vals)) < 2:
                continue

            # H(bag)
            h_bag = _entropy_list(bags)
            if h_bag < 0.01:
                continue

            # MI(bag, dim_k)
            mi_k = _mi_two_lists(bags, dims_vals)

            # MI(bag, dim_j) for all other dims j != k
            other_mis = []
            for j in range(n_dims):
                if j == k:
                    continue
                other_vals = [ctx[j] for _, ctx in multi]
                other_mis.append(_mi_two_lists(bags, other_vals))

            if other_mis and h_bag > 0:
                max_other = max(other_mis)
                gap = (mi_k - max_other) / h_bag
                scores.append(max(0.0, min(1.0, gap)))

        return round(sum(scores) / len(scores), 4) if scores else 0.0

    def _mutual_information(self) -> float:
        """MI(signal; context) from joint frequency table."""
        if len(self.observations) < 10:
            return 0.0

        joint: dict[tuple, int] = defaultdict(int)
        sig_count: dict[tuple, int] = defaultdict(int)
        ctx_count: dict[tuple, int] = defaultdict(int)
        total = 0

        for seq, ctx, _ in self.observations:
            joint[(seq, ctx)] += 1
            sig_count[seq] += 1
            ctx_count[ctx] += 1
            total += 1

        if total == 0:
            return 0.0

        mi = 0.0
        for (s, c), count in joint.items():
            p_sc = count / total
            p_s = sig_count[s] / total
            p_c = ctx_count[c] / total
            if p_s > 0 and p_c > 0 and p_sc > 0:
                mi += p_sc * math.log2(p_sc / (p_s * p_c))

        return round(mi, 4)

    def _economic_delta(self, lobsters) -> float:
        """Compare avg credits of lobsters that spoke recently vs those that didn't."""
        if not lobsters:
            return 0.0

        # Track who spoke in recent observations (last 100)
        recent_speakers = set()
        for seq, ctx, _ in self.observations[-100:]:
            pass  # We don't store speaker IDs in observations

        # Use lobster state directly: those with speaking > 0 credits
        speakers = [l for l in lobsters.values() if l.credits_earned > l.credits_spent and l.agent_type == "social"]
        silent = [l for l in lobsters.values() if l.agent_type in ("greedy", "random")]

        if not speakers or not silent:
            return 0.0

        avg_speakers = sum(l.credits_earned - l.credits_spent for l in speakers) / len(speakers)
        avg_silent = sum(l.credits_earned - l.credits_spent for l in silent) / len(silent)

        return round(avg_speakers - avg_silent, 2)

    def reset(self):
        self.observations.clear()
        self.latest = {}


# --- Helper functions ---

def _hamming(a: tuple, b: tuple) -> int:
    """Hamming distance between two tuples."""
    return sum(1 for x, y in zip(a, b) if x != y)


def _levenshtein(a: tuple, b: tuple) -> int:
    """Levenshtein distance between two sequences."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def _rank(values: list[float]) -> list[float]:
    """Assign ranks to values (average rank for ties)."""
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def _spearman(x: list, y: list) -> float:
    """Spearman rank correlation."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    rx = _rank([float(v) for v in x])
    ry = _rank([float(v) for v in y])
    n = len(rx)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mean_rx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - mean_ry) ** 2 for i in range(n)))
    if den_x == 0 or den_y == 0:
        return 0.0
    return round(num / (den_x * den_y), 4)


def _entropy_list(values: list) -> float:
    """Shannon entropy of a list of discrete values."""
    counts: dict = defaultdict(int)
    for v in values:
        counts[v] += 1
    total = len(values)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            h -= p * math.log2(p)
    return h


def _mi_two_lists(xs: list, ys: list) -> float:
    """Mutual information between two aligned lists of discrete values."""
    n = len(xs)
    if n == 0:
        return 0.0
    joint: dict[tuple, int] = defaultdict(int)
    x_count: dict = defaultdict(int)
    y_count: dict = defaultdict(int)
    for x, y in zip(xs, ys):
        joint[(x, y)] += 1
        x_count[x] += 1
        y_count[y] += 1
    mi = 0.0
    for (x, y), c in joint.items():
        p_xy = c / n
        p_x = x_count[x] / n
        p_y = y_count[y] / n
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * math.log2(p_xy / (p_x * p_y))
    return mi
